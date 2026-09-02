"""Compile an exported model with ST Edge AI Core the way the Neuro N6
Arduino core does, and report what the build will see.

This answers PK-1.3 (the weights budget) and, more importantly, turns
"should deploy" into "compiles for this NPU": the epoch report says how
much of the network landed on hardware, and the memory table says what
it costs against the external-flash weights pool and npuRAM.

Best-effort by design: stedgeai is a separate ST install, so when it is
absent the export still succeeds and report.json says `verified: false`
with the reason. When it IS present and the compile fails, the export
fails loudly - a model the target cannot build is not a deliverable.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Mirrors tools/neuron6_model_gen.py in the Arduino core exactly: the one
# blessed compiler configuration known to run on the board.
NEURAL_ART_OPTIONS = ("--native-float --cache-maintenance --Ocache-opt "
                      "--enable-virtual-mem-pools --Os --optimization 3 "
                      "--Omax-ca-pipe 4 --Oauto-sched --all-buffers-info")
GENERATE_INTERFACE_OPTIONS = ("--input-data-type", "uint8", "--inputs-ch-position", "chlast")

MPOOL = Path(__file__).parent / "neuron6" / "NeuroN6_MemPool.mpool"

# The weights pool the mempool declares runs 0x90300000..0x91D80000
# (26.5 MB), but the last 1 MB is also OTA app slot B
# (n6_ota_layout.h: N6_OTA_SLOT_B_OFF 0x01C80000). The safe ceiling for
# packed weights is therefore 25.5 MB; headroom is reported against that.
POOL_BASE = 0x90300000
SAFE_POOL_END = 0x91C80000
SAFE_POOL_BYTES = SAFE_POOL_END - POOL_BASE

COMPILE_TIMEOUT_S = 15 * 60


def find_stedgeai() -> str | None:
    """Same search order as the core: env override, then ST's default
    install roots (newest version first), then PATH."""
    env = os.environ.get("NEURON6_STEDGEAI")
    if env:
        return env if os.path.isfile(env) else None
    if sys.platform.startswith("win"):
        pats = [r"C:\ST\STEdgeAI\*\Utilities\windows\stedgeai.exe",
                r"C:\Program Files\STMicroelectronics\STEdgeAI\*\Utilities\windows\stedgeai.exe"]
    elif sys.platform == "darwin":
        pats = ["/Applications/STEdgeAI/*/Utilities/mac/stedgeai",
                os.path.expanduser("~/STEdgeAI/*/Utilities/mac/stedgeai")]
    else:
        pats = [os.path.expanduser("~/STEdgeAI/*/Utilities/linux/stedgeai"),
                "/opt/ST/STEdgeAI/*/Utilities/linux/stedgeai"]
    for p in pats:
        hits = sorted(glob.glob(p), reverse=True)
        if hits:
            return hits[0]
    return shutil.which("stedgeai")


_UNIT = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}


def _to_bytes(num: str, unit: str) -> int:
    return int(round(float(num) * _UNIT[unit.lower()]))


def parse_generate_report(text: str) -> dict:
    """Pull the numbers PixelKit reports out of *_generate_report.txt.

    Tolerant of layout drift: every field is optional and missing ones
    are simply absent, so a newer stedgeai never breaks an export."""
    out: dict = {}
    m = re.search(r"Total number of epochs\s+(\d+)", text)
    if m:
        out["epochsTotal"] = int(m.group(1))
    m = re.search(r"pure software \(SW\) epochs\s+(\d+)", text)
    if m:
        out["epochsSoftware"] = int(m.group(1))
    m = re.search(r"hybrid epochs[^\d\n]*(\d+)", text)
    if m:
        out["epochsHybrid"] = int(m.group(1))
    m = re.search(r"pure hardware \(HW or EC\) epochs\s+(\d+)", text)
    if m:
        out["epochsHardware"] = int(m.group(1))
    m = re.search(r"^model:\s+macc=([\d,]+)", text, re.M)
    if m:
        out["macc"] = int(m.group(1).replace(",", ""))

    # Memory table rows look like:
    #   OCTAFLASH  [0x90300000 - 0x91D80000]:  898.188 kB /  26.500 MB ( 3.31 % used) -- weights: ...
    #   npuRAM     [0x34200000 - 0x343C0000]:  448.000 kB /   1.750 MB (25.00 % used) -- ...
    row = re.compile(
        r"^\s*(\w+)\s+\[0x[0-9A-Fa-f]+ - 0x[0-9A-Fa-f]+\]:\s+([\d.]+)\s*(B|kB|MB|GB)\s*/\s+([\d.]+)\s*(B|kB|MB|GB)",
        re.M)
    for name, used, uu, cap, cu in row.findall(text):
        if name == "OCTAFLASH":
            out["weightsBytes"] = _to_bytes(used, uu)
            out["poolBytes"] = _to_bytes(cap, cu)
        elif name == "npuRAM":
            out["activationsBytes"] = _to_bytes(used, uu)
            out["npuRamBytes"] = _to_bytes(cap, cu)
    # Software epochs by op type: the report's "what fell off the NPU".
    sw_ops = re.findall(r"^\|\s*epoch_\d+\s*\|\s*SW\s*\|\s*(\w+)", text, re.M)
    if sw_ops:
        counts: dict[str, int] = {}
        for op in sw_ops:
            counts[op] = counts.get(op, 0) + 1
        out["softwareOps"] = counts
    return out


def verify(model_path: Path, keep_dir: Path, name: str = "m") -> dict:
    """Run `stedgeai generate` exactly as the core's pragma pipeline does.

    The compiler runs in a SHORT temp directory: its tracer workspace
    nests several levels deep and, under a dataset path, overflowed
    Windows' 260-char MAX_PATH ("The filename or extension is too long").
    Only stedgeai.log and the generate report are copied into `keep_dir`
    for a human debugging a build. `name` is the generated C symbol,
    which is discarded, so it stays short too.

    Returns a JSON-ready dict. `verified` is True only when the compiler
    ran AND succeeded. Never raises for a missing tool; raises
    RuntimeError when the tool is present and rejects the model."""
    exe = find_stedgeai()
    if not exe:
        return {"verified": False, "available": False,
                "reason": "ST Edge AI Core (stedgeai) not installed; install it from st.com "
                          "or set NEURON6_STEDGEAI to verify exports against the NPU"}
    if not MPOOL.is_file():
        # A packaging fault, not a model fault: say so rather than crash
        # the export (the desktop build copies engine/gd by extension).
        return {"verified": False, "available": True,
                "reason": f"Neuro N6 memory-pool file missing from this install ({MPOOL.name})"}
    keep_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pkn6-"))
    try:
        out_dir, ws_dir = work / "out", work / "ws"
        out_dir.mkdir()
        ws_dir.mkdir()
        shutil.copyfile(MPOOL, work / MPOOL.name)
        # Copy the model in too, so every path the tool derives is short.
        model_local = work / "model.onnx"
        shutil.copyfile(model_path, model_local)
        profile = {"Globals": {}, "Profiles": {"neuron6": {
            "memory_pool": "./" + MPOOL.name, "options": NEURAL_ART_OPTIONS}}}
        profile_path = work / "neuron6_neural_art.json"
        profile_path.write_text(json.dumps(profile, indent=1), "utf-8")

        cmd = [exe, "generate", "-m", str(model_local), "--target", "stm32n6",
               "--st-neural-art", "neuron6@" + str(profile_path).replace("\\", "/"),
               "-n", name, "--enable-epoch-controller",
               "--output", str(out_dir), "--workspace", str(ws_dir),
               *GENERATE_INTERFACE_OPTIONS]
        t0 = time.time()
        log_path = work / "stedgeai.log"
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                      cwd=str(work), timeout=COMPILE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"stedgeai generate timed out after {COMPILE_TIMEOUT_S // 60} min")
        log = log_path.read_text("utf-8", errors="replace") if log_path.exists() else ""
        shutil.copyfile(log_path, keep_dir / "stedgeai.log")
        report_path = next(iter(out_dir.glob("*_generate_report.txt")), None)
        if report_path:
            shutil.copyfile(report_path, keep_dir / "generate_report.txt")
        if proc.returncode != 0:
            err = next((ln.strip() for ln in log.splitlines() if "ERROR" in ln.upper()), "")
            tail = "\n".join(log.strip().splitlines()[-6:])
            raise RuntimeError(
                f"ST Edge AI rejected the model (exit {proc.returncode}): {err or tail}")

        info: dict = {"verified": True, "available": True,
                      "seconds": round(time.time() - t0, 1)}
        m = re.search(r"ST Edge AI Core\s+(v[\w.\-]+)", log)
        if m:
            info["stedgeai"] = m.group(1)
        if report_path:
            info.update(parse_generate_report(report_path.read_text("utf-8", errors="replace")))
        # The packed weights blob is the authoritative footprint: it is the
        # exact byte count the core's pragma pipeline packs into flash.
        blob = next(iter(out_dir.glob("*_atonbuf.xSPI1.raw")), None)
        if blob:
            info["weightsBytes"] = blob.stat().st_size
        if "weightsBytes" in info:
            info["poolSafeBytes"] = SAFE_POOL_BYTES
            info["poolRemainingBytes"] = max(0, SAFE_POOL_BYTES - info["weightsBytes"])
            info["poolUsedPct"] = round(100.0 * info["weightsBytes"] / SAFE_POOL_BYTES, 2)
        return info
    finally:
        shutil.rmtree(work, ignore_errors=True)
