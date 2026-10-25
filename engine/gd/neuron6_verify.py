"""Compile an exported model with ST Edge AI Core as the Neuro N6 Arduino core does."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Must match tools/neuron6_model_gen.py in the Arduino core.
NEURAL_ART_OPTIONS = ("--native-float --cache-maintenance --Ocache-opt "
                      "--enable-virtual-mem-pools --Os --optimization 3 "
                      "--Omax-ca-pipe 4 --Oauto-sched --all-buffers-info")
GENERATE_INTERFACE_OPTIONS = ("--input-data-type", "uint8", "--inputs-ch-position", "chlast")

# The Neuro N6 memory map, as the Arduino core builds it: ST Edge AI's own
# stm32n6.mpool with the board's pools. Each pool is (fname, name, the
# template pool whose figures it takes, changes to them, offset, size).
MPOOL_NAME = "NeuroN6_MemPool.mpool"
MPOOL_TEMPLATE = ("scripts", "N6_scripts", "my_mpools", "stm32n6.mpool")
ONCHIP_SRAM_KB = 1792
POOLS = (
    ("AXIFLEXMEM", "flexMEM", "AXIFLEXMEM", {}, "0x34000000", ("0", "KBYTES")),
    ("AXISRAM1", "cpuRAM1", "AXISRAM1", {}, "0x34080000", ("0", "KBYTES")),
    ("AXISRAM2", "cpuRAM2", "AXISRAM2", {}, "0x34100000", ("0", "KBYTES")),
    ("AXISRAM3", "npuRAM", "AXISRAM3", {}, "0x34200000", ("1792", "KBYTES")),
    ("xSPI2", "PSRAM", "xSPI1", {"constants_preferred": "false"}, "0x71000000", ("14", "MBYTES")),
    # Ends at 0x91C80000, where OTA slot B begins.
    ("xSPI1", "OCTAFLASH", "xSPI2", {"byteWidth": 2, "freqRatio": 5.0}, "0x90300000", ("26112", "KBYTES")),
)

COMPILE_TIMEOUT_S = 15 * 60


def mpool_template(exe: str | None) -> Path | None:
    """ST Edge AI's stm32n6.mpool in the install that holds `exe`, or None."""
    if not exe:
        return None
    for root in list(Path(exe).resolve().parents)[:4]:
        p = root.joinpath(*MPOOL_TEMPLATE)
        if p.is_file():
            return p
    return None


def build_mpool(template: Path) -> dict:
    """The Neuro N6 memory map from ST's stm32n6.mpool. Raises ValueError when
    the template lacks a pool the map takes its figures from."""
    t = json.loads(template.read_text("utf-8"))
    by_fname = {p.get("fname"): p for p in t["memory"]["mempools"]}
    pools = []
    for fname, name, src, changes, offset, (size, unit) in POOLS:
        if src not in by_fname:
            raise ValueError(f"{template.name} has no {src} pool")
        pools.append({"fname": fname, "name": name,
                      "fformat": by_fname[src].get("fformat", "FORMAT_RAW"),
                      "prop": {**by_fname[src]["prop"], **changes},
                      "offset": {"value": offset, "magnitude": "BYTES"},
                      "size": {"value": size, "magnitude": unit}})
    params = [p for p in (t.get("params") or {}).get("param", [])
              if p.get("paramname") != "max_onchip_sram_size"]
    params.insert(0, {"paramname": "max_onchip_sram_size", "value": str(ONCHIP_SRAM_KB),
                      "magnitude": "KBYTES"})
    return {**t, "params": {**(t.get("params") or {}), "param": params},
            "memory": {**t["memory"], "mempools": pools}}


def weights_pool_bytes(mpool: Path | None = None) -> int | None:
    """Bytes in the xSPI1 external-flash weights pool: the board's, or that of
    the .mpool file `mpool`; None when the file is unreadable."""
    if mpool is None:
        pools = [{"fname": f, "size": {"value": v, "magnitude": u}}
                 for f, _name, _src, _changes, _offset, (v, u) in POOLS]
    else:
        try:
            pools = json.loads(mpool.read_text("utf-8"))["memory"]["mempools"]
        except (OSError, ValueError, KeyError):
            return None
    try:
        p = next(p for p in pools if p.get("fname") == "xSPI1")
        unit = {"BYTES": 1, "KBYTES": 1024, "MBYTES": 1024 * 1024}[p["size"]["magnitude"].upper()]
        return int(p["size"]["value"], 0) * unit
    except (ValueError, KeyError, StopIteration):
        return None


# ll_aton_version.h in the Neuro N6 core and in an ST Edge AI install.
CORE_LL_ATON = ("cores", "NeuroN6", "AI_Runtime", "Npu", "ll_aton", "ll_aton_version.h")
INSTALL_LL_ATON = ("Middlewares", "ST", "AI", "Npu", "ll_aton", "ll_aton_version.h")


def ll_aton_version(header: Path) -> tuple[int, ...] | None:
    """(major, minor, micro, dev) from an ll_aton_version.h, or None."""
    try:
        text = header.read_text("utf-8", errors="replace")
    except OSError:
        return None
    found = [re.search(rf"LL_ATON_VERSION_{k}\s+\((\d+)\)", text)
             for k in ("MAJOR", "MINOR", "MICRO", "DEV")]
    if not all(found):
        return None
    return tuple(int(m.group(1)) for m in found)


def _version_key(name: str) -> list:
    """Sort key of a version folder name, the core's rule: numbers compare as
    numbers and a release sorts above its text suffixes, so 10.1 > 10.0.1 >
    10.0 > 10.0-beta > 9.0."""
    parts = [(2, int(x), "") if x.isdigit() else (0, 0, x) for x in re.split(r"[.\-_]", name)]
    return parts + [(1, 0, "")]


def core_ll_aton() -> tuple[int, ...] | None:
    """ll_aton version of the Neuro N6 core a board build uses: NEURON6_CORE,
    the sketchbook's hardware folder, then Boards Manager installs, newest
    first. None when no core is found."""
    home = Path.home()
    cores = [Path(os.environ["NEURON6_CORE"])] if os.environ.get("NEURON6_CORE") else []
    user = os.environ.get("ARDUINO_DIRECTORIES_USER")
    for sketchbook in [Path(user)] if user else [home / "Documents" / "Arduino", home / "Arduino"]:
        cores.append(sketchbook / "hardware" / "Ohmlab" / "stm32n6")
    data = os.environ.get("ARDUINO_DIRECTORIES_DATA")
    for d in [Path(data)] if data else [Path(os.environ.get("LOCALAPPDATA", home)) / "Arduino15",
                                         home / "Library" / "Arduino15", home / ".arduino15"]:
        cores += sorted((d / "packages" / "Ohmlab" / "hardware" / "stm32n6").glob("*"),
                        key=lambda p: _version_key(p.name), reverse=True)
    for core in cores:
        v = ll_aton_version(core.joinpath(*CORE_LL_ATON))
        if v is not None:
            return v
    return None


def _stedgeai_patterns() -> list[str]:
    """ST's install locations, in the core's order."""
    if sys.platform.startswith("win"):
        return [r"C:\ST\STEdgeAI\*\Utilities\windows\stedgeai.exe",
                r"C:\Program Files\STMicroelectronics\STEdgeAI\*\Utilities\windows\stedgeai.exe"]
    if sys.platform == "darwin":
        return ["/Applications/STEdgeAI/*/Utilities/mac/stedgeai",
                os.path.expanduser("~/STEdgeAI/*/Utilities/mac/stedgeai")]
    return ["/opt/ST/STEdgeAI/*/Utilities/linux/stedgeai",
            os.path.expanduser("~/STEdgeAI/*/Utilities/linux/stedgeai")]


def find_stedgeai() -> str | None:
    """Find stedgeai as the core's tools/neuron6_model_gen.py does: of ST's
    install roots, newest version first, then PATH, the first install whose
    ll_aton matches the core's, else the first found. NEURON6_STEDGEAI
    overrides the search."""
    env = os.environ.get("NEURON6_STEDGEAI")
    if env:
        return env if os.path.isfile(env) else None
    cands = []
    for p in _stedgeai_patterns():
        # <version>/Utilities/<os>/stedgeai
        cands += sorted(glob.glob(p), key=lambda c: _version_key(Path(c).parents[2].name), reverse=True)
    which = shutil.which("stedgeai")
    if which:
        cands.append(which)
    want = core_ll_aton()
    first = None
    for c in dict.fromkeys(os.path.abspath(c) for c in cands):
        if not os.path.isfile(c):
            continue
        first = first or c
        install = Path(c).parent.parent.parent
        if want is not None and ll_aton_version(install.joinpath(*INSTALL_LL_ATON)) == want:
            return c
    return first


_UNIT = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}


def _to_bytes(num: str, unit: str) -> int:
    return int(round(float(num) * _UNIT[unit.lower()]))


def parse_generate_report(text: str) -> dict:
    """Numbers from *_generate_report.txt; a field the report lacks is left out."""
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

    # Memory table rows:
    #   OCTAFLASH  [0x90300000 - 0x91D80000]:  898.188 kB /  26.500 MB ( 3.31 % used)
    #   npuRAM     [0x34200000 - 0x343C0000]:  448.000 kB /   1.750 MB (25.00 % used)
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
    # Ops that run in software epochs, by type.
    sw_ops = re.findall(r"^\|\s*epoch_\d+\s*\|\s*SW\s*\|\s*(\w+)", text, re.M)
    if sw_ops:
        counts: dict[str, int] = {}
        for op in sw_ops:
            counts[op] = counts.get(op, 0) + 1
        out["softwareOps"] = counts
    return out


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass


# Compilers running now. Each runs in its own process group, which the
# desktop's stop signal for the engine does not reach, so the engine stops
# them as it shuts down.
_RUNNING: set[subprocess.Popen] = set()
_RUNNING_LOCK = threading.Lock()
_STOPPING = threading.Event()


def allow_compilers() -> None:
    """Allow compiles again, as an engine starts in this process."""
    _STOPPING.clear()


def stop_running_compilers() -> None:
    """Stop the compilers running now and start no more."""
    _STOPPING.set()
    with _RUNNING_LOCK:
        procs = list(_RUNNING)
    for proc in procs:
        _kill_tree(proc)


def _did_not_run(returncode: int, log: str) -> bool:
    """True when the compiler crashed or never started: a Windows status code
    of 0xC0000000 or above, or a failure that printed nothing."""
    return (returncode & 0xFFFFFFFF) >= 0xC0000000 or (returncode != 0 and not log.strip())


def verify(model_path: Path, keep_dir: Path, name: str = "m") -> dict:
    """Run `stedgeai generate` as the core does; returns a JSON-ready dict.

    The compiler runs in a short temp directory; its workspace can exceed
    Windows' 260-char MAX_PATH. Raises RuntimeError when the tool rejects the model."""
    exe = find_stedgeai()
    if not exe:
        return {"verified": False, "available": False,
                "reason": "ST Edge AI Core (stedgeai) not installed; install it from st.com "
                          "or set NEURON6_STEDGEAI to verify exports against the NPU"}
    template = mpool_template(exe)
    try:
        if template is None:
            raise ValueError("no " + "/".join(MPOOL_TEMPLATE) + " in its install")
        mpool = build_mpool(template)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"verified": False, "available": True,
                "reason": f"ST Edge AI Core's memory-pool template could not be used ({e}), "
                          "so this export is not verified"}
    keep_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pkn6-"))
    try:
        out_dir, ws_dir = work / "out", work / "ws"
        out_dir.mkdir()
        ws_dir.mkdir()
        (work / MPOOL_NAME).write_text(json.dumps(mpool, indent=1), "utf-8")
        # Copied in so every path the tool derives is short.
        model_local = work / "model.onnx"
        shutil.copyfile(model_path, model_local)
        profile = {"Globals": {}, "Profiles": {"neuron6": {
            "memory_pool": "./" + MPOOL_NAME, "options": NEURAL_ART_OPTIONS}}}
        profile_path = work / "neuron6_neural_art.json"
        profile_path.write_text(json.dumps(profile, indent=1), "utf-8")

        cmd = [exe, "generate", "-m", str(model_local), "--target", "stm32n6",
               "--st-neural-art", "neuron6@" + str(profile_path).replace("\\", "/"),
               "-n", name, "--enable-epoch-controller",
               "--output", str(out_dir), "--workspace", str(ws_dir),
               *GENERATE_INTERFACE_OPTIONS]
        t0 = time.time()
        log_path = work / "stedgeai.log"
        # stedgeai.exe is a launcher; its own process group lets a timeout kill the compiler children.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform.startswith("win") else 0
        for attempt in range(2):
            if _STOPPING.is_set():
                return {"verified": False, "available": True,
                        "reason": "PixelKit was closing, so this export is not verified"}
            with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
                try:
                    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                            cwd=str(work), creationflags=flags,
                                            start_new_session=not sys.platform.startswith("win"))
                except OSError as e:
                    # Not executable, the wrong architecture, blocked by antivirus.
                    return {"verified": False, "available": True,
                            "reason": f"ST Edge AI could not start ({e}), so this export is not "
                                      "verified; the board build compiles it"}
                with _RUNNING_LOCK:
                    _RUNNING.add(proc)
                try:
                    proc.wait(timeout=COMPILE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    _kill_tree(proc)
                    lf.close()
                    shutil.copyfile(log_path, keep_dir / "stedgeai.log")
                    return {"verified": False, "available": True,
                            "reason": f"ST Edge AI took more than {COMPILE_TIMEOUT_S // 60} min, so "
                                      "this export is not verified; the board build compiles it"}
                finally:
                    with _RUNNING_LOCK:
                        _RUNNING.discard(proc)
            log = log_path.read_text("utf-8", errors="replace") if log_path.exists() else ""
            if not _did_not_run(proc.returncode, log):
                break
        shutil.copyfile(log_path, keep_dir / "stedgeai.log")
        if _did_not_run(proc.returncode, log):
            return {"verified": False, "available": True,
                    "reason": f"ST Edge AI did not run (exit 0x{proc.returncode & 0xFFFFFFFF:08X}, "
                              "twice), so this export is not verified; the board build "
                              "compiles it"}
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
        # The packed weights blob is the exact byte count the core packs into flash.
        blob = next(iter(out_dir.glob("*_atonbuf.xSPI1.raw")), None)
        if blob:
            info["weightsBytes"] = blob.stat().st_size
        pool = weights_pool_bytes()
        if "weightsBytes" in info and pool:
            info["poolSafeBytes"] = pool
            info["poolRemainingBytes"] = max(0, pool - info["weightsBytes"])
            info["poolUsedPct"] = round(100.0 * info["weightsBytes"] / pool, 2)
        return info
    finally:
        shutil.rmtree(work, ignore_errors=True)
