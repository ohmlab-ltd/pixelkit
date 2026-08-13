// First-run runtime bootstrap for the slim Windows installer.
//
// The NSIS installer ships only the Electron shell, the engine sources and
// the UI build (~a couple hundred MB). On first launch this module sets up
// the Python runtime the engine needs, into a per-user directory:
//
//   %LOCALAPPDATA%\PixelKit\runtime\py        relocatable CPython
//   %LOCALAPPDATA%\PixelKit\runtime\runtime.json   "install completed" marker
//
// Steps: download python-build-standalone CPython -> extract with the
// system tar.exe (present on Windows 10 1803+) -> pip install torch
// (CUDA build when nvidia-smi is present, CPU build otherwise) -> pip
// install the engine requirements -> import sanity check -> marker.
// pip's wheel cache makes retries and reinstalls cheap.

'use strict';

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const https = require('https');
const path = require('path');

// Pinned interpreter (updated deliberately; the fallback query keeps a
// deleted release from bricking first runs).
const PBS_PINNED_URL =
  'https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13%2B20260807-x86_64-pc-windows-msvc-install_only.tar.gz';
const PBS_LATEST_API =
  'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest';
const PBS_ASSET_MATCH = ['cpython-3.12', 'x86_64-pc-windows-msvc-install_only.tar.gz'];

const TORCH_CUDA_INDEX = 'https://download.pytorch.org/whl/cu126';
const TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu';

function runtimeRoot() {
  const base =
    process.env.LOCALAPPDATA || path.join(require('os').homedir(), 'AppData', 'Local');
  return path.join(base, 'PixelKit', 'runtime');
}

function runtimePython() {
  return path.join(runtimeRoot(), 'py', 'python.exe');
}

function runtimeReady() {
  return (
    fs.existsSync(path.join(runtimeRoot(), 'runtime.json')) && fs.existsSync(runtimePython())
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function httpGet(url, { json = false } = {}) {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      { headers: { 'User-Agent': 'PixelKit-bootstrap' } },
      (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(httpGet(res.headers.location, { json }));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          try {
            resolve(json ? JSON.parse(body) : body);
          } catch (e) {
            reject(e);
          }
        });
      }
    );
    req.on('error', reject);
  });
}

function download(url, dest, onProgress) {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      { headers: { 'User-Agent': 'PixelKit-bootstrap' } },
      (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(download(res.headers.location, dest, onProgress));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        const total = parseInt(res.headers['content-length'] || '0', 10);
        let received = 0;
        const out = fs.createWriteStream(dest);
        res.on('data', (chunk) => {
          received += chunk.length;
          if (onProgress) onProgress(received, total);
        });
        res.pipe(out);
        out.on('finish', () => out.close(resolve));
        out.on('error', reject);
        res.on('error', reject);
      }
    );
    req.on('error', reject);
  });
}

function run(cmd, args, { onLine, log } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    let lastLines = [];
    const handle = (buf) => {
      const text = buf.toString();
      if (log) log.write(text);
      for (const raw of text.split(/\r?\n/)) {
        const line = raw.trim();
        if (!line) continue;
        lastLines = [...lastLines.slice(-4), line];
        if (onLine) onLine(line);
      }
    };
    proc.stdout.on('data', handle);
    proc.stderr.on('data', handle);
    proc.on('error', reject);
    proc.on('exit', (code) => {
      if (code === 0) return resolve();
      reject(new Error(`${path.basename(cmd)} ${args[0] || ''} exited ${code}:\n${lastLines.join('\n')}`));
    });
  });
}

function hasNvidiaGpu() {
  try {
    return spawnSync('nvidia-smi', ['-L'], { windowsHide: true }).status === 0;
  } catch {
    return false;
  }
}

async function resolveInterpreterUrl() {
  // Prefer the pin; fall back to the latest release if the pin is gone.
  try {
    await new Promise((resolve, reject) => {
      const req = https.request(
        PBS_PINNED_URL,
        { method: 'HEAD', headers: { 'User-Agent': 'PixelKit-bootstrap' } },
        (res) => {
          res.resume();
          res.statusCode < 400 ? resolve() : reject(new Error(`HTTP ${res.statusCode}`));
        }
      );
      req.on('error', reject);
      req.end();
    });
    return PBS_PINNED_URL;
  } catch {
    const rel = await httpGet(PBS_LATEST_API, { json: true });
    const asset = (rel.assets || []).find(
      (a) =>
        PBS_ASSET_MATCH.every((m) => a.browser_download_url.includes(m)) &&
        !a.browser_download_url.endsWith('.sha256')
    );
    if (!asset) throw new Error('no matching python-build-standalone asset found');
    return asset.browser_download_url;
  }
}

const mb = (n) => Math.round(n / (1024 * 1024));

// ---------------------------------------------------------------------------
// The bootstrap
// ---------------------------------------------------------------------------

// Returns the runtime python path; throws with a readable message on failure.
// onStatus(text) drives the splash screen; log is an open WriteStream.
async function ensureWindowsRuntime({ requirementsPath, onStatus, log }) {
  const status = (t) => onStatus && onStatus(t);
  const root = runtimeRoot();
  const py = runtimePython();

  if (runtimeReady()) return py;
  if (!fs.existsSync(requirementsPath)) {
    throw new Error(`engine requirements not found at ${requirementsPath}`);
  }

  fs.mkdirSync(root, { recursive: true });

  // Disk preflight: CUDA torch unpacks to ~7 GB, CPU ~1.5 GB; demand
  // headroom up front so the failure is one readable sentence instead
  // of a pip ENOSPC stack halfway through.
  const needGb = hasNvidiaGpu() ? 9 : 3;
  try {
    const { bavail, bsize } = fs.statfsSync(root);
    const freeGb = (bavail * bsize) / 1024 ** 3;
    if (freeGb < needGb) {
      throw new Error(
        `not enough disk space for the AI runtime: ${freeGb.toFixed(1)} GB free, ` +
          `~${needGb} GB needed (in ${root})`
      );
    }
  } catch (e) {
    if (e && /not enough disk space/.test(String(e.message))) throw e;
    // statfs unsupported -> skip the preflight rather than block install.
  }

  // 1. Interpreter ----------------------------------------------------------
  if (!fs.existsSync(py)) {
    status('Locating Python runtime…');
    const url = await resolveInterpreterUrl();
    const tarPath = path.join(root, 'python.tar.gz');
    status('Downloading Python…');
    await download(url, tarPath, (got, total) =>
      status(`Downloading Python… ${mb(got)}${total ? ` / ${mb(total)}` : ''} MB`)
    );

    const size = fs.statSync(tarPath).size;
    if (size < 5 * 1024 * 1024) {
      throw new Error(`interpreter download looks corrupt (${mb(size)} MB)`);
    }

    status('Unpacking Python…');
    const pyDir = path.join(root, 'py');
    fs.rmSync(pyDir, { recursive: true, force: true });
    // Windows 10 1803+ ships bsdtar as System32\tar.exe. Use it by absolute
    // path: a PATH "tar" may be Git's GNU tar, which reads "C:\..." as a
    // remote host:file archive and fails.
    const systemTar = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
    const tarCmd = fs.existsSync(systemTar) ? systemTar : 'tar';
    await run(tarCmd, ['-xzf', tarPath, '-C', root], { log });
    // The tarball's top-level directory is "python/".
    fs.renameSync(path.join(root, 'python'), pyDir);
    fs.rmSync(tarPath, { force: true });
    if (!fs.existsSync(py)) throw new Error('python.exe missing after extraction');
  }

  // 2. PyTorch (the big one) ------------------------------------------------
  const cuda = hasNvidiaGpu();
  const flavour = cuda ? 'CUDA' : 'CPU';
  status(`Installing PyTorch (${flavour}${cuda ? ', ~3.5 GB' : ''} — one time)…`);
  const pipBase = ['-m', 'pip', 'install', '--no-warn-script-location', '--disable-pip-version-check'];
  await run(py, [...pipBase, 'torch', 'torchvision', '--index-url', cuda ? TORCH_CUDA_INDEX : TORCH_CPU_INDEX], {
    log,
    onLine: (l) => status(`Installing PyTorch (${flavour}) — ${l.slice(0, 80)}`),
  });

  // 3. Engine dependencies --------------------------------------------------
  status('Installing engine dependencies…');
  await run(py, [...pipBase, '-r', requirementsPath], {
    log,
    onLine: (l) => status(`Installing engine dependencies — ${l.slice(0, 80)}`),
  });

  // 4. Sanity + marker ------------------------------------------------------
  status('Checking the runtime…');
  const check = spawnSync(py, ['-c', 'import torch, transformers, fastapi, cv2, PIL'], {
    windowsHide: true,
  });
  if (check.status !== 0) {
    throw new Error(`runtime sanity check failed:\n${String(check.stderr || '').slice(-800)}`);
  }

  fs.writeFileSync(
    path.join(root, 'runtime.json'),
    JSON.stringify(
      { python: path.basename(py), torch: flavour.toLowerCase(), completedAt: new Date().toISOString() },
      null,
      2
    )
  );
  status('Runtime ready.');
  return py;
}

module.exports = { ensureWindowsRuntime, runtimePython, runtimeReady };
