// First-run setup of the Windows Python runtime in %LOCALAPPDATA%\PixelKit\runtime.

'use strict';

const { spawn, spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const https = require('https');
const path = require('path');

const PBS_PINNED_URL =
  'https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13%2B20260807-x86_64-pc-windows-msvc-install_only.tar.gz';
// As in the release's SHA256SUMS.
const PBS_PINNED_SHA256 = '6cf2be701aa7e9470454c9c86285c1bcc1832518d63e39c3e34e9d8ea1cbb99f';
const PBS_LATEST_API =
  'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest';
const PBS_ASSET_MATCH = ['cpython-3.12', 'x86_64-pc-windows-msvc-install_only.tar.gz'];

// A download or probe with no bytes for this long has stalled.
const STALL_MS = 60000;

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

// Comments and blank lines are left out, so editing them triggers no reinstall.
function requirementsHash(requirementsPath) {
  const lines = fs
    .readFileSync(requirementsPath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.replace(/(^|\s)#.*$/, '').trim())
    .filter(Boolean);
  return crypto.createHash('sha256').update(lines.join('\n')).digest('hex');
}

function readMarker() {
  try {
    return JSON.parse(fs.readFileSync(path.join(runtimeRoot(), 'runtime.json'), 'utf8'));
  } catch {
    return null;
  }
}

// True when the runtime was installed for these requirements.
function runtimeReady(requirementsPath) {
  const marker = readMarker();
  if (!marker || !fs.existsSync(runtimePython())) return false;
  if (!requirementsPath || !fs.existsSync(requirementsPath)) return true;
  return marker.requirements === requirementsHash(requirementsPath);
}

function httpGet(url, { json = false, hops = 0 } = {}) {
  return new Promise((resolve, reject) => {
    if (hops > 10) return reject(new Error(`too many redirects for ${url}`));
    const req = https.get(
      url,
      { headers: { 'User-Agent': 'PixelKit-bootstrap' } },
      (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(httpGet(res.headers.location, { json, hops: hops + 1 }));
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
    req.setTimeout(STALL_MS, () => req.destroy(new Error(`no response from ${url}`)));
    req.on('error', reject);
  });
}

function fileSha256(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    fs.createReadStream(file)
      .on('data', (chunk) => hash.update(chunk))
      .on('end', () => resolve(hash.digest('hex')))
      .on('error', reject);
  });
}

function download(url, dest, onProgress, hops = 0) {
  return new Promise((resolve, reject) => {
    if (hops > 10) return reject(new Error(`too many redirects for ${url}`));
    const req = https.get(
      url,
      { headers: { 'User-Agent': 'PixelKit-bootstrap' } },
      (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(download(res.headers.location, dest, onProgress, hops + 1));
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
    req.setTimeout(STALL_MS, () => req.destroy(new Error('download stalled')));
    req.on('error', reject);
  });
}

// The running tar or pip child, killed by abort() when the app quits.
let _activeChild = null;

function abort() {
  const proc = _activeChild;
  if (!proc) return;
  try {
    if (process.platform === 'win32') {
      // Kill the tree: pip spawns its own children on Windows.
      spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { windowsHide: true });
    } else {
      proc.kill('SIGTERM');
    }
  } catch {
    /* already gone */
  }
}

function run(cmd, args, { onLine, log, cwd } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    _activeChild = proc;
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
    proc.on('error', (e) => {
      if (_activeChild === proc) _activeChild = null;
      reject(e);
    });
    // 'close', not 'exit': close fires after stdio has drained into the log.
    proc.on('close', (code) => {
      if (_activeChild === proc) _activeChild = null;
      if (code === 0) return resolve();
      reject(new Error(`${path.basename(cmd)} ${args[0] || ''} exited ${code}:\n${lastLines.join('\n')}`));
    });
  });
}

function hasNvidiaGpu() {
  try {
    return spawnSync('nvidia-smi', ['-L'], { windowsHide: true, timeout: 20000 }).status === 0;
  } catch {
    return false;
  }
}

// The version a requirements file pins for `name`.
function pinnedVersion(file, name) {
  const m = fs.readFileSync(file, 'utf8').match(new RegExp(`^${name}==(\\S+)`, 'm'));
  return m ? m[1] : null;
}

function installedTorch(py) {
  const r = spawnSync(py, ['-c', 'import torch; print(torch.__version__)'], {
    windowsHide: true,
    timeout: 120000,
  });
  return r.status === 0 ? String(r.stdout).trim() : null;
}

// The pinned interpreter, or the newest release's when the pinned file is
// gone; either way with the SHA-256 the download must match.
async function resolveInterpreter() {
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
      req.setTimeout(STALL_MS, () => req.destroy(new Error(`no response from ${PBS_PINNED_URL}`)));
      req.on('error', reject);
      req.end();
    });
    return { url: PBS_PINNED_URL, sha256: PBS_PINNED_SHA256 };
  } catch {
    const rel = await httpGet(PBS_LATEST_API, { json: true });
    const asset = (rel.assets || []).find(
      (a) =>
        PBS_ASSET_MATCH.every((m) => a.browser_download_url.includes(m)) &&
        !a.browser_download_url.endsWith('.sha256')
    );
    if (!asset) throw new Error('no matching python-build-standalone asset found');
    const digest = /^sha256:([0-9a-f]{64})$/.exec(asset.digest || '');
    if (!digest) throw new Error(`no published checksum for ${asset.name}`);
    return { url: asset.browser_download_url, sha256: digest[1] };
  }
}

const mb = (n) => Math.round(n / (1024 * 1024));

// Returns the runtime python path; throws a readable message on failure.
// onStatus(text) drives the splash screen; log is an open WriteStream.
async function ensureWindowsRuntime({ requirementsPath, onStatus, log }) {
  const status = (t) => onStatus && onStatus(t);
  const root = runtimeRoot();
  const py = runtimePython();

  if (runtimeReady(requirementsPath)) return py;
  const updating = !!readMarker() && fs.existsSync(py);
  if (!fs.existsSync(requirementsPath)) {
    throw new Error(`engine requirements not found at ${requirementsPath}`);
  }

  fs.mkdirSync(root, { recursive: true });

  // Disk preflight: CUDA torch unpacks to ~7 GB, CPU torch to ~1.5 GB.
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
    // statfs unsupported: no preflight.
  }

  if (!fs.existsSync(py)) {
    status('Locating Python runtime…');
    const { url, sha256 } = await resolveInterpreter();
    const tarPath = path.join(root, 'python.tar.gz');
    status('Downloading Python…');
    await download(url, tarPath, (got, total) =>
      status(`Downloading Python… ${mb(got)}${total ? ` / ${mb(total)}` : ''} MB`)
    );

    const got = await fileSha256(tarPath);
    if (got !== sha256) {
      fs.rmSync(tarPath, { force: true });
      throw new Error(`the Python download does not match its published checksum (got ${got})`);
    }

    status('Unpacking Python…');
    const pyDir = path.join(root, 'py');
    fs.rmSync(pyDir, { recursive: true, force: true });
    // System32\tar.exe, not PATH tar: GNU tar reads "C:\..." as a remote host:file.
    const systemTar = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
    const tarCmd = fs.existsSync(systemTar) ? systemTar : 'tar';
    // Relative names: tar.exe cannot open absolute paths outside the system code page.
    await run(tarCmd, ['-xzf', path.basename(tarPath)], { log, cwd: root });
    // The tarball's top-level directory is "python/".
    fs.renameSync(path.join(root, 'python'), pyDir);
    fs.rmSync(tarPath, { force: true });
    if (!fs.existsSync(py)) throw new Error('python.exe missing after extraction');
  }

  const cuda = hasNvidiaGpu();
  const flavour = cuda ? 'CUDA' : 'CPU';
  const torchRequirements = path.join(
    path.dirname(requirementsPath),
    cuda ? 'requirements-win-torch-cuda.txt' : 'requirements-win-torch-cpu.txt'
  );
  // Every package must match a hash in its requirements file.
  const pipBase = [
    '-m', 'pip', 'install', '--require-hashes', '--no-warn-script-location', '--disable-pip-version-check',
  ];
  if (installedTorch(py) !== pinnedVersion(torchRequirements, 'torch')) {
    status(`Installing PyTorch (${flavour}${cuda ? ', ~3.5 GB' : ''})…`);
    // Its dependencies come from PyPI with the engine's.
    await run(
      py,
      [...pipBase, '--no-deps', '-r', torchRequirements, '--index-url', cuda ? TORCH_CUDA_INDEX : TORCH_CPU_INDEX],
      { log, onLine: (l) => status(`Installing PyTorch (${flavour}): ${l.slice(0, 80)}`) }
    );
  }

  const depsVerb = updating ? 'Updating' : 'Installing';
  status(`${depsVerb} engine dependencies…`);
  await run(py, [...pipBase, '-r', requirementsPath], {
    log,
    onLine: (l) => status(`${depsVerb} engine dependencies: ${l.slice(0, 80)}`),
  });

  status('Checking the runtime…');
  // Includes the export stack: onnxruntime fails to import without the VC++ runtime.
  const check = spawnSync(py, ['-c', 'import torch, transformers, fastapi, cv2, PIL, onnx, onnxruntime'], {
    windowsHide: true,
    timeout: 300000,
  });
  if (check.status !== 0) {
    throw new Error(`runtime sanity check failed:\n${String(check.stderr || '').slice(-800)}`);
  }

  fs.writeFileSync(
    path.join(root, 'runtime.json'),
    JSON.stringify(
      {
        python: path.basename(py),
        torch: flavour.toLowerCase(),
        requirements: requirementsHash(requirementsPath),
        completedAt: new Date().toISOString(),
      },
      null,
      2
    )
  );
  status('Runtime ready.');
  return py;
}

module.exports = { ensureWindowsRuntime, runtimePython, runtimeReady, abort };
