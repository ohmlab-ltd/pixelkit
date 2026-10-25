// PixelKit desktop shell: the local engine on 127.0.0.1:8001 in a native window.

'use strict';

const { app, BrowserWindow, Menu, dialog, screen, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

const bootstrap = require('./bootstrap');
const { checkForUpdates } = require('./updater');

const ENGINE_ORIGIN = 'http://127.0.0.1:8001';
const HEALTH_URL = `${ENGINE_ORIGIN}/api/health`;
const APP_URL = `${ENGINE_ORIGIN}/app`;
const ALLOWED_ORIGINS = new Set(['http://127.0.0.1:8001', 'http://localhost:8001']);

const HEALTH_TIMEOUT_MS = 1000; // per request
const STARTUP_TIMEOUT_MS = 60000; // total wait for a spawned engine
const POLL_INTERVAL_MS = 500;
const SHUTDOWN_GRACE_MS = 4000; // after SIGTERM, before SIGKILL

app.setName('PixelKit');

// Single instance

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    const win = mainWindow || splashWindow;
    if (win && !win.isDestroyed()) {
      if (win.isMinimized()) win.restore();
      win.show();
      win.focus();
    }
  });
}

let splashWindow = null;
let mainWindow = null;

// external: an engine found running, never killed. proc: the engine this shell spawned.
const engine = { external: false, proc: null, exited: false };

// Engine management

// { state, version }: 'up' when a PixelKit engine answered, 'busy' when a
// connection was accepted but no answer came in time, 'down' otherwise.
function probeEngine() {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        return resolve({ state: 'down' });
      }
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          const h = JSON.parse(body);
          resolve(h.app === 'pixelkit' ? { state: 'up', version: h.version } : { state: 'down' });
        } catch {
          resolve({ state: 'down' });
        }
      });
      res.on('error', () => resolve({ state: 'down' }));
    });
    req.on('timeout', () => {
      req.destroy();
      resolve({ state: 'busy' });
    });
    req.on('error', () => resolve({ state: 'down' }));
  });
}

async function checkHealth() {
  return (await probeEngine()).state === 'up';
}

// POST JSON to the engine; resolves { status, body }. No timeout: an import
// can run for minutes.
function postJson(pathname, payload) {
  return new Promise((resolve, reject) => {
    const data = Buffer.from(JSON.stringify(payload));
    const req = http.request(
      `${ENGINE_ORIGIN}${pathname}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': data.length },
      },
      (res) => {
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          let parsed = null;
          try {
            parsed = JSON.parse(body);
          } catch {}
          resolve({ status: res.statusCode, body: parsed });
        });
      }
    );
    req.on('error', reject);
    req.end(data);
  });
}

function engineLogPath() {
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Logs', 'PixelKit', 'engine.log');
  }
  return path.join(app.getPath('logs'), 'engine.log');
}

// The macOS dmg carries runtime/py in resources/; the Windows install has none
// and bootstraps one on first run.
function bundledPaths() {
  const res = process.resourcesPath || '';
  const python =
    process.platform === 'win32'
      ? path.join(res, 'runtime', 'py', 'python.exe')
      : path.join(res, 'runtime', 'py', 'bin', 'python3');
  return {
    python,
    engineCwd: path.join(res, 'engine', 'gd'),
    uiDir: path.join(res, 'ui'),
  };
}

function resolvePython() {
  if (process.env.PIXELKIT_PYTHON) return process.env.PIXELKIT_PYTHON;
  const b = bundledPaths();
  if (app.isPackaged) {
    if (fs.existsSync(b.python)) return b.python;
    return bootstrap.runtimePython(); // ensureEngine bootstraps it first
  }
  const venvPython =
    process.platform === 'win32'
      ? path.join(__dirname, '..', 'engine', '.venv', 'Scripts', 'python.exe')
      : path.join(__dirname, '..', 'engine', '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  if (fs.existsSync(bootstrap.runtimePython())) return bootstrap.runtimePython();
  return 'python3';
}

function spawnEngine() {
  const python = resolvePython();
  const b = bundledPaths();
  const cwd =
    app.isPackaged && fs.existsSync(b.engineCwd)
      ? b.engineCwd
      : path.join(__dirname, '..', 'engine', 'gd');
  const logFile = engineLogPath();

  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  const log = fs.createWriteStream(logFile, { flags: 'a' });
  log.write(`\n--- PixelKit engine start ${new Date().toISOString()} (${python}) ---\n`);

  const proc = spawn(python, ['server.py'], {
    cwd,
    env: {
      ...process.env,
      // Python's piped stdio on Windows is not UTF-8 by default; non-ASCII output raises.
      PYTHONIOENCODING: 'utf-8',
      ...(app.isPackaged && fs.existsSync(bundledPaths().uiDir)
        ? { PIXELKIT_UI_DIR: bundledPaths().uiDir }
        : {}),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    // A process group of its own, so stopping it reaches its children.
    detached: process.platform !== 'win32',
  });
  proc.stdout.pipe(log);
  proc.stderr.pipe(log);
  // An event from a replaced process must not mark the current engine dead.
  proc.on('exit', () => {
    if (engine.proc === proc) engine.exited = true;
  });
  proc.on('error', () => {
    if (engine.proc === proc) engine.exited = true;
  });

  engine.proc = proc;
  engine.exited = false;
  engine.external = false;
}

function bootstrapRequirementsPath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'engine', 'requirements-win.txt')
    : path.join(__dirname, '..', 'engine', 'requirements-win.txt');
}

// Retries until setup succeeds or Quit is chosen; returns false on Quit.
async function runFirstRunBootstrap() {
  const requirementsPath = bootstrapRequirementsPath();
  const logFile = path.join(path.dirname(engineLogPath()), 'bootstrap.log');
  fs.mkdirSync(path.dirname(logFile), { recursive: true });

  for (;;) {
    const log = fs.createWriteStream(logFile, { flags: 'a' });
    log.write(`\n--- PixelKit runtime bootstrap ${new Date().toISOString()} ---\n`);
    try {
      setSplashStatus('Preparing first run…');
      await bootstrap.ensureWindowsRuntime({ requirementsPath, onStatus: setSplashStatus, log });
      log.end();
      return true;
    } catch (err) {
      log.write(`\nBootstrap failed: ${err && err.stack ? err.stack : err}\n`);
      log.end();
      const opts = {
        type: 'error',
        title: 'PixelKit setup',
        message: 'PixelKit could not finish setting up its runtime.',
        detail:
          `${(err && err.message) || err}\n\nLog: ${logFile}\n\n` +
          'First run needs an internet connection. The AI runtime is downloaded once.',
        buttons: ['Retry', 'Quit'],
        defaultId: 0,
        cancelId: 1,
      };
      const choice =
        splashWindow && !splashWindow.isDestroyed()
          ? dialog.showMessageBoxSync(splashWindow, opts)
          : dialog.showMessageBoxSync(opts);
      if (choice !== 0) return false;
    }
  }
}

let bootstrapQuit = false;
let versionMismatch = null;
let portUnresponsive = false;

async function ensureEngine() {
  let running = await probeEngine();
  // A busy engine is waited for: a second one could not bind the port.
  const busyDeadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (running.state === 'busy' && Date.now() < busyDeadline) {
    setSplashStatus('Waiting for the running engine…');
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    running = await probeEngine();
  }
  if (running.state === 'busy') {
    portUnresponsive = true;
    return false;
  }
  if (running.state === 'up') {
    if (running.version && running.version !== app.getVersion()) {
      versionMismatch = running.version;
      return false;
    }
    engine.external = true;
    return true;
  }

  if (process.platform === 'win32' && app.isPackaged && !process.env.PIXELKIT_PYTHON) {
    const b = bundledPaths();
    if (!fs.existsSync(b.python) && !bootstrap.runtimeReady(bootstrapRequirementsPath())) {
      const ok = await runFirstRunBootstrap();
      if (!ok) {
        bootstrapQuit = true;
        app.quit();
        return false;
      }
    }
  }

  setSplashStatus('Starting engine…');
  spawnEngine();

  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (engine.exited) return false;
    if (await checkHealth()) return true;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return false;
}

// Crash watch

let crashWatchTimer = null;
let restartInFlight = false;
const restartTimes = []; // timestamps of recent automatic restarts

async function restartEngine() {
  if (shuttingDown) return false;
  // A healthy engine on the port is adopted. One this shell spawned stays
  // owned, so Quit still stops it.
  if (await checkHealth()) {
    if (engine.proc && !engine.external && !engine.exited) return true;
    engine.proc = null;
    engine.exited = false;
    engine.external = true;
    return true;
  }
  // A hung owned engine still holds the port, so it is stopped before respawning.
  if (engine.proc && !engine.external && !engine.exited) {
    await stopOwnedEngine();
  }
  if (shuttingDown) return false;
  spawnEngine();
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (engine.exited) return false;
    if (await checkHealth()) return true;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return false;
}

function startCrashWatch() {
  if (crashWatchTimer) clearInterval(crashWatchTimer);
  let misses = 0;
  let busySince = 0;
  crashWatchTimer = setInterval(async () => {
    if (shuttingDown || restartInFlight) return;
    if (!mainWindow || mainWindow.isDestroyed()) return;

    const ownedDead = Boolean(engine.proc) && !engine.external && engine.exited;
    const state = ownedDead ? 'down' : (await probeEngine()).state;
    if (state === 'up') {
      misses = 0;
      busySince = 0;
      return;
    }
    // A busy engine, as in a long export or delete, counts as hung after two minutes.
    if (state === 'busy') {
      if (!busySince) busySince = Date.now();
      if (Date.now() - busySince < 120000) return;
    } else {
      busySince = 0;
    }
    misses += 1;
    if (!ownedDead && misses < 3) return;

    restartInFlight = true;
    misses = 0;
    try {
      // Crashloop guard: at most 2 automatic restarts per 5 minutes.
      const now = Date.now();
      while (restartTimes.length && now - restartTimes[0] > 5 * 60 * 1000) restartTimes.shift();
      let ok = false;
      if (restartTimes.length < 2) {
        restartTimes.push(now);
        ok = await restartEngine();
      }
      if (ok) {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.reload();
        return;
      }
      const choice = dialog.showMessageBoxSync(mainWindow, {
        type: 'error',
        title: 'PixelKit',
        message: 'The PixelKit engine stopped and could not be restarted.',
        detail: `See the engine log:\n${engineLogPath()}`,
        buttons: ['Try again', 'Quit'],
        defaultId: 0,
        cancelId: 1,
      });
      if (choice === 0) {
        restartTimes.length = 0;
        if (await restartEngine()) {
          if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.reload();
        }
      } else {
        app.quit();
      }
    } finally {
      restartInFlight = false;
    }
  }, 5000);
}

// Kills the engine and its children: the process tree on Windows, the process group elsewhere.
function killEngineTree(proc, signal) {
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { windowsHide: true });
    } else {
      process.kill(-proc.pid, signal);
    }
  } catch {
    /* already gone */
  }
}

function stopOwnedEngine() {
  return new Promise((resolve) => {
    const proc = engine.proc;
    if (!proc || engine.external || engine.exited) return resolve();
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      killEngineTree(proc, 'SIGKILL');
      setTimeout(finish, 1000);
    }, SHUTDOWN_GRACE_MS);
    proc.once('exit', finish);
    killEngineTree(proc, 'SIGTERM');
  });
}

// Splash window

function splashHtml(body) {
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
    html,body{margin:0;height:100%;overflow:hidden;background:#101014;color:#e8e8ec;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      -webkit-user-select:none;user-select:none}
    body{display:flex;align-items:center;justify-content:center;-webkit-app-region:drag}
    .card{text-align:center;padding:0 32px}
    h1{font-size:22px;font-weight:600;letter-spacing:.4px;margin:0 0 10px}
    p{margin:6px 0;color:#9a9aa4}
    .err{color:#ff7a76}
    .spin{width:22px;height:22px;margin:16px auto 0;border-radius:50%;
      border:2.5px solid #33333c;border-top-color:#8f7bff;animation:s 0.9s linear infinite}
    @keyframes s{to{transform:rotate(360deg)}}
    #st{min-height:1.5em;max-width:380px;margin:12px auto 0;font-size:12px;color:#8a8a94;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  </style></head><body><div class="card">${body}</div></body></html>`;
  return 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
}

function setSplashStatus(text) {
  if (!splashWindow || splashWindow.isDestroyed()) return;
  splashWindow.webContents
    .executeJavaScript(
      `(el => { if (el) el.textContent = ${JSON.stringify(String(text))}; })(document.getElementById('st'))`
    )
    .catch(() => {});
}

function createSplash() {
  splashWindow = new BrowserWindow({
    width: 440,
    height: 260,
    frame: false,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    show: true,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  splashWindow.on('closed', () => {
    splashWindow = null;
  });
  splashWindow.loadURL(
    splashHtml('<h1>PixelKit</h1><p>Starting up…</p><div class="spin"></div><p id="st"></p>')
  );
}

// True on any HTTP response from the engine port, an error status included.
function portOccupied() {
  return new Promise((resolve) => {
    const req = http.get(`${ENGINE_ORIGIN}/`, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

function showSplashError(kind) {
  const logFile = engineLogPath().replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const body =
    kind === 'version'
      ? '<h1>PixelKit</h1>' +
        `<p class="err">A PixelKit engine from version ${esc(versionMismatch)} is still running.</p>` +
        `<p>Quit it (or the PixelKit window that started it) and open PixelKit ${esc(app.getVersion())} again.</p>`
      : kind === 'port'
      ? '<h1>PixelKit</h1>' +
        '<p class="err">Port 8001 is in use by another application.</p>' +
        '<p>Close whatever is listening on 127.0.0.1:8001 and start PixelKit again.</p>'
      : kind === 'unresponsive'
      ? '<h1>PixelKit</h1>' +
        '<p class="err">A process on 127.0.0.1:8001 accepts connections but does not answer.</p>' +
        '<p>If it is an earlier PixelKit engine, end it, then start PixelKit again.</p>'
      : '<h1>PixelKit</h1>' +
        '<p class="err">The engine failed to start.</p>' +
        `<p>See engine.log:<br><small>${logFile}</small></p>`;
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.setSize(480, 280);
    splashWindow.loadURL(splashHtml(body));
  }
}

// Window state

const stateFile = () => path.join(app.getPath('userData'), 'window-state.json');

function loadWindowState() {
  const defaults = { width: 1280, height: 820, isMaximized: false };
  let state;
  try {
    state = { ...defaults, ...JSON.parse(fs.readFileSync(stateFile(), 'utf8')) };
  } catch {
    return defaults;
  }
  if (typeof state.x === 'number' && typeof state.y === 'number') {
    const visible = screen.getAllDisplays().some((d) => {
      const a = d.workArea;
      return (
        state.x >= a.x - 8 &&
        state.y >= a.y - 8 &&
        state.x < a.x + a.width &&
        state.y < a.y + a.height
      );
    });
    if (!visible) {
      delete state.x;
      delete state.y;
    }
  }
  state.width = Math.max(state.width || defaults.width, 1100);
  state.height = Math.max(state.height || defaults.height, 700);
  return state;
}

function saveWindowState(win) {
  try {
    const bounds = win.getNormalBounds();
    const state = { ...bounds, isMaximized: win.isMaximized() };
    fs.mkdirSync(app.getPath('userData'), { recursive: true });
    fs.writeFileSync(stateFile(), JSON.stringify(state, null, 2));
  } catch {
    // best effort
  }
}

// Main window

function isInternalUrl(url) {
  try {
    return ALLOWED_ORIGINS.has(new URL(url).origin);
  } catch {
    return false;
  }
}

// The licences of the components PixelKit ships, beside the app's own.
function openNotices() {
  const file = app.isPackaged
    ? path.join(process.resourcesPath, 'THIRD-PARTY-NOTICES.md')
    : path.join(__dirname, '..', 'THIRD-PARTY-NOTICES.md');
  shell.openPath(file).then((err) => {
    if (err) dialog.showErrorBox('PixelKit', `Could not open ${file}: ${err}`);
  });
}

function openExternally(url) {
  if (/^https?:/i.test(url)) shell.openExternal(url);
}

const APP_ICON = path.join(__dirname, 'build', 'icon.png');

function createMainWindow() {
  const state = loadWindowState();
  mainWindow = new BrowserWindow({
    x: state.x,
    y: state.y,
    width: state.width,
    height: state.height,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    icon: APP_ICON,
    backgroundColor: '#101014',
    ...(process.platform === 'darwin' ? { titleBarStyle: 'hiddenInset' } : {}),
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });

  if (state.isMaximized) mainWindow.maximize();

  const wc = mainWindow.webContents;

  wc.setWindowOpenHandler(({ url }) => {
    openExternally(url);
    return { action: 'deny' };
  });

  wc.on('will-navigate', (event, url) => {
    if (!isInternalUrl(url)) {
      event.preventDefault();
      openExternally(url);
    }
  });

  mainWindow.once('ready-to-show', () => {
    if (mainWindow) mainWindow.show();
    if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
  });

  // Without a retry, a failed load leaves the splash up: ready-to-show never fires.
  let loadRetries = 0;
  wc.on('did-fail-load', (_e, code, desc, url, isMainFrame) => {
    if (!isMainFrame || loadRetries >= 5) return;
    loadRetries += 1;
    setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
    }, 1000 * loadRetries);
  });

  // 'close' comes before the page's beforeunload, so the prompt below knows
  // whether a close or a reload is being held up.
  let closeRequested = false;
  mainWindow.on('close', () => {
    closeRequested = true;
    saveWindowState(mainWindow);
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // ExportModal holds the page with beforeunload while an export builds.
  // Electron would block the unload silently; preventDefault lets it go ahead.
  wc.on('will-prevent-unload', (event) => {
    const closing = closeRequested;
    closeRequested = false;
    const choice = dialog.showMessageBoxSync(mainWindow, {
      type: 'warning',
      title: 'PixelKit',
      message: closing
        ? 'An export is still being built. Close PixelKit anyway?'
        : 'An export is still being built. Reload anyway?',
      buttons: [closing ? 'Close' : 'Reload', 'Cancel'],
      defaultId: 1,
      cancelId: 1,
    });
    if (choice === 0) event.preventDefault();
  });

  mainWindow.loadURL(APP_URL);
}

// Menu

let importBusy = false;

async function importLegacyBackup() {
  if (importBusy) return;
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const picked = await dialog.showOpenDialog(mainWindow, {
    title: 'Import legacy PixelKit data',
    message: 'Choose a nightly backup zip',
    properties: ['openFile'],
    filters: [{ name: 'PixelKit backup', extensions: ['zip'] }],
  });
  if (picked.canceled || !picked.filePaths[0]) return;
  const src = picked.filePaths[0];
  importBusy = true;
  const alive = () => mainWindow && !mainWindow.isDestroyed();
  try {
    const r = await postJson('/api/import-legacy', { path: src });
    if (!alive()) return;
    if (r.status === 200) {
      await dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'PixelKit',
        message: 'Legacy data imported.',
        detail: 'Your Projects and datasets are in the workspace now. The window will reload.',
      });
      if (alive()) mainWindow.webContents.reload();
    } else {
      const detail = (r.body && r.body.detail) || `engine answered ${r.status}`;
      dialog.showMessageBox(mainWindow, {
        type: 'error',
        title: 'PixelKit',
        message: 'Legacy import failed.',
        detail: String(detail),
      });
    }
  } catch (err) {
    if (alive()) {
      dialog.showMessageBox(mainWindow, {
        type: 'error',
        title: 'PixelKit',
        message: 'Legacy import failed.',
        detail: String((err && err.message) || err),
      });
    }
  } finally {
    importBusy = false;
  }
}

function openEngineLogFolder() {
  const logFile = engineLogPath();
  if (fs.existsSync(logFile)) shell.showItemInFolder(logFile);
  else shell.openPath(path.dirname(logFile));
}

const ISSUES_URL = 'https://github.com/ohmlab-ltd/pixelkit/issues';

function buildMenu() {
  const isMac = process.platform === 'darwin';
  app.setAboutPanelOptions({ applicationName: 'PixelKit', applicationVersion: app.getVersion() });

  const template = [
    ...(isMac
      ? [
          {
            label: app.name,
            submenu: [
              { role: 'about' },
              { type: 'separator' },
              { role: 'hide' },
              { role: 'hideOthers' },
              { role: 'unhide' },
              { type: 'separator' },
              { role: 'quit' },
            ],
          },
        ]
      : []),
    {
      label: 'File',
      submenu: [
        { label: 'Import Legacy Backup…', click: () => importLegacyBackup() },
        ...(isMac ? [] : [{ type: 'separator' }, { role: 'quit' }]),
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        ...(isMac ? [{ role: 'pasteAndMatchStyle' }] : []),
        { role: 'delete' },
        { role: 'selectAll' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        { label: 'Open Engine Log Folder', click: () => openEngineLogFolder() },
        { label: 'Report an Issue…', click: () => shell.openExternal(ISSUES_URL) },
        { label: 'Third-Party Notices', click: () => openNotices() },
        ...(isMac ? [] : [{ type: 'separator' }, { role: 'about' }]),
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// Lifecycle

app.whenReady().then(async () => {
  // For dev runs; packaged builds get the dock icon from electron-builder.
  if (process.platform === 'darwin' && app.dock) {
    try { app.dock.setIcon(APP_ICON); } catch {}
  }
  if (!gotLock) return;
  buildMenu();
  createSplash();

  const ok = await ensureEngine();
  if (ok) {
    createMainWindow();
    startCrashWatch();
    if (app.isPackaged) {
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) checkForUpdates(mainWindow);
      }, 5000);
    }
  } else if (versionMismatch) {
    showSplashError('version');
  } else if (portUnresponsive) {
    showSplashError('unresponsive');
  } else if (!bootstrapQuit) {
    // Stopped first, so the port probe does not report it as another application.
    await stopOwnedEngine();
    showSplashError((await portOccupied()) ? 'port' : undefined);
  }
});

app.on('activate', () => {
  // macOS dock click with no windows open.
  if (mainWindow === null && splashWindow === null) {
    checkHealth().then((ok) => {
      if (ok && mainWindow === null) createMainWindow();
    });
  }
});

app.on('window-all-closed', () => {
  // Quits on macOS too: quitting is what stops a spawned engine.
  app.quit();
});

let shuttingDown = false;
app.on('will-quit', (event) => {
  if (shuttingDown) return;
  // Set first: a restart in flight must not spawn an engine that outlives the app.
  shuttingDown = true;
  bootstrap.abort();
  if (engine.proc && !engine.external && !engine.exited) {
    event.preventDefault();
    stopOwnedEngine().then(() => app.exit(0));
  }
});
