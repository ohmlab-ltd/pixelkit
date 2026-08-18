// PixelKit desktop shell.
//
// Wraps the local PixelKit engine (FastAPI on 127.0.0.1:8001) in a native
// window. If an engine is already running (e.g. started from a terminal, or
// mid-download of model weights) it is REUSED and never killed; otherwise the
// shell spawns its own engine and shuts it down on quit.

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

const HEALTH_TIMEOUT_MS = 1000; // per-request timeout
const STARTUP_TIMEOUT_MS = 60000; // total time to wait for a spawned engine
const POLL_INTERVAL_MS = 500;
const SHUTDOWN_GRACE_MS = 4000; // how long to wait for SIGTERM'd engine

app.setName('PixelKit');

// ---------------------------------------------------------------------------
// Single instance
// ---------------------------------------------------------------------------

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

// engine.external === true  -> we found a running engine and must never kill it
// engine.proc !== null      -> we spawned it and own its lifecycle
const engine = { external: false, proc: null, exited: false };

// ---------------------------------------------------------------------------
// Engine management
// ---------------------------------------------------------------------------

function checkHealth() {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        return resolve(false);
      }
      // Identity check: something else listening on 8001 must not be
      // adopted as "the engine" just because it answers 200.
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          resolve(JSON.parse(body).app === 'pixelkit');
        } catch {
          resolve(false);
        }
      });
      res.on('error', () => resolve(false));
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

// POST JSON to the engine with no timeout (legacy imports can run for
// minutes); resolves {status, body}.
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

// Packaged app: a fat bundle (macOS dmg) carries runtime/py inside
// resources/; the slim Windows install has no bundled runtime and instead
// bootstraps one into %LOCALAPPDATA%\PixelKit\runtime on first run
// (bootstrap.js). Dev runs use the repo venv.
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
    if (fs.existsSync(b.python)) return b.python; // fat bundle (macOS dmg)
    return bootstrap.runtimePython(); // slim install - ensureEngine bootstraps it first
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
      // Piped stdio on Windows defaults Python to the legacy codepage;
      // a log line with ≥/²/→ then raises inside the printing code.
      // The engine also reconfigures its own streams - this covers
      // interpreter startup and any subprocesses it spawns.
      PYTHONIOENCODING: 'utf-8',
      ...(app.isPackaged && fs.existsSync(bundledPaths().uiDir)
        ? { PIXELKIT_UI_DIR: bundledPaths().uiDir }
        : {}),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.pipe(log);
  proc.stderr.pipe(log);
  // Tag the handlers to THIS proc: a stale exit event from a replaced
  // process must never mark the current engine dead (that stale flag
  // used to wedge the crash watch into a restart loop over a healthy
  // engine and let will-quit skip shutdown of a live one).
  proc.on('exit', () => {
    if (engine.proc === proc) engine.exited = true;
  });
  proc.on('error', () => {
    // e.g. python binary not found; poll loop will time out and report.
    if (engine.proc === proc) engine.exited = true;
  });

  engine.proc = proc;
  engine.exited = false;
  engine.external = false;
}

// Slim Windows installs set up their Python runtime on first launch.
// Loops on failure (Retry / Quit); returns false only when the user quits.
async function runFirstRunBootstrap() {
  const requirementsPath = app.isPackaged
    ? path.join(process.resourcesPath, 'engine', 'requirements-win.txt')
    : path.join(__dirname, '..', 'engine', 'requirements-win.txt');
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
      log.write(`\nBOOTSTRAP FAILED: ${err && err.stack ? err.stack : err}\n`);
      log.end();
      const opts = {
        type: 'error',
        title: 'PixelKit setup',
        message: 'PixelKit could not finish setting up its runtime.',
        detail:
          `${(err && err.message) || err}\n\nLog: ${logFile}\n\n` +
          'First run needs an internet connection - the AI runtime is downloaded once.',
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

async function ensureEngine() {
  if (await checkHealth()) {
    engine.external = true; // someone else owns it -> never kill it
    return true;
  }

  // Slim Windows install: make sure a runtime exists before spawning.
  if (process.platform === 'win32' && app.isPackaged && !process.env.PIXELKIT_PYTHON) {
    const b = bundledPaths();
    if (!fs.existsSync(b.python) && !bootstrap.runtimeReady()) {
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
    if (engine.exited) return false; // crashed / python missing
    if (await checkHealth()) return true;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return false;
}

// ---------------------------------------------------------------------------
// Crash watch - if the engine dies mid-session, restart it (bounded) and
// reload the window; if it won't come back, tell the user instead of
// leaving a dead page.
// ---------------------------------------------------------------------------

let crashWatchTimer = null;
let restartInFlight = false;
const restartTimes = []; // timestamps of recent automatic restarts

async function restartEngine() {
  // Something healthy may already own the port (the user started their
  // own engine, or ours recovered) - adopt it instead of spawning into
  // a guaranteed bind failure.
  if (await checkHealth()) {
    engine.proc = null;
    engine.exited = false;
    engine.external = true;
    return true;
  }
  // A hung-but-alive owned process still holds the port; kill and wait
  // before respawning, or the replacement dies on the bind.
  if (engine.proc && !engine.external && !engine.exited) {
    await stopOwnedEngine();
  }
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
  crashWatchTimer = setInterval(async () => {
    if (shuttingDown || restartInFlight) return;
    if (!mainWindow || mainWindow.isDestroyed()) return;

    const ownedDead = Boolean(engine.proc) && !engine.external && engine.exited;
    const healthy = ownedDead ? false : await checkHealth();
    if (healthy) {
      misses = 0;
      return;
    }
    misses += 1;
    if (!ownedDead && misses < 3) return; // ride out transient blips

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

function stopOwnedEngine() {
  return new Promise((resolve) => {
    const proc = engine.proc;
    if (!proc || engine.external || engine.exited) return resolve();
    const timer = setTimeout(resolve, SHUTDOWN_GRACE_MS);
    proc.once('exit', () => {
      clearTimeout(timer);
      resolve();
    });
    try {
      proc.kill('SIGTERM');
    } catch {
      clearTimeout(timer);
      resolve();
    }
  });
}

// ---------------------------------------------------------------------------
// Splash window
// ---------------------------------------------------------------------------

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

// Update the one-line status on the splash (no-op once it's closed).
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

// Any HTTP response on the engine port (even an error) means something
// else owns it - checkHealth() already said it isn't a PixelKit engine.
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
  const body =
    kind === 'port'
      ? '<h1>PixelKit</h1>' +
        '<p class="err">Port 8001 is in use by another application.</p>' +
        '<p>Close whatever is listening on 127.0.0.1:8001 and start PixelKit again.</p>'
      : '<h1>PixelKit</h1>' +
        '<p class="err">The engine failed to start.</p>' +
        `<p>See engine.log:<br><small>${logFile}</small></p>`;
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.setSize(480, 280);
    splashWindow.loadURL(splashHtml(body));
  }
}

// ---------------------------------------------------------------------------
// Window state persistence
// ---------------------------------------------------------------------------

const stateFile = () => path.join(app.getPath('userData'), 'window-state.json');

function loadWindowState() {
  const defaults = { width: 1280, height: 820, isMaximized: false };
  let state;
  try {
    state = { ...defaults, ...JSON.parse(fs.readFileSync(stateFile(), 'utf8')) };
  } catch {
    return defaults;
  }
  // Only honour saved x/y if they land on a currently attached display.
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

// ---------------------------------------------------------------------------
// Main window
// ---------------------------------------------------------------------------

function isInternalUrl(url) {
  try {
    return ALLOWED_ORIGINS.has(new URL(url).origin);
  } catch {
    return false;
  }
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

  // target=_blank and window.open -> system browser, never a new Electron window.
  wc.setWindowOpenHandler(({ url }) => {
    openExternally(url);
    return { action: 'deny' };
  });

  // Any navigation away from the local engine -> system browser.
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

  // A transient failure loading the app page (engine mid-listen, socket
  // reset) would otherwise leave an eternal splash: ready-to-show never
  // fires and nothing retries. Retry the load a few times.
  let loadRetries = 0;
  wc.on('did-fail-load', (_e, code, desc, url, isMainFrame) => {
    if (!isMainFrame || loadRetries >= 5) return;
    loadRetries += 1;
    setTimeout(() => {
      if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
    }, 1000 * loadRetries);
  });

  mainWindow.on('close', () => saveWindowState(mainWindow));
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.loadURL(APP_URL);
}

// ---------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------

let importBusy = false;

async function importLegacyBackup() {
  if (importBusy) return; // one minutes-long import at a time
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const picked = await dialog.showOpenDialog(mainWindow, {
    title: 'Import legacy PixelKit data',
    message: 'Choose a nightly backup zip or an old backend data folder',
    properties: ['openFile'],
    filters: [{ name: 'PixelKit backup', extensions: ['zip'] }],
  });
  if (picked.canceled || !picked.filePaths[0]) return;
  const src = picked.filePaths[0];
  importBusy = true;
  const alive = () => mainWindow && !mainWindow.isDestroyed();
  try {
    const r = await postJson('/api/import-legacy', { path: src });
    if (!alive()) return; // window closed during a long import
    if (r.status === 200) {
      await dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'PixelKit',
        message: 'Legacy data imported.',
        detail: 'Your projects are in the workspace now; the window will reload.',
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
        ...(isMac ? [] : [{ type: 'separator' }, { role: 'about' }]),
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  // Dev-mode dock icon (packaged builds get it from electron-builder).
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
  } else if (!bootstrapQuit) {
    // Startup timed out: reap our own spawn first, both to avoid a
    // zombie engine behind the error splash and so the port probe
    // can't blame "another application" for a process we started.
    await stopOwnedEngine();
    showSplashError((await portOccupied()) ? 'port' : undefined);
  }
});

app.on('activate', () => {
  // macOS dock click with no windows: recreate if the engine is up.
  if (mainWindow === null && splashWindow === null) {
    checkHealth().then((ok) => {
      if (ok && mainWindow === null) createMainWindow();
    });
  }
});

app.on('window-all-closed', () => {
  // Single-window shell: closing the window quits the app on every platform,
  // which is what tears down a spawned engine (a reused one is left alone).
  app.quit();
});

let shuttingDown = false;
app.on('will-quit', (event) => {
  if (shuttingDown) return;
  // Quitting mid-bootstrap: kill the tar/pip child so a multi-GB
  // install doesn't keep running headless and race the next launch.
  bootstrap.abort();
  if (engine.proc && !engine.external && !engine.exited) {
    shuttingDown = true;
    event.preventDefault();
    stopOwnedEngine().then(() => app.exit(0));
  }
});
