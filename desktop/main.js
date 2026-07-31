// PixelKit desktop shell.
//
// Wraps the local PixelKit engine (FastAPI on 127.0.0.1:8001) in a native
// window. If an engine is already running (e.g. started from a terminal, or
// mid-download of model weights) it is REUSED and never killed; otherwise the
// shell spawns its own engine and shuts it down on quit.

'use strict';

const { app, BrowserWindow, Menu, screen, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

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
      res.resume(); // drain
      resolve(res.statusCode === 200);
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

function engineLogPath() {
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Logs', 'PixelKit', 'engine.log');
  }
  return path.join(app.getPath('logs'), 'engine.log');
}

// Resolution order: PIXELKIT_PYTHON env var, the repo venv next to this
// directory, then whatever `python3` is on PATH.
// TODO(packaging): when the app is packaged, the engine + a bundled Python
// env should live in process.resourcesPath; add that lookup here.
function resolvePython() {
  if (process.env.PIXELKIT_PYTHON) return process.env.PIXELKIT_PYTHON;
  const venvPython =
    process.platform === 'win32'
      ? path.join(__dirname, '..', 'engine', '.venv', 'Scripts', 'python.exe')
      : path.join(__dirname, '..', 'engine', '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) return venvPython;
  return 'python3';
}

function spawnEngine() {
  const python = resolvePython();
  const cwd = path.join(__dirname, '..', 'engine', 'gd');
  const logFile = engineLogPath();

  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  const log = fs.createWriteStream(logFile, { flags: 'a' });
  log.write(`\n--- PixelKit engine start ${new Date().toISOString()} (${python}) ---\n`);

  const proc = spawn(python, ['server.py'], {
    cwd,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.pipe(log);
  proc.stderr.pipe(log);
  proc.on('exit', () => {
    engine.exited = true;
  });
  proc.on('error', () => {
    // e.g. python binary not found; poll loop will time out and report.
    engine.exited = true;
  });

  engine.proc = proc;
  engine.external = false;
}

async function ensureEngine() {
  if (await checkHealth()) {
    engine.external = true; // someone else owns it -> never kill it
    return true;
  }

  spawnEngine();

  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (engine.exited) return false; // crashed / python missing
    if (await checkHealth()) return true;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return false;
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
  </style></head><body><div class="card">${body}</div></body></html>`;
  return 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
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
    splashHtml('<h1>PixelKit</h1><p>Starting engine…</p><div class="spin"></div>')
  );
}

function showSplashError() {
  const logFile = engineLogPath().replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const body =
    '<h1>PixelKit</h1>' +
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

  mainWindow.on('close', () => saveWindowState(mainWindow));
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.loadURL(APP_URL);
}

// ---------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------

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
    ...(isMac ? [] : [{ role: 'help', submenu: [{ role: 'about' }] }]),
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
  } else {
    showSplashError();
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
  if (engine.proc && !engine.external && !engine.exited) {
    shuttingDown = true;
    event.preventDefault();
    stopOwnedEngine().then(() => app.exit(0));
  }
});
