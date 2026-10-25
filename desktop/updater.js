// Update check: a dialog offers the download page when GitHub has a newer release.

'use strict';

const { app, dialog, shell } = require('electron');
const fs = require('fs');
const https = require('https');
const path = require('path');

const REPO = 'ohmlab-ltd/pixelkit';
const RELEASES_PAGE = `https://github.com/${REPO}/releases/latest`;

const stateFile = () => path.join(app.getPath('userData'), 'update-check.json');

function loadState() {
  try {
    return JSON.parse(fs.readFileSync(stateFile(), 'utf8'));
  } catch {
    return {};
  }
}

function saveState(state) {
  try {
    fs.mkdirSync(app.getPath('userData'), { recursive: true });
    fs.writeFileSync(stateFile(), JSON.stringify(state, null, 2));
  } catch {
    // best effort
  }
}

// "v1.2.3" or "1.2.3" -> [1, 2, 3]; null for anything else, prereleases included.
function parseVersion(tag) {
  const m = /^v?(\d+)\.(\d+)\.(\d+)$/.exec((tag || '').trim());
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}

function isNewer(a, b) {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] > b[i];
  }
  return false;
}

function fetchLatestRelease() {
  return new Promise((resolve, reject) => {
    const req = https.get(
      `https://api.github.com/repos/${REPO}/releases/latest`,
      { headers: { 'User-Agent': `PixelKit/${app.getVersion()}` } },
      (res) => {
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode}`));
        }
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            reject(e);
          }
        });
      }
    );
    req.setTimeout(10000, () => req.destroy(new Error('update check timed out')));
    req.on('error', reject);
  });
}

async function checkForUpdates(parentWindow) {
  if (process.env.PIXELKIT_NO_UPDATE_CHECK) return;
  let release;
  try {
    release = await fetchLatestRelease();
  } catch {
    return;
  }

  const latest = parseVersion(release.tag_name);
  const current = parseVersion(app.getVersion());
  if (!latest || !current || !isNewer(latest, current)) return;
  if (loadState().skippedVersion === release.tag_name) return;
  if (!parentWindow || parentWindow.isDestroyed()) return;

  const { response } = await dialog.showMessageBox(parentWindow, {
    type: 'info',
    title: 'PixelKit update',
    message: `PixelKit ${release.tag_name.replace(/^v/, '')} is available`,
    detail: `You have ${app.getVersion()}. Download opens the release page on GitHub; run the new installer from there to update.`,
    buttons: ['Download', 'Skip this version', 'Later'],
    defaultId: 0,
    cancelId: 2,
  });

  if (response === 0) shell.openExternal(release.html_url || RELEASES_PAGE);
  else if (response === 1) saveState({ skippedVersion: release.tag_name });
}

module.exports = { checkForUpdates };
