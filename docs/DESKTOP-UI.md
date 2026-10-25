# Desktop UI

The `/app` route renders a fixed application frame. The page does not
scroll; each pane scrolls on its own. The frame's components are in
`ui/app/shell/`.

## Layout

```
+--------------------------------------------------------------+
| TitleBar                                                     |
+----+--------------+------------------------------------------+
| A  | SideBar      | Content                                  |
| c  |              |                                          |
| t  |              |                                          |
| i  |              |                                          |
| v  |              |                                          |
+----+--------------+------------------------------------------+
| StatusBar                                                    |
+--------------------------------------------------------------+
```

| Component | File | Size | Contents |
|---|---|---|---|
| Title bar | `TitleBar.tsx` | 36px high | "PixelKit" on the left, the open dataset or project name in the centre. It is the window's drag region. Under Electron on macOS the name is inset 76px to clear the window buttons. |
| Activity bar | `ActivityBar.tsx` | 48px wide | Explorer and Guide at the top, Settings at the bottom. The active item has a 2px accent bar on its left. Explorer closes any open dataset or project page and shows the side bar; Guide hides it. |
| Side bar | `SideBar.tsx`, `ExplorerPane.tsx` | 260px wide | The workspace tree: projects, their datasets, and each open dataset's sections (Overview, References, Dataset, Augmentations). Datasets outside any project are listed at the root. The header has New dataset, Refresh and Collapse buttons. |
| Content | `app/app/page.tsx` | remaining space | `HomeView` for the Explorer activity, `GuideView` for Guide. The scroll container is marked `data-app-scroll`. |
| Status bar | `StatusBar.tsx` | 24px high | Left: engine status dot, device (CUDA, Metal or CPU) and VRAM in use, the running job's progress, the workspace folder name. Right: the app version. |

The tree reads `GET /api/containers` and `GET /api/projects`. The status
bar polls `GET /api/health` every 5 s and `GET /api/jobs/active` every 2 s
while a job is queued or running, every 6 s otherwise.

## Views

| View | Where it shows |
|---|---|
| `HomeView` | Content pane: workspace, projects and dataset creation |
| `GuideView` | Content pane |
| `ProjectViewV2Stub` | An overlay over the content pane when a dataset is open, with its own scroll container |
| `SettingsView` | Full-screen overlay, from the Settings button |
| `SetupWizard` | Modal on first run while model weights are missing |

## URLs

| URL | Opens |
|---|---|
| `/app/<id>` | a dataset |
| `?tab=guide`, `?tab=workspaces` | the Guide or Explorer activity |
| `?project=<id>` | a project page in `HomeView` |
| `?profile=1` | Settings |

## Theme

The UI is dark. The tokens are set once on `:root` in `ui/app/globals.css`,
with `color-scheme: dark`.

| Token | Value |
|---|---|
| `--background` | `rgb(10 10 12)` |
| `--foreground` | `rgb(242 244 248)` |
| `--fg-soft` | `rgb(201 206 214)` |
| `--fg-muted`, `--muted` | `rgb(153 161 172)` |
| `--fg-dim` | `rgb(109 117 129)` |
| `--fg-faint` | `rgb(69 76 87)` |
| `--panel` | `rgba(255,255,255,0.02)` |
| `--surface`, `--panel-solid` | `rgb(15 15 17)` |
| `--surface-2` | `rgb(21 21 24)` |
| `--surface-hover` | `rgba(255,255,255,0.045)` |
| `--line-soft` | `rgba(255,255,255,0.05)` |
| `--line`, `--border` | `rgba(255,255,255,0.08)` |
| `--line-strong` | `rgba(255,255,255,0.16)` |
| `--accent`, `--accent-orange` | `rgb(255 121 0)` |
| `--accent-dim` | `rgba(255,121,0,0.14)` |
| `--accent-contrast` | `rgb(26 18 5)`, text on the accent |
| `--ok` | `rgb(61 220 151)` |
| `--warn` | `rgb(245 184 61)` |
| `--bad` | `rgb(255 93 93)` |
| `--ease-out` | `cubic-bezier(0.16, 1, 0.3, 1)` |

Tailwind has three colours, `background`, `foreground` and `accent`, which
take opacity modifiers such as `text-foreground/55`. Everything else uses the
tokens directly, as in `bg-[var(--panel)]`.

## Style

| Element | Rule |
|---|---|
| Micro-labels: pane headers, status segments, stat keys | `.pk-micro`: mono, 11px, uppercase, `0.12em` tracking, `--fg-dim` |
| Numbers | `.pk-num`: tabular figures |
| Panels and cards | `.pk-card`: `--panel` fill, 1px `--line` border, no shadow |
| Buttons | `.pk-btn`: 6px radius, 1px border, `--surface-hover` on hover. `.pk-btn-primary`: `--accent` fill, `--accent-contrast` text |
| Accent | Active states, primary actions and the focus ring. Status dots use `--ok`, `--warn` and `--bad`. |
| Motion | `.pk-up` and `.pk-pop`, on `--ease-out` |
| Shell | 13px text, 24px tree rows, radius at most 6px |
