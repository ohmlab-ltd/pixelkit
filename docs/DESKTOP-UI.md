# Desktop UI shell

The `/app` route renders a fixed-viewport, VS Code-style application frame
instead of the old website chrome (floating TopNav capsule + marketing-scale
headings + Footer). The page never scrolls; individual panes do. All shell
components live in `ui/app/shell/`.

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│ TitleBar (36px, draggable)          <dataset name when open> │
├───┬──────────────┬───────────────────────────────────────────┤
│ A │  SideBar     │  Content                                  │
│ c │  (260px,     │  (fills remainder, scrolls internally)    │
│ t │  collapsible)│                                           │
│ 48│              │                                           │
├───┴──────────────┴───────────────────────────────────────────┤
│ StatusBar (24px)                                             │
└──────────────────────────────────────────────────────────────┘
```

### TitleBar - `shell/TitleBar.tsx`
- 36px tall, `-webkit-app-region: drag` so a packaged Electron window can be
  moved by grabbing it.
- Left: "PixelKit" (13px, medium). When running inside Electron on macOS
  (`navigator.userAgent.includes("Electron")` + platform check) the label gets
  76px of left padding to clear the native traffic-light buttons.
- Centre: the open dataset/project name, when one is open. Right: empty.

### ActivityBar - `shell/ActivityBar.tsx`
- 48px vertical icon column: Explorer, Models, Guide, then a spacer and the
  Settings gear pinned at the bottom. Inline stroke SVGs at 20px.
- Active item: 2px left accent bar (`--accent`) + full-opacity icon.
  Inactive: 55% opacity, 85% on hover.
- Clicking the active item again collapses/expands the side bar. Guide has no
  side-bar pane, so selecting it hides the side bar entirely.

### SideBar - `shell/SideBar.tsx` (260px)
- **Explorer pane** (`shell/ExplorerPane.tsx`): "EXPLORER" header (11px
  uppercase, tracking-wide) with "+" (new dataset - fires the
  `requestNewDataset()` event on `lib/appNav.ts`, which HomeView answers by
  opening its existing CreateDatasetModal onboarding entry) and a refresh
  button. Below it, the workspace tree: Projects (containers) as expandable
  nodes with their datasets as children; datasets outside any Project at root
  level. Rows are 24px / 13px; dataset rows show a right-aligned dim image
  count; the open dataset gets a foreground/8% background. Clicking a dataset
  goes through the same `openProj` path the workspace cards use (correct `v2`
  flag; container-child datasets pass the container id so "Back to project"
  works).
  - Data: `GET /api/containers` → `{containers: [{id, name, n_datasets, …}]}`
    and `GET /api/projects?owner=local&viewer=local&offset=0&limit=1000` →
    `{total, items}` where each item carries `id`, `name`, `n_images`, `v2`,
    `createdBy` and `container: {id, name} | null` (the reverse map used to
    parent datasets under their Project).
- **Models pane** (`shell/ModelsPane.tsx`): compact rows over `lib/models.ts`
  - label, state (not downloaded / downloading NN% with a thin progress bar /
  ready / loaded) and small Download / Load / Unload text-buttons calling the
  existing `downloadModel` / `loadModel` / `unloadModel` functions.

### Content pane
- The single scroll container for embedded views, marked `data-app-scroll`
  (GuideView and ScrollToTop target it for scroll-to-top).
- Explorer/Models activity → `HomeView` (kept mounted so returning from a
  dataset is instant). Guide activity → `GuideView`.

### StatusBar - `shell/StatusBar.tsx`
- 24px, 12px font, 1px top border. Segments highlight `foreground/8%` on hover.
- Left: engine dot (5s `GET /api/health` poll; green up / red down) + device
  label (`/api/settings` → "Metal" / "CUDA" / "CPU"), then the workspace
  folder name.
- Right: SAM 3 state (4s `GET /api/models/status` poll: "not installed" /
  "downloading NN%" / "ready" / "loaded"; clicking opens Settings), a compact
  theme toggle (same `useTheme` logic as the old ThemeToggle), and
  `v0.1.0-dev`.

## Tokens

The UI runs on the studio design language shared with the sibling products:
dark is the primary/brand look (near-black neutral ground, flat panels,
hairline borders, one orange accent); light mode mirrors the same system in
adapted neutrals via the `html.dark` class mechanism (dark is the first-visit
default).

### Colour system (`app/globals.css`)

| Token | Dark (default) | Light |
|---|---|---|
| `--background` | `rgb(10 10 12)` near-black neutral | `rgb(246 246 248)` |
| `--foreground` / `--fg` | `rgb(242 244 248)` | `rgb(23 25 30)` |
| `--fg-soft` | `rgb(201 206 214)` | `rgb(63 67 76)` |
| `--fg-muted` / `--muted` | `rgb(153 161 172)` | `rgb(92 97 108)` |
| `--fg-dim` | `rgb(109 117 129)` | `rgb(109 114 125)` |
| `--fg-faint` | `rgb(69 76 87)` | `rgb(155 160 171)` |
| `--panel` | `rgba(255,255,255,0.02)` flat panel | `rgba(23,25,30,0.025)` |
| `--surface` / `--panel-solid` | `rgb(15 15 17)` | `rgb(255 255 255)` |
| `--surface-2` | `rgb(21 21 24)` | `rgb(238 238 241)` |
| `--surface-hover` | `rgba(255,255,255,0.045)` | `rgba(23,25,30,0.05)` |
| `--line-soft` | `rgba(255,255,255,0.05)` | `rgba(23,25,30,0.06)` |
| `--line` / `--border` | `rgba(255,255,255,0.08)` hairline | `rgba(23,25,30,0.10)` (`--border` 0.12) |
| `--line-strong` | `rgba(255,255,255,0.16)` | `rgba(23,25,30,0.18)` |
| `--accent` / `--accent-orange` | `#ff7900` PixelKit orange | `#c04a00` (deepened for AA) |
| `--accent-dim` | `rgba(255,121,0,0.14)` | `rgba(192,74,0,0.12)` |
| `--accent-contrast` (text ON accent) | `rgb(26 18 5)` | `#ffffff` |
| `--ok` (alias `--success`) | `#3ddc97` | `#0c7f52` |
| `--warn` (alias `--warning`) | `#f5b83d` | `#9c6500` |
| `--bad` (alias `--destructive`) | `#ff5d5d` | `#d92626` |
| `--ease-out` | `cubic-bezier(0.16, 1, 0.3, 1)` | same |

Legacy var names (`--background-rgb`, `--foreground-rgb`, `--accent-rgb`,
`--surface-rgb`, `--border`, `--success`, `--destructive`, `--brand-1/2/3`, …)
are mapped onto this system so every existing consumer shifts automatically;
the `--brand-*` gradient triplets are flattened to the single accent. New code
should prefer the ramp vars (or the matching Tailwind colours `ok` / `warn` /
`bad` / `panel` / `line*` / `fg-*` / `accent-dim`).

### Language rules

| Rule | Value |
|---|---|
| Micro-labels (pane headers, status segments, stat keys, eyebrows) | uppercase mono, 11px, `tracking-[0.12em]`, `--fg-dim` (`.pk-micro`) |
| Counts / figures | `tabular-nums` (`.pk-num`), quiet weight (400–500) |
| Panels / cards | flat `--panel` + 1px `--line` hairline, no elevation (`.pk-card`) |
| Buttons | compact flat, 6px radius (`rounded-md`), hairline border, hover `--surface-hover` (`.pk-btn`); primary = flat `--accent` fill + `--accent-contrast` label (`.pk-btn-primary`) |
| Accent usage | sparing: active states, primary actions, focus ring; status dots stay semantic `--ok`/`--warn`/`--bad` |
| Motion | one vocabulary: rise/fade in once on `--ease-out` (`.pk-up`, `.pk-pop`), state changes ease |

### Shell metrics

| Token | Value |
|---|---|
| Title bar height | 36px (`h-9`) |
| Activity bar width | 48px (`w-12`), icons 20px |
| Side bar width | 260px |
| Status bar height | 24px (`h-6`), mono 11px uppercase segments |
| Shell base font | 13px |
| Tree row height | 24px (`h-6`) |
| Pane headers | `.pk-micro` (11px uppercase mono, tracked) |
| Borders | flat 1px hairline `var(--border)` / `var(--line)` |
| Border radius | ≤ 6px (no pill shapes in shell chrome) |

## Where the legacy views live now

| View | Location in the shell |
|---|---|
| `HomeView` (workspace + full V2 onboarding) | Content pane, Explorer/Models activities. Densified: headings dropped to text-2xl, sections widened to `max-w-[1400px]`, Footer removed. Onboarding logic untouched. |
| `GuideView` | Content pane, Guide activity (scrolls within the pane; unchanged content, still serves the standalone `/guide` route). |
| `ProjectViewV2Stub` / `ProjectView` (dataset views) | Fixed overlays above the content region - below the title bar, above the status bar, right of the activity/side bars - each with its own scroll container. Internals untouched. |
| `SettingsView` | Unchanged full-screen overlay; opens from the Settings gear and the SAM 3 status segment. |
| `SetupWizard` | Unchanged modal; mounts on first run when model weights are missing. |
| `TopNav` | Deleted - replaced by TitleBar + ActivityBar + StatusBar. |
| `ScrollToTop` | Now discovers its nearest scrollable ancestor (pane/overlay) instead of assuming window scroll. |

Deep-links keep working: `/app/<id>` opens a dataset, `?tab=guide` (and legacy
`?tab=workspaces`) select the activity, `?project=<id>` opens a Project page
inside HomeView, `?profile=1` reopens Settings.
