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

### TitleBar — `shell/TitleBar.tsx`
- 36px tall, `-webkit-app-region: drag` so a packaged Electron window can be
  moved by grabbing it.
- Left: "PixelKit" (13px, medium). When running inside Electron on macOS
  (`navigator.userAgent.includes("Electron")` + platform check) the label gets
  76px of left padding to clear the native traffic-light buttons.
- Centre: the open dataset/project name, when one is open. Right: empty.

### ActivityBar — `shell/ActivityBar.tsx`
- 48px vertical icon column: Explorer, Models, Guide, then a spacer and the
  Settings gear pinned at the bottom. Inline stroke SVGs at 20px.
- Active item: 2px left accent bar (`--accent`) + full-opacity icon.
  Inactive: 55% opacity, 85% on hover.
- Clicking the active item again collapses/expands the side bar. Guide has no
  side-bar pane, so selecting it hides the side bar entirely.

### SideBar — `shell/SideBar.tsx` (260px)
- **Explorer pane** (`shell/ExplorerPane.tsx`): "EXPLORER" header (11px
  uppercase, tracking-wide) with "+" (new dataset — fires the
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
  — label, state (not downloaded / downloading NN% with a thin progress bar /
  ready / loaded) and small Download / Load / Unload text-buttons calling the
  existing `downloadModel` / `loadModel` / `unloadModel` functions.

### Content pane
- The single scroll container for embedded views, marked `data-app-scroll`
  (GuideView and ScrollToTop target it for scroll-to-top).
- Explorer/Models activity → `HomeView` (kept mounted so returning from a
  dataset is instant). Guide activity → `GuideView`.

### StatusBar — `shell/StatusBar.tsx`
- 24px, 12px font, 1px top border. Segments highlight `foreground/8%` on hover.
- Left: engine dot (5s `GET /api/health` poll; green up / red down) + device
  label (`/api/settings` → "Metal" / "CUDA" / "CPU"), then the workspace
  folder name.
- Right: SAM 3 state (4s `GET /api/models/status` poll: "not installed" /
  "downloading NN%" / "ready" / "loaded"; clicking opens Settings), a compact
  theme toggle (same `useTheme` logic as the old ThemeToggle), and
  `v0.1.0-dev`.

## Tokens

| Token | Value |
|---|---|
| Title bar height | 36px (`h-9`) |
| Activity bar width | 48px (`w-12`), icons 20px |
| Side bar width | 260px |
| Status bar height | 24px (`h-6`), 12px font |
| Shell base font | 13px |
| Tree row height | 24px (`h-6`) |
| Pane headers | 11px uppercase, tracking-wide |
| Borders | flat 1px `var(--border)` / foreground-mix |
| Border radius | ≤ 6px (no pill shapes in shell chrome) |

Both themes work off the existing CSS variables in `app/globals.css`
(`--background`, `--foreground`, `--border`, `--accent`, `--success`,
`--destructive`); the shell introduces no new colour tokens.

## Where the legacy views live now

| View | Location in the shell |
|---|---|
| `HomeView` (workspace + full V2 onboarding) | Content pane, Explorer/Models activities. Densified: headings dropped to text-2xl, sections widened to `max-w-[1400px]`, Footer removed. Onboarding logic untouched. |
| `GuideView` | Content pane, Guide activity (scrolls within the pane; unchanged content, still serves the standalone `/guide` route). |
| `ProjectViewV2Stub` / `ProjectView` (dataset views) | Fixed overlays above the content region — below the title bar, above the status bar, right of the activity/side bars — each with its own scroll container. Internals untouched. |
| `SettingsView` | Unchanged full-screen overlay; opens from the Settings gear and the SAM 3 status segment. |
| `SetupWizard` | Unchanged modal; mounts on first run when model weights are missing. |
| `TopNav` | Deleted — replaced by TitleBar + ActivityBar + StatusBar. |
| `ScrollToTop` | Now discovers its nearest scrollable ancestor (pane/overlay) instead of assuming window scroll. |

Deep-links keep working: `/app/<id>` opens a dataset, `?tab=guide` (and legacy
`?tab=workspaces`) select the activity, `?project=<id>` opens a Project page
inside HomeView, `?profile=1` reopens Settings.
