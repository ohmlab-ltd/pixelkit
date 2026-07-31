"use client";

import { useEffect, useRef, useState } from "react";
import { HomeView } from "../HomeView";
import { ProjectView } from "../ProjectView";
import { ScrollToTop } from "../components/ScrollToTop";
import { GuideView } from "../GuideView";
import { SettingsView } from "../SettingsView";
import { SetupWizard, setupNeeded } from "../SetupWizard";
import { TitleBar } from "../shell/TitleBar";
import { ActivityBar, type ActivityKey } from "../shell/ActivityBar";
import { SideBar } from "../shell/SideBar";
import { StatusBar } from "../shell/StatusBar";
import { broadcastCurrentTab, onAppNavigate, requestNewDataset } from "@/lib/appNav";
import type { ReferenceImage } from "../v2/OnboardReferencesV2";
import { ProjectViewV2Stub } from "../v2/ProjectViewV2Stub";
import { patchProjectMeta, readProjectMeta } from "@/lib/projectMetaCache";
import { apiFetch } from "@/lib/apiFetch";
import { fetchModelsStatus } from "@/lib/models";

// Result of the full V2 onboarding flow. HomeView owns every stage
// (name + labels + references) and fires onV2Begin once with the
// complete payload, we just mount the post-onboarding stub.
type V2Result = {
  name: string;
  labels: string[];
  references: ReferenceImage[];
  projectId: string | null;
  // Owner username pulled from the manifest. Used to render the curator's
  // @handle in the project chrome (not the viewer's).
  owner?: string | null;
  // The caller's effective WRITE access, resolved from /access. True for the
  // dataset's creator AND for any editor/owner of the Project it belongs to —
  // so a Project editor can edit a dataset another member created. Undefined
  // until resolved (read-only is the safe default meanwhile).
  writable?: boolean;
  // The Project (container) this dataset was opened FROM, if any. Drives the
  // dataset view's "Back to project" button + return navigation.
  fromProjectId?: string | null;
  // First-load handoff hint from onboarding:
  //   "general"  → HomeView's "Reading between the labels…" overlay
  //                already carried through as "Loading project…";
  //                ProjectViewV2Stub should NOT show its own full-
  //                screen mount loader (would be a second full-screen
  //                takeover on top of HomeView's).
  //   "specific" → user just finished references; render a smaller
  //                "Loading project…" loader card inside the page
  //                rather than the full-screen mount loader.
  //   null       → normal load, ProjectViewV2Stub uses its default
  //                full-screen mount loader.
  firstLoad?: "general" | "specific" | null;
};

// Legacy tab identity kept for the appNav bus + ProjectViewV2Stub's
// originTab prop ("Back to …" copy). The desktop shell's activity
// state maps onto it: guide ↔ "guide", everything else "workspaces".
type OriginTab = "workspaces" | "guide";

export default function Page() {
  // Desktop-shell state: which activity-bar item is active, and
  // whether the 260px side bar is expanded. Clicking the active
  // activity icon again collapses/expands the side bar.
  const [activity, setActivity] = useState<ActivityKey>("explorer");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [openProject, setOpenProject] = useState<string | null>(null);
  const [openProjectName, setOpenProjectName] = useState<string>("");
  const [profileOpen, setProfileOpen] = useState(false);
  // V2 post-onboarding stub. HomeView owns the entire onboarding
  // (name + labels + references) inline so the V2 path here is a
  // single piece of state, flipping back to V1 is one tweak.
  const [openV2Project, setOpenV2Project] = useState<V2Result | null>(null);
  // First-run setup wizard. Shown when the engine reports missing model
  // weights AND the user hasn't dismissed it before ("pk-setup-dismissed").
  const [setupOpen, setSetupOpen] = useState(false);

  const [projectOriginTab, setProjectOriginTab] = useState<OriginTab>("workspaces");
  // Set when the user deep-links to a project id that the backend
  // doesn't recognise (deleted, never existed, wrong owner).
  // Drives the centred "This project cannot be found…" overlay.
  const [notFoundProjectId, setNotFoundProjectId] = useState<string | null>(null);

  // Browser-URL sync. /app stays the SPA root; /app/<id> renders
  // the same page but with the matching project pre-opened. We
  // push the URL on every open so a reload or back-button keeps
  // the user on the right project, and the workspace card +
  // sidebar tree can both produce shareable deep-links.
  const syncUrl = (id: string | null) => {
    if (typeof window === "undefined") return;
    const target = id ? `/app/${id}` : "/app";
    if (window.location.pathname === target) return;
    window.history.pushState(null, "", target);
  };

  const openProj = (
    id: string,
    owner: string = "",
    displayName: string = "",
    v2 = false,
    fromProjectId?: string,
  ) => {
    setProjectOriginTab(activity === "guide" ? "guide" : "workspaces");
    setNotFoundProjectId(null);
    syncUrl(id);
    if (v2) {
      // Opening a V2 dataset replaces any open V1 view (the sidebar
      // tree can switch datasets without going through the workspace).
      setOpenProject(null);
      // V2 project: seed name + labels from the per-project meta
      // cache (populated by HomeView's workspace list refresh) so
      // the page paints with chips, dataset health, and the
      // expected reference count already wired up. The manifest
      // fetch that follows just confirms / refines those values.
      // References themselves hydrate from manifest in a follow-up
      // effect inside ProjectViewV2Stub. Mounting immediately with
      // the cached labels avoids the few-hundred-ms flash where the
      // user sees an empty header before chips appear.
      const cached = readProjectMeta(id);
      setOpenV2Project({
        name: cached?.name ?? displayName,
        labels: cached?.labels ?? [],
        references: [],
        projectId: id,
        // Cache hit → use it. Else the caller-supplied hint (set by
        // /projects card clicks). Else null until the manifest fetch
        // below resolves. readOnly defaults to true while null so a
        // direct refresh of someone else's project never flashes the
        // owner chrome.
        owner: owner || cached?.owner || null,
        fromProjectId: fromProjectId ?? null,
      });
      // Slim /overview projection, multi-MB legacy /api/projects/{id}
      // returns the whole manifest, which is 3 s+ on big projects when
      // we only need name + tags + owner here. imports_limit=0 hits the
      // backend's sidecar fast-path so this is ~200 ms regardless of
      // project size. Use apiFetch so this dedups against the same URL
      // fetched inside ProjectViewV2Stub when both mount together.
      apiFetch(`/api/v2/projects/${id}/overview?imports_limit=0`)
        .then((r) => (r.ok ? r.json() : null))
        .then((m) => {
          if (!m) return;
          const resolvedOwner = m.owner ?? m.createdBy ?? null;
          setOpenV2Project((cur) =>
            cur && cur.projectId === id
              ? {
                  ...cur,
                  name: m.name ?? cur.name,
                  labels: m.tags ?? cur.labels,
                  // Manifest is canonical for the owner string, overrides the
                  // caller-supplied hint so deep-links from /app/<id> (where the
                  // owner arg is "") still display the curator's handle.
                  owner: resolvedOwner ?? cur.owner ?? null,
                }
              : cur,
          );
          // Cache the owner for the next /app/<id> refresh so readOnly
          // resolves on first paint instead of after this fetch returns.
          if (resolvedOwner) {
            patchProjectMeta(id, { owner: resolvedOwner });
          }
        })
        .catch(() => { /* silent, page still works with empty refs */ });
      // Resolve WRITE access in parallel: editors of a Project can edit a
      // dataset another member created, so read-only must follow write access,
      // not ownership. Default stays read-only until this lands.
      apiFetch(`/api/v2/projects/${id}/access`)
        .then((r) => (r.ok ? r.json() : null))
        .then((a) => {
          if (!a) return;
          setOpenV2Project((cur) =>
            cur && cur.projectId === id ? { ...cur, writable: !!a.writable } : cur,
          );
        })
        .catch(() => { /* silent — owner check still grants the creator access */ });
      return;
    }
    setOpenV2Project(null);
    setOpenProject(id);
    setOpenProjectName(displayName);
  };

  // Portable build: no accounts. Everything renders as the local user —
  // there is no logged-out state (the /login route doesn't exist).
  const loggedIn = true;

  // A `?tab=` query param deep-links into a specific view without needing a
  // separate top-level route per view (legacy values: "workspaces" maps to
  // the Explorer activity). `?profile=1` re-opens the settings pane after a
  // hard reload (e.g. post-settings-save); strip the param afterwards so
  // back/refresh don't reopen it forever.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const search = new URLSearchParams(window.location.search);
    const t = search.get("tab");
    if (t === "guide") setActivity("guide");
    else if (t === "workspaces") setActivity("explorer");
    if (search.get("profile") === "1") {
      setProfileOpen(true);
      const cleaned = new URLSearchParams(window.location.search);
      cleaned.delete("profile");
      const next = cleaned.toString();
      window.history.replaceState(null, "", next ? `?${next}` : window.location.pathname);
    }
  }, []);

  // First-run setup: ask the engine once whether the model weights are in
  // place. Engine may be down / restarting — swallow the error and skip the
  // wizard (Settings can always reach it later).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await fetchModelsStatus();
        if (cancelled) return;
        if (
          setupNeeded(status) &&
          window.localStorage.getItem("pk-setup-dismissed") !== "1"
        ) {
          setSetupOpen(true);
        }
      } catch {
        /* engine unreachable — no wizard */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Broadcast the resolved view so components on the legacy appNav bus
  // keep working. The Explorer/Models activities both surface the
  // workspace, so they broadcast as "workspaces".
  useEffect(() => {
    broadcastCurrentTab(activity === "guide" ? "guide" : "workspaces");
  }, [activity]);

  // Hydrate openProject from the URL on first mount. /app/<id>
  // means "open this project on load" so deep-links + reloads
  // bring the user straight back to the same view they shared.
  // Also listens for popstate so the browser back/forward
  // buttons toggle the project overlay correctly.
  const urlHydratedRef = useRef(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handle = async () => {
      const m = window.location.pathname.match(/^\/app\/([^\/?#]+)/);
      const idFromUrl = m ? m[1] : null;
      if (idFromUrl) {
        if (
          openProject !== idFromUrl
          && (openV2Project?.projectId ?? null) !== idFromUrl
          && notFoundProjectId !== idFromUrl
        ) {
          // Optimistic open: mount the V2 view IMMEDIATELY and let
          // ProjectViewV2Stub's /initial fetch surface 404 + display
          // name once it lands. The previous serial probe here was
          // the dominant cause of "10 s to first paint" on slow
          // connections, it forced an /overview round-trip BEFORE
          // the stub could even mount, blocking /initial from firing
          // for a full RTT. Cached name from readProjectMeta seeds
          // the title; missing name renders as "" until /initial
          // arrives a moment later.
          const cached = readProjectMeta(idFromUrl);
          setNotFoundProjectId(null);
          openProj(idFromUrl, "", cached?.name ?? "", true);
          // Fire /initial in parallel JUST for the 404 / name update.
          // The stub itself ALSO calls /initial on mount and the two
          // requests dedup at apiFetch's in-flight Map, net one
          // network round-trip. We don't await this so the optimistic
          // mount above isn't blocked.
          void apiFetch(`/api/v2/projects/${idFromUrl}/initial?n=20`, { cache: "no-store" })
            .then(async (r) => {
              if (r.status === 404) {
                setNotFoundProjectId(idFromUrl);
                setOpenProject(null);
                setOpenV2Project(null);
                return;
              }
              if (!r.ok) return;
              const m = (await r.json()) as { name?: string };
              if (m?.name) {
                // Re-write the optimistic placeholder name once the
                // canonical comes back. Cheap setState, React dedups
                // identical strings.
                setOpenV2Project((cur) =>
                  cur && cur.projectId === idFromUrl
                    ? { ...cur, name: m.name ?? cur.name }
                    : cur,
                );
              }
            })
            .catch(() => { /* network blip, retry on next nav */ });
        }
      } else {
        setNotFoundProjectId(null);
        setOpenProject(null);
        setOpenV2Project(null);
      }
    };
    if (!urlHydratedRef.current) {
      urlHydratedRef.current = true;
      void handle();
    }
    const listener = () => { void handle(); };
    window.addEventListener("popstate", listener);
    return () => window.removeEventListener("popstate", listener);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Listen for in-app navigation events. Lets any nested component ask to
  // switch views without prop-drilling. Legacy tabs (pricing / projects /
  // terminal) no longer exist in the portable build, ignore those events.
  useEffect(() => {
    return onAppNavigate((next) => {
      if (next !== "workspaces" && next !== "guide") return;
      setActivity(next === "guide" ? "guide" : "explorer");
      setOpenProject(null);
      setOpenV2Project(null);
      setProfileOpen(false);
      syncUrl(null);
    });
  }, []);

  // Portable build: the single local user, no session round-trip.
  const user = {
    name: "Local",
    username: "local",
    email: "local@pixelkit.local",
    image: null,
  };

  // Activity-bar click: switch views, or collapse/expand the side bar
  // when the active icon is clicked again. Guide has no side-bar pane.
  const handleActivitySelect = (key: ActivityKey) => {
    if (key === activity) {
      if (key !== "guide") setSidebarOpen((v) => !v);
      return;
    }
    setActivity(key);
    setProfileOpen(false);
    if (key === "guide") {
      // Guide replaces the content pane — close any dataset overlay so
      // the user actually sees it (mirrors the old tab-switch flow).
      setOpenProject(null);
      setOpenV2Project(null);
      setNotFoundProjectId(null);
      syncUrl(null);
    } else {
      setSidebarOpen(true);
    }
  };

  // Side-bar "+": make the workspace visible again, then hand off to
  // HomeView's existing onboarding entry (the CreateDatasetModal) via
  // the appNav event bus — HomeView stays mounted for every
  // non-guide activity, so the listener is guaranteed to be attached.
  const handleNewDataset = () => {
    setActivity("explorer");
    setOpenProject(null);
    setOpenV2Project(null);
    setProfileOpen(false);
    setNotFoundProjectId(null);
    syncUrl(null);
    requestNewDataset();
  };

  const openDatasetId = openProject ?? openV2Project?.projectId ?? null;
  const titleBarName = openV2Project
    ? openV2Project.name
    : openProject
    ? (openProjectName || openProject)
    : "";

  const sidebarVisible = sidebarOpen && activity !== "guide";
  // Dataset views (and the not-found message) render as fixed overlays
  // above the content pane: below the title bar, above the status bar,
  // and to the right of the activity bar + side bar — so the window
  // chrome and the Explorer tree stay reachable while a dataset is
  // open. Their internal modals (BoxEditor, viewers) are fixed
  // inset-0 with higher z-indices and still cover everything, which
  // matches their previous behaviour.
  const overlayCls = [
    "fixed top-9 bottom-6 right-0 z-[200] bg-[var(--background)]",
    sidebarVisible ? "left-[308px]" : "left-12",
  ].join(" ");

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[var(--background)] text-[var(--foreground)]">
      <TitleBar title={titleBarName} />
      <div className="flex min-h-0 flex-1">
        <ActivityBar
          activity={activity}
          onSelect={handleActivitySelect}
          onSettings={() => setProfileOpen(true)}
          settingsActive={profileOpen}
        />
        {sidebarVisible && (
          <SideBar
            pane={activity === "models" ? "models" : "explorer"}
            username={user.username}
            selectedDatasetId={openDatasetId}
            onOpenDataset={(ds) => {
              if (openDatasetId === ds.id) return;
              setProfileOpen(false);
              // Same code path as the workspace cards (openProj), with
              // the dataset's real v2 flag; datasets inside a Project
              // carry the container id so the V2 view's back button
              // returns to that Project page.
              openProj(ds.id, ds.owner, ds.name, ds.v2, ds.containerId ?? undefined);
            }}
            onNewDataset={handleNewDataset}
          />
        )}
        {/* Content pane: the ONLY scroll container for the embedded
            legacy views (HomeView / GuideView). data-app-scroll lets
            those views target it for their scroll-to-top calls. */}
        <div
          data-app-scroll
          className="relative min-w-0 flex-1 overflow-y-auto overscroll-contain"
        >
          {/* HomeView stays mounted under the project overlay so coming
              back from a project is instant, its `projects` state, the
              card grid, scroll position, and any in-flight pollers all
              survive the round-trip. Dataset views render on top as
              fixed overlays, so HomeView is hidden visually while one
              of those is up. The Models activity only swaps the side
              bar pane, so the workspace stays in the content slot. */}
          {loggedIn && activity !== "guide" && (
            <div hidden={!!profileOpen || !!openV2Project || !!openProject || !!notFoundProjectId}>
              <HomeView
                onOpen={openProj}
                onV2Begin={(name, labels, references, projectId, firstLoad) => {
                  // Reflect the freshly-created project in the address bar
                  // (/app/<id>) right away. Without this the URL stayed /app
                  // until the user navigated off and re-opened the project
                  // via openProj (which calls syncUrl) — so a reload or
                  // share-link mid-session lost the project.
                  if (projectId) syncUrl(projectId);
                  setOpenV2Project({
                    name,
                    labels,
                    references,
                    projectId,
                    // The creator is always the owner, without this the
                    // readOnly default fires and hides the drop card on
                    // the freshly-created project page until the
                    // /api/projects/<id> fetch resolves, which broke
                    // uploads on first-mount.
                    owner: user.username,
                    firstLoad: firstLoad ?? null,
                  });
                }}
                username={user.username}
                userImage={user.image}
                loggedIn={loggedIn}
              />
              {/* Back-to-top inside the workspace too — it finds the
                  content pane as its scroll container. The `hidden`
                  parent eats the button when an overlay is up. */}
              <ScrollToTop />
            </div>
          )}
          {activity === "guide" && <GuideView />}
        </div>
      </div>
      <StatusBar onOpenSettings={() => setProfileOpen(true)} />

      {notFoundProjectId ? (
        // Deep-link to a project the backend doesn't know about.
        // The shell chrome stays up; centre a quiet message in the
        // content region. No retry / back button — the Explorer tree
        // and activity bar remain reachable.
        <div className={overlayCls}>
          <main className="grid h-full place-items-center px-6">
            <p className="text-base text-foreground/85 font-light">
              This project cannot be found...
            </p>
          </main>
        </div>
      ) : profileOpen ? (
        <SettingsView onClose={() => setProfileOpen(false)} />
      ) : openV2Project ? (
        // Keyed by project id so switching datasets from the sidebar
        // tree remounts the view cleanly (it hydrates on mount).
        <div
          key={openV2Project.projectId ?? "v2-onboarding"}
          className={`${overlayCls} overflow-y-auto overscroll-contain`}
        >
          <ProjectViewV2Stub
            projectName={openV2Project.name}
            labels={openV2Project.labels}
            references={openV2Project.references}
            projectId={openV2Project.projectId}
            username={user.username}
            userImage={user.image}
            ownerUsername={openV2Project.owner ?? null}
            // Read-only is the SAFE default: until the manifest fetch
            // confirms this viewer owns the project, render the public
            // view (no drop card, no annotations card, etc). The owner
            // gets a brief flash of read-only at most, which is the
            // lesser evil compared to a non-owner momentarily seeing
            // owner-only chrome on every refresh of /app/<id>.
            readOnly={
              // Editable if you own it OR you have write access via the Project
              // (editor+). Read-only is the safe default until those resolve.
              !(
                (!!openV2Project.owner && openV2Project.owner === user.username) ||
                openV2Project.writable === true
              )
            }
            // Origin tab feeds the "Back to …" copy on the loader +
            // header.
            originTab={projectOriginTab}
            // When the dataset was opened from inside a Project, the back button
            // reads "Back to project" and returns to that Project page.
            backToProjectId={openV2Project.fromProjectId ?? null}
            // First-load hint from onboarding. "general" suppresses
            // the project's full-screen mount loader because HomeView
            // is still showing its overlay; "specific" swaps in the
            // smaller in-page "Loading project…" card.
            firstLoad={openV2Project.firstLoad ?? null}
            onClose={() => {
              const proj = openV2Project.fromProjectId;
              if (proj) {
                // Return to the Project page: pushing ?project=<id> + a synthetic
                // popstate clears this dataset view (app/page popstate handler)
                // AND re-opens the Project (HomeView's ?project deep-link).
                window.history.pushState(null, "", `/app?project=${proj}`);
                window.dispatchEvent(new PopStateEvent("popstate"));
              } else {
                setOpenV2Project(null);
                syncUrl(null);
              }
            }}
            onReferencesChange={(next) =>
              setOpenV2Project((cur) =>
                cur ? { ...cur, references: next } : cur,
              )
            }
          />
          {/* Back-to-top is rendered INSIDE ProjectViewV2Stub so it can
              suppress itself while the image viewer or augmentations
              viewer modal is open. */}
        </div>
      ) : openProject ? (
        // No key here: ProjectView refetches itself when `name` changes
        // (rename + tree switches), and keying by name would force a
        // full remount on every rename.
        <div className={`${overlayCls} overflow-y-auto overscroll-contain`}>
          <ProjectView
            name={openProject}
            initialDisplayName={openProjectName}
            username={user.username}
            // Portable build: single local user, every legacy (V1) dataset on
            // this machine is theirs — always editable.
            readOnly={false}
            onClose={() => { setOpenProject(null); syncUrl(null); }}
            onRename={(newName) => { setOpenProject(newName); syncUrl(newName); }}
          />
          <ScrollToTop />
        </div>
      ) : null}
      {setupOpen && (
        <SetupWizard
          onClose={() => {
            window.localStorage.setItem("pk-setup-dismissed", "1");
            setSetupOpen(false);
          }}
        />
      )}
    </div>
  );
}
