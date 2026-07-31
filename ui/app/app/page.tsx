"use client";

import { useEffect, useRef, useState } from "react";
import { HomeView } from "../HomeView";
import { ProjectView } from "../ProjectView";
import { ScrollToTop } from "../components/ScrollToTop";
import { GuideView } from "../GuideView";
import { SettingsView } from "../SettingsView";
import { SetupWizard, setupNeeded } from "../SetupWizard";
import { TopNav, NavTab } from "../TopNav";
import { broadcastCurrentTab, onAppNavigate } from "@/lib/appNav";
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

export default function Page() {
  const [tab, setTab] = useState<NavTab>("workspaces");
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

  const [projectOriginTab, setProjectOriginTab] = useState<NavTab>("workspaces");
  // Set when the user deep-links to a project id that the backend
  // doesn't recognise (deleted, never existed, wrong owner).
  // Drives the centred "This project cannot be found…" overlay.
  const [notFoundProjectId, setNotFoundProjectId] = useState<string | null>(null);

  // Browser-URL sync. /app stays the SPA root; /app/<id> renders
  // the same page but with the matching project pre-opened. We
  // push the URL on every open so a reload or back-button keeps
  // the user on the right project, and the workspace card +
  // public feed can both produce shareable deep-links.
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
    setProjectOriginTab(tab);
    syncUrl(id);
    if (v2) {
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
    setOpenProject(id);
    setOpenProjectName(displayName);
  };

  // Portable build: no accounts. Everything renders as the local user —
  // there is no logged-out state (the /login route doesn't exist).
  const loggedIn = true;

  // A `?tab=` query param deep-links into a specific tab without needing a
  // separate top-level route per tab. `?profile=1` re-opens the settings
  // pane after a hard reload (e.g. post-settings-save); strip the param
  // afterwards so back/refresh don't reopen it forever.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const search = new URLSearchParams(window.location.search);
    const t = search.get("tab");
    const valid: NavTab[] = ["workspaces", "guide"];
    if (t && (valid as string[]).includes(t)) setTab(t as NavTab);
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

  // Broadcast the resolved tab so root-layout components (e.g.
  // ScrollToTop) can gate themselves on the current section without
  // having to read query params, the URL doesn't update when tabs
  // change in place, so URL-driven gates miss every tab switch.
  useEffect(() => {
    broadcastCurrentTab(tab);
  }, [tab]);

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
  // switch tabs without prop-drilling. Legacy tabs (pricing / projects /
  // terminal) no longer exist in the portable build, ignore those events.
  useEffect(() => {
    return onAppNavigate((next) => {
      if (next !== "workspaces" && next !== "guide") return;
      setTab(next);
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

  return (
    <>
      <TopNav
        current={(openProject || openV2Project) ? projectOriginTab : tab}
        onNavigate={(t) => {
          // Clicking the tab you're already on is a "scroll to top"
          // gesture, not a no-op. Only counts when no project (V1
          // OR V2) / profile pane is open over the tab, otherwise
          // the click should close that and switch tabs as normal.
          const onSameTab =
            t === tab && !openProject && !openV2Project && !profileOpen;
          if (onSameTab) {
            window.scrollTo({ top: 0, behavior: "smooth" });
            return;
          }
          setTab(t);
          setOpenProject(null);
          setOpenV2Project(null);
          setProfileOpen(false);
          setNotFoundProjectId(null);
          syncUrl(null);
        }}
        onProfile={() => {
          // Already on the profile pane → smooth-scroll to top
          // instead of a no-op (mirrors the iOS "tap status bar"
          // pattern). Otherwise open the pane.
          if (profileOpen) {
            window.scrollTo({ top: 0, behavior: "smooth" });
          } else {
            setProfileOpen(true);
          }
        }}
        onHome={() => {
          if (profileOpen) {
            window.scrollTo({ top: 0, behavior: "smooth" });
            return;
          }
          // Hard navigation so the marketing home re-mounts cleanly.
          // router.push("/") sometimes left the /app SPA shell up
          // because the layout boundary isn't crossed; window.location
          // forces a real GET and the marketing routes load fresh.
          window.location.href = "/";
        }}
        user={user}
      />
      {/* HomeView stays mounted under the project overlay so coming
          back from a project is instant, its `projects` state, the
          card grid, scroll position, and any in-flight pollers all
          survive the round-trip. ProjectView / ProjectViewV2Stub /
          ProfileView render on top as fixed-position overlays, so
          HomeView is hidden visually while one of those is up.
          Only show HomeView when the user actually wants the
          workspace tab and they're logged in, anonymous viewers
          and other tabs render through the lower switch below. */}
      {loggedIn && tab === "workspaces" && (
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
          {/* Back-to-top inside the workspace too, the user's own
              project grid scrolls just like the public feed. The
              `hidden` parent eats the button when an overlay is up. */}
          <ScrollToTop />
        </div>
      )}
      {notFoundProjectId ? (
        // Deep-link to a project the backend doesn't know about.
        // Keep TopNav above and Footer below; centre a quiet
        // white-on-dark message between them. No retry / back
        // button, the user can hit the nav at the top.
        <main className="min-h-[calc(100vh-9rem)] grid place-items-center px-6">
          <p className="text-base text-foreground/85 font-light">
            This project cannot be found...
          </p>
        </main>
      ) : profileOpen ? (
        <SettingsView onClose={() => setProfileOpen(false)} />
      ) : openV2Project ? (
        <>
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
          //
          // Previously this also gated on projectOriginTab === "projects"
          // which was the bug: on hard-refresh of a public project URL
          // the tab defaulted to "workspaces" and readOnly stayed false
          // until the next navigation. Dropping that gate.
          readOnly={
            // Editable if you own it OR you have write access via the Project
            // (editor+). Read-only is the safe default until those resolve.
            !(
              (!!openV2Project.owner && openV2Project.owner === user.username) ||
              openV2Project.writable === true
            )
          }
          // Origin tab feeds the "Back to …" copy on the loader +
          // header, opening from /projects reads "Back to projects"
          // even when the project happens to be the viewer's own.
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
        {/* Back-to-top is now rendered INSIDE ProjectViewV2Stub
            so it can suppress itself while the image viewer or
            augmentations viewer modal is open — those modals
            cover the dataset section the button was meant to
            return to, and the floating affordance over the modal
            chrome looked stray. */}
        </>
      ) : openProject ? (
        <>
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
        </>
      ) : tab === "guide" ? (
        <GuideView />
      ) : (
        // Workspace renders via the always-mounted block above.
        null
      )}
      {setupOpen && (
        <SetupWizard
          onClose={() => {
            window.localStorage.setItem("pk-setup-dismissed", "1");
            setSetupOpen(false);
          }}
        />
      )}
    </>
  );
}
