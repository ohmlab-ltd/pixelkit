"use client";

import { useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { HomeView } from "../HomeView";
import { ProjectView } from "../ProjectView";
import { ProjectsView } from "../ProjectsView";
import { ScrollToTop } from "../components/ScrollToTop";
import { PricingView } from "../PricingView";
import { ProfileView } from "../ProfileView";
import { TerminalView } from "../TerminalView";
import { GuideView } from "../GuideView";
import { TopNav, NavTab } from "../TopNav";
import { broadcastCurrentTab, onAppNavigate } from "@/lib/appNav";
import type { ReferenceImage } from "../v2/OnboardReferencesV2";
import { ProjectViewV2Stub } from "../v2/ProjectViewV2Stub";
import { patchProjectMeta, readProjectMeta } from "@/lib/projectMetaCache";
import { apiFetch, primeBackendToken } from "@/lib/apiFetch";
import { installApiAuth, ensureAuthCookie } from "@/lib/apiAuth";

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
  const { data: session, status } = useSession();
  const router = useRouter();

  // Default tab depends on auth: signed-in visitors land on Workspace,
  // anonymous visitors land on Demo. Resolved once status is known so the
  // first paint goes to the right place without a flicker.
  const [tab, setTab] = useState<NavTab>("workspaces");
  const [tabResolved, setTabResolved] = useState(false);
  const [openProject, setOpenProject] = useState<string | null>(null);
  const [openProjectOwner, setOpenProjectOwner] = useState<string>("");
  const [openProjectName, setOpenProjectName] = useState<string>("");
  const [profileOpen, setProfileOpen] = useState(false);
  // V2 post-onboarding stub. HomeView owns the entire onboarding
  // (name + labels + references) inline so the V2 path here is a
  // single piece of state, flipping back to V1 is one tweak.
  const [openV2Project, setOpenV2Project] = useState<V2Result | null>(null);

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
    setOpenProjectOwner(owner);
    setOpenProjectName(displayName);
  };

  // Compute these BEFORE any early returns so the hook order below stays
  // stable across renders (loading -> authenticated transitions).
  // Portable build: no accounts. Everything renders as the local user —
  // never show a logged-out state (the /login route doesn't exist).
  const loggedIn = true;
  void status;
  const username = (session?.user?.username ?? "") as string;

  // OAuth users with no username yet, finish onboarding before they can do
  // anything else. Logged-out viewers fall through and see the public chrome.
  useEffect(() => {
    if (status === "authenticated" && session?.user && !session.user.username) {
      router.replace("/onboard");
    }
  }, [status, session, router]);

  // Anonymous users who land on the workspaces tab (which used to
  // fall through to DemoView) get bounced to /login instead. Public
  // tabs (projects feed, pricing, guide) stay accessible without auth.
  // Preserve the originating URL via ?callbackUrl= so NextAuth lands
  // the user back where they were after signing in.
  useEffect(() => {
    if (status !== "unauthenticated") return;
    if (!tabResolved) return;
    if (tab !== "workspaces") return;
    const callbackUrl =
      typeof window !== "undefined"
        ? window.location.pathname + window.location.search
        : "/app";
    router.replace(`/login?callbackUrl=${encodeURIComponent(callbackUrl)}`);
  }, [status, tab, tabResolved, router]);

  // Pick the landing tab based on auth state. We only set it once (on first
  // resolution) so subsequent navigations the user makes aren't clobbered.
  // A `?tab=` query param wins over the auth-based default, that's how
  // links from /profile or /pricing-related CTAs deep-link into a specific
  // tab without needing a separate top-level route per tab.
  useEffect(() => {
    if (tabResolved || status === "loading") return;
    if (typeof window !== "undefined") {
      const search = new URLSearchParams(window.location.search);
      const t = search.get("tab");
      const valid: NavTab[] = ["workspaces", "projects", "guide", "pricing", "terminal"];
      const fromUrl = t && (valid as string[]).includes(t) ? (t as NavTab) : null;
      if (fromUrl) setTab(fromUrl);
      else if (status === "authenticated") setTab("workspaces");
      else setTab("projects");
      // `?profile=1` re-opens the profile pane after a hard reload
      // (e.g. post-settings-save). Strip the param afterwards so the
      // URL stays clean and back/refresh don't reopen profile forever.
      if (search.get("profile") === "1") {
        setProfileOpen(true);
        const cleaned = new URLSearchParams(window.location.search);
        cleaned.delete("profile");
        const next = cleaned.toString();
        window.history.replaceState(null, "", next ? `?${next}` : window.location.pathname);
      }
    } else if (status === "authenticated") {
      setTab("workspaces");
    } else {
      setTab("projects");
    }
    setTabResolved(true);
  }, [status, tabResolved]);

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

  // Listen for in-app navigation events. Lets any nested component
  // (PlanPill, Upgrade buttons, billing warnings, AI-Review-upgrade link)
  // ask to switch tabs without prop-drilling or relying on URL
  // changes, those don't reliably re-trigger this page since /app
  // is a single SPA route.
  useEffect(() => {
    return onAppNavigate((tab) => {
      setTab(tab);
      setOpenProject(null);
      setOpenV2Project(null);
      setProfileOpen(false);
      syncUrl(null);
    });
  }, []);

  // Heartbeat, pings the backend every 15s while a logged-in user has the
  // app open, so the Terminal's "Live now" panel reflects current usage.
  // Pure presence, never persisted. Declared above the early return so the
  // hook order is identical on every render.
  const backendToken = (session?.user as { backendToken?: string | null } | undefined)?.backendToken ?? null;
  // Prime apiFetch's token cache the moment the session resolves, so
  // the first backend call on project-open fires immediately instead
  // of waiting on a getSession() → /api/auth/session round-trip. This
  // is what makes "click project → data loads" feel instant rather
  // than stalling for a few seconds before the requests even leave.
  useEffect(() => {
    if (status === "loading") return;
    primeBackendToken(backendToken);
    // Phase 0 security: patch fetch so every backend call carries the bearer
    // (covers legacy plain-fetch sites), and set the pk_auth cookie so private
    // <img> image loads authenticate against the new project guards too.
    installApiAuth();
    if (backendToken) ensureAuthCookie();
  }, [status, backendToken]);
  useEffect(() => {
    if (!loggedIn || !username) return;
    const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";
    const ping = () => {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (backendToken) headers["Authorization"] = `Bearer ${backendToken}`;
      fetch(`${API}/api/heartbeat`, {
        method: "POST",
        headers,
        body: JSON.stringify({ username }),
        keepalive: true,
      }).catch(() => {});
    };
    ping();
    const id = window.setInterval(ping, 15000);
    return () => window.clearInterval(id);
  }, [loggedIn, username, backendToken]);

  // Don't flash anything while the session is still resolving.
  if (status === "loading") return null;

  const user = loggedIn
    ? {
        name: session!.user.name ?? session!.user.username ?? "User",
        username: session!.user.username ?? "",
        email: session!.user.email ?? "",
        image: session!.user.image ?? null,
      }
    : { name: "", username: "", email: "", image: null };

  // Gate the Terminal link to the single operator account. The page itself
  // already requires a backend token, but hiding the link in nav keeps it
  // out of sight for everyone else.
  const ADMIN_USERNAMES = ["hamish", "mukund"];
  const isAdmin = loggedIn && ADMIN_USERNAMES.includes(user.username);

  return (
    <>
      <TopNav
        current={(openProject || openV2Project) ? projectOriginTab : tab}
        onNavigate={(t) => {
          if (t === "terminal" && !isAdmin) return;
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
        loggedIn={loggedIn}
        // Terminal entry now lives on the profile page (admins only), not the
        // top bar.
        showTerminal={false}
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
        <ProfileView
          user={user}
          onJumpWorkspaces={() => {
            setProfileOpen(false);
            setOpenProject(null);
            setOpenV2Project(null);
            setTab("workspaces");
          }}
          onJumpProjects={() => {
            setProfileOpen(false);
            setOpenProject(null);
            setOpenV2Project(null);
            setTab("projects");
          }}
          // Operator terminal entry, admins (@hamish / @mukund) only. Moved here
          // from the top bar.
          showTerminal={isAdmin}
          onJumpTerminal={() => {
            setProfileOpen(false);
            setOpenProject(null);
            setOpenV2Project(null);
            setTab("terminal");
          }}
        />
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
          // Read-only only when viewing *someone else's* project from the
          // community Projects tab. Own projects stay editable wherever
          // they're opened from.
          readOnly={
            projectOriginTab === "projects" &&
            !!openProjectOwner &&
            openProjectOwner !== user.username
          }
          onClose={() => { setOpenProject(null); syncUrl(null); }}
          onRename={(newName) => { setOpenProject(newName); syncUrl(newName); }}
        />
        <ScrollToTop />
        </>
      ) : tab === "workspaces" ? (
        // Logged-in workspace renders via the always-mounted block
        // above; anonymous viewers are redirected to /login by the
        // effect higher up, so we render null while that bounce
        // completes (one tick) instead of flashing any fallback UI.
        null
      ) : tab === "terminal" ? (
        <TerminalView username={user.username} />
      ) : tab === "pricing" ? (
        <PricingView />
      ) : tab === "guide" ? (
        <GuideView />
      ) : (
        <>
          <ProjectsView onOpen={openProj} username={user.username} loggedIn={loggedIn} />
          {/* Back-to-top, only mounted on the Projects feed where the
              user actually scrolls through a long grid. No global
              mount, no cross-component broadcast required. */}
          <ScrollToTop />
        </>
      )}


    </>
  );
}
