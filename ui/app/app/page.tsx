"use client";

import { useEffect, useRef, useState } from "react";
import { HomeView } from "../HomeView";
import { ScrollToTop } from "../components/ScrollToTop";
import { GuideView } from "../GuideView";
import { SettingsView } from "../SettingsView";
import { SetupWizard, setupNeeded } from "../SetupWizard";
import { TitleBar } from "../shell/TitleBar";
import { ActivityBar, type ActivityKey } from "../shell/ActivityBar";
import { SideBar } from "../shell/SideBar";
import { StatusBar } from "../shell/StatusBar";
import { requestNewDataset } from "@/lib/appNav";
import type { ReferenceImage } from "../v2/ReferenceImageEditor";
import { ProjectViewV2Stub, type ProjectTab } from "../v2/ProjectViewV2Stub";
import { readProjectMeta } from "@/lib/projectMetaCache";
import { apiFetch } from "@/lib/apiFetch";
import { fetchModelsStatus } from "@/lib/models";

// The open dataset.
type V2Result = {
  name: string;
  labels: string[];
  references: ReferenceImage[];
  projectId: string | null;
  // The Project this dataset was opened from, for "Back to project".
  fromProjectId?: string | null;
  // "onboarding" skips the view's mount loader, as HomeView's overlay is up.
  firstLoad?: "onboarding" | null;
};

export default function Page() {
  const [activity, setActivity] = useState<ActivityKey>("explorer");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);
  const [openV2Project, setOpenV2Project] = useState<V2Result | null>(null);
  // Open dataset's section, shared by the Explorer tree and the view.
  const [datasetSection, setDatasetSection] = useState<ProjectTab>("overview");
  const [setupOpen, setSetupOpen] = useState(false);

  // Set when a deep-linked project id is unknown to the backend.
  const [notFoundProjectId, setNotFoundProjectId] = useState<string | null>(null);

  // Pushes /app/<id> for an open dataset, /app for none.
  const syncUrl = (id: string | null) => {
    if (typeof window === "undefined") return;
    const target = id ? `/app/${id}` : "/app";
    if (window.location.pathname === target) return;
    window.history.pushState(null, "", target);
  };

  // Pushes target and fires popstate. This page reads the path, HomeView
  // reads ?project.
  const navigateSpa = (target: string) => {
    if (typeof window === "undefined") return;
    if (window.location.pathname + window.location.search !== target) {
      window.history.pushState(null, "", target);
    }
    window.dispatchEvent(new PopStateEvent("popstate"));
  };

  const openProj = (id: string, displayName: string = "", fromProjectId?: string) => {
    setNotFoundProjectId(null);
    setDatasetSection("overview");
    syncUrl(id);
    // Name and labels come from the meta cache first, then the fetch below.
    const cached = readProjectMeta(id);
    setOpenV2Project({
      name: cached?.name ?? displayName,
      labels: cached?.labels ?? [],
      references: [],
      projectId: id,
      fromProjectId: fromProjectId ?? null,
    });
    // imports_limit=0 takes the backend's sidecar fast path. apiFetch
    // dedups the request with ProjectViewV2Stub's.
    apiFetch(`/api/v2/projects/${id}/overview?imports_limit=0`)
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => {
        if (!m) return;
        setOpenV2Project((cur) =>
          cur && cur.projectId === id
            ? {
                ...cur,
                name: m.name ?? cur.name,
                labels: m.tags ?? cur.labels,
              }
            : cur,
        );
      })
      .catch(() => { /* the page works without it */ });
  };

  // ?tab= selects a view. ?profile=1 opens settings once and is removed.
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

  // First-run setup wizard, shown when the model weights are missing.
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
        /* engine unreachable, no wizard */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Opens the dataset named in the URL, on mount and on popstate.
  const urlHydratedRef = useRef(false);
  // Open dataset id for the popstate handler, whose effect has no deps.
  const openIdRef = useRef<string | null>(null);
  useEffect(() => {
    openIdRef.current = openV2Project?.projectId ?? null;
  }, [openV2Project]);

  // A rename on the dataset page or in its settings reaches the title bar.
  useEffect(() => {
    const handler = (e: Event) => {
      const d = (e as CustomEvent<{ projectId?: string; name?: string }>).detail;
      const name = d?.name;
      if (!d?.projectId || !name) return;
      setOpenV2Project((cur) => (cur && cur.projectId === d.projectId ? { ...cur, name } : cur));
    };
    window.addEventListener("pixelkit-project-meta-changed", handler);
    return () => window.removeEventListener("pixelkit-project-meta-changed", handler);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handle = async () => {
      const m = window.location.pathname.match(/^\/app\/([^\/?#]+)/);
      const idFromUrl = m ? m[1] : null;
      if (idFromUrl) {
        if (
          openIdRef.current !== idFromUrl
          && notFoundProjectId !== idFromUrl
        ) {
          // Opens at once. The /initial request below reports a 404 or the
          // name, and apiFetch dedups it with the stub's own /initial.
          const cached = readProjectMeta(idFromUrl);
          setNotFoundProjectId(null);
          openProj(idFromUrl, cached?.name ?? "");
          void apiFetch(`/api/v2/projects/${idFromUrl}/initial?n=20`, { cache: "no-store" })
            .then(async (r) => {
              if (r.status === 404) {
                // Only if this dataset is still the open one.
                if (openIdRef.current === idFromUrl) {
                  setNotFoundProjectId(idFromUrl);
                  setOpenV2Project(null);
                }
                return;
              }
              if (!r.ok) return;
              const m = (await r.json()) as { name?: string };
              if (m?.name) {
                setOpenV2Project((cur) =>
                  cur && cur.projectId === idFromUrl
                    ? { ...cur, name: m.name ?? cur.name }
                    : cur,
                );
              }
            })
            .catch(() => { /* network error */ });
        }
      } else {
        setNotFoundProjectId(null);
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

  // The Explorer icon returns to the workspace and expands the side bar.
  const handleActivitySelect = (key: ActivityKey) => {
    if (key === "guide") {
      if (activity === "guide") return;
      setActivity("guide");
      setProfileOpen(false);
      setOpenV2Project(null);
      setNotFoundProjectId(null);
      syncUrl(null);
      return;
    }
    setActivity("explorer");
    setSidebarOpen(true);
    setProfileOpen(false);
    // Closes the dataset overlay and HomeView's Project page.
    navigateSpa("/app");
  };

  // Opens a Project page through HomeView's ?project link.
  const handleOpenProjectPage = (containerId: string) => {
    setProfileOpen(false);
    setNotFoundProjectId(null);
    navigateSpa(`/app?project=${encodeURIComponent(containerId)}`);
  };

  // Side-bar "+": HomeView's appNav listener opens the CreateDatasetModal.
  const handleNewDataset = () => {
    setActivity("explorer");
    setOpenV2Project(null);
    setProfileOpen(false);
    setNotFoundProjectId(null);
    syncUrl(null);
    requestNewDataset();
  };

  const openDatasetId = openV2Project?.projectId ?? null;
  const titleBarName = openV2Project ? openV2Project.name : "";

  const sidebarVisible = sidebarOpen && activity !== "guide";
  // Dataset views and the not-found message cover the content pane.
  const overlayCls = [
    "fixed top-9 bottom-6 right-0 z-[200] bg-[var(--background)]",
    sidebarVisible ? "left-[308px]" : "left-12",
  ].join(" ");

  // Content area's left edge, on <html> so document.body portals can read it.
  useEffect(() => {
    document.documentElement.style.setProperty(
      "--pk-content-left",
      sidebarVisible ? "308px" : "48px",
    );
    return () => {
      document.documentElement.style.removeProperty("--pk-content-left");
    };
  }, [sidebarVisible]);

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
            selectedDatasetId={openDatasetId}
            activeSection={openV2Project ? datasetSection : null}
            onOpenDataset={(ds) => {
              if (openDatasetId === ds.id) return;
              setProfileOpen(false);
              openProj(ds.id, ds.name, ds.containerId ?? undefined);
            }}
            onOpenSection={(ds, section) => {
              setProfileOpen(false);
              if (openDatasetId !== ds.id) {
                // openProj resets the section, so it runs first.
                openProj(ds.id, ds.name, ds.containerId ?? undefined);
              }
              setDatasetSection(section);
            }}
            onOpenProject={handleOpenProjectPage}
            onNewDataset={handleNewDataset}
            onCollapse={() => setSidebarOpen(false)}
          />
        )}
        {/* Scroll container for HomeView and GuideView. */}
        <div
          data-app-scroll
          className="relative min-w-0 flex-1 overflow-y-auto overscroll-contain"
        >
          {/* HomeView stays mounted under the overlays to keep its state. */}
          {activity !== "guide" && (
            <div hidden={!!profileOpen || !!openV2Project || !!notFoundProjectId}>
              <HomeView
                onOpen={openProj}
                onV2Begin={(name, labels, references, projectId, firstLoad) => {
                  if (projectId) syncUrl(projectId);
                  setDatasetSection("overview");
                  setOpenV2Project({
                    name,
                    labels,
                    references,
                    projectId,
                    firstLoad: firstLoad ?? null,
                  });
                }}
              />
              <ScrollToTop />
            </div>
          )}
          {activity === "guide" && <GuideView />}
        </div>
      </div>
      <StatusBar />

      {notFoundProjectId ? (
        <div className={overlayCls}>
          <main className="grid h-full place-items-center px-6">
            <p className="text-base text-foreground/85 font-light">
              This dataset could not be found.
            </p>
          </main>
        </div>
      ) : profileOpen ? (
        <SettingsView onClose={() => setProfileOpen(false)} />
      ) : openV2Project ? (
        // Keyed by id so another dataset remounts the view.
        <div
          key={openV2Project.projectId ?? "v2-onboarding"}
          data-dataset-scroll
          className={`${overlayCls} overflow-y-auto overscroll-contain`}
        >
          <ProjectViewV2Stub
            projectName={openV2Project.name}
            labels={openV2Project.labels}
            references={openV2Project.references}
            projectId={openV2Project.projectId}
            backToProjectId={openV2Project.fromProjectId ?? null}
            firstLoad={openV2Project.firstLoad ?? null}
            section={datasetSection}
            onSectionChange={setDatasetSection}
            onClose={() => {
              const proj = openV2Project.fromProjectId;
              if (proj) {
                // The popstate closes this view and opens the Project page.
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
