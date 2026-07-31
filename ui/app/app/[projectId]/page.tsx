"use client";

// Dynamic route, /app/<projectId>. Lets people share a link
// directly to a specific project without going through the
// workspace card. Renders the same Page component as /app; the
// shell reads window.location.pathname on mount to figure out
// which project to open. Without this route, deep-link reloads
// 404'd because Next.js doesn't accept a path the file-router
// didn't generate.

import AppPage from "../page";

export default function ProjectRoute() {
  return <AppPage />;
}
