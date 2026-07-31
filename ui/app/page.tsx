import { redirect } from "next/navigation";

// Portable build has no marketing landing page — the root goes straight
// to the workspace.
export default function Home() {
  redirect("/app");
}
