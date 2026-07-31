import { redirect } from "next/navigation";

const VALID = new Set([
  "overview",
  "projects",
  "dataset",
  "labelling",
  "augmentations",
  "stats",
  "settings",
  "reference",
]);

export default async function GuideSectionRedirect({
  params,
}: {
  params: Promise<{ section: string }>;
}) {
  const { section } = await params;
  // Old deep-link shape: /guide/<section> 301s into the canonical
  // /guide?section=<section> URL so existing inbound links + bookmarks
  // keep working.
  if (section === "overview" || !VALID.has(section)) redirect("/guide");
  redirect(`/guide?section=${section}`);
}
