// Server component: renders the Guide content at a clean /guide
// URL so Google indexes the long-form content (and so social cards
// have a real page to scrape). The interactive section-switcher
// chrome lives in GuideView, a client component imported here —
// Next.js SSRs its initial state, the user gets the Overview
// section rendered in the HTML straight away.

import type { Metadata } from "next";
import { GuideView } from "../GuideView";

const SITE_URL = (
  process.env.NEXT_PUBLIC_APP_URL
  ?? "https://pixelkit.ai"
).replace(/\/+$/, "");

const GUIDE_TITLE = "Guide — Auto-labelling, augmentations and dataset export";
const GUIDE_DESCRIPTION =
  "How to use Pixel Kit: create a project, upload images, auto-label with one click, review predictions, generate augmentations, and export your dataset. Step-by-step walkthrough with screenshots.";

export const metadata: Metadata = {
  title: GUIDE_TITLE,
  description: GUIDE_DESCRIPTION,
  alternates: { canonical: `${SITE_URL}/guide` },
  openGraph: {
    type: "article",
    url: `${SITE_URL}/guide`,
    title: GUIDE_TITLE,
    description: GUIDE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: GUIDE_TITLE,
    description: GUIDE_DESCRIPTION,
  },
};

export default function GuidePage() {
  return <GuideView />;
}
