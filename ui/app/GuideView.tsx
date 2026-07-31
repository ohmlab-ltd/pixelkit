"use client";

// SPA tab view for the Guide. Mounted by /app/page.tsx when tab === "guide".
// Owns its own section state (overview / projects / dataset / ...) and syncs
// it to the URL as ?tab=guide&section=<key> via replaceState, same pattern
// as the other tabs in the SPA.

import { useEffect, useState, type ReactNode } from "react";
import { Footer } from "./Footer";

export type GuideSectionKey =
  | "overview"
  | "projects"
  | "dataset"
  | "labelling"
  | "augmentations"
  | "stats"
  | "settings"
  | "teams"
  | "derived"
  | "reference";

const GUIDE_SECTIONS: { key: GuideSectionKey; title: string; blurb: string }[] = [
  { key: "overview", title: "Overview", blurb: "What PixelKit does and how the moving parts fit together." },
  { key: "projects", title: "Create a dataset", blurb: "Creating a dataset, picking general vs specific, managing references." },
  { key: "dataset", title: "Dataset", blurb: "Adding images, searching Openverse, video trims, filters, bulk delete." },
  { key: "labelling", title: "Labelling", blurb: "Auto-labelling, the Annotations card, manual edits, fast Review mode." },
  { key: "augmentations", title: "Augmentations", blurb: "Every dial, the Randomise button, the per-tile viewer." },
  { key: "stats", title: "Dataset stats", blurb: "The health score, AI insights, the variation plot, near-duplicate review." },
  { key: "settings", title: "Settings & themes", blurb: "Renaming, recolouring labels, visibility, light / dark, export." },
  { key: "teams", title: "Projects & teams", blurb: "Group datasets into a Project, invite members, owner / editor / viewer roles, activity." },
  { key: "derived", title: "Derived datasets", blurb: "Crop each detection into a child dataset: ROI squares, inherited or fresh labels, live sync." },
  { key: "reference", title: "Reference", blurb: "Keyboard shortcuts, plan limits, content-safety policies." },
];

export function GuideView() {
  const [section, setSection] = useState<GuideSectionKey>("overview");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const s = params.get("section");
    const valid = GUIDE_SECTIONS.map((x) => x.key) as string[];
    if (s && valid.includes(s)) setSection(s as GuideSectionKey);
  }, []);

  const go = (key: GuideSectionKey) => {
    setSection(key);
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("tab", "guide");
    if (key === "overview") url.searchParams.delete("section");
    else url.searchParams.set("section", key);
    window.history.replaceState(null, "", url.toString());
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <main className="min-h-[calc(100vh-9rem)] mx-auto max-w-6xl px-6 pt-8 pb-16">
      <nav
        aria-label="Guide sections"
        className="flex flex-wrap gap-1.5 mb-8 pb-2 border-b border-foreground/[0.06]"
      >
        {GUIDE_SECTIONS.map((s) => {
          const isActive = s.key === section;
          return (
            <button
              key={s.key}
              onClick={() => go(s.key)}
              className={[
                "px-3.5 h-9 inline-flex items-center rounded-full text-xs uppercase tracking-wider transition-colors duration-[90ms]",
                isActive
                  ? "bg-foreground/[0.08] text-[var(--foreground)] font-medium"
                  : "text-foreground/55 hover:text-foreground/90 hover:bg-foreground/[0.04]",
              ].join(" ")}
            >
              {s.title}
            </button>
          );
        })}
      </nav>

      <div className="grid gap-10">
        {section === "overview" && <OverviewSection go={go} />}
        {section === "projects" && <ProjectsSection go={go} />}
        {section === "dataset" && <DatasetSection go={go} />}
        {section === "labelling" && <LabellingSection go={go} />}
        {section === "augmentations" && <AugmentationsSection go={go} />}
        {section === "stats" && <StatsSection go={go} />}
        {section === "settings" && <SettingsSection go={go} />}
        {section === "teams" && <TeamsSection go={go} />}
        {section === "derived" && <DerivedSection go={go} />}
        {section === "reference" && <ReferenceSection />}
      </div>
      <Footer />
    </main>
  );
}

// ---------- Shared helpers ----------

function GuideHeader({
  eyebrow,
  title,
  intro,
}: {
  eyebrow: string;
  title: string;
  intro: ReactNode;
}) {
  return (
    <header>
      <div className="text-[11px] uppercase tracking-[0.24em] font-mono text-[var(--muted)] mb-2">
        {eyebrow}
      </div>
      <h1 className="text-5xl md:text-6xl font-medium tracking-tight text-[var(--foreground)]">
        {title}
      </h1>
      <p className="mt-4 text-base text-[var(--muted)] leading-relaxed max-w-2xl">
        {intro}
      </p>
    </header>
  );
}

function GuideSection({
  n,
  title,
  children,
}: {
  n: number;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="scroll-mt-24">
      <h2 className="flex items-center gap-3 text-2xl font-medium tracking-tight">
        <span className="pk-accent-bar" style={{ height: "1.3rem" }} aria-hidden />
        <span className="text-[var(--muted)] font-mono tabular-nums">
          {String(n).padStart(2, "0")}
        </span>
        {title}
      </h2>
      <div className="mt-4 grid gap-4 text-[15px] leading-relaxed text-[var(--foreground)]/85 [&_p]:m-0">
        {children}
      </div>
    </section>
  );
}

function Strong({ children }: { children: ReactNode }) {
  return <span className="font-medium text-[var(--foreground)]">{children}</span>;
}

function Bullets({ items }: { items: string[] }) {
  return (
    <ul className="list-disc list-outside ml-5 marker:text-[var(--muted)] grid gap-1.5">
      {items.map((it, i) => (
        // eslint-disable-next-line react/no-danger
        <li key={i} dangerouslySetInnerHTML={{ __html: it }} />
      ))}
    </ul>
  );
}

function Steps({ items }: { items: [string, string][] }) {
  return (
    <ol className="grid gap-2.5">
      {items.map(([title, body], i) => (
        <li
          key={i}
          className="flex items-start gap-3 rounded-xl border border-foreground/[0.08] bg-foreground/[0.02] px-4 py-3"
        >
          <span className="shrink-0 grid place-items-center h-6 w-6 rounded-full bg-foreground/[0.08] text-[10px] font-mono text-[var(--foreground)] tabular-nums mt-0.5">
            {i + 1}
          </span>
          <div className="min-w-0">
            <div className="text-sm font-medium text-[var(--foreground)]">{title}</div>
            <div
              className="mt-0.5 text-[13px] text-[var(--muted)] leading-relaxed"
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: body }}
            />
          </div>
        </li>
      ))}
    </ol>
  );
}

function ShortcutTable({ rows }: { rows: [string, string][] }) {
  return (
    <div className="rounded-xl border border-foreground/[0.08] overflow-hidden">
      <table className="w-full text-sm">
        <tbody className="divide-y divide-foreground/[0.06]">
          {rows.map(([key, action], i) => (
            <tr key={i}>
              <td className="px-4 py-2.5 w-32">
                <kbd className="inline-block rounded border border-foreground/15 bg-foreground/[0.04] px-1.5 py-0.5 font-mono text-[11px] text-[var(--foreground)]">
                  {key}
                </kbd>
              </td>
              <td className="px-4 py-2.5 text-[var(--foreground)]/85">{action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Callout({
  tone,
  children,
}: {
  tone: "info" | "warn";
  children: ReactNode;
}) {
  const classes =
    tone === "warn"
      ? "border-amber-500/40 bg-amber-500/[0.08]"
      : "border-foreground/15 bg-foreground/[0.03]";
  return (
    <div className={`rounded-xl border px-4 py-3 text-[14px] leading-relaxed ${classes}`}>
      {children}
    </div>
  );
}

function NextUp({
  next,
  go,
}: {
  next: GuideSectionKey;
  go: (k: GuideSectionKey) => void;
}) {
  const target = GUIDE_SECTIONS.find((s) => s.key === next);
  if (!target) return null;
  return (
    <button
      onClick={() => go(target.key)}
      className="mt-4 w-full text-left inline-flex items-center justify-between gap-4 rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02] hover:bg-orange-500/[0.04] hover:border-orange-400/35 transition-colors px-5 py-4 group"
    >
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-[0.18em] font-mono text-[var(--muted)] mb-1">
          Next up
        </div>
        <div className="text-base font-medium text-[var(--foreground)]">{target.title}</div>
        <p className="mt-0.5 text-sm text-[var(--muted)] leading-relaxed">{target.blurb}</p>
      </div>
      <span aria-hidden className="text-[var(--muted)] group-hover:text-[var(--foreground)] transition-colors text-xl shrink-0">
        →
      </span>
    </button>
  );
}

// ---------- New helpers: LLM prompt + figures ----------

const LLM_PROMPT = `You are a helpful assistant for someone using PixelKit, a SaaS for building labelled computer-vision datasets. Answer my questions in plain English. If something is unclear or undocumented in what follows, say so rather than guessing.

==============================================================
WHAT PIXELKIT IS
==============================================================
PixelKit turns images into a labelled, augmented dataset that's ready for training.

The five-step flow:
  1. Create a project. Name it and define your labels.
  2. Add images. Upload, import from video, or search Openverse.
  3. Label. One click runs auto-labelling.
  4. Augment. Choose realistic variations to expand the dataset.
  5. Export. Download with a train/val split.

==============================================================
PROJECTS
==============================================================
When you create a project, pick the labels you want PixelKit to find.

If your labels are common things (cat, dog, person, car), PixelKit
already knows what they look like. You go straight to the dataset.

If your labels are visually similar or specific to your domain
(hare vs rabbit, your part A vs your part B), PixelKit will ask for
a few example photos of each. This helps it tell them apart.

You can add or remove reference examples later from the project
page.

Project names are display-only; the URL uses a stable identifier.
A profanity guard runs on names and labels.

==============================================================
ADDING IMAGES
==============================================================
Three ways to add images:

  - Upload from your computer (drag and drop, or click to browse).
  - Import frames from a video. Pick a start and end time and an
    fps. PixelKit extracts the frames as still images.
  - Search Openverse. Type what you're looking for and PixelKit
    shows you a grid of CC-licensed images you can pull in.

Each image runs through a content-safety check before it lands in
the dataset.

The dataset gallery shows a thumbnail for every image. Indicators
on each thumbnail include a sparkle icon (augmentations exist),
an Unsure pill (a detection is borderline), and the per-image
verdict if you've used Review mode.

A filter chip rail above the gallery lets you narrow to All /
Unlabelled / Unrated / Good / Bad / Unsure.

==============================================================
LABELLING
==============================================================
Start labelling runs auto-labelling across every unlabelled image.

Click-to-detect: switch the viewer into click mode, then tap any
object. PixelKit labels it.

Manual mode: a toggle that lets you draw boxes by hand. Use it when
you want full control or when auto-labelling missed something.

The image viewer can also delete the current image entirely (a Delete
button in the header, the same as the gallery tile trash); it then
moves to the next image. The detection count in the header is live, so
it updates as you add or delete boxes. Large 4K images load a fast
preview first, then sharpen to full resolution.

Borderline detections are flagged "Unsure" so you can confirm them
quickly. They appear as an amber pill on the dataset thumbnail, in
the Dataset stats summary, and next to the detection in the
viewer's box list. Resolve by confirming or deleting.

Review mode: a focused review interface. Use arrow keys to approve
or reject images one at a time.

Deleting a label removes every detection carrying that label and
takes down any augmentations that referenced it. Irreversible;
export first if you might want the annotations back.

==============================================================
AUGMENTATIONS
==============================================================
Augmentations create realistic variations of your images so the
trained model handles real-world conditions better.

Categories:
  Camera & sensor - simulates real camera artefacts.
  Distortion - perspective, scale, rotation, hue.
  Occlusion - simulates things blocking the view.
  Domain randomisation - varies the background and lighting.

For each enabled category you can pick the strength and how often
it's applied.

You can set how many augmented copies to make per image. Off
creates none and clears the existing ones.

Augmentations re-generate in the background after manual edits.

==============================================================
DATASET STATS
==============================================================
PixelKit shows a Health score (0-100) summarising how trainable
your dataset is, plus a short list of the biggest things to fix.

The Insights area also shows an AI-written lead recommendation (what
to improve next) above the rule-based suggestion cards. It only
regenerates on a material change (crossing an image or label-count
tier, or changing the label set), so it stays token-frugal, and it
never tells you to add labels to a deliberately single-class dataset.

The variation map shows your dataset as dots. Pictures that look
the same to a model sit close together. Pictures that look
different spread out. Duplicates are highlighted.

You can flick through duplicates in the Review duplicates modal
and keep one per group.

==============================================================
SETTINGS, THEMES, EXPORT
==============================================================
The gear icon at the top of any project page opens Settings.

You can rename the project, set its visibility (private projects
require Pro), pick the cover image, and rename / recolour labels.

Export in YOLO, COCO, or Pascal VOC. Choose your train/val split.
The split is reproducible - the same image always lands in the
same set, even across re-exports.

You can filter out detections that would be too small at your
target input shape.

Light and dark mode toggle in the top-right corner.

Deleting a project is destructive and irreversible. Type the
project name to confirm.

==============================================================
PROJECTS (TEAMS) AND ROLES
==============================================================
A Project (capital P) is a team container that groups several
datasets, with shared members, a cover, a privacy setting, and an
activity timeline. It is different from a single dataset (which is
one labelled image collection).

Create a Project from the New project tile on the Workspace. Add
datasets with "Add existing" or "+ New dataset". Privacy defaults to
private; a private Project and its datasets are hidden from Community.

Members have a role:
  - Owner: full control, including rename, cover, privacy, members,
    and deleting the Project.
  - Editor: can add and create datasets, upload, label, augment, and
    edit datasets created by other members, but cannot rename or
    delete the Project, change its cover, or manage members.
  - Viewer: read-only.

Deleting a dataset from a Project asks whether to remove it from the
Project (it survives as a standalone dataset) or delete it entirely.
Deleting entirely is creator-only: even the owner can only detach
someone else's dataset, never destroy it.

==============================================================
DERIVED DATASETS
==============================================================
A derived dataset is a cropped child of a parent dataset: one image
per detection, each cropped to its box. Useful for training on
individual objects.

Create one from the Derived datasets panel on a dataset's overview.
Options: keep the parent labels on each crop or create new labels
(crops arrive unlabelled for a fresh taxonomy); ROI mode forces every
crop to an exact 1:1 square; an optional fixed crop size resizes every
crop to one size (N x N) so all derived images match, handy for
training; context padding and a minimum image size; and whether to
group the crop dataset in a workspace Project.

In "create new labels" mode no boxes, masks or reference images are
copied; each crop is blank and the original parent label shows as a
muted reference while you label. The child re-syncs from the parent
one way (Sync now), and a crop you delete will not come back. A
parent can have many derived datasets, but you cannot derive from a
derived dataset.

==============================================================
WORKSPACE AND COMMUNITY
==============================================================
The Workspace tab shows your own area: your Projects (team
containers) in a row across the top, then the datasets you own below.
The Community tab is the public feed (it used to be called Projects):
a carousel of public Projects above a grid of public datasets,
sortable by Trending / Newest / Most liked.

Opening any public dataset gives you a read-only view of the full
dataset, stats, and variation plot.

Star a dataset to pin it to the top of your own workspace; heart a
dataset for a public Like that feeds the Trending sort.

==============================================================
SHORTCUTS
==============================================================
Image viewer:
  Left / right arrow - previous / next image
  Esc - close viewer
  1-9 - relabel the box under the cursor
  Delete - delete the selected box
  B - toggle box layer
  L - toggle label-chip layer
  M - toggle mask layer
  + / - - zoom in / out
  0 - reset zoom and pan to fit-on-screen

Review mode:
  Left arrow - Good
  Right arrow - Bad
  Space - Unsure
  Esc - close Review mode

Global:
  / - focus search input
  ? - open shortcuts cheatsheet
  Esc - close any modal or panel

==============================================================
PLAN LIMITS
==============================================================
Free: small per-project image cap, fixed monthly Openverse quota,
public projects only.

Pro: three credit tiers. Larger caps, private projects, priority
labelling queue.

Beta: Pro-level limits for 30 days from the day a beta code is
redeemed.

Hitting a cap mid-upload shows an inline note explaining which
limit was reached. The Pricing page is canonical.

==============================================================
CONTENT SAFETY AND LICENSING
==============================================================
Every image entering PixelKit is checked before it's saved. Adult
content is rejected with a clear error.

Openverse imports carry their original CC licence and source URL;
both are stamped into exports. Direct uploads carry no licence
metadata - you're responsible for your right to use the images,
especially for any redistributable model.

==============================================================
HOW TO RESPOND
==============================================================
- Answer in plain English and stay grounded in the information above.
- Do not use em-dashes in your responses. Prefer commas, periods, semicolons, or parentheses.
- When I ask a how-to question, give a step-by-step answer that maps to the UI described.
- If I ask about general computer-vision dataset best practice (class balance, augmentation strategy, annotation quality, train/val splits, etc.) you can answer from your own knowledge.
- If I ask about an implementation detail not covered here (specific thresholds, server architecture, internal models), say you do not know rather than guessing.

I'll send my actual question in the next message.`;

function LLMPromptBox() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(LLM_PROMPT);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard blocked. Fall back to selecting the textarea so
      // the user can hit Cmd/Ctrl+C themselves.
      const ta = document.getElementById("llm-prompt-ta") as HTMLTextAreaElement | null;
      if (ta) {
        ta.focus();
        ta.select();
      }
    }
  };

  return (
    <div className="rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02] overflow-hidden">
      <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-3">
        <div className="min-w-0">
          <div className="text-[10px] uppercase tracking-[0.18em] font-mono text-[var(--muted)] mb-1">
            Ask any LLM about PixelKit
          </div>
          <h3 className="text-lg font-medium tracking-tight text-[var(--foreground)]">
            Copy this prompt into ChatGPT, Claude, Gemini, or your model of choice.
          </h3>
          <p className="mt-1 text-[13px] text-[var(--muted)] leading-relaxed">
            It primes the model with what PixelKit is and what it can do, then hands the floor to your question.
          </p>
        </div>
        <button
          type="button"
          onClick={copy}
          className={[
            "shrink-0 inline-flex items-center gap-2 rounded-full px-4 py-2 text-[13px] font-medium transition-colors",
            copied
              ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-200"
              : "bg-foreground text-background hover:opacity-90",
          ].join(" ")}
          aria-live="polite"
        >
          {copied ? (
            <>
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="5 12 10 17 19 7" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <rect x="9" y="9" width="13" height="13" rx="2" />
                <path d="M5 15V5a2 2 0 0 1 2-2h10" />
              </svg>
              Copy prompt
            </>
          )}
        </button>
      </div>
      <textarea
        id="llm-prompt-ta"
        readOnly
        value={LLM_PROMPT}
        onFocus={(e) => e.currentTarget.select()}
        spellCheck={false}
        className="block w-full max-h-72 resize-none bg-foreground/[0.03] border-t border-foreground/[0.06] px-5 py-4 font-mono text-[12px] leading-relaxed text-[var(--foreground)]/85 focus:outline-none focus:bg-foreground/[0.05]"
        rows={10}
      />
    </div>
  );
}

function Figure({
  caption,
  children,
}: {
  caption?: string;
  children: ReactNode;
}) {
  return (
    <figure className="rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02] overflow-hidden">
      <div className="p-5 sm:p-6">{children}</div>
      {caption && (
        <figcaption className="border-t border-foreground/[0.06] px-5 py-2.5 text-[12px] text-[var(--muted)] leading-relaxed">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

// Five-step flow at the top of the overview. Inline SVG so it themes
// with currentColor and needs no external asset.
function FlowDiagram() {
  const steps = [
    { label: "Create", icon: <FolderIcon /> },
    { label: "Add", icon: <ImageIcon /> },
    { label: "Label", icon: <CrosshairIcon /> },
    { label: "Augment", icon: <SparkleIcon /> },
    { label: "Export", icon: <DownloadIcon /> },
  ];
  return (
    <div className="flex items-center justify-between gap-2 sm:gap-3 overflow-x-auto">
      {steps.map((s, i) => (
        <div key={s.label} className="flex items-center gap-2 sm:gap-3 shrink-0">
          <div className="flex flex-col items-center gap-2">
            <div className="grid place-items-center h-12 w-12 sm:h-14 sm:w-14 rounded-2xl bg-foreground/[0.06] text-[var(--foreground)]">
              {s.icon}
            </div>
            <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--muted)]">
              {s.label}
            </div>
          </div>
          {i < steps.length - 1 && (
            <svg viewBox="0 0 24 24" className="h-4 w-4 text-foreground/35 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="13 6 19 12 13 18" />
            </svg>
          )}
        </div>
      ))}
    </div>
  );
}

function GeneralVsSpecificDiagram() {
  const general = ["#5b8def", "#f06292", "#ffb74d", "#81c784"];
  const specific = ["#7c8aa6", "#8c97b0", "#7a89a3", "#92a0b8"];
  return (
    <div className="grid sm:grid-cols-2 gap-4">
      <div className="rounded-xl bg-foreground/[0.03] p-4">
        <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--muted)] mb-3">
          General
        </div>
        <div className="grid grid-cols-4 gap-2">
          {general.map((c, i) => (
            <div key={i} className="aspect-square rounded-lg grid place-items-center" style={{ backgroundColor: `${c}22` }}>
              <span className="block h-5 w-5 rounded-full" style={{ backgroundColor: c }} />
            </div>
          ))}
        </div>
        <p className="mt-3 text-[12px] text-[var(--muted)] leading-relaxed">
          Distinct categories. No references needed. PixelKit already recognises them.
        </p>
      </div>
      <div className="rounded-xl bg-foreground/[0.03] p-4">
        <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--muted)] mb-3">
          Specific
        </div>
        <div className="grid grid-cols-4 gap-2">
          {specific.map((c, i) => (
            <div key={i} className="aspect-square rounded-lg grid place-items-center" style={{ backgroundColor: `${c}22` }}>
              <span className="block h-5 w-5 rounded-full" style={{ backgroundColor: c }} />
            </div>
          ))}
        </div>
        <p className="mt-3 text-[12px] text-[var(--muted)] leading-relaxed">
          Visually similar classes. Upload references so the model can tell them apart.
        </p>
      </div>
    </div>
  );
}

function PipelineDiagram() {
  const Box = ({ title, sub }: { title: string; sub: string }) => (
    <div className="flex-1 min-w-0 rounded-xl bg-foreground/[0.04] px-3 py-3 text-center">
      <div className="text-[13px] font-medium text-[var(--foreground)]">{title}</div>
      <div className="mt-0.5 text-[11px] text-[var(--muted)]">{sub}</div>
    </div>
  );
  const Arrow = () => (
    <svg viewBox="0 0 24 24" className="h-4 w-4 text-foreground/35 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="13 6 19 12 13 18" />
    </svg>
  );
  return (
    <div className="flex items-center gap-2">
      <Box title="References" sub="anchor images" />
      <Arrow />
      <Box title="Auto-label" sub="boxes + masks" />
      <Arrow />
      <Box title="PixelKit" sub="picks the label" />
      <Arrow />
      <Box title="Labelled" sub="per-image annotations" />
    </div>
  );
}

function VariationPlotMock() {
  const dots = [
    { x: 22, y: 28, c: "#5b8def", r: 6 },
    { x: 30, y: 32, c: "#5b8def", r: 5 },
    { x: 35, y: 26, c: "#5b8def", r: 7 },
    { x: 72, y: 22, c: "#f06292", r: 6, halo: true },
    { x: 76, y: 26, c: "#f06292", r: 6, halo: true },
    { x: 68, y: 60, c: "#81c784", r: 8 },
    { x: 75, y: 66, c: "#81c784", r: 5 },
    { x: 60, y: 70, c: "#81c784", r: 6 },
    { x: 20, y: 75, c: "#ffb74d", r: 7 },
    { x: 28, y: 70, c: "#ffb74d", r: 5 },
    { x: 50, y: 50, c: "#5b8def", r: 5 },
    { x: 45, y: 40, c: "#5b8def", r: 4 },
    { x: 40, y: 80, c: "#ffb74d", r: 4 },
  ];
  return (
    <svg viewBox="0 0 100 100" className="w-full max-w-md mx-auto block" aria-hidden>
      <rect x="0" y="0" width="100" height="100" rx="4" fill="rgb(var(--foreground-rgb) / 0.03)" />
      {/* faint axes */}
      <line x1="6" y1="94" x2="94" y2="94" stroke="rgb(var(--foreground-rgb) / 0.15)" strokeWidth="0.4" />
      <line x1="6" y1="94" x2="6" y2="6" stroke="rgb(var(--foreground-rgb) / 0.15)" strokeWidth="0.4" />
      {dots.map((d, i) => (
        <g key={i}>
          {d.halo && <circle cx={d.x} cy={d.y} r={d.r + 2.5} fill="none" stroke="#f59e0b" strokeWidth="0.8" opacity="0.85" />}
          <circle cx={d.x} cy={d.y} r={d.r * 0.4} fill={d.c} opacity="0.9" />
        </g>
      ))}
    </svg>
  );
}

function AugmentationGridMock() {
  const cells = [
    { label: "Original", bg: "#3a3a3a", hue: 0 },
    { label: "Rotation", bg: "#3a3a3a", hue: 0, rot: 12 },
    { label: "Hue shift", bg: "#4a3a3a", hue: 0 },
    { label: "Blur", bg: "#3a3a3a", hue: 0, blur: true },
    { label: "Occlusion", bg: "#3a3a3a", hue: 0, occ: true },
    { label: "Background", bg: "#2a4a3a", hue: 0 },
  ];
  return (
    <div className="grid grid-cols-3 gap-2">
      {cells.map((c, i) => (
        <div key={i} className="aspect-[4/3] rounded-lg overflow-hidden relative grid place-items-center" style={{ backgroundColor: c.bg }}>
          <div
            className="h-1/2 w-1/2 rounded-md"
            style={{
              backgroundColor: "rgba(255,255,255,0.85)",
              transform: c.rot ? `rotate(${c.rot}deg)` : undefined,
              filter: c.blur ? "blur(2.5px)" : undefined,
            }}
          />
          {c.occ && (
            <div className="absolute inset-0">
              <div className="absolute w-1/4 h-1/3 bg-black/55" style={{ top: "35%", left: "40%" }} />
            </div>
          )}
          <div className="absolute top-1.5 left-1.5 rounded-full bg-black/55 text-white/90 text-[9px] uppercase tracking-wider font-mono px-1.5 py-0.5">
            {c.label}
          </div>
        </div>
      ))}
    </div>
  );
}

function HealthBadgeMock() {
  const factors = [
    { name: "Uniqueness", v: 0.78, weight: 40 },
    { name: "Balance", v: 0.86, weight: 20 },
    { name: "Coverage", v: 0.92, weight: 20 },
    { name: "Confidence", v: 0.81, weight: 20 },
  ];
  return (
    <div className="grid sm:grid-cols-[160px_1fr] gap-5 items-center">
      <div className="grid place-items-center">
        <div className="rounded-full bg-emerald-500/15 text-emerald-700 dark:text-emerald-200 px-5 py-3 inline-flex items-center gap-3">
          <span className="text-[10px] uppercase tracking-[0.18em] font-mono opacity-80">Health</span>
          <span className="text-2xl font-mono tabular-nums">82</span>
        </div>
      </div>
      <div className="grid gap-2">
        {factors.map((f) => (
          <div key={f.name} className="grid grid-cols-[110px_1fr_44px] items-center gap-3">
            <div className="text-[12px] text-[var(--muted)]">
              {f.name}{" "}
              <span className="text-foreground/35">({f.weight}%)</span>
            </div>
            <div className="h-1.5 rounded-full bg-foreground/[0.08] overflow-hidden">
              <div className="h-full rounded-full bg-foreground/55" style={{ width: `${Math.round(f.v * 100)}%` }} />
            </div>
            <div className="text-[12px] font-mono tabular-nums text-right text-[var(--foreground)]/80">{Math.round(f.v * 100)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ViewerOverlayMock() {
  return (
    <svg viewBox="0 0 200 130" className="w-full max-w-md mx-auto block" aria-hidden>
      <rect x="0" y="0" width="200" height="130" rx="6" fill="#2a2a2a" />
      {/* simulated subject */}
      <ellipse cx="92" cy="78" rx="44" ry="34" fill="#cccccc" opacity="0.85" />
      <ellipse cx="92" cy="78" rx="44" ry="34" fill="#5b8def" opacity="0.18" />
      {/* mask outline */}
      <ellipse cx="92" cy="78" rx="44" ry="34" fill="none" stroke="#5b8def" strokeWidth="1.2" opacity="0.95" />
      {/* bounding box */}
      <rect x="42" y="38" width="100" height="80" fill="none" stroke="#5b8def" strokeWidth="1" strokeDasharray="3 2" />
      {/* label chip */}
      <rect x="42" y="28" width="48" height="10" rx="5" fill="#5b8def" />
      <text x="46" y="35.5" fontSize="6" fontFamily="ui-sans-serif" fill="#fff" fontWeight="600">cat · 0.94</text>
    </svg>
  );
}

function ReviewModeMock() {
  return (
    <div className="grid sm:grid-cols-3 gap-3">
      <div className="rounded-xl bg-foreground/[0.03] p-3">
        <div className="text-[11px] uppercase tracking-wider font-mono text-emerald-700 dark:text-emerald-300 mb-2">
          ← Good
        </div>
        <div className="aspect-[4/3] rounded-lg bg-foreground/[0.06] grid place-items-center text-[11px] text-[var(--muted)]">
          confirmed
        </div>
      </div>
      <div className="rounded-xl bg-foreground/[0.04] p-3 outline outline-2 outline-foreground/15">
        <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--muted)] mb-2">
          Reviewing
        </div>
        <div className="aspect-[4/3] rounded-lg bg-foreground/[0.08] grid place-items-center text-[11px] text-[var(--foreground)]/70">
          current image
        </div>
      </div>
      <div className="rounded-xl bg-foreground/[0.03] p-3">
        <div className="text-[11px] uppercase tracking-wider font-mono text-red-600 dark:text-red-300 mb-2">
          Bad →
        </div>
        <div className="aspect-[4/3] rounded-lg bg-foreground/[0.06] grid place-items-center text-[11px] text-[var(--muted)]">
          rejected
        </div>
      </div>
    </div>
  );
}

function FilterChipsMock() {
  const chips = [
    { label: "All", count: 128, active: true },
    { label: "Unlabelled", count: 4 },
    { label: "Unrated", count: 96 },
    { label: "Good", count: 18 },
    { label: "Bad", count: 7 },
    { label: "Unsure", count: 3, tone: "amber" as const },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((c) => (
        <span
          key={c.label}
          className={[
            "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[12px]",
            c.active
              ? "bg-foreground text-background font-medium"
              : c.tone === "amber"
                ? "bg-amber-500/15 text-amber-800 dark:text-amber-200"
                : "bg-foreground/[0.06] text-[var(--foreground)]/80",
          ].join(" ")}
        >
          {c.label}
          <span className="font-mono tabular-nums opacity-70 text-[11px]">{c.count}</span>
        </span>
      ))}
    </div>
  );
}

// ---------- Tiny icon set used in the flow diagram ----------

function FolderIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
    </svg>
  );
}
function ImageIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="10" r="1.6" />
      <polyline points="4 18 9 13 13 16 20 9" />
    </svg>
  );
}
function CrosshairIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="2.5" />
      <line x1="12" y1="2" x2="12" y2="5" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="2" y1="12" x2="5" y2="12" />
      <line x1="19" y1="12" x2="22" y2="12" />
    </svg>
  );
}
function SparkleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 3v3" />
      <path d="M12 18v3" />
      <path d="M5 12H2" />
      <path d="M22 12h-3" />
      <path d="M6 6l2 2" />
      <path d="M18 18l-2-2" />
      <path d="M6 18l2-2" />
      <path d="M18 6l-2 2" />
    </svg>
  );
}
function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

// ---------- Sections ----------

function OverviewSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Guide"
        title="Build a labelled dataset from scratch"
        intro={
          <>
            PixelKit turns a pile of images into a labelled, augmented computer-vision
            dataset that&rsquo;s ready to train a model on. You give it pictures, you
            tell it the objects you care about, and it draws the boxes and masks for
            you. Every section below walks through a specific stage of the flow.
          </>
        }
      />

      <LLMPromptBox />

      <GuideSection n={1} title="The flow at a glance">
        <p>
          Most projects follow the same five-step rhythm. You don&rsquo;t need to
          know anything about computer vision to use it. Every step has sensible
          defaults.
        </p>
        <Figure caption="Create the project, add media, label, augment, export.">
          <FlowDiagram />
        </Figure>
        <Bullets
          items={[
            "<strong>Create a project</strong>. Name it, define the labels you want to detect.",
            "<strong>Add images</strong>. Drop your own, paste video files, or search Openverse from inside the app.",
            "<strong>Label</strong>. One click runs auto-labelling across the whole dataset.",
            "<strong>Augment</strong>. Configurable rotations, lighting, occlusions, background swaps, and more.",
            "<strong>Export</strong>. Download with train / val split, segmentations, and size-class filters baked in.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Section guide">
        <p>
          Pick a section to dive in. The sub-nav above the title stays on every
          page so you can jump around without losing your place.
        </p>
        <ul className="grid gap-3 sm:grid-cols-2">
          {GUIDE_SECTIONS.slice(1).map((s, i) => (
            <li key={s.key}>
              <button
                onClick={() => go(s.key)}
                className="w-full text-left block rounded-2xl border border-foreground/[0.08] bg-foreground/[0.02] hover:bg-orange-500/[0.04] hover:border-orange-400/35 transition-colors px-5 py-4 group h-full"
              >
                <div className="text-[10px] uppercase tracking-[0.18em] font-mono text-[var(--muted)] mb-1">
                  {String(i + 1).padStart(2, "0")}
                </div>
                <div className="text-base font-medium text-[var(--foreground)]">{s.title}</div>
                <p className="mt-0.5 text-sm text-[var(--muted)] leading-relaxed">{s.blurb}</p>
              </button>
            </li>
          ))}
        </ul>
      </GuideSection>

    </>
  );
}

function ProjectsSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 01"
        title="Create a dataset"
        intro={
          <>
            A dataset is a single collection of labelled images plus everything
            attached to it: labels, reference images, augmentations, settings.
            This section walks through creating one, choosing the right dataset
            type, and managing references when needed. (A <Strong>Project</Strong>
            {" "}is a separate thing, a team container that groups several
            datasets, covered in the Projects &amp; teams section.)
          </>
        }
      />

      <GuideSection n={1} title="Creating a dataset">
        <p>
          Datasets are private by default on the Pro plan. On the Free plan
          they live in the Community feed where anyone can see them. From the
          Workspace tab:
        </p>
        <Steps
          items={[
            ["Click + Add Dataset", "A small card slides in with a name field."],
            ["Type a name", "Anything you like. You can rename it later from Settings."],
            ["Click Continue", "PixelKit takes you to the labels stage."],
            ["Type the labels you want to detect", "Press Enter, comma, or full stop between each one. Each chip gets a random colour you&rsquo;ll see across the whole dataset."],
            ["Click Done", "PixelKit classifies the dataset as general or specific (see below) and routes you accordingly."],
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="General vs specific datasets">
        <p>
          When you click Done, PixelKit classifies the label set into one of two
          flavours.
        </p>
        <Figure caption="General labels skip the references step. Specific labels add a reference upload so similar classes can be told apart.">
          <GeneralVsSpecificDiagram />
        </Figure>
        <ul className="grid gap-4">
          <li className="rounded-xl border border-foreground/10 bg-foreground/[0.02] px-5 py-4">
            <Strong>General datasets</Strong>{" "}
            <span className="text-[var(--muted)]">
              e.g. <em>cat, dog, person, bicycle</em>
            </span>
            <p className="mt-2 text-sm">
              Labels describe distinct, common categories. PixelKit skips
              references and routes you straight to the dataset page. Auto-labelling
              already knows what a cat looks like.
            </p>
          </li>
          <li className="rounded-xl border border-foreground/10 bg-foreground/[0.02] px-5 py-4">
            <Strong>Specific datasets</Strong>{" "}
            <span className="text-[var(--muted)]">
              e.g. <em>hare, rabbit, alpaca, llama</em>, or rare and
              fine-grained variants
            </span>
            <p className="mt-2 text-sm">
              Labels are visually similar or very specific. PixelKit asks for a
              handful of reference images for each label. These help it tell
              visually-similar classes apart.
            </p>
          </li>
        </ul>
      </GuideSection>

      <GuideSection n={3} title="Reference images">
        <p>
          References are only needed for specific datasets. Each one is a clear
          photo of the labelled object: close-up, well-lit, little background
          clutter. The reference grid stays editable on the project page so
          you can keep updating them.
        </p>
        <Figure caption="Reference photos help PixelKit pick the right label when classes look similar.">
          <PipelineDiagram />
        </Figure>
        <Steps
          items={[
            ["Drop or click to upload", "Up to 20 reference images across all labels."],
            ["Pick the label for each one", "PixelKit guesses from context, but you can change it."],
            ["Continue", "Once each label has at least one solid reference, click Continue to move to the dataset page."],
          ]}
        />
        <Callout tone="info">
          <Strong>Editing references later:</Strong> open the Reference images
          section on the project page (it&rsquo;s collapsible) to add, remove,
          or re-label references. The change is picked up the next time you
          run auto-labelling.
        </Callout>
      </GuideSection>

      <GuideSection n={4} title="Cancelling out of onboarding">
        <p>
          If you bail before finishing the onboarding flow (Cancel or Esc), the
          half-created project is removed from the workspace automatically.
        </p>
      </GuideSection>

      <GuideSection n={5} title="The naming card">
        <Bullets
          items={[
            "Project names can be any text. They&rsquo;re display-only. The URL uses a stable UUID under the hood.",
            "A profanity guard runs on both the client and the server. Blocked terms surface as an inline red error before the request even leaves your browser.",
            "Names are editable any time from project Settings.",
          ]}
        />
      </GuideSection>

      <NextUp next="dataset" go={go} />
    </>
  );
}

function DatasetSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 02"
        title="Dataset"
        intro={
          <>
            Adding images and videos, searching Openverse from inside the app,
            video trim and frames-per-second selection, the verdict filter,
            bulk delete, and the per-tile chrome on every thumbnail.
          </>
        }
      />

      <GuideSection n={1} title="Adding images">
        <p>
          On the project page, the <em>Drop media here</em> card is where your
          dataset lives. Drag and drop images (JPG, PNG, WebP, HEIC, AVIF, BMP,
          TIFF, GIF) or videos (MP4, MOV, WebM, M4V, AVI, MKV), or click the
          card to open a file picker.
        </p>
        <Bullets
          items={[
            "Every image runs through a content-safety check before it&rsquo;s saved. Anything flagged as adult content is rejected with a clear error.",
            "Image orientation from EXIF metadata is honoured so iPhone photos don&rsquo;t end up sideways.",
            "Upload progress streams into the dataset gallery one tile at a time, with blurred placeholders for in-flight files.",
            "Up to 100 MB per video, then the file gets split into per-frame images on the server.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Video trim and frame rate">
        <p>
          Dropping a video opens a small modal before it&rsquo;s sent to the
          server. Drag the in / out handles to trim, set the frames-per-second
          rate, and preview how many frames the import will produce. The video
          is sliced server-side and each frame is treated like a fresh image.
        </p>
        <Bullets
          items={[
            "<strong>Trim handles</strong> let you crop to the relevant section of the clip so you don&rsquo;t flood the dataset with near-identical frames.",
            "<strong>FPS slider</strong> ranges from a single frame per second up to the source frame rate. Lower FPS produces more varied frames.",
            "<strong>Preview count</strong> updates as you drag so you can dial in roughly how many images the import will create.",
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Searching Openverse">
        <p>
          Click <em>Don&rsquo;t have images?</em> on the project page and
          PixelKit opens an inline Openverse search. Openverse aggregates
          openly-licensed photos. You can grab matching images directly into
          your dataset without leaving the app.
        </p>
        <Steps
          items={[
            ["Type what you’re looking for", "e.g. &ldquo;potholes&rdquo;, &ldquo;factory pipes&rdquo;, &ldquo;forklifts&rdquo;."],
            ["Tap Search", "You get a preview grid of candidate images."],
            ["Tap Yes, this is what I’m looking for", "Confirms the search direction."],
            ["Choose how many", "Slider from 1 up to your remaining import quota."],
            ["Pull images", "PixelKit downloads, content-safety-checks, and adds them to your dataset. <strong>Check the CC licence on each image before reusing it.</strong> Licences vary per image. Some require attribution, some restrict commercial use, some prohibit derivatives. The footer&rsquo;s Openverse policy page covers attribution and takedown."],
          ]}
        />
        <Callout tone="info">
          <Strong>Licensing:</Strong> every Openverse image carries its
          original CC licence on import. The Openverse policy page in the
          footer covers attribution and takedown for redistribution.
        </Callout>
      </GuideSection>

      <GuideSection n={4} title="The dataset gallery">
        <p>
          Each thumbnail in the gallery carries a small set of indicators:
        </p>
        <Bullets
          items={[
            "<strong>Sparkle icon (top-left)</strong>. Image has augmentations generated. Click it to open the Augmentations viewer.",
            "<strong>Unsure pill (top-left, amber)</strong>. At least one detection on the image was flagged borderline.",
            "<strong>×&nbsp;icon (top-right)</strong>. Delete this image. Single-tile delete.",
            "<strong>Detection count + label chips (bottom)</strong>. How many objects were found, in what colour each label is.",
            "<strong>Labelling overlay</strong>. Appears briefly while the active label job is processing this image.",
            "<strong>Verdict chip</strong>. If you&rsquo;ve marked the image good, bad, or unsure in Review mode, the corresponding pill paints in the top-left.",
          ]}
        />
      </GuideSection>

      <GuideSection n={5} title="Verdict filter">
        <p>
          The Filter chip rail above the gallery lets you scope the visible
          tiles by the verdict you&rsquo;ve given them. Counts in each pill
          update live as you tag images in Review mode or as new uploads land.
        </p>
        <Figure caption="Filter pills above the dataset gallery: All, Unlabelled, Unrated, Good, Bad, Unsure.">
          <FilterChipsMock />
        </Figure>
        <Bullets
          items={[
            "<strong>All</strong>. Default. Every tile.",
            "<strong>Unlabelled</strong>. Images with nothing detected.",
            "<strong>Unrated</strong>. Images you haven&rsquo;t reviewed yet.",
            "<strong>Good / Bad / Unsure</strong>. Whatever verdict you assigned in Review mode.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Bulk select &amp; delete">
        <p>
          A floating Select-mode toggle sits at the top of the gallery. Clicking
          it puts every tile into checkbox mode. Click any tile to add it to
          the selection. A bottom action bar shows the current count alongside
          Select all, Delete N, and Cancel buttons.
        </p>
        <Callout tone="warn">
          <Strong>Bulk delete is permanent.</Strong> The selected images, their
          labelled-preview thumbnails, and their augmentations are all removed
          in one go.
        </Callout>
      </GuideSection>

      <GuideSection n={7} title="The image viewer">
        <p>
          Click any tile to open the full-screen viewer. The canvas paints
          every detection (boxes, masks, label chips) over the original image.
          Manual editing controls live in the toolbar at the top, covered in
          the next section.
        </p>
        <Figure caption="The viewer overlays each detection on the image with a label chip carrying the class and confidence.">
          <ViewerOverlayMock />
        </Figure>
        <Bullets
          items={[
            "Arrow keys cycle through neighbouring images.",
            "Esc closes the viewer and returns to the gallery.",
            "Toolbar pills toggle the box, label, and mask display layers, plus the Click-to-detect and Manual labelling modes covered in the Labelling section.",
            "<strong>Clear all annotations</strong>. Wipes every box and mask on the current image in one modal-confirmed click. Useful for restarting from scratch on a difficult shot. Cleared images are treated as unlabelled, so the next Start labelling run will re-label them.",
            "<strong>Delete image</strong>. Removes the current image from the dataset entirely (the same as the gallery tile&rsquo;s trash) and moves on to the next one, or closes the viewer if it was the last.",
            "<strong>Live detection count</strong>. The header count next to the resolution reflects the boxes on the canvas, so it updates the moment you add or delete a detection.",
            "Large originals (e.g. 4K) load a fast display version first, then sharpen to full resolution in the background, so the viewer opens quickly and zoom stays crisp.",
          ]}
        />
      </GuideSection>

      <NextUp next="labelling" go={go} />
    </>
  );
}

function LabellingSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 03"
        title="Labelling"
        intro={
          <>
            One-click auto-labelling, the Annotations card
            controls, manual edits in the box editor, fast verdict review
            with Review mode, and how to purge a label entirely.
          </>
        }
      />

      <GuideSection n={1} title="Auto-labelling">
        <p>
          When you click <Strong>Start labelling</Strong> on the Annotations
          card, PixelKit&rsquo;s auto-labelling runs across every unlabelled
          image in your dataset. For each label you defined, you get back
          labelled detections for matching objects.
        </p>
        <p>
          For specific datasets, your reference photos help PixelKit choose
          the right label when classes look similar. The result is a
          per-detection label assignment with a confidence score. Borderline
          cases are flagged as <em>Unsure</em>.
        </p>
        <Callout tone="info">
          <Strong>Adaptive start button.</Strong> The Start labelling button
          reshapes its copy based on what would happen if you clicked it now.
          On a fresh project it reads &ldquo;Start labelling&rdquo;. After
          you&rsquo;ve added a new label to a partially-labelled project it
          flips to &ldquo;Label new images and new labels&rdquo; so it&rsquo;s
          obvious what the run will cover.
        </Callout>
      </GuideSection>

      <GuideSection n={2} title="Watching the job">
        <p>
          The progress card at the top of the project page tracks the labelling
          job:
        </p>
        <Bullets
          items={[
            "An ETA based on how long earlier images took.",
            "A live counter of processed images.",
            "The filename of the image currently in flight.",
            "A small Labelling overlay on the matching dataset thumbnail.",
            "Whimsical phrase rotator that paints something different every few seconds. Mostly there so the wait doesn&rsquo;t feel like nothing&rsquo;s happening.",
          ]}
        />
        <p>
          You can keep working on other tabs while it runs. The job completes
          whether you&rsquo;re looking or not. The card auto-dismisses a few
          seconds after &ldquo;Labelling complete&rdquo; lands.
        </p>
      </GuideSection>

      <GuideSection n={3} title="Auto-augment on finish">
        <p>
          When the labelling job lands, if the project already has an
          augmentation config saved (i.e. you&rsquo;ve clicked Update on the
          Augmentations tab at least once), PixelKit kicks off a fresh
          augmentation pass automatically. Every newly-labelled image picks
          up the same augmentations you have set without a second manual
          click.
        </p>
      </GuideSection>

      <GuideSection n={4} title="The Annotations card">
        <p>
          The Annotations card on the project page is the labelling control
          panel. It opens collapsed. The header shows the current label
          count, the detection-sensitivity mode, and synonym handling.
        </p>
        <Bullets
          items={[
            "<strong>Add / remove labels</strong>. Every change cascades through every image&rsquo;s chip rail.",
            "<strong>Detection mode</strong>. Switch between Normal and Stricter. Stricter favours fewer false positives at the cost of missing some objects.",
            "<strong>Synonyms</strong>. When on, common alternates of your label words also match (e.g. &ldquo;car&rdquo; matches &ldquo;automobile&rdquo;).",
            "<strong>Confidence slider</strong>. Lifts or lowers the cut-off below which detections are dropped. The slider previews live as you drag.",
            "<strong>Start labelling</strong>. Runs the pipeline. You can re-run any time to incorporate new images, edited labels, or a flipped detection mode.",
          ]}
        />
        <Callout tone="info">
          <Strong>Credit.</Strong> The small credit row
          under the Annotations heading links to Meta&rsquo;s SAM page.
          Required attribution, one of the conditions of using the model.
        </Callout>
      </GuideSection>

      <GuideSection n={5} title="Editing labels manually">
        <p>
          Click any tile in the dataset gallery to open the image viewer.
          From there:
        </p>
        <Bullets
          items={[
            "<strong>Move or resize</strong> a box by dragging its edges.",
            "<strong>Relabel</strong> by hovering a box and pressing a number key. The legend in the header shows which digit maps to which label.",
            "<strong>Delete</strong> by clicking a box and pressing Delete, or by clicking the × on its row in the right-hand sidebar.",
            "<strong>Draw</strong> a new box with the + Add box tool. PixelKit segments and labels it for you the moment you release the drag.",
            "<strong>Paint a mask</strong> onto an existing box with the brush / eraser pair.",
            "<strong>Toggle layers</strong>. Boxes, Labels and Masks each have their own pill toggle in the toolbar, useful for decluttering while you work.",
          ]}
        />
        <p>
          Every edit saves automatically. Augmentations re-generate in the
          background after each save so the augmentation set always reflects
          the latest annotations. No manual re-run needed.
        </p>
      </GuideSection>

      <GuideSection n={6} title="Click-to-detect">
        <p>
          The viewer toolbar has a click-to-detect mode. Activate it, then
          click anywhere on the image. PixelKit labels what you clicked, in
          one gesture. Designed to feel instant.
        </p>
        <Bullets
          items={[
            "Use it to top up a partially-labelled image without drawing a box by hand.",
            "If nothing useful matches under the click, the gesture is cancelled rather than placing a guess. Click somewhere else, or switch to Manual mode below.",
            "Works on labelled and unlabelled images alike. The new annotation sits alongside any boxes already there.",
          ]}
        />
      </GuideSection>

      <GuideSection n={7} title="Manual labelling mode">
        <p>
          Some objects are too unusual, too occluded, or too small for
          automatic detection to land cleanly. The Manual mode toggle in
          the viewer header switches PixelKit into a fully manual flow
          for the current image.
        </p>
        <Bullets
          items={[
            "Click-to-detect and auto-segment when drawing a new box are switched off.",
            "You draw the box geometry by hand, then type the label into the picker that pops up.",
            "Mask painting still works, so you can refine the outline manually.",
            "Flip the toggle back off to return to the assisted defaults.",
          ]}
        />
        <Callout tone="info">
          <Strong>Useful when</Strong> auto-labelling cannot find the object you are
          looking at, or when you want the box exactly where you draw it
          without any model-driven adjustment.
        </Callout>
      </GuideSection>

      <GuideSection n={8} title="Unsure detections">
        <p>
          When PixelKit isn&rsquo;t confident about a detection&rsquo;s label,
          it flags the detection as <em>Unsure</em>. You&rsquo;ll see this
          surface in three places:
        </p>
        <Bullets
          items={[
            "An amber Unsure pill on the dataset thumb.",
            "An &ldquo;n unsure&rdquo; row on the Dataset stats card.",
            "An Unsure chip next to the affected detection in the image viewer&rsquo;s box list.",
          ]}
        />
        <p>
          Click into the viewer and either confirm (relabel via 1-9) or delete
          the box. The Unsure count drops live as you go.
        </p>
      </GuideSection>

      <GuideSection n={9} title="Review mode">
        <p>
          Review mode is a fast keyboard-or-swipe interface for tagging every
          image with a verdict (good, bad, or unsure) without opening the full
          viewer. It paints the current image with all its detections so you
          can decide at a glance whether the labelling is worth keeping.
        </p>
        <Figure caption="Swipe or arrow-left to mark the current image good, swipe or arrow-right to mark it bad, space to mark it unsure. Verdicts feed the gallery filter chips.">
          <ReviewModeMock />
        </Figure>
        <Bullets
          items={[
            "<strong>← Good</strong>. Confirms the labelling on this image.",
            "<strong>→ Bad</strong>. Flags the image for follow-up. Doesn&rsquo;t delete anything.",
            "<strong>Space / Unsure</strong>. Park for later review.",
            "<strong>Tap a label chip</strong> to filter visible boxes by that label as you scrub through.",
            "<strong>Hover-highlight</strong>. Hovering a detection box dims everything else so the focus pops.",
            "<strong>Verdict persists</strong>. Stored on the image, surfaces in the gallery filter chips, and survives reloads.",
          ]}
        />
      </GuideSection>

      <GuideSection n={10} title="Deleting a label entirely">
        <p>
          Removing a label chip in the Annotations card triggers a confirm
          modal because the action is destructive. Confirming it purges every
          detection on that label across the whole dataset, removes the
          label&rsquo;s reference data (for specific datasets), and tears down
          any augmentations that referenced it.
        </p>
        <Bullets
          items={[
            "A progress card shows the purge in flight, the same shape as the labelling progress card.",
            "Other detections on affected images are kept. Only the boxes carrying the removed label are touched.",
            "The label disappears from every chip rail and from the colour palette.",
          ]}
        />
        <Callout tone="warn">
          <Strong>Irreversible.</Strong> Once confirmed, the label and every
          detection on it are gone. Take an export if you might want the
          annotations back.
        </Callout>
      </GuideSection>

      <NextUp next="augmentations" go={go} />
    </>
  );
}

function AugmentationsSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 04"
        title="Augmentations"
        intro={
          <>
            Augmentations create variations of every image (rotations, lighting,
            occlusions, background swaps, and more) so a trained model
            generalises beyond the exact framing of your originals. This
            section covers every dial.
          </>
        }
      />

      <GuideSection n={1} title="The four categories">
        <p>
          Open the Augmentations tab. The dials sit under four collapsible
          category cards. Click any heading to expand it. Clicking a
          category&rsquo;s enable checkbox auto-expands it too.
        </p>
        <Bullets
          items={[
            "<strong>Camera &amp; sensor</strong>. Simulates the artefacts of a real-world camera.",
            "<strong>Distortion</strong>. Perspective, scale, rotation, hue.",
            "<strong>Occlusion</strong>. Simulates things blocking the camera over a configurable fraction of the detected objects.",
            "<strong>Domain randomisation</strong>. Varies the background and lighting.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="A look at the output">
        <p>
          Every Update kicks off a fresh generation that produces 1, 2, or 3
          augmented copies per source image, depending on the per-image count.
          The Augmentations viewer (covered below) shows you exactly what
          PixelKit produced.
        </p>
        <Figure caption="A handful of augmentations applied to the same source image. Real previews look photographic, not block-shape.">
          <AugmentationGridMock />
        </Figure>
      </GuideSection>

      <GuideSection n={3} title="Per-augmentation controls">
        <p>
          Inside each category, every sub-augmentation has a consistent set
          of controls:
        </p>
        <Bullets
          items={[
            "<strong>Enable checkbox</strong>. Opens the controls. Ticked categories are included in every Update.",
            "<strong>Frequency toggle</strong>. <em>All</em> applies the augmentation to every copy. <em>Random</em> applies to roughly half so the augmented set has more variety.",
            "<strong>Strength dial(s)</strong>. Each augmentation has one or more sliders. Drag to see a live preview on a sample dataset image.",
            "<strong>Random image button</strong>. Under each preview, swaps to a different source image so you can see the augmentation on a few different shots before committing.",
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Per-image count">
        <p>
          At the top of the Augmentations tab, the <em>Per image</em> slider
          sets how many augmented copies to generate per dataset image:
        </p>
        <Bullets
          items={[
            "<strong>Off</strong>. No augmentations. Hitting Update in this mode deletes every existing augmentation (button label flips to Clear all).",
            "<strong>1 / 2 / 3</strong>. Fixed copy count per image.",
            "<strong>Random</strong>. PixelKit picks 1, 2 or 3 per image; re-runs are reproducible.",
          ]}
        />
      </GuideSection>

      <GuideSection n={5} title="Randomise + Update">
        <p>
          Two buttons in the top-right of the page drive the pipeline.
        </p>
        <Bullets
          items={[
            "<strong>Randomise</strong>. Rolls fresh values for every pure-dial augmentation in one click. Skips augmentations that need uploaded assets (background swap, object overlay). Auto-opens every category card so the new values land visibly.",
            "<strong>Update</strong>. Generates augmentations for every dataset image based on the current settings. The job runs in the background. The same progress card you see during labelling tracks it, mounted directly on the Augmentations tab too so you can stay on this page.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Background swap">
        <p>
          Background randomisation places the detected objects from your
          dataset images onto a different background. Useful for teaching the
          model that the same object can appear in many environments.
        </p>
        <Bullets
          items={[
            "Upload up to three backgrounds. They are compressed in the browser before upload so the page does not stall on slow connections.",
            "Backgrounds run through the same content-safety check as dataset uploads.",
            "Mix any number of backgrounds. PixelKit picks one at random per generated copy.",
          ]}
        />
      </GuideSection>

      <GuideSection n={7} title="The Augmentations viewer">
        <p>
          Once Update has run, every dataset thumbnail picks up a small sparkle
          icon. Click it to open the Augmentations viewer for that image: a
          grid of the generated variations alongside the original.
        </p>
        <Bullets
          items={[
            "<strong>Hover a tile</strong> to see the annotations overlay: polygons and bounding boxes coloured by size validity against the project&rsquo;s target input shape (green / orange / red).",
            "<strong>Click the × on a tile</strong> to delete that single augmentation. The remaining count updates live and the dataset-thumbnail sparkle hides when the last copy is gone.",
            "<strong>Bounding boxes turn red</strong> when they will be too small to detect at your chosen input shape. A fast visual indicator that the annotation will be hard for the model to learn.",
          ]}
        />
      </GuideSection>

      <GuideSection n={8} title="Auto-regen after edits">
        <p>
          The augmentation set always reflects the latest annotations. Two
          triggers re-run augmentations automatically:
        </p>
        <Bullets
          items={[
            "When the labelling job finishes (any newly-labelled images get covered).",
            "When you commit a manual edit in the image viewer (a debounced regen kicks in after the save lands).",
          ]}
        />
        <p>
          You never need to click Update again after a label or annotation
          change. It just happens.
        </p>
        <Callout tone="info">
          The auto-regen skips while another generate job is already running
          for the project. Edits during an in-flight run pile up and the next
          regen picks them all up at once.
        </Callout>
      </GuideSection>

      <NextUp next="stats" go={go} />
    </>
  );
}

function StatsSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 05"
        title="Dataset stats"
        intro={
          <>
            The Dataset stats card sits at the top of the Dataset tab. It
            surfaces counts, label distribution, a 0-100 health score,
            a 2-D variation plot, and a Review duplicates modal for finding
            and removing near-identical images.
          </>
        }
      />

      <GuideSection n={1} title="The summary row">
        <p>
          The card opens collapsed showing a one-line summary so it
          doesn&rsquo;t dominate the page. Expand it for the full breakdown.
        </p>
        <Bullets
          items={[
            "<strong>Image count</strong>. How many images you&rsquo;ve added to the dataset.",
            "<strong>Detection count</strong>. Total objects detected across the dataset.",
            "<strong>Label count</strong>. How many distinct labels have at least one detection.",
            "<strong>Augmentation count</strong>. Total generated copies across all images.",
            "<strong>Near-dup count</strong>. Flagged when one or more images cluster too tightly with another (amber-tinted, click to open the review modal).",
            "<strong>Health badge</strong>. The 0-100 score, covered below.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Health score">
        <p>
          The health score is a single 0-to-100 number summarising how
          trainable your dataset is. Hover the badge for a short breakdown
          of the biggest things to fix.
        </p>
        <Figure caption="A green badge is healthy, amber is workable, red flags a meaningful issue. Hover the badge in-product to see the breakdown.">
          <HealthBadgeMock />
        </Figure>
        <p>
          The score blends several signals about your dataset and surfaces
          the lowest-scoring one as the recommended next thing to fix.
          Following the suggestion is usually the cheapest win.
        </p>
      </GuideSection>

      <GuideSection n={3} title="AI insights">
        <p>
          Above the rule-based suggestion cards, the dataset overview shows a
          short <Strong>AI insight</Strong>: a single lead recommendation,
          written by a language model, on the most useful thing to do next. The
          cards beneath it cover the deterministic checks (duplicates, detection
          coverage, label balance).
        </p>
        <Bullets
          items={[
            "It is token-frugal by design. The insight only regenerates on a <strong>material change</strong> (crossing an image or labelled-count tier, or changing the set of labels), not on every upload. The rest of the time it is served from cache.",
            "It respects your label set. A single-class dataset is treated as a deliberate choice, so it will not nag you to add more labels.",
            "If the model is unavailable, the rule-based cards still show, so the Insights area is never empty.",
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Label distribution">
        <p>
          The middle column of the expanded card shows a bar per label
          sorted by count, with the label&rsquo;s project colour as the
          fill and a count + percentage on the right. Useful for spotting
          imbalances at a glance. If one class accounts for 80% of all
          detections, your model will learn it disproportionately.
        </p>
      </GuideSection>

      <GuideSection n={5} title="Image variation plot">
        <p>
          The right column shows a 2-D plot of the dataset. Each dot is one
          image. The position reflects how visually similar it is to every
          other image. Tight clusters mean lots of pictures of basically the
          same thing. Wide spread means good variation.
        </p>
        <Figure caption="Each dot is an image, coloured by primary label. An amber halo flags near-duplicates. Click a dot to jump to that tile in the dataset gallery.">
          <VariationPlotMock />
        </Figure>
        <Bullets
          items={[
            "<strong>Dot colour</strong> matches the primary label of the image.",
            "<strong>Dot radius</strong> scales softly with the number of objects in the image.",
            "<strong>Amber halo</strong> around a dot flags it as a near-duplicate of at least one other image. Consider removing one or the other.",
            "<strong>Smaller dots clustered around each main one</strong> are the augmentations generated from that image.",
            "<strong>Click a dot</strong> to scroll the dataset gallery to that image, dim everything else, and highlight the matched tile. Clicking an augmentation dot jumps to its parent image with an &ldquo;Augmentation&rdquo; badge briefly flashing over the cover.",
            "<strong>Soft spinner</strong>. The plot shows a spinner while it is still computing, then fills in as the data lands.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Near-duplicate detection &amp; review">
        <p>
          PixelKit identifies pairs of images that look very similar to a
          model. Both members of each flagged pair get an amber halo on the
          variation plot, and the summary row shows a single &ldquo;n
          near-dup&rdquo; count.
        </p>
        <p>
          Clicking that count opens the <Strong>Review duplicates</Strong>
          modal. From there you can scan every flagged group, pick which
          image to keep, and act in bulk.
        </p>
        <Bullets
          items={[
            "<strong>Near / Exact toggle</strong>. Switch between visually-similar pairs and pixel-identical duplicates.",
            "<strong>Per-group keeper</strong>. Each group shows one image as the keeper (emerald outline) and the rest as candidates for deletion.",
            "<strong>Per-group actions</strong>. Delete the n duplicates in one click, or mark the group as &ldquo;Not duplicates&rdquo; to ignore it next time.",
            "<strong>Select &amp; bulk-delete</strong>. Tick individual non-keeper images across multiple groups and delete them all in one go.",
            "<strong>Select all duplicates</strong>. One click to mark every non-keeper across every group.",
          ]}
        />
      </GuideSection>

      <GuideSection n={7} title="Live updates">
        <p>
          The card refreshes itself on every meaningful change to your
          dataset: new uploads, deletions, manual annotation edits, and
          augmentation generate / delete events. No reload required.
        </p>
      </GuideSection>

      <NextUp next="settings" go={go} />
    </>
  );
}

function SettingsSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 06"
        title="Settings & themes"
        intro={
          <>
            Project settings, label management, visibility, the light / dark
            toggle, the workspace and public-projects feeds, and exporting
            the finished dataset.
          </>
        }
      />

      <GuideSection n={1} title="Opening Settings">
        <p>
          Click the gear icon at the top of any project page to open the
          Settings panel. From there you can rename the project, change its
          visibility, set the cover image, manage label colours, and delete
          the project entirely.
        </p>
      </GuideSection>

      <GuideSection n={2} title="Renaming the project">
        <p>
          Type the new name and click <em>Rename</em>. Profanity guard runs
          on the new value before the request is sent. Blocked terms surface
          as an inline red error.
        </p>
        <p>
          The renamed project propagates everywhere it&rsquo;s shown in the
          same tick (workspace card, public feed, page title) without
          waiting for a refresh.
        </p>
      </GuideSection>

      <GuideSection n={3} title="Visibility">
        <Bullets
          items={[
            "<strong>Free plan</strong>. Projects are public. Anyone on the platform can see the images, labels and exports. Don&rsquo;t upload anything sensitive.",
            "<strong>Pro plan</strong>. Toggle a project to private. Private projects don&rsquo;t appear in the community feed and are only visible to you.",
            "<strong>Beta tier</strong>. Same surface area as Pro for the duration of the access window, including the private toggle.",
          ]}
        />
        <p>
          The visibility toggle in Settings takes effect immediately. The
          padlock icon at the top of the project page repaints in the same
          tick across every place the project is shown.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Cover image">
        <p>
          Pick which thumbnail represents the project across the workspace
          and public feed. The Cover image section in Settings shows every
          uploaded image as a small grid. Click any of them to set it as the
          cover. Click the small reset icon (visible when a custom cover is
          set) to revert to PixelKit&rsquo;s default pick.
        </p>
      </GuideSection>

      <GuideSection n={5} title="Renaming and recolouring labels">
        <p>
          The Label colours section in Settings lists every label in the
          project, each row showing the current swatch and the count of
          detections carrying that label.
        </p>
        <Bullets
          items={[
            "<strong>Rename</strong> a label by clicking the pencil icon. The change cascades to every chip in the workspace card, dataset gallery, and image viewer instantly. Profanity guard applies.",
            "<strong>Recolour</strong> by clicking the swatch. Pick from the eight-tone palette or paste a hex value. Colours persist into the per-project meta cache so they survive reloads with no re-fetch.",
            "<strong>Reset</strong> the colour to the default palette pick by clicking the reset icon when a custom colour is set.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Export modal">
        <p>
          Click the <Strong>Export</Strong> button at the top of the project
          page to open the export modal. PixelKit packages your dataset into
          a single download with annotations in your chosen format.
        </p>
        <Bullets
          items={[
            "<strong>Train / val split</strong>. A slider sets the percentage of images to go into the training split. The remainder lands in val. Splits are reproducible across re-exports.",
            "<strong>Boxes / Segmentations</strong>. Toggle which annotation kinds to include. At least one has to be on.",
            "<strong>Size-class filters</strong>. Include or exclude detections that fall into Tiny / Small / Medium / Large buckets at your chosen input shape.",
            "<strong>Include augmentations</strong>. Optional. When on, every generated copy ships alongside its source.",
            "<strong>Animated modal</strong>. Slides in with a backdrop blur. Esc closes it without committing.",
          ]}
        />
      </GuideSection>

      <GuideSection n={7} title="Deleting a project">
        <p>
          At the bottom of Settings, the destructive section wipes the
          project permanently. Type the project name to confirm before the
          delete button enables. Once gone:
        </p>
        <Bullets
          items={[
            "Every reference image is removed.",
            "Every imported image is removed.",
            "Every annotation, augmentation, and labelled-preview thumbnail is removed.",
            "The project itself is taken off the server.",
          ]}
        />
        <Callout tone="warn">
          <Strong>No undo.</Strong> Once you confirm the delete, the data is
          gone. Take an export first if you might want it back.
        </Callout>
      </GuideSection>

      <GuideSection n={8} title="Workspace, Community, and read-only view">
        <p>
          Two separate tabs in the top-nav:
        </p>
        <Bullets
          items={[
            "<strong>Workspace</strong> is your own area: your Projects (team containers) across the top row, then the datasets you own below. Paginated, loads the top page first and infinite-scrolls more as you go.",
            "<strong>Community</strong> is the public feed (it used to be called Projects). A carousel of public Projects sits above the grid of public datasets, which is sortable by Trending, Newest, or Most liked.",
          ]}
        />
        <p>
          Click any public dataset to open it in read-only view. You see the
          full dataset gallery, stats card, variation plot, and image viewer
          (without edit controls). Owner-only chrome (the Drop card, Annotations
          controls, Settings, and Export) is hidden.
        </p>
      </GuideSection>

      <GuideSection n={9} title="Favourites and likes">
        <p>
          Two distinct gestures, two different signals.
        </p>
        <Bullets
          items={[
            "<strong>Favourite</strong>. The star button on a project card. Favourites pin to the top of your workspace and the public feed sort. Useful for projects you want quick access to.",
            "<strong>Like</strong>. The heart button on a public project. Likes are public and feed into the Trending sort. They don&rsquo;t change a project&rsquo;s position in your own workspace.",
          ]}
        />
      </GuideSection>

      <GuideSection n={10} title="Light and dark mode">
        <p>
          The moon / sun toggle in the top-right of every page switches
          between dark and light modes. The choice is stored in your browser
          so it persists across navigations and refreshes. It also syncs
          across tabs you have open. PixelKit defaults to light mode for new
          visitors.
        </p>
      </GuideSection>

      <NextUp next="teams" go={go} />
    </>
  );
}

function TeamsSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 07"
        title="Projects & teams"
        intro={
          <>
            A <Strong>Project</Strong> is a team container that groups several
            datasets together with shared members and roles. It is the layer
            above an individual dataset, useful once more than one person is
            working on related datasets.
          </>
        }
      />

      <GuideSection n={1} title="Projects vs datasets">
        <p>Two things share similar names, so it is worth being precise:</p>
        <Bullets
          items={[
            "A <strong>dataset</strong> is one collection of labelled images (the thing you build in the Create a dataset, Labelling, and Augmentations sections).",
            "A <strong>Project</strong> is a container that holds many datasets, plus members, a cover photo, a privacy setting, and an activity timeline.",
          ]}
        />
        <p>
          On the Workspace tab your Projects sit in a row across the top, above
          the grid of your individual datasets.
        </p>
      </GuideSection>

      <GuideSection n={2} title="Creating a Project">
        <Steps
          items={[
            ["Click the New project tile", "It sits at the start of the Projects row on the Workspace."],
            ["Name it and pick a cover", "The cover photo is optional. It shows on the Project hero and card."],
            ["Set the privacy", "The toggle is on for private (the default), off for public. A private Project is hidden from the Community feed, and so are its datasets."],
            ["Create", "You land on the Project page, ready to add datasets and members."],
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Adding datasets">
        <p>On the Project page, the Datasets section has two buttons:</p>
        <Bullets
          items={[
            "<strong>Add existing</strong> pulls one of your standalone datasets into the Project.",
            "<strong>+ New dataset</strong> creates a fresh dataset directly inside the Project, using the same naming and labels flow as the Workspace.",
          ]}
        />
        <p>
          A dataset can belong to a Project and still be edited exactly as you
          always edit it. Datasets created by other members can be added too.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Members and roles">
        <p>
          Each member of a Project has one of three roles, which control what
          that person can do inside it.
        </p>
        <ul className="grid gap-4">
          <li className="rounded-xl border border-foreground/10 bg-foreground/[0.02] px-5 py-4">
            <Strong>Owner</Strong>
            <p className="mt-2 text-sm">
              Full control. Everything an editor can do, plus renaming the
              Project, changing its cover, setting its privacy, adding and
              removing members, and deleting the Project.
            </p>
          </li>
          <li className="rounded-xl border border-foreground/10 bg-foreground/[0.02] px-5 py-4">
            <Strong>Editor</Strong>
            <p className="mt-2 text-sm">
              Can do almost everything inside the Project: add and create
              datasets, upload images, run labelling and augmentation, and edit
              datasets created by other members. An editor cannot rename or
              delete the Project, change its cover, or manage members.
            </p>
          </li>
          <li className="rounded-xl border border-foreground/10 bg-foreground/[0.02] px-5 py-4">
            <Strong>Viewer</Strong>
            <p className="mt-2 text-sm">
              Read-only access to the Project and its datasets.
            </p>
          </li>
        </ul>
        <Callout tone="info">
          <Strong>Deleting a dataset</Strong> from a Project asks what should
          happen: <em>remove from project</em> detaches it (it survives as a
          standalone dataset), or <em>delete entirely</em> destroys it. Deleting
          entirely is creator-only: even the Project owner can only detach a
          dataset somebody else created, never permanently delete it.
        </Callout>
      </GuideSection>

      <GuideSection n={5} title="Cover, privacy, and activity">
        <Bullets
          items={[
            "<strong>Cover</strong>. Upload a cover photo for the Project hero and card. A small cover is AI-upscaled on the GPU so the full-width banner stays sharp.",
            "<strong>Privacy</strong>. A private Project (and every dataset in it) is hidden from the Community feed. Public Projects appear in the Community carousel.",
            "<strong>Activity</strong>. The Project page carries a timeline of recent events across all its datasets: uploads, labelling runs, members added, datasets added or removed.",
            "<strong>Derived icon</strong>. A dataset that is a cropped child of another shows a small branch icon next to its name (see the next section).",
          ]}
        />
      </GuideSection>

      <NextUp next="derived" go={go} />
    </>
  );
}

function DerivedSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 08"
        title="Derived datasets"
        intro={
          <>
            A derived dataset is a cropped child of a parent dataset: one image
            per detection, each cropped to its box. It is handy for training a
            focused classifier on individual objects. The child stays linked to
            the parent and re-syncs from it one way.
          </>
        }
      />

      <GuideSection n={1} title="What it is">
        <p>
          From a normal dataset&rsquo;s overview, the <Strong>Derived
          datasets</Strong> panel creates a child whose images are the
          parent&rsquo;s detections, cropped out one per image. A person photo
          with three boxes becomes three crops (one person, one helmet, one
          glove), each its own image with a single box.
        </p>
        <p>
          The link is one way. Edits to the parent flow down to the child on the
          next sync; the child never writes back to the parent.
        </p>
      </GuideSection>

      <GuideSection n={2} title="Creating one">
        <p>Click <Strong>Create cropped dataset</Strong> in the panel. The modal has:</p>
        <Bullets
          items={[
            "<strong>Name</strong>. Pre-filled from the parent (a dataset called <em>people</em> suggests <em>People Crops</em>). Editable.",
            "<strong>Label source</strong>. <em>Keep parent labels</em> carries each detection&rsquo;s label onto its crop. <em>Create new labels</em> brings the crops in unlabelled so you can build a fresh taxonomy.",
            "<strong>Labels to include</strong>. Pick which of the parent&rsquo;s labels to crop.",
            "<strong>ROI mode (square crops)</strong>. Forces every crop to an exact 1:1 square centred on the object, so long thin objects are not cropped to slivers.",
            "<strong>Fixed crop size</strong>. Optional. Turn it on and a slider sets one size: every crop is resized to exactly that N x N, so all derived images match (handy for training). It forces square crops and overrides the minimum size.",
            "<strong>Context padding</strong> and <strong>minimum image size</strong> sliders, to keep surrounding context and to scale tiny crops up.",
            "<strong>Group in a workspace Project</strong>. On keeps the crop dataset alongside its parent in a Project; off creates a standalone dataset.",
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Keep vs create new labels">
        <p>The label source choice changes what the crops arrive with.</p>
        <Bullets
          items={[
            "<strong>Keep parent labels</strong>. Each crop carries the parent&rsquo;s box, segmentation, and label. Reference images and label colours come across too.",
            "<strong>Create new labels</strong>. Each crop is a blank ROI canvas: no box, no segmentation, no label, and no reference images are copied. You draw and label everything fresh. The crop&rsquo;s original parent label is still shown as a muted &ldquo;from {label}&rdquo; reference on the gallery card and in the viewer header, so you always know what it was.",
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Live sync">
        <p>
          The child re-derives from the parent&rsquo;s current state, on creation
          and whenever you hit <Strong>Sync now</Strong> on the child&rsquo;s
          banner.
        </p>
        <Bullets
          items={[
            "A detection removed from the parent removes its crop from the child.",
            "A crop you delete in the child is remembered, so a later sync will not bring it back.",
            "In create-new mode, boxes you have drawn yourself are preserved across syncs; only unwanted inherited boxes get cleaned up.",
          ]}
        />
      </GuideSection>

      <GuideSection n={5} title="Limits">
        <Bullets
          items={[
            "A parent can have <strong>many</strong> derived datasets (different label selections or crop settings).",
            "You cannot derive from a derived dataset. The lineage stays one level deep (no derivatives of derivatives).",
          ]}
        />
      </GuideSection>

      <NextUp next="reference" go={go} />
    </>
  );
}

function ReferenceSection() {
  return (
    <>
      <GuideHeader
        eyebrow="Section 09"
        title="Reference"
        intro={
          <>
            Keyboard shortcuts, plan limits (including the Beta tier), and
            the content-safety policy. The bits you reach for sporadically,
            kept in one place.
          </>
        }
      />

      <GuideSection n={1} title="Image viewer shortcuts">
        <p>
          Most of the time you&rsquo;ll be in the image viewer when you want
          a shortcut. The toolbar surfaces the obvious ones. This table
          covers everything.
        </p>
        <ShortcutTable
          rows={[
            ["←  /  →", "Cycle to the previous / next image in the gallery."],
            ["Esc", "Close the viewer and return to the dataset gallery."],
            ["1 - 9", "Relabel the box under the cursor to label N. The legend in the header shows which digit maps to which label."],
            ["Delete", "Delete the currently-selected box."],
            ["B", "Toggle the box-display layer."],
            ["L", "Toggle the label-chip layer."],
            ["M", "Toggle the mask layer."],
            ["+ / -", "Zoom the canvas in / out."],
            ["0", "Reset the zoom + pan to fit-on-screen."],
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Review mode shortcuts">
        <ShortcutTable
          rows={[
            ["←", "Mark current image Good and advance."],
            ["→", "Mark current image Bad and advance."],
            ["Space", "Mark current image Unsure and advance."],
            ["Esc", "Close Review mode and return to the dataset gallery."],
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Project + workspace shortcuts">
        <ShortcutTable
          rows={[
            ["/", "Focus the search input (when one is visible on the current page)."],
            ["?", "Open the global shortcuts cheatsheet from anywhere in the app."],
            ["Esc", "Close any open modal or panel."],
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Plan limits">
        <p>
          The Free tier exists so you can try PixelKit end-to-end without a
          card. Pro removes the caps and unlocks private projects. Beta gives
          a 30-day Pro-equivalent window once a redeem code is applied. The
          exact numbers can shift (the Pricing page is canonical), but at
          the time of writing:
        </p>
        <Bullets
          items={[
            "<strong>Free</strong>. Small per-project image cap, fixed monthly Openverse import quota, public projects only.",
            "<strong>Pro</strong>. Three credit tiers. Large per-project image cap, large monthly import quota, private projects allowed, priority labelling queue.",
            "<strong>Beta</strong>. Pro-level limits for 30 days from the day a beta code is redeemed. Same private-project access as Pro.",
          ]}
        />
        <Callout tone="info">
          Hit the cap mid-upload and the gallery shows an inline note
          explaining which limit was reached. Free-tier users can upgrade
          from the Pricing tab. Pro users can contact us for bespoke
          ceilings.
        </Callout>
      </GuideSection>

      <GuideSection n={5} title="Content safety">
        <p>
          Every image entering PixelKit is checked before it lands in your
          dataset. The check runs server-side and applies to all three
          ingress paths:
        </p>
        <Bullets
          items={[
            "<strong>Direct uploads</strong>. Drag-and-drop and file-picker.",
            "<strong>Openverse imports</strong>. Even though Openverse pre-filters, we re-check every fetched image.",
            "<strong>Background uploads</strong>. The backgrounds you upload for the augmentation swap.",
          ]}
        />
        <p>
          Anything classified as adult content is rejected with a clear
          error. The image is never written to disk. Nothing flagged at
          upload makes it into your dataset, into the public feed, or into
          any export.
        </p>
      </GuideSection>

      <GuideSection n={6} title="Profanity guard">
        <p>
          Project names, label names, and any other user-typed text gets
          run through a profanity filter before it&rsquo;s saved. The list
          covers the obvious cases plus the common bypass spellings. If
          you&rsquo;re working on a legitimate dataset that needs a flagged
          word, get in touch and we&rsquo;ll add an exception.
        </p>
      </GuideSection>

      <GuideSection n={7} title="Licensing of imported imagery">
        <p>
          Openverse aggregates openly-licensed images from across the web,
          but the licence varies per image. PixelKit records the licence
          and the source URL for every Openverse import and stamps both
          into your exports so the trail stays intact.
        </p>
        <p>
          Direct uploads carry no licence metadata. You&rsquo;re responsible
          for ensuring you have the right to use the images you upload,
          especially for any model you intend to redistribute.
        </p>
      </GuideSection>

      <GuideSection n={8} title="Getting in touch">
        <p>
          The feedback link in the footer goes to a small form that
          forwards to the team. Bugs, feature requests, exception
          requests, and general feedback all welcome.
        </p>
      </GuideSection>
    </>
  );
}
