"use client";

// Guide view. Mounted by /app/page.tsx for ?tab=guide.

import { useEffect, useState, type ReactNode } from "react";

type GuideSectionKey =
  | "overview"
  | "create"
  | "dataset"
  | "labelling"
  | "augmentations"
  | "stats"
  | "settings"
  | "projects"
  | "derived"
  | "training"
  | "reference";

const GUIDE_SECTIONS: { key: GuideSectionKey; title: string; blurb: string }[] = [
  { key: "overview", title: "Overview", blurb: "What PixelKit does and how the moving parts fit together." },
  { key: "create", title: "Create a dataset", blurb: "Creating a dataset, naming, labels, optional reference images." },
  { key: "dataset", title: "Dataset", blurb: "Adding images, searching Openverse, video trims, filters, bulk delete." },
  { key: "labelling", title: "Labelling", blurb: "Auto-labelling, detection settings, manual edits, fast Review mode." },
  { key: "augmentations", title: "Augmentations", blurb: "Every dial, the Randomise button, the per-tile viewer." },
  { key: "stats", title: "Dataset stats", blurb: "The health score, insights, the variation plot, near-duplicate review." },
  { key: "settings", title: "Settings & export", blurb: "Renaming, recolouring labels, annotation snapshots, export." },
  { key: "projects", title: "Projects", blurb: "Group datasets into a Project: shared labels, covers, activity." },
  { key: "derived", title: "Derived datasets", blurb: "Crop each detection into a child dataset: ROI squares, inherited or fresh labels, live sync." },
  { key: "training", title: "Train for Neuro N6", blurb: "Train a YOLOX on this dataset, check it before and after, verify it for the NPU, and export the board bundle." },
  { key: "reference", title: "Reference", blurb: "Keyboard shortcuts and reference tables." },
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
    document.querySelector<HTMLElement>("[data-app-scroll]")?.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <main className="min-h-full mx-auto max-w-6xl px-6 pt-8 pb-16">
      <nav
        aria-label="Guide sections"
        className="flex flex-wrap gap-1.5 mb-8 pb-2 border-b border-[var(--line-soft)]"
      >
        {GUIDE_SECTIONS.map((s) => {
          const isActive = s.key === section;
          return (
            <button
              key={s.key}
              onClick={() => go(s.key)}
              className={[
                "px-3 h-8 inline-flex items-center rounded-md border font-mono text-[11px] uppercase tracking-[0.12em] transition-colors duration-[90ms]",
                isActive
                  ? "border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--foreground)] font-medium"
                  : "border-[var(--line)] text-[var(--fg-muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)]",
              ].join(" ")}
            >
              {s.title}
            </button>
          );
        })}
      </nav>

      <div className="grid gap-10">
        {section === "overview" && <OverviewSection go={go} />}
        {section === "create" && <CreateSection go={go} />}
        {section === "dataset" && <DatasetSection go={go} />}
        {section === "labelling" && <LabellingSection go={go} />}
        {section === "augmentations" && <AugmentationsSection go={go} />}
        {section === "stats" && <StatsSection go={go} />}
        {section === "settings" && <SettingsSection go={go} />}
        {section === "projects" && <ProjectsSection go={go} />}
        {section === "derived" && <DerivedSection go={go} />}
        {section === "training" && <TrainingSection go={go} />}
        {section === "reference" && <ReferenceSection />}
      </div>
    </main>
  );
}

// Shared helpers

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
      <div className="pk-micro mb-2">
        {eyebrow}
      </div>
      <h1 className="pk-page-title text-2xl md:text-3xl text-[var(--foreground)]">
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
      <h2 className="flex items-center gap-3 text-lg font-semibold tracking-tight">
        <span className="pk-accent-bar" style={{ height: "1.05rem" }} aria-hidden />
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
          className="flex items-start gap-3 rounded-md border border-[var(--line)] bg-[var(--panel)] px-4 py-3"
        >
          <span className="shrink-0 grid place-items-center h-6 w-6 rounded-full bg-[var(--surface-hover)] text-[10px] font-mono text-[var(--foreground)] tabular-nums mt-0.5">
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
    <div className="rounded-md border border-[var(--line)] overflow-hidden">
      <table className="w-full text-sm">
        <tbody className="divide-y divide-[var(--line-soft)]">
          {rows.map(([key, action], i) => (
            <tr key={i}>
              <td className="px-4 py-2.5 w-32">
                <kbd className="inline-block rounded border border-[var(--line)] bg-[var(--panel)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--foreground)]">
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
      ? "border-[var(--line)] border-l-2 border-l-[var(--warn)] bg-[var(--panel)]"
      : "border-[var(--line)] bg-[var(--panel)]";
  return (
    <div className={`rounded-md border px-4 py-3 text-[14px] leading-relaxed ${classes}`}>
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
      className="mt-4 w-full text-left inline-flex items-center justify-between gap-4 rounded-lg border border-[var(--line)] bg-[var(--panel)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] transition-colors px-5 py-4 group"
    >
      <div className="min-w-0">
        <div className="pk-micro mb-1">
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

// LLM prompt and figures

const LLM_PROMPT = `You are a helpful assistant for someone using PixelKit, a free desktop app for building labelled computer-vision datasets and training detectors for the Ohm Lab Neuro N6 board. Answer my questions in plain English. If something is unclear or undocumented in what follows, say so rather than guessing.

==============================================================
WHAT PIXELKIT IS
==============================================================
PixelKit turns images into a labelled, augmented dataset that's ready for training.

The five-step flow:
  1. Create a dataset. Name it and define your labels.
  2. Add images. Upload, import from video, or search Openverse.
  3. Label. One click runs auto-labelling on every unlabelled image.
  4. Augment. Choose realistic variations to expand the dataset.
  5. Export. Download with a train/val split.

==============================================================
DATASETS
==============================================================
When you create a dataset, pick the labels you want PixelKit to find.
Confirming the labels creates the dataset and opens it straight away.

Reference images are optional on every dataset. If your labels are
visually similar or specific to your domain (hare vs rabbit, your
part A vs your part B), add a few example photos of each from the
References tab on the dataset page. They help PixelKit tell the
classes apart. You can add or remove references at any time.

A new dataset's name can be up to 120 characters. You can rename a
dataset at any time, from its settings or by clicking its name at the
top of the dataset page.

==============================================================
ADDING IMAGES
==============================================================
Three ways to add images:

  - Upload from your computer (drag and drop, or click to browse).
  - Import frames from a video. Pick a start and end time and an
    fps. PixelKit extracts the frames as still images.
  - Search Openverse. Type what you're looking for and PixelKit
    shows you a grid of CC-licensed images you can pull in.

The dataset gallery shows a thumbnail for every image. Indicators
on each thumbnail include a sparkle icon (augmentations exist),
an Unsure pill (a detection is borderline), and an Unlabelled or
Background pill.

A filter chip rail above the gallery lets you narrow to All /
Unlabelled / Unrated / Good / Bad / Unsure. Unrated shows the images
with boxes and no verdict yet. A pill with no images is hidden.

==============================================================
LABELLING
==============================================================
The labelling button at the top of the dataset page (Label images,
Label new images and so on) runs auto-labelling across every
unlabelled image.

Auto-labelling uses SAM 3. It needs a Hugging Face token with access
to facebook/sam3, set in the first-run setup or in Settings, and a GPU
unless Compute is set to CPU.

Click-to-detect: switch the viewer into click mode, then click any
object. PixelKit labels it.

Manual mode: a toggle in the viewer header that switches off
click-to-detect and the automatic mask and label for drawn boxes, so a
box stays where you draw it and you type its label.

The image viewer can also delete the current image entirely (a Delete
button in the header, the same as the × on a gallery tile); it then
moves to the next image. The detection count in the header is live, so
it updates as you add or delete boxes. Large 4K images load a fast
preview first, then sharpen to full resolution.

Borderline detections are flagged "Unsure" so you can confirm them
quickly. They appear as an amber pill on the dataset thumbnail, in
the Dataset health counts, and next to the detection in the
viewer's box list. Delete the box, or press its label's key to keep
it. Once an image is edited, Dataset health stops counting its
detections as unsure; the chip in the box list and the pill on the
thumbnail stay.

Review mode: a focused review interface. The arrow keys give each
image a verdict: good, bad or unsure; a swipe right or left marks it
good or bad. The Review button
above the gallery opens the unrated images, or the images of the
active Good, Bad or Unsure pill; its menu picks another set. The
Good, Unsure and Bad pills in the image viewer header also set a
verdict.

Deleting a label removes every detection carrying that label,
including the boxes on augmented copies. Irreversible; export first
if you might want the annotations back. Ctrl + Z brings the label
back without its boxes.

==============================================================
AUGMENTATIONS
==============================================================
Augmentations create realistic variations of your images so the
trained model handles real-world conditions better.

Categories:
  Camera & sensor: simulates real camera artefacts.
  Distortion: perspective, scale, rotation, hue.
  Occlusion: simulates things blocking the view.
  Domain randomisation: varies the background and lighting.

For each enabled category you can pick the strength and how often
it's applied.

You can set how many augmented copies to make per image. Off
creates none and clears the existing ones.

Augmentations are generated when you click Update. They are not
regenerated after labelling or manual edits, so click Update again
after either. Until then, the copies of an image whose boxes changed
are left out of training and exports.

==============================================================
DATASET STATS
==============================================================
PixelKit shows a Health score (0-100) summarising how trainable
your dataset is. Uniqueness (few near-duplicates) counts for 40% of
it; label balance, coverage and confidence count for 20% each. The
Overview tab also shows insight cards for duplicates, images with no
detections and underrepresented labels.

The variation map shows your dataset as dots. Pictures that look
the same to a model sit close together. Pictures that look
different spread out. Near-duplicates are highlighted. Clicking a
dot shows that image in the Dataset gallery.

You can review duplicates in the Dataset health window and keep one
per group.

==============================================================
SETTINGS AND EXPORT
==============================================================
The gear icon at the top of a dataset page opens its settings.

You can rename the dataset, set the target input shape, recolour
labels, pick the cover image, save annotation snapshots and delete
the dataset. Labels are renamed in the Labels row on the dataset page.

Export in YOLO, COCO, Pascal VOC, CVAT, Label Studio or PNG masks.
Choose your train/val split.
The split is reproducible. The same image always lands in the
same set, even across re-exports. CVAT and Label Studio exports
have no split: every image is in one folder and one annotation file.

You can leave out boxes that would be too small or borderline at
your target input shape.

Deleting a dataset is destructive and irreversible. From its
settings or the Workspace you type the dataset name to confirm;
Delete entirely on a Project page deletes it at once.

==============================================================
PROJECTS
==============================================================
A Project (capital P) is a folder that groups several related
datasets, with a cover and an activity timeline. It is different
from a single dataset (which is one labelled image collection).

Create a Project from the New project tile on the Workspace. Add
datasets with "Add existing", which moves one or more datasets into
the Project, or "+ New dataset".

The Project page also has a Labels card: label usage across every
dataset in the Project, with rename. Renaming onto an existing
label merges the two everywhere.

Deleting a dataset from a Project asks whether to remove it from the
Project (it survives as a standalone dataset) or delete it entirely.
Delete entirely deletes it at once, with no typed confirmation.

==============================================================
IMPORTING AN EXISTING DATASET
==============================================================
The Import a labelled dataset step (Import it, on the labels step of
a new dataset) accepts Pascal VOC (Annotations/*.xml), COCO (a JSON
with images and annotations) and YOLO (labels/*.txt with classes.txt
or data.yaml), as a folder or a zip. Every box comes in editable;
polygon masks are kept.

==============================================================
TRAINING FOR THE NEURO N6
==============================================================
The Train for Neuro N6 card on a dataset's overview trains a
YOLOX detector (Nano, or Tiny for more accuracy at about five times
the compute) on this machine (on a GPU, or slowly on the CPU when Compute
is set to CPU in Settings), fine-tunes it with int8
simulated so the deployed model keeps most of its accuracy, and exports a
sketch folder ready to build: the sketch, the int8 model, a labels
header, report.json, and the YOLOX NOTICE.txt and LICENSE.txt. When
ST Edge AI Core is installed the export is
compiled for the board's NPU and the run shows how much runs in
hardware and how much flash it uses. A model the compiler rejects is
kept with the reason instead of exported.

Options: epochs, input size, model size, the train/val split (per
class, so every class is measured; or pick another dataset as the
held-out set), whether generated augmentations join the training
set: only copies of training images do, never copies of val images,
and Start from (continue from an earlier run's weights). A pre-flight
line under the options says what the run will use and warns about
classes with few boxes, classes the val split cannot measure, objects
too small for the input size, and duplicate images across the split.

While it trains the run shows live loss and AP@0.5 curves and the
time left. The batch size comes from free GPU memory and is halved
automatically if the card runs out. Every float epoch is checkpointed: a
run interrupted by a crash or a closed app shows as Interrupted with
a Resume button.

Each finished run has Details: the curves, the confidence threshold
(calibrated on the val split for the best F1 and written into the
labels header), per-class precision and recall at that threshold, a
confusion matrix, a gallery of the val images the model got most
wrong (missed, spurious, wrong class), and every val image beside its
ground truth. Label images runs the exported
model over the dataset's unlabelled images so you can review and
correct instead of drawing from scratch; hand-edited images are never
touched. From a source install, the same run can be started from a
terminal with pixelkit train.

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
crop to one size (N x N) so all derived images match; context
padding and a minimum image size; and whether to
group the crop dataset in a workspace Project.

In "create new labels" mode no boxes, masks or reference images are
copied; each crop is blank and the original parent label shows as a
muted reference while you label. The child re-syncs from the parent
one way, a few seconds after each parent edit or when you click Sync
now, and a crop you delete will not come back. A
parent can have many derived datasets, but you cannot derive from a
derived dataset.

==============================================================
WORKSPACE
==============================================================
The Explorer shows everything in your workspace: Projects (folders
of datasets) as expandable nodes in the tree, with standalone
datasets at the root level. Everything is stored as plain files in
the workspace folder on this machine: a folder named PixelKit in the
home folder unless it is changed under Settings, Workspace. A JPEG
is stored as uploaded; other image formats are stored as JPEG, and
Openverse images keep the format they were downloaded in.

==============================================================
SHORTCUTS
==============================================================
On a Mac, use Cmd where this list says Ctrl.

Image viewer:
  Left / right arrow: previous / next image
  Esc: deselect the box or leave draw mode, then close the viewer
  1-9, Q-P, A-L: relabel the box under the cursor (labels 1 to 28)
  Delete or Backspace: delete the selected box
  B: draw a box
  X: click to detect
  M: edit the mask of the box under the cursor
  Ctrl + wheel: zoom (the wheel alone pans)
  Ctrl + Z: undo the last box edit

Review mode:
  Left arrow: Bad
  Right arrow: Good
  Up or down arrow: Unsure
  Esc: close Review mode

Dataset page:
  Ctrl + Z: undo the last label add, rename or removal, or the last
    dataset rename; a removed label comes back without its boxes

Desktop app:
  Ctrl + 0: reset the window zoom
  Ctrl + plus / minus: zoom the window in / out

==============================================================
LIMITS
==============================================================
PixelKit runs on this machine, so dataset size is bounded by your
disk space and labelling speed by your hardware. An image or a video
can be up to 100 MB, a board recording up to 2 GB.

==============================================================
LICENSING
==============================================================
PixelKit keeps the title, creator and licence of each Openverse
import. An export that includes the image files adds ATTRIBUTIONS.txt,
which credits every Openverse image, its crops and its augmented
copies. Check each licence's terms before you reuse an image. Direct
uploads carry no licence metadata: you are responsible for your right
to use the images, especially for any redistributable model.

A trained model's export includes NOTICE.txt and LICENSE.txt: the
model is a modified YOLOX, licensed under the Apache License 2.0.
Keep both files with the model when you share it.

==============================================================
HOW TO RESPOND
==============================================================
- Answer in plain English and stay grounded in the information above.
- Do not use em-dashes in your responses. Prefer commas, periods, semicolons, or parentheses.
- When I ask a how-to question, give a step-by-step answer that maps to the UI described.
- If I ask about general computer-vision dataset best practice (class balance, augmentation strategy, annotation quality, train/val splits, etc.) you can answer from your own knowledge.
- If I ask about an implementation detail not covered here, such as specific thresholds, how the engine works inside or internal models, say you do not know rather than guessing.

I'll send my actual question in the next message.`;

function LLMPromptBox() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(LLM_PROMPT);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard blocked: select the text for a manual copy.
      const ta = document.getElementById("llm-prompt-ta") as HTMLTextAreaElement | null;
      if (ta) {
        ta.focus();
        ta.select();
      }
    }
  };

  return (
    <div className="rounded-lg border border-[var(--line)] bg-[var(--panel)] overflow-hidden">
      <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-3">
        <div className="min-w-0">
          <div className="pk-micro mb-1">
            Ask any LLM about PixelKit
          </div>
          <h3 className="text-base font-semibold tracking-tight text-[var(--foreground)]">
            Copy this prompt into the LLM of your choice.
          </h3>
          <p className="mt-1 text-[13px] text-[var(--muted)] leading-relaxed">
            It tells the model what PixelKit is and what it can do, then waits for your question.
          </p>
        </div>
        <button
          type="button"
          onClick={copy}
          className={[
            "shrink-0 inline-flex items-center gap-2 rounded-md border px-4 py-2 text-[13px] font-medium transition-colors",
            copied
              ? "border-[var(--line)] bg-[var(--panel)] text-[var(--ok)]"
              : "border-transparent bg-[var(--accent)] text-[var(--accent-contrast)] hover:brightness-105",
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
        className="block w-full max-h-72 resize-none bg-[var(--panel)] border-t border-[var(--line-soft)] px-5 py-4 font-mono text-[12px] leading-relaxed text-[var(--foreground)]/85 focus:outline-none focus:bg-[var(--surface-hover)]"
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
    <figure className="rounded-lg border border-[var(--line)] bg-[var(--panel)] overflow-hidden">
      <div className="p-5 sm:p-6">{children}</div>
      {caption && (
        <figcaption className="border-t border-[var(--line-soft)] px-5 py-2.5 text-[12px] text-[var(--muted)] leading-relaxed">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

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
            <div className="grid place-items-center h-12 w-12 sm:h-14 sm:w-14 rounded-lg border border-[var(--line)] bg-[var(--panel)] text-[var(--foreground)]">
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

function PipelineDiagram() {
  const Box = ({ title, sub }: { title: string; sub: string }) => (
    <div className="flex-1 min-w-0 rounded-md border border-[var(--line-soft)] bg-[var(--panel)] px-3 py-3 text-center">
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
      <Box title="Auto-label" sub="boxes and masks" />
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
      {/* axes */}
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
    { name: "Balance", v: 0.86 },
    { name: "Coverage", v: 0.92 },
    { name: "Confidence", v: 0.81 },
    { name: "Uniqueness", v: 0.78 },
  ];
  return (
    <div className="grid sm:grid-cols-[160px_1fr] gap-5 items-center">
      <div className="grid place-items-center">
        <div className="rounded-full border border-[var(--line)] bg-[var(--panel)] text-[var(--ok)] px-5 py-3 inline-flex items-center gap-3">
          <span className="text-2xl font-mono tabular-nums">
            83<span className="ml-1 text-sm text-[var(--muted)]">/100</span>
          </span>
          <span className="text-[10px] uppercase tracking-[0.18em] font-mono opacity-80">Healthy</span>
        </div>
      </div>
      <div className="grid gap-2">
        {factors.map((f) => (
          <div key={f.name} className="grid grid-cols-[110px_1fr_44px] items-center gap-3">
            <div className="text-[12px] text-[var(--muted)]">
              {f.name}
            </div>
            <div className="h-1.5 rounded-full bg-foreground/[0.08] overflow-hidden">
              <div className="h-full rounded-full bg-foreground/55" style={{ width: `${Math.round(f.v * 100)}%` }} />
            </div>
            <div className="text-[12px] font-mono tabular-nums text-right text-[var(--foreground)]/80">{Math.round(f.v * 100)}%</div>
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
      <rect x="42" y="28" width="20" height="10" rx="5" fill="#5b8def" />
      <text x="46" y="35.5" fontSize="6" fontFamily="ui-sans-serif" fill="#fff" fontWeight="600">cat</text>
    </svg>
  );
}

function ReviewModeMock() {
  return (
    <div className="grid sm:grid-cols-3 gap-3">
      <div className="rounded-md border border-[var(--line-soft)] bg-[var(--panel)] p-3">
        <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--bad)] mb-2">
          ← Bad
        </div>
        <div className="aspect-[4/3] rounded-md bg-foreground/[0.06] grid place-items-center text-[11px] text-[var(--muted)]">
          rejected
        </div>
      </div>
      <div className="rounded-md border border-[var(--line-soft)] bg-[var(--panel)] p-3">
        <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--warn)] mb-2">
          ↓ Unsure
        </div>
        <div className="aspect-[4/3] rounded-md bg-foreground/[0.06] grid place-items-center text-[11px] text-[var(--muted)]">
          parked
        </div>
      </div>
      <div className="rounded-md border border-[var(--line-soft)] bg-[var(--panel)] p-3">
        <div className="text-[11px] uppercase tracking-wider font-mono text-[var(--ok)] mb-2">
          Good →
        </div>
        <div className="aspect-[4/3] rounded-md bg-foreground/[0.06] grid place-items-center text-[11px] text-[var(--muted)]">
          confirmed
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
    { label: "Unsure", count: 3 },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((c) => (
        <span
          key={c.label}
          className={[
            "inline-flex items-center gap-1.5 rounded-md border px-3 py-1 text-[12px]",
            c.active
              ? "border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--foreground)] font-medium"
              : "border-[var(--line)] bg-[var(--panel)] text-[var(--foreground)]/80",
          ].join(" ")}
        >
          {c.label}
          <span className="font-mono tabular-nums opacity-70 text-[11px]">{c.count}</span>
        </span>
      ))}
    </div>
  );
}

// Flow diagram icons

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

// Sections

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
          Most datasets follow the same five steps.
        </p>
        <Figure caption="Create the dataset, add media, label, augment, export.">
          <FlowDiagram />
        </Figure>
        <Bullets
          items={[
            "<strong>Create a dataset</strong>. Name it and define the labels you want to detect.",
            "<strong>Add images</strong>. Drop your own images or videos, or search Openverse from inside the app.",
            "<strong>Label</strong>. One click runs auto-labelling on every unlabelled image.",
            "<strong>Augment</strong>. Configurable rotations, lighting, occlusions, background swaps, and more.",
            "<strong>Export</strong>. Download with a train / val split, segmentations, and an optional filter for boxes too small to detect.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Section guide">
        <p>
          Pick a section. The sub-nav above the title stays on every page.
        </p>
        <ul className="grid gap-3 sm:grid-cols-2">
          {GUIDE_SECTIONS.slice(1).map((s, i) => (
            <li key={s.key}>
              <button
                onClick={() => go(s.key)}
                className="w-full text-left block rounded-lg border border-[var(--line)] bg-[var(--panel)] hover:bg-[var(--surface-hover)] hover:border-[var(--line-strong)] transition-colors px-5 py-4 group h-full"
              >
                <div className="pk-micro pk-num mb-1">
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

function CreateSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 01"
        title="Create a dataset"
        intro={
          <>
            A dataset is a single collection of labelled images plus everything
            attached to it: labels, reference images, augmentations, settings.
            This section walks through creating one and adding optional
            reference images. (A <Strong>Project</Strong>
            {" "}is a separate thing: a folder that groups several
            datasets, covered in the Projects section.)
          </>
        }
      />

      <GuideSection n={1} title="Creating a dataset">
        <p>
          Datasets live on this machine and are only visible to you. From the
          Workspace page, which the Explorer icon in the activity bar opens:
        </p>
        <Steps
          items={[
            ["Click + Add Dataset", "A New dataset dialog opens with a name field."],
            ["Type a name", "Anything you like. You can rename it later from Settings."],
            ["Click Continue", "PixelKit takes you to the labels stage."],
            ["Type the labels you want to detect", "Press Enter, comma, or full stop between each one. Each chip gets a random colour you&rsquo;ll see across the whole dataset."],
            ["Click Done", "PixelKit creates the dataset and opens it."],
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Reference images (optional)">
        <p>
          Reference images are optional. Add examples to improve label
          matching. Each one is a clear photo of the labelled object:
          close-up, well-lit, little background clutter. They matter most
          when your labels are visually similar (hare vs rabbit) or specific
          to your domain; for distinct, common categories auto-labelling
          usually works without them.
        </p>
        <Figure caption="Reference photos help PixelKit pick the right label when classes look similar.">
          <PipelineDiagram />
        </Figure>
        <Steps
          items={[
            ["Open the References tab", "On the dataset page, in the left section nav."],
            ["Drop or click to upload", "Each label has its own section. The section is the label, so PixelKit knows exactly what each reference shows."],
            ["Keep them editable", "Add, remove, or re-label references at any time. Changes are picked up the next time you run auto-labelling."],
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Cancelling a new dataset">
        <p>
          If you cancel before finishing (Cancel or Esc), the half-created
          dataset is removed from the workspace automatically.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Dataset names">
        <Bullets
          items={[
            "A new dataset&rsquo;s name can be up to 120 characters.",
                        "Rename a dataset any time from its settings, or by clicking its name at the top of the dataset page.",
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
            bulk delete, and the indicators on every thumbnail.
          </>
        }
      />

      <GuideSection n={1} title="Adding images">
        <p>
          On the Dataset tab, the <em>Drop images or click to browse</em> card
          is where you add media. Drag and drop images (JPG, PNG, WebP, HEIC,
          AVIF, BMP, TIFF, GIF) or videos (MP4, MOV, WebM, M4V, MKV) in a
          codec the app can play, or click the card to open a file picker.
          The Add images button at the top of the page opens the same picker.
        </p>
        <Bullets
          items={[
                        "Image orientation from EXIF metadata is honoured so iPhone photos don&rsquo;t end up sideways.",
            "Each image gets a tile in the gallery as soon as you drop it, marked processing&hellip; until it has uploaded, and a Processing images card counts them.",
            "Up to 100 MB per video (2 GB per recording from the board), split into per-frame images on this machine before upload.",
            "Already have annotations? Click <em>Import it</em> on the labels step of a new dataset. It takes Pascal VOC, COCO or YOLO (folder or zip); every box comes in editable and polygon masks are kept.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Video trim and frame rate">
        <p>
          Dropping a video opens a small modal before anything is uploaded.
          Drag the in / out handles to trim, set the frames-per-second rate,
          and preview how many frames the import will produce. The video is
          sliced into frames locally and each frame is treated like a fresh
          image.
        </p>
        <Bullets
          items={[
            "<strong>Trim handles</strong> let you crop to the relevant section of the clip so you don&rsquo;t flood the dataset with near-identical frames.",
            "<strong>FPS slider</strong> ranges from half a frame to ten frames per second. Lower FPS produces more varied frames.",
            "<strong>Preview count</strong> updates as you drag so you can dial in roughly how many images the import will create.",
            "<strong>Recordings from the board</strong>. Neuro Studio&rsquo;s Record button saves the Neuro N6 camera&rsquo;s stream as an <em>.mjpeg</em> file. Drop it here like a video: PixelKit splits it into its frames as they are, trimmed in frames and sampled as one frame in N. Pictures from the camera the model will run on are the best training data.",
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Searching Openverse">
        <p>
          Click <em>Import from web</em> on the Dataset tab and PixelKit
          opens an Openverse search. Openverse aggregates openly-licensed
          photos; PixelKit asks only for licences that allow commercial use.
          You can grab matching images directly into your dataset without
          leaving the app.
        </p>
        <Steps
          items={[
            ["Type what you\u2019re looking for", "e.g. &ldquo;potholes&rdquo;, &ldquo;factory pipes&rdquo;, &ldquo;forklifts&rdquo;."],
            ["Click Search", "You get a preview grid of candidate images."],
            ["Click Yes, this is what I\u2019m looking for", "Confirms the search direction."],
            ["Choose how many", "A slider from 10 to 250 images."],
            ["Pull images", "PixelKit fetches them. Click an image to leave it out, then click Add to dataset. <strong>Check the CC licence on each image before reusing it.</strong> Licences vary per image. Some require attribution, some prohibit derivatives."],
          ]}
        />
        <Callout tone="info">
          <Strong>Licensing:</Strong> PixelKit keeps each Openverse
          image&rsquo;s title, creator and licence, and an export that
          includes the images adds ATTRIBUTIONS.txt crediting them. Check a
          licence&rsquo;s terms before you redistribute the image.
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
            "<strong>Box count and label chips (bottom)</strong>. How many objects were found, and the colour of each label.",
            "<strong>Labelling overlay</strong>. Appears briefly while the active label job is processing this image.",
            "<strong>Unlabelled or Background pill (top-left)</strong>. The image has no boxes yet, or is marked as background.",
          ]}
        />
      </GuideSection>

      <GuideSection n={5} title="Verdict filter">
        <p>
          The Filter chip rail above the gallery lets you scope the visible
          tiles by the verdict you&rsquo;ve given them. Counts in each pill
          update live as you tag images in Review mode or as new uploads land.
        </p>
        <Figure caption="Filter pills above the dataset gallery: All, Unlabelled, Unrated, Good, Bad, Unsure. A pill with no images is hidden.">
          <FilterChipsMock />
        </Figure>
        <Bullets
          items={[
            "<strong>All</strong>. Default. Every tile.",
            "<strong>Unlabelled</strong>. Images with nothing detected, other than those marked background.",
            "<strong>Unrated</strong>. Images with boxes that have no verdict yet. Images without boxes, background images included, are not counted.",
            "<strong>Good / Bad / Unsure</strong>. The verdict you gave in Review mode or with the Good, Unsure and Bad pills in the image viewer header.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Bulk select &amp; delete">
        <p>
          The Select button at the top of the gallery puts every tile into
          selection mode. Click any tile to add it to the selection. A bottom
          action bar shows the current count alongside Select all, Mark N
          background, Delete N, and Cancel buttons.
        </p>
        <Callout tone="warn">
          <Strong>Bulk delete is permanent.</Strong> Select all takes the
          images the current filter shows. The selected images and their
          labelled-preview thumbnails are deleted in one go, and their
          augmented copies leave the dataset; the copies&rsquo; files are
          deleted at the next Update.
        </Callout>
      </GuideSection>

      <GuideSection n={7} title="The image viewer">
        <p>
          Click any tile to open the full-screen viewer. The canvas paints
          every detection (boxes, masks, label chips) over the original image.
          Manual editing controls live in the toolbar at the top, covered in
          the next section.
        </p>
        <Figure caption="The viewer draws each detection&rsquo;s box and mask on the image, with a chip naming its label.">
          <ViewerOverlayMock />
        </Figure>
        <Bullets
          items={[
            "Arrow keys cycle through neighbouring images.",
            "Esc closes the viewer and returns to the gallery.",
            "Toolbar pills toggle the box, label, and mask display layers. Click to detect and the Manual mode in the header are covered in the Labelling section.",
            "<strong>Clear all</strong>. Removes every box and mask on the current image in one click, with no confirmation. Useful for restarting from scratch on a difficult shot. Cleared images are treated as unlabelled, so the next labelling run labels them again.",
            "<strong>Background</strong>. Shown instead of Clear all on an image with no boxes: marks it as checked, with nothing in it to label. It then counts as labelled, labelling runs leave it alone, and training learns from it as a negative example, which is what stops a detector firing on an empty scene. Drawing a box on it unmarks it. Select mode marks many at once (<em>Mark N background</em>), and an imported dataset&rsquo;s images whose annotation lists no objects arrive marked.",
            "<strong>Delete image</strong>. Removes the current image from the dataset entirely (the same as the × on its gallery tile) and moves on to the next one, or to the previous one if it was the last. Deleting the only image closes the viewer.",
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
            One-click auto-labelling, the detection settings, manual edits
            in the box editor, fast verdict review with Review mode, and how
            to remove a label entirely.
          </>
        }
      />

      <GuideSection n={1} title="Auto-labelling">
        <p>
          When you click <Strong>Label images</Strong> at the top of the
          dataset page, PixelKit&rsquo;s auto-labelling runs across every
          unlabelled image in your dataset. For each label you defined, you
          get back labelled detections for matching objects.
        </p>
        <p>
          If you&rsquo;ve added reference images, they help PixelKit choose
          the right label when classes look similar. The result is a
          per-detection label assignment with a confidence score. Borderline
          cases are flagged as <em>Unsure</em>.
        </p>
        <Callout tone="info">
          <Strong>Before the first run.</Strong> Auto-labelling,
          click-to-detect and the masks for drawn boxes use SAM 3 on this
          machine. Its weights download once, after you give PixelKit a
          Hugging Face read token for an account that has accepted
          Meta&rsquo;s licence for facebook/sam3, in the first-run setup or
          under Settings, Hugging Face access. Labelling needs a GPU; without
          one it stays off unless Compute is set to CPU in Settings, where it
          takes minutes per image.
        </Callout>
        <Callout tone="info">
          <Strong>Adaptive button.</Strong> The labelling button says what a
          click will do. On a new dataset it reads &ldquo;Label images&rdquo;
          (or &ldquo;Label&rdquo; and the label&rsquo;s name when there is
          only one). Once some images are labelled it reads &ldquo;Label new
          images&rdquo;. After you add labels it names the new label, or reads
          &ldquo;Label new labels&rdquo; for several. After you change the
          detection settings it offers &ldquo;Relabel all&rdquo;, and with
          nothing left to do it reads &ldquo;All images labelled&rdquo;.
        </Callout>
      </GuideSection>

      <GuideSection n={2} title="Watching the job">
        <p>
          A progress card on the Dataset tab tracks the labelling job:
        </p>
        <Bullets
          items={[
            "An ETA based on how long earlier images took.",
            "A live counter of processed images.",
            "A progress bar.",
            "A small Labelling overlay on the matching dataset thumbnail.",
            "A status line that changes every few seconds.",
          ]}
        />
        <p>
          You can keep working on other tabs while it runs; the status bar
          shows the job too. The job completes whether you&rsquo;re looking or
          not. The × on the card cancels it. The card closes a few seconds
          after &ldquo;Labelling complete&rdquo; appears.
        </p>
      </GuideSection>

      <GuideSection n={3} title="Augmentations after labelling">
        <p>
          Labelling does not generate augmentations. After a labelling run,
          click Update on the Augmentations tab so the newly labelled images
          get their augmented copies.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Detection settings">
        <p>
          The Detection settings panel on the Dataset tab holds the
          labelling controls. It opens collapsed. The header shows whether
          the settings are default or custom.
        </p>
        <Bullets
          items={[
            "<strong>Detection confidence</strong>. How sure the model has to be before keeping a box, from 10% to 95%. Lower picks up more candidates; higher trims false positives.",
            "<strong>Mask precision</strong>. How tight each mask outline is. Higher trims softer pixels for cleaner edges.",
            "<strong>Minimum object size</strong>. Drops same-label fragments smaller than this fraction of the largest match. 0 keeps every hit.",
            "<strong>Resolution</strong>. <em>Downscale</em> runs one pass per image. <em>Tile · native</em> slices large images into native-resolution tiles, which catches small objects but is slower.",
            "<strong>Clear all annotations</strong> removes every detection and edited box in the dataset after you type a confirmation. <strong>Default</strong> restores the default settings.",
          ]}
        />
        <Callout tone="info">
          <Strong>Labels.</Strong> Add, rename and remove labels in the Labels
          row under the stats strip: click + to add one, click a label to
          rename it, and hover it for the × that removes it. New detection
          settings apply to the next labelling run; once every image is
          labelled, the labelling button offers Relabel all.
        </Callout>
      </GuideSection>

      <GuideSection n={5} title="Editing labels manually">
        <p>
          Click any tile in the dataset gallery to open the image viewer.
          From there:
        </p>
        <Bullets
          items={[
            "<strong>Move or resize</strong> a box by dragging it or its handles.",
            "<strong>Relabel</strong> by hovering a box and pressing its label&rsquo;s key: 1 to 9, then Q to P and A to L for labels 10 to 28. In a window at least 1280 px wide, the legend in the header shows which key maps to which label.",
            "<strong>Delete</strong> by clicking a box and pressing Delete, or by clicking the × on its row in the right-hand sidebar.",
            "<strong>Draw</strong> a new box with the + Add box tool (or press B). PixelKit segments and labels it for you the moment you release the drag.",
            "<strong>Edit a mask</strong>. Select a box and click Edit mask (or press M over a box), then paint with the brush or eraser.",
            "<strong>Toggle layers</strong>. Boxes, Labels and Masks each have their own pill toggle in the toolbar, useful for decluttering while you work.",
          ]}
        />
        <p>
          Every edit saves automatically, and Ctrl + Z undoes the last box
          edit. Augmentations are not regenerated after an edit: an edited
          image&rsquo;s copies are left out of training and exports until you
          click Update on the Augmentations tab.
        </p>
      </GuideSection>

      <GuideSection n={6} title="Click-to-detect">
        <p>
          The viewer toolbar has a click-to-detect mode (Click to detect, or
          press X). Turn it on, then click an object on the image. PixelKit
          labels what you clicked, in one gesture.
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
            "The Unsure detections count in Dataset health.",
            "An Unsure chip next to the affected detection in the image viewer&rsquo;s box list.",
          ]}
        />
        <p>
          Open the image in the viewer and delete the box, or press its
          label&rsquo;s key to keep it. Once an image is edited, Dataset
          health stops counting its detections as unsure; the Unsure chip in
          the box list and the pill on the tile stay.
        </p>
      </GuideSection>

      <GuideSection n={9} title="Review mode">
        <p>
          Review mode is a fast keyboard-or-swipe interface for giving images
          a verdict: good, bad or unsure, without opening the full viewer.
          The Review button above the gallery opens the unrated images, or the
          images of the active Good, Bad or Unsure pill; its menu picks
          another set. It paints
          the current image with all its detections so you can decide at a
          glance whether the labelling is worth keeping.
        </p>
        <Figure caption="Swipe right or arrow-right to mark the current image good, swipe left or arrow-left to mark it bad, up/down arrow to mark it unsure. Verdicts feed the gallery filter chips.">
          <ReviewModeMock />
        </Figure>
        <Bullets
          items={[
            "<strong>→ Good</strong>. Confirms the labelling on this image.",
            "<strong>← Bad</strong>. Flags the image for follow-up. Doesn&rsquo;t delete anything.",
            "<strong>↑ / ↓ Unsure</strong>. Park for later review.",
            "<strong>Annotations panel</strong>. Lists every detection on the current image with its label and score.",
            "<strong>Hover-highlight</strong>. Hovering a box highlights it and its row in the annotations panel.",
            "<strong>Verdict persists</strong>. Stored on the image, surfaces in the gallery filter chips, and survives reloads.",
          ]}
        />
      </GuideSection>

      <GuideSection n={10} title="Deleting a label entirely">
        <p>
          Removing a label from the Labels row (hover it and click ×) asks
          for confirmation because the action is destructive. Confirming it
          deletes every detection on that label across the whole dataset,
          removes the label&rsquo;s reference data (if it has any), and
          drops the label&rsquo;s boxes from the augmented copies.
        </p>
        <Bullets
          items={[
            "A progress card shows the removal while it runs, like the labelling progress card.",
            "Other detections on affected images are kept. Only the boxes carrying the removed label are touched.",
            "The label disappears from every chip rail and from the colour palette.",
          ]}
        />
        <Callout tone="warn">
          <Strong>Irreversible.</Strong> Once confirmed, every detection on the
          label is gone. Ctrl + Z brings the label back without its boxes.
          Take an export if you might want the annotations back.
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
        <Figure caption="A handful of augmentations applied to the same source image. Real output is photographic; this is a sketch.">
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
            "<strong>Enable checkbox</strong>. Opens the controls. Ticked augmentations are included in every Update. Camera &amp; sensor has one checkbox for its nine dials, and a dial at 0 is off.",
            "<strong>Frequency toggle</strong>. <em>All</em> applies the augmentation to every copy. <em>Random</em> applies to roughly half so the augmented set has more variety.",
            "<strong>Strength dial(s)</strong>. Each augmentation has one or more sliders. Release a slider to update the preview on a sample dataset image.",
            "<strong>Random image button</strong>. Under each preview, swaps to a different source image so you can see the augmentation on a few different shots before committing.",
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Per-image count">
        <p>
          At the top of the Augmentations tab, the <em>Augmentations per
          image</em> control sets how many augmented copies to generate per
          dataset image:
        </p>
        <Bullets
          items={[
            "<strong>Off</strong>. No augmentations. Clicking Update in this mode deletes every existing augmentation (the button reads Clear all).",
            "<strong>1 / 2 / 3</strong>. Fixed copy count per image.",
            "<strong>Random</strong>. PixelKit picks 1, 2 or 3 at random for each image.",
          ]}
        />
      </GuideSection>

      <GuideSection n={5} title="Randomise and Update">
        <p>
          Two buttons in the top-right of the page drive the pipeline.
        </p>
        <Bullets
          items={[
            "<strong>Randomise</strong>. Rolls fresh values for most dials, and a new per-image count, in one click. Lens distortion, lens glare, low resolution, pixelation and background swap keep their settings. Opens every category card so you can see the new values.",
            "<strong>Update</strong>. Generates augmentations for every dataset image based on the current settings. The job runs in the background, with a progress card on the Augmentations tab and on the Dataset tab.",
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
            "Upload up to six backgrounds. PixelKit keeps each as a JPEG at most 1280 px on the long side.",
            "PixelKit picks one of them at random for each generated copy.",
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
            "<strong>Hover a tile</strong> to see the annotations overlay: polygons and bounding boxes coloured by size validity against the dataset&rsquo;s target input shape (green, orange or red).",
            "<strong>Click the × on a tile</strong> to delete that single augmentation. The remaining count updates live and the dataset-thumbnail sparkle hides when the last copy is gone.",
            "<strong>Bounding boxes turn red</strong> when they will be too small to detect at your chosen input shape.",
          ]}
        />
      </GuideSection>

      <GuideSection n={8} title="Keeping augmentations current">
        <p>
          Augmentations are generated when you click Update. They are not
          regenerated automatically, so click Update again:
        </p>
        <Bullets
          items={[
            "After a labelling run, so newly labelled images get their copies.",
            "After manual edits in the image viewer, so the copies carry the corrected annotations. Until then, an edited image&rsquo;s copies are left out of training and exports, and the Train card&rsquo;s pre-flight says how many images need Update.",
          ]}
        />
        <p>
          Update regenerates every copy with the current settings.
        </p>
        <Callout tone="info">
          Update is disabled while a generation job is running for the
          dataset, and refused while a training run on the dataset is
          reading the copies.
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
            Stats appear in three places: the stats strip under the dataset
            name, the Overview tab, and the Dataset health window. Together
            they show counts, label distribution, a 0-100 health score, a
            2-D variation plot, and a review for finding and removing
            near-identical images.
          </>
        }
      />

      <GuideSection n={1} title="The summary row">
        <p>
          The stats strip under the dataset name shows the main numbers on
          every tab. The Overview tab adds insight cards and the full charts.
        </p>
        <Bullets
          items={[
            "<strong>Images</strong>. How many images you&rsquo;ve added to the dataset.",
            "<strong>Detections</strong>. Total boxes across the dataset.",
            "<strong>Labels</strong>. How many labels the dataset has.",
            "<strong>Augmentations</strong>. Total generated copies across all images.",
            "<strong>Health</strong>. The 0-100 score and its rating (Healthy, Fair or Needs work). Click it to open Dataset health.",
            "<strong>Insights</strong> on the Overview tab flag near-duplicates, images with no detections, and an underrepresented label.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Health score">
        <p>
          The health score is a single 0-to-100 number summarising how
          trainable your dataset is. The Overview tab shows it with its four
          factors; Dataset health also explains each factor.
        </p>
        <Figure caption="Healthy is 75 and above, Fair 45 to 74, Needs work below 45. Uniqueness counts for 40% of the score, the other factors for 20% each.">
          <HealthBadgeMock />
        </Figure>
        <p>
          Uniqueness is the share of images that are not near-duplicates,
          balance is how evenly detections spread across labels, coverage is
          the share of images with a detection or marked background, and
          confidence is the share of detections not flagged unsure. Start
          with the lowest factor.
        </p>
      </GuideSection>

      <GuideSection n={3} title="Label distribution">
        <p>
          The Overview tab shows a donut of detections per label, with counts
          for the largest six. Dataset health lists every label as a bar with
          its count and percentage. Useful for spotting imbalances at a
          glance. If one class accounts for 80% of all detections, your model
          will learn it disproportionately.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Image variation plot">
        <p>
          The Overview tab and Dataset health both show a 2-D plot of the
          dataset. Each dot is one image. The position reflects how visually
          similar it is to every other image. Tight clusters mean many
          pictures of nearly the same thing. Wide spread means good variation.
        </p>
        <Figure caption="Each dot is an image, coloured by primary label. An amber halo flags near-duplicates. Hover a dot to see which image it is. Click it to show the image in the Dataset gallery.">
          <VariationPlotMock />
        </Figure>
        <Bullets
          items={[
            "<strong>Dot colour</strong> matches the primary label of the image.",
            "<strong>Dot radius</strong> grows with the number of detections in the image.",
            "<strong>Amber halo</strong> around a dot flags it as a near-duplicate of at least one other image. Consider removing one or the other.",
            "<strong>Smaller, fainter dots</strong> are the augmentations generated from your images.",
            "<strong>Hover a dot</strong> to see the file name, its label, its detection count and whether it is a near-duplicate. Click it to show the image in the Dataset gallery.",
            "<strong>Progress ring</strong>. While image embeddings are still computing, the plot shows how far along they are, then fills in.",
          ]}
        />
      </GuideSection>

      <GuideSection n={5} title="Near-duplicate detection &amp; review">
        <p>
          PixelKit identifies pairs of images that look very similar to a
          model. Both members of each flagged pair get an amber halo on the
          variation plot, and the Insights row on the Overview tab shows how
          many images are near-duplicates.
        </p>
        <p>
          Clicking that card opens Dataset health, where the <Strong>Review duplicates</Strong>
          {" "}panel lists every flagged group. From there you can pick which
          image to keep and act in bulk.
        </p>
        <Bullets
          items={[
            "<strong>Near duplicates / 100% duplicates</strong>. Switch between visually similar images and exact copies.",
            "<strong>Per-group keeper</strong>. Each group shows one image as the keeper (green outline, marked Keep) and the rest as candidates for deletion.",
            "<strong>Per-group actions</strong>. Delete the n duplicates in one click, or mark the group as &ldquo;Not duplicates&rdquo; to ignore it next time.",
            "<strong>Select and delete</strong>. Click non-keeper images across several groups, then delete them all in one go.",
            "<strong>No duplicates to review</strong> shows once every group is resolved.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Live updates">
        <p>
          The stats refresh after every meaningful change to your dataset:
          new uploads, deletions, annotation edits, labelling runs, and
          augmentation changes. No reload required.
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
        title="Settings & export"
        intro={
          <>
            Dataset settings, label colours, annotation snapshots, and
            exporting the finished dataset.
          </>
        }
      />

      <GuideSection n={1} title="Opening Settings">
        <p>
          Click the gear icon at the top of a dataset page to open its
          settings. From there you can rename the dataset, set the target
          input shape, recolour labels, set the cover image, save annotation
          snapshots, and delete the dataset entirely.
        </p>
      </GuideSection>

      <GuideSection n={2} title="Renaming the dataset">
        <p>
          Type the new name and click <em>Rename</em>, or click the dataset
          name at the top of the page, type the new name and press Enter.
        </p>
        <p>
          The new name shows everywhere without a refresh.
        </p>
      </GuideSection>

      <GuideSection n={3} title="Your data stays local">
        <Bullets
          items={[
            "Everything (images, labels, annotations and exports) lives on this machine. Nothing is uploaded to a cloud service.",
            "Deleting a dataset removes its files from disk permanently; there is no remote copy to restore from.",
          ]}
        />
        <p>
          Back up the workspace folder if you want an extra copy of your
          datasets.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Cover image">
        <p>
          Pick which image represents the dataset on the workspace. The
          Cover image section in Settings shows the dataset&rsquo;s images
          and references as a grid. Click any of them to set it as the
          cover, or click Upload image to use a separate picture. Reset
          (visible when a cover is set) reverts to PixelKit&rsquo;s default
          pick.
        </p>
      </GuideSection>

      <GuideSection n={5} title="Renaming and recolouring labels">
        <p>
          The Label colours section in Settings lists every label in the
          dataset with its current colour.
        </p>
        <Bullets
          items={[
            "<strong>Rename</strong> a label by clicking its name in the Labels row on the dataset page. The change shows in the dataset gallery and image viewer at once.",
            "<strong>Recolour</strong> by hovering the colour dot in Settings. Pick one of the eight palette colours, or any colour from the colour picker. Colours are saved with the dataset.",
            "<strong>Reset</strong> the colour to the default palette pick by clicking Reset, shown when a custom colour is set.",
          ]}
        />
      </GuideSection>

      <GuideSection n={6} title="Export modal">
        <p>
          Click the <Strong>Export</Strong> button at the top of the dataset
          page to open the export modal. PixelKit packages your dataset into
          a single download with annotations in your chosen format.
        </p>
        <Bullets
          items={[
            "<strong>Train / val split</strong>. A slider sets the percentage of images to go into the training split; the remainder lands in val. Splits are reproducible across re-exports. CVAT and Label Studio exports have no split: every image is in one folder and one annotation file.",
            "<strong>Boxes / Segmentations</strong>. Toggle which annotation kinds to include. At least one has to be on.",
            "<strong>Filter by detection size</strong>. Leave out red boxes (too small for the model to detect at your target input shape) or orange boxes (borderline).",
            "<strong>Include image files</strong>. Off exports annotations only. On bundles every image and its augmentations in the same archive.",
            "<strong>Formats</strong>. YOLO, COCO, Pascal VOC, CVAT, Label Studio or PNG masks. Pascal VOC has boxes only; PNG masks need segmentations.",
          ]}
        />
      </GuideSection>

      <GuideSection n={7} title="Deleting a dataset">
        <p>
          At the bottom of Settings, Delete this dataset removes the dataset
          permanently. Type the dataset name to confirm before the delete
          button enables. Once gone:
        </p>
        <Bullets
          items={[
            "Every reference image is removed.",
            "Every imported image is removed.",
            "Every annotation, augmentation, and labelled-preview thumbnail is removed.",
            "The dataset folder is removed from the workspace.",
          ]}
        />
        <Callout tone="warn">
          <Strong>No undo.</Strong> Once you confirm the delete, the data is
          gone. Take an export first if you might want it back.
        </Callout>
      </GuideSection>

      <GuideSection n={8} title="Your workspace on disk">
        <p>
          Everything lives in the workspace folder, a folder named PixelKit in
          your home folder unless you change it under Settings, Workspace; the
          status bar shows its name. It holds one folder per Project and one
          per dataset, the images in <code>images/</code> and one annotation
          file per image in <code>annotations/</code>. A JPEG is stored as
          uploaded; other image formats are stored as JPEG, and Openverse
          images keep the format they were downloaded in. The Explorer tree
          mirrors that layout. Back up or sync the folder like any other
          files; the format is documented in the repository&rsquo;s
          docs/WORKSPACE.md.
        </p>
      </GuideSection>

      <GuideSection n={9} title="Annotation snapshots">
        <p>
          The dataset settings panel can save a snapshot of the current
          annotation state (labels, boxes and masks, never the images) and
          restore any earlier one. Restoring first snapshots the current
          state automatically, so it&rsquo;s always undoable.
        </p>
      </GuideSection>

      <NextUp next="projects" go={go} />
    </>
  );
}

function ProjectsSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 07"
        title="Projects"
        intro={
          <>
            A <Strong>Project</Strong> is a folder that groups several
            related datasets together, with a shared cover, an activity
            timeline, and project-wide label tools. It is the layer above
            an individual dataset.
          </>
        }
      />

      <GuideSection n={1} title="Projects vs datasets">
        <p>Two things share similar names:</p>
        <Bullets
          items={[
            "A <strong>dataset</strong> is one collection of labelled images (the thing you build in the Create a dataset, Labelling, and Augmentations sections).",
            "A <strong>Project</strong> is a container that holds many datasets, plus a cover photo and an activity timeline.",
          ]}
        />
        <p>
          On the Workspace page your Projects sit in a row across the top, above
          the grid of your individual datasets.
        </p>
      </GuideSection>

      <GuideSection n={2} title="Creating a Project">
        <Steps
          items={[
            ["Click the New project tile", "It sits at the start of the Projects row on the Workspace."],
            ["Name it and pick a cover", "The cover photo is optional. It shows on the Project hero and card."],
            ["Click Create project", "You land on the Project page, ready to add datasets."],
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Adding datasets">
        <p>On the Project page, the Datasets section has two buttons:</p>
        <Bullets
          items={[
            "<strong>Add existing</strong> moves datasets into the Project: tick one or more.",
            "<strong>+ New dataset</strong> creates a fresh dataset directly inside the Project, using the same naming and labels flow as the Workspace.",
          ]}
        />
        <p>
          A dataset can belong to a Project and still be edited exactly as
          you always edit it.
        </p>
      </GuideSection>

      <GuideSection n={4} title="Labels across the Project">
        <p>
          The Project page&rsquo;s <Strong>Labels</Strong> card lists every
          label used across the Project&rsquo;s datasets with its box and
          dataset counts. Rename a label there and it is rewritten
          everywhere at once: dataset tag lists, every annotation, and the
          colour/alias maps.
        </p>
        <Callout tone="info">
          <Strong>Merging.</Strong> Renaming onto a label that already
          exists merges the two. Use it when datasets spell the same label
          differently, such as &ldquo;bolt&rdquo; and &ldquo;bolts&rdquo;.
        </Callout>
        <Callout tone="info">
          <Strong>Deleting a dataset</Strong> from a Project asks what should
          happen: <em>remove from project</em> detaches it (it survives as a
          standalone dataset), or <em>delete entirely</em> deletes it at
          once, with no typed confirmation and no undo.
        </Callout>
      </GuideSection>

      <GuideSection n={5} title="Cover and activity">
        <Bullets
          items={[
            "<strong>Cover</strong>. Upload a cover photo for the Project hero and card.",
            "<strong>Activity</strong>. The Project page carries a timeline of recent events across all its datasets: uploads, labelling runs, datasets added or removed.",
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
            per detection, each cropped to its box. Use it to train on
            individual objects. The child stays linked to
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
            "<strong>Fixed crop size</strong>. Optional. Turn it on and a slider sets one size: every crop is resized to exactly that N x N, so all derived images match. It forces square crops and overrides the minimum size.",
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
          The child re-derives from the parent&rsquo;s current state on
          creation, a few seconds after each change to the parent, and
          whenever you click <Strong>Sync now</Strong> on the child&rsquo;s
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

      <NextUp next="training" go={go} />
    </>
  );
}

function TrainingSection({ go }: { go: (k: GuideSectionKey) => void }) {
  return (
    <>
      <GuideHeader
        eyebrow="Section 09"
        title="Train for Neuro N6"
        intro={
          <>
            Turn a labelled dataset into a detector that runs on the Ohm Lab
            Neuro N6 board. Training happens on this machine, and the sketch,
            the int8 model and its labels header land in one zip. With ST Edge
            AI Core installed, the run also compiles the model for the
            board&rsquo;s NPU to check that it fits.
          </>
        }
      />

      <GuideSection n={1} title="Starting a run">
        <p>
          Open a dataset&rsquo;s overview and find the{" "}
          <strong>Train for Neuro N6</strong> card. Pick the options, read
          the pre-flight line under them, then Start training. The job runs
          in the background and reports in the status bar; you can keep
          labelling while it trains.
        </p>
        <Bullets
          items={[
            "<strong>Epochs</strong>. 50 is the default; small datasets are usually fine with fewer.",
            "<strong>Input</strong>. The square resolution the model sees (256 by default). Bigger inputs see smaller objects at the cost of speed on the board.",
            "<strong>Model</strong>. Nano (0.9 M parameters, under 1 MB of weights) or Tiny (5 M parameters, about 5 MB, more accurate, about five times the compute).",
            "<strong>Split</strong>. The share of images that train; the rest is held out per class, so every class with two or more images gets a val image and is measured.",
            "<strong>Augmentations</strong>. Generated copies of training images join the training set (with their transformed boxes). Copies of val images are never used, so the val score stays honest.",
            "<strong>Start from</strong>. Continue from an earlier run&rsquo;s weights instead of the COCO checkpoint, for a second pass on a grown dataset.",
            "<strong>Evaluate on</strong>. Score on a separate dataset instead of a held-out split: everything here trains, everything labelled there is scored.",
            "<strong>Pre-flight</strong>. Before you start, the card says how many images and augmentations the run will use and warns about classes with few boxes, classes the val split cannot measure, objects under 8 px at the input size, and identical images on both sides of the split. Too few labelled images blocks the run.",
            "<strong>GPU</strong>. Training needs a CUDA card or Apple Silicon; it runs on the CPU only when Compute is set to CPU in Settings, and then slowly. On a single GPU, training pauses between steps for click-to-detect, drawn boxes and auto-labelling, so you can keep labelling while it trains; the run is a little slower meanwhile. With a second GPU installed, training uses the card labelling is not on.",
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="What the run does">
        <Bullets
          items={[
            "<strong>Float training</strong>. A YOLOX detector, warm-started from COCO or a previous run, trained on the dataset&rsquo;s labelled images and their augmentations. The batch size comes from free GPU memory and is halved automatically if the card runs out.",
            "<strong>Live curves</strong>. The run shows the loss every epoch and AP@0.5 every few epochs as it trains, with the time left. Every float epoch is checkpointed: a run interrupted by a crash, Cancel training or a closed app shows as Interrupted and resumes from its last epoch with the Resume button (an interruption during the int8 fine-tuning restarts that short phase).",
            "<strong>Quantisation-aware fine-tuning</strong>. Eight extra epochs with int8 simulated throughout the network, so the exported int8 model keeps most of the float model&rsquo;s accuracy. The run reports both scores.",
            "<strong>Threshold calibration</strong>. On the int8 model&rsquo;s val predictions the confidence threshold with the best F1 is chosen and written into the labels header, with the precision and recall it gives.",
            "<strong>Export and verify</strong>. The int8 model is written with the board&rsquo;s input contract built in and, when ST Edge AI Core is installed, compiled for the NPU so the run can report how much of it runs in hardware and how much flash it needs.",
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="Reading a run">
        <p>Each finished run in the card shows its summary, and Details opens the rest:</p>
        <Bullets
          items={[
            "<strong>AP50 float and int8</strong>. Accuracy on the held-out split for the float model and for the exact int8 file you deploy, with the per-class numbers in Details.",
            "<strong>The threshold line</strong>. The calibrated confidence threshold and its F1; Details shows per-class precision and recall at that threshold and the confusion matrix (truth by prediction, with a background row and column for spurious boxes and misses).",
            "<strong>Worst images</strong>. The val images with the most errors, drawn: green is the truth, thin blue a correct detection, yellow a miss, red a spurious box, orange the right place with the wrong class. Click one to enlarge it.",
            "<strong>Compare every val image with its ground truth</strong>. Every val image twice, side by side: the truth it was scored against, and what the exported int8 model reports at the shipped threshold, in the same colours, worst first. Arrow keys step through them.",
            "<strong>The NPU line</strong>. <em>Fully on NPU</em> means zero software epochs; otherwise it says how many of the compiler&rsquo;s epochs run on the CPU instead. Weights are shown against the board&rsquo;s flash pool.",
            "<strong>Rejected</strong>. If the compiler refuses the model the run is kept with the reason instead of shipping something that cannot build.",
            "<strong>Label images</strong>. Runs the exported model over the dataset&rsquo;s unlabelled images at its calibrated threshold, so you review and correct instead of drawing from scratch. Hand-edited images are never touched.",
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Putting it on the board">
        <p>
          Click <strong>N6 export</strong> to download the zip, unzip it, open
          the sketch inside in the Arduino IDE and upload it. The IDE needs
          the Neuro N6 board package, 0.1.3 or later (add{" "}
          <span className="font-mono break-all">https://raw.githubusercontent.com/ohmlab-ltd/Neuro-N6-Arduino-Core/main/package_ohmlab_index.json</span>{" "}
          to the Boards Manager URLs and install <strong>Neuro N6</strong>),
          and ST Edge AI Core, which the core runs to compile the model and
          flash it with the sketch; on an older core the build stops with an
          error naming the version it needs. The sketch
          declares the camera, feeds the model at the size it was trained on
          and hands its output to the core&rsquo;s PixelKit post-processor,
          so the boxes appear on Neuro Studio&rsquo;s live video with the
          class names and the calibrated threshold from the labels header.
          Comments in the sketch say what to change to draw the boxes on the
          TFT carrier&rsquo;s display instead. While the sketch runs,
          Neuro Studio shows that threshold as a <strong>confidence</strong>{" "}
          slider: drag it on your own scene, then write the value you settle
          on into the labels header. To stream over WiFi instead, on a board
          with the MAYA-W476 radio, change{" "}
          <span className="font-mono">STREAM_OVER_WIFI</span> to 1 at the top
          of the sketch and upload again: the board hosts its own network.
          With the board on USB, pick{" "}
          <strong>WiFi (the board&rsquo;s network)</strong> in Neuro Studio&rsquo;s
          connection menu and join the network it shows; the Serial Monitor
          prints its name and password too. From a source install,
          where <span className="font-mono">pip install -e engine</span> adds
          the <span className="font-mono">pixelkit</span> command, the same run
          starts from a terminal with{" "}
          <span className="font-mono">pixelkit train &lt;dataset&gt;</span>.
          The local endpoints behind the card are documented in the
          repository&rsquo;s docs/API.md.
        </p>
      </GuideSection>

      <NextUp next="reference" go={go} />
    </>
  );
}


function ReferenceSection() {
  return (
    <>
      <GuideHeader
        eyebrow="Section 10"
        title="Reference"
        intro={
          <>
            Keyboard shortcuts and the other reference tables, kept in one
            place. On a Mac, use Cmd where a table says Ctrl.
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
            ["Esc", "Deselect the box or leave draw mode first, then close the viewer and return to the dataset gallery."],
            ["1-9, Q-P, A-L", "Relabel the box under the cursor: 1 to 9 for the first nine labels, then the Q to P and A to L rows for labels 10 to 28. In a window at least 1280 px wide, the legend in the header shows which key maps to which label."],
            ["Delete", "Delete the currently-selected box (Backspace works too)."],
            ["B", "Draw a box (the Add box tool)."],
            ["X", "Click to detect."],
            ["M", "Edit the mask of the box under the cursor, or of the selected box."],
            ["Ctrl + wheel", "Zoom the canvas. The wheel alone, Alt-drag or a middle-button drag pans it."],
            ["Ctrl + Z", "Undo the last box edit."],
          ]}
        />
      </GuideSection>

      <GuideSection n={2} title="Review mode shortcuts">
        <ShortcutTable
          rows={[
            ["→", "Mark current image Good and advance."],
            ["←", "Mark current image Bad and advance."],
            ["↑ / ↓", "Mark current image Unsure and advance."],
            ["Esc", "Close Review mode and return to the dataset gallery."],
          ]}
        />
      </GuideSection>

      <GuideSection n={3} title="App shortcuts">
        <ShortcutTable
          rows={[
            ["Ctrl + Z", "On a dataset page, undo the last label add, rename or removal, or the last dataset rename. A removed label comes back without its boxes."],
            ["Ctrl + 0", "Reset the window zoom (View menu). Ctrl + plus and Ctrl + minus zoom in and out."],
            ["Esc", "Close the image viewer, Review mode or most dialogs."],
          ]}
        />
      </GuideSection>

      <GuideSection n={4} title="Limits">
        <p>
          PixelKit runs on this machine, so dataset size is bounded by your
          disk space and labelling speed by your hardware. An image or a
          video can be up to 100 MB, a board recording up to 2 GB.
        </p>
      </GuideSection>

      <GuideSection n={5} title="Licensing">
        <p>
          Openverse aggregates openly-licensed images from across the web,
          and the licence varies per image. PixelKit keeps the title,
          creator and licence of each Openverse import, and an export that
          includes the image files adds ATTRIBUTIONS.txt, which credits every
          Openverse image, its crops and its augmented copies. Check each
          licence&rsquo;s terms before you reuse an image.
        </p>
        <p>
          Direct uploads carry no licence metadata. You are responsible for
          your right to use the images, especially in a redistributable
          dataset or model.
        </p>
        <p>
          A trained model&rsquo;s export includes NOTICE.txt and LICENSE.txt:
          the model is a modified YOLOX, licensed under the Apache License
          2.0. Keep both files with the model when you share it.
        </p>
      </GuideSection>

      <GuideSection n={6} title="Getting in touch">
        <p>
          Report bugs and request features on GitHub. In the desktop app,
          Help &gt; Report an Issue&hellip; opens
          github.com/ohmlab-ltd/pixelkit/issues.
        </p>
      </GuideSection>
    </>
  );
}
