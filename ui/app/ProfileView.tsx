"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { signOut, useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Footer } from "./Footer";
import { resizeForUpload } from "@/lib/resize";
import { invalidateUser } from "@/lib/userCache";
import type { PlanId } from "@/lib/plans";
import { CREDIT_COSTS, creditsUsed } from "@/lib/plans";
import { PlanPill, refreshPlan, usePlan, type PlanData } from "./PlanPill";
import { Tooltip } from "./Tooltip";
import { navigateAppTo } from "@/lib/appNav";

type Props = {
  user: { name: string; username?: string; email: string; image?: string | null };
  onJumpWorkspaces: () => void;
  onJumpProjects: () => void;
  /** Operator terminal entry. Only rendered when showTerminal is true (admins). */
  showTerminal?: boolean;
  onJumpTerminal?: () => void;
};

export function ProfileView({ user, onJumpWorkspaces, onJumpProjects, showTerminal = false, onJumpTerminal }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const { update } = useSession();
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Local preview that wins until the session refresh lands; otherwise the
  // avatar would briefly snap back to the old image after upload.
  const [localImage, setLocalImage] = useState<string | null>(null);
  const usage = usePlan();

  // Webhook-independent subscription reconciliation. Fires the
  // sync POST exactly once on mount and INTENTIONALLY does NOT call
  // session.update() or refreshPlan() afterwards. Both of those
  // triggered re-renders that were running the effect on a loop ,
  // a fresh `update` ref from useSession plus an invalidated plan
  // cache from refreshPlan() caused the profile page to flicker
  // every few hundred ms and never settle.
  //
  // Net behaviour: subscription state lands in the DB on this mount,
  // but the UI only picks it up on the NEXT page load / navigation.
  // Trade-off accepted to stop the flicker. A user-clickable "Sync"
  // affordance can be added later if eager refresh is needed.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        await fetch("/api/billing/sync-subscription", { method: "POST" });
        if (cancelled) return;
      } catch {
        /* network blip, leave previous state alone */
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const onPick = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const resized = await resizeForUpload(file);
      const fd = new FormData();
      fd.append("file", resized);
      const r = await fetch("/api/users/avatar", { method: "POST", body: fd });
      const text = await r.text();
      let data: { image?: string; error?: string } = {};
      try { data = JSON.parse(text); } catch { /* surface raw body */ }
      if (!r.ok) {
        console.error("[avatar] upload failed", r.status, text);
        setError(data.error || `Upload failed (${r.status}). Check browser console / backend log.`);
        return;
      }
      if (!data.image) {
        setError("Upload succeeded but no image URL returned.");
        return;
      }
      setLocalImage(data.image);
      // Drop our own entry from the lookup cache so the new avatar appears
      // immediately on Projects cards instead of waiting out the TTL.
      if (user.username) invalidateUser(user.username);
      await update();
    } catch (e) {
      console.error("[avatar] threw", e);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const displayImage = localImage ?? user.image ?? null;
  const initials = user.name
    .split(/\s+/)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .filter(Boolean)
    .slice(0, 2)
    .join("") || (user.email[0]?.toUpperCase() ?? "?");

  return (
    <main className="min-h-screen bg-[var(--background)]">
      <section className="mx-auto max-w-6xl px-6 pt-12 pb-8">
        <div className="pk-card relative rounded-2xl overflow-hidden">
          {/* Ambient hero background: the user's avatar, blown up and heavily
              blurred so it reads as a soft colour + gradient wash. A theme-aware
              surface scrim sits on top so every bit of content above stays
              readable regardless of how light or dark the photo is. */}
          <div aria-hidden className="pointer-events-none absolute inset-0">
            {displayImage ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={displayImage}
                alt=""
                referrerPolicy="no-referrer"
                className="h-full w-full scale-150 object-cover blur-3xl opacity-70"
              />
            ) : (
              <div
                className="h-full w-full opacity-60"
                style={{ backgroundImage: "linear-gradient(135deg, #6366f1 0%, #ec4899 60%, #f59e0b 100%)", filter: "blur(48px)" }}
              />
            )}
            {/* Surface scrim: keeps the avatar colour bleeding through while
                guaranteeing the foreground text reads in both themes. */}
            <div className="absolute inset-0 bg-[var(--surface)]/62" />
          </div>
          <div className="relative z-10 px-6 sm:px-8 py-7">
            <div className="flex flex-wrap items-center gap-5">
              <div className="relative group">
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  disabled={uploading}
                  aria-label="Change profile picture"
                  className="block h-24 w-24 sm:h-28 sm:w-28 rounded-full overflow-hidden shadow-xl outline-none focus-visible:ring-2 focus-visible:ring-white"
                >
                  {displayImage ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={displayImage}
                      alt={user.name}
                      className="h-full w-full object-cover"
                      referrerPolicy="no-referrer"
                    />
                  ) : (
                    <span
                      className="h-full w-full grid place-items-center text-4xl font-semibold tracking-wide text-[var(--foreground)]"
                      style={{ backgroundImage: "linear-gradient(135deg, #6366f1 0%, #ec4899 60%, #f59e0b 100%)" }}
                    >
                      {initials}
                    </span>
                  )}
                  <span className="pointer-events-none absolute inset-0 rounded-full bg-black/60 text-white text-[10px] uppercase tracking-wider grid place-items-center opacity-0 group-hover:opacity-100 transition-opacity">
                    {uploading ? "Uploading…" : "Change photo"}
                  </span>
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onPick(f);
                    if (fileRef.current) fileRef.current.value = "";
                  }}
                />
              </div>

              <div className="flex-1 min-w-0 pb-1">
                <div className="flex items-end justify-between gap-4 flex-wrap">
                  <div className="min-w-0">
                    <div className="flex items-center gap-4 flex-wrap">
                      <h1 className="text-3xl md:text-4xl font-semibold tracking-tight truncate">{user.name}</h1>
                      <PlanPill
                        plan={usage?.plan ?? "free"}
                        planName={usage?.planName ?? "Free"}
                        size="lg"
                      />
                    </div>
                    <div className="flex items-baseline gap-3 flex-wrap mt-1.5">
                      {user.username && (
                        <span className="font-mono text-[var(--muted)] text-sm">@{user.username}</span>
                      )}
                      {user.email && (
                        <span className="text-[var(--muted)] text-sm truncate">{user.email}</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {usage?.plan === "free" && (
                      <button
                        type="button"
                        onClick={() => navigateAppTo("pricing")}
                        className="rounded-full bg-[var(--accent-orange)] text-black px-4 py-2 text-xs font-semibold tracking-wide hover:opacity-90 transition-opacity"
                      >
                        Upgrade
                      </button>
                    )}
                    <button
                      onClick={() => signOut({ callbackUrl: "/" })}
                      className="rounded-full border border-foreground/15 hover:border-foreground/30 hover:bg-foreground/[0.04] px-4 py-2 text-sm transition-colors"
                    >
                      Sign out
                    </button>
                  </div>
                </div>
              </div>
            </div>
            {error && <div className="mt-3 text-xs text-red-400">{error}</div>}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pt-2 pb-6 grid gap-8">
        <BillingWarning usage={usage} />
        <SubscriptionEndingNotice usage={usage} />
        <UsagePanel usage={usage} />
      </section>

      <section className="mx-auto max-w-6xl px-6 pt-2 pb-24">
        <div className="mb-4 flex items-center gap-2.5">
          <span className="pk-accent-bar" aria-hidden />
          <h2 className="pk-section-title text-xl">Jump back in</h2>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <button
            onClick={onJumpWorkspaces}
            className="group text-left pk-card pk-card-hover rounded-2xl p-6 flex items-start gap-4"
          >
            <span className="grid h-11 w-11 place-items-center rounded-xl bg-orange-500/12 text-orange-600 dark:text-orange-400 shrink-0">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z" />
              </svg>
            </span>
            <div className="grid gap-1 min-w-0">
              <div className="text-xs text-[var(--muted)] uppercase tracking-wider">Your kit</div>
              <div className="text-xl font-semibold tracking-tight group-hover:text-[var(--foreground)]">Workspaces &rarr;</div>
              <div className="text-sm text-[var(--muted)]">Private datasets you&apos;re labelling and training on.</div>
            </div>
          </button>
          <button
            onClick={onJumpProjects}
            className="group text-left pk-card pk-card-hover rounded-2xl p-6 flex items-start gap-4"
          >
            <span className="grid h-11 w-11 place-items-center rounded-xl bg-amber-500/12 text-amber-600 dark:text-amber-400 shrink-0">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <circle cx="12" cy="12" r="9" />
                <path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" />
              </svg>
            </span>
            <div className="grid gap-1 min-w-0">
              <div className="text-xs text-[var(--muted)] uppercase tracking-wider">Community</div>
              <div className="text-xl font-semibold tracking-tight group-hover:text-[var(--foreground)]">Community &rarr;</div>
              <div className="text-sm text-[var(--muted)]">Public datasets shared by other PixelKit users.</div>
            </div>
          </button>
        </div>
        {/* Operator terminal entry, admins (@hamish / @mukund) only. Moved here
            from the top bar so it's tucked away on the profile page. */}
        {showTerminal && onJumpTerminal && (
          <button
            onClick={onJumpTerminal}
            className="group mt-4 w-full text-left pk-card pk-card-hover rounded-2xl p-6 flex items-start gap-4"
          >
            <span className="grid h-11 w-11 place-items-center rounded-xl bg-foreground/[0.06] text-[var(--foreground)] shrink-0">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <polyline points="4 17 10 11 4 5" />
                <line x1="12" y1="19" x2="20" y2="19" />
              </svg>
            </span>
            <div className="grid gap-1 min-w-0">
              <div className="text-xs text-[var(--muted)] uppercase tracking-wider">Admin</div>
              <div className="text-xl font-semibold tracking-tight group-hover:text-[var(--foreground)]">Terminal &rarr;</div>
              <div className="text-sm text-[var(--muted)]">Operator terminal. Visible to admins only.</div>
            </div>
          </button>
        )}
      </section>

      <section className="mx-auto max-w-6xl px-6 pt-2 pb-24">
        <SettingsPanel
          plan={usage?.plan ?? "free"}
          subscriptionStatus={usage?.subscriptionStatus ?? null}
          betaExpiresAt={usage?.betaExpiresAt ?? null}
        />
      </section>

      <Footer />
    </main>
  );
}

// Stripe statuses that indicate the user *intended* to be on a paid
// plan but billing is currently broken or paused. We only nag people
// who were paying, `null` (never subscribed) doesn't trigger.
const NEEDS_ATTENTION_STATUSES = new Set([
  "past_due",
  "unpaid",
  "incomplete",
  "incomplete_expired",
  "canceled",
  "paused",
]);

function billingWarningCopy(status: string): { title: string; body: string } {
  switch (status) {
    case "past_due":
    case "unpaid":
      return {
        title: "Payment failed",
        body: "Your last payment didn't go through. Update your card to keep your Pro plan active.",
      };
    case "incomplete":
      return {
        title: "Finish your payment",
        body: "Your subscription is waiting on a successful first payment. Update your details to activate Pro.",
      };
    case "incomplete_expired":
      return {
        title: "Payment expired",
        body: "Your initial payment never completed. Restart checkout to activate Pro.",
      };
    case "canceled":
      return {
        title: "Subscription cancelled",
        body: "Your Pro subscription was cancelled. Restart it any time to lift the free-tier limits.",
      };
    case "paused":
      return {
        title: "Subscription paused",
        body: "Your Pro subscription is paused. Resume it to lift the free-tier limits.",
      };
    default:
      return {
        title: "Subscription needs attention",
        body: "Your Pro subscription isn't active. Update your payment method to continue.",
      };
  }
}

// Shown when a user has cancelled their Pro subscription via the
// portal but the period they've already paid for hasn't expired yet
// (Stripe `cancel_at_period_end`). Distinct from BillingWarning,
// which fires only when status is unhealthy. This banner is purely
// informational, they keep Pro features until the date, plus a
// Reinstate button that flips `cancel_at_period_end` back to false.
// Stripe keeps the existing period intact, so reinstating doesn't
// trigger an early charge.
function SubscriptionEndingNotice({ usage }: { usage: PlanData | null }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!usage) return null;
  if (!usage.subscriptionCancelAt) return null;
  if (usage.plan === "free") return null;

  const end = new Date(usage.subscriptionCancelAt);
  const dateLabel = end.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
  const planLabel = usage.planName;

  const reinstate = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/billing/reinstate", { method: "POST" });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        // Subscription already ended in Stripe → kick them through a
        // fresh checkout instead of leaving them on a dead-end error.
        if (body?.code === "subscription_ended" || body?.code === "no_subscription") {
          navigateAppTo("pricing");
          return;
        }
        throw new Error(body?.error || `http ${r.status}`);
      }
      // Refresh the page so /api/users/usage re-runs and the banner
      // disappears. Same hard-reload pattern we use for settings
      // saves to keep the JWT + UI consistent.
      window.location.href = "/app?profile=1";
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-amber-500/30 bg-amber-500/[0.06] p-5">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="min-w-0">
          <div className="text-xs uppercase tracking-wider text-amber-700 dark:text-amber-200/85 font-mono">Subscription</div>
          <h2 className="mt-0.5 text-lg font-medium tracking-tight text-[var(--foreground)]">
            {planLabel} expiring {dateLabel}
          </h2>
        </div>
        <button
          type="button"
          onClick={reinstate}
          disabled={busy}
          className="rounded-full border border-amber-300/50 dark:border-amber-300/30 bg-amber-300/[0.12] dark:bg-amber-300/[0.06] hover:border-amber-400/70 dark:hover:border-amber-300/55 hover:bg-amber-300/[0.2] dark:hover:bg-amber-300/[0.10] text-amber-800 dark:text-amber-100 px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {busy ? "Reinstating…" : "Reinstate"}
        </button>
      </div>

      <p className="text-sm text-foreground/70 leading-relaxed">
        Your subscription is set to cancel. You&rsquo;ll keep {planLabel} features until <span className="text-[var(--foreground)]">{dateLabel}</span>, then drop back to the Free plan.
      </p>

      <p className="mt-3 text-[12px] text-foreground/45 leading-relaxed">
        Reinstate to keep going, you won&rsquo;t be charged again until {dateLabel}.
      </p>

      {error && <p className="mt-3 text-[11px] text-rose-700 dark:text-rose-300/90">{error}</p>}
    </div>
  );
}

function BillingWarning({ usage }: { usage: PlanData | null }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!usage) return null;
  const status = usage.subscriptionStatus;
  if (!status || !NEEDS_ATTENTION_STATUSES.has(status)) return null;

  const { title, body } = billingWarningCopy(status);

  const openPortal = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/billing/portal", { method: "POST" });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        throw new Error(data.error || `http ${r.status}`);
      }
      const data = await r.json();
      if (data.url) {
        window.location.href = data.url;
        return;
      }
      throw new Error("no portal url");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-amber-300/25 bg-amber-300/[0.04] p-4 sm:p-5 flex flex-wrap items-start gap-4 justify-between">
      <div className="flex items-start gap-3 min-w-0">
        <span aria-hidden className="mt-1 h-1.5 w-1.5 rounded-full bg-amber-300/80 shrink-0" />
        <div className="min-w-0">
          <div className="text-xs uppercase tracking-wider text-amber-700 dark:text-amber-200/85 font-mono">
            {title}
          </div>
          <p className="mt-1 text-sm text-foreground/75 leading-relaxed">{body}</p>
          {error && (
            <p className="mt-1.5 text-[11px] text-rose-700 dark:text-rose-300/90">{error}</p>
          )}
        </div>
      </div>
      <button
        type="button"
        onClick={openPortal}
        disabled={busy}
        className={[
          "rounded-full border border-amber-300/50 dark:border-amber-300/30 bg-amber-300/[0.12] dark:bg-amber-300/[0.06] hover:border-amber-400/70 dark:hover:border-amber-300/55 hover:bg-amber-300/[0.2] dark:hover:bg-amber-300/[0.10] text-amber-800 dark:text-amber-100 px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors shrink-0",
          busy ? "opacity-60 cursor-not-allowed" : "",
        ].join(" ")}
      >
        {busy ? "Opening…" : "Update payment"}
      </button>
    </div>
  );
}

function UsagePanel({ usage }: { usage: PlanData | null }) {
  const limits = usage?.limits;
  const u = usage?.usage;
  const labelled = u?.imagesLabelledThisPeriod ?? 0;
  const uploaded = u?.imagesUploadedThisPeriod ?? 0;
  const stored = u?.imagesStoredNow ?? 0;
  const projects = u?.projects ?? 0;
  const projectLimit = limits?.projects ?? 3;
  const creditsTotal = limits?.creditsPerMonth ?? 0;
  // creditsUsed lives in lib/plans so the gate + display share the
  // same arithmetic. Round to one decimal for legibility, the
  // underlying counters are integers so partial credits only come
  // from storage division.
  const creditsConsumed = usage
    ? creditsUsed({
        labelledThisMonth: labelled,
        uploadedThisMonth: uploaded,
        storedNow: stored,
      })
    : 0;
  const creditsExceeded = creditsTotal > 0 && creditsConsumed >= creditsTotal;
  const projectExceeded = projects >= projectLimit;

  return (
    <div className="rounded-2xl border border-foreground/10 bg-[var(--surface)] shadow-[var(--shadow-soft)] p-5">
      <div className="flex items-center justify-between mb-4 gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-foreground/45">Usage this period</div>
          <h2 className="text-lg font-medium tracking-tight mt-0.5">
            {usage ? `${usage.planName} plan` : "Loading…"}
          </h2>
          {usage && (
            <p className="mt-1 text-[11px] text-foreground/45">
              {usage.subscriptionCancelAt && usage.plan !== "free"
                ? `Subscription ends ${formatResetDate(usage.subscriptionCancelAt)}`
                : `Resets ${formatResetDate(usage.cycle.end)}`}
            </p>
          )}
        </div>
        {usage?.plan === "free" && (creditsExceeded || projectExceeded) && (
          <button
            type="button"
            onClick={() => navigateAppTo("pricing")}
            className="rounded-full border border-foreground/15 bg-foreground/[0.04] hover:border-foreground/30 hover:bg-foreground/[0.08] text-foreground/80 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors"
          >
            Upgrade for more
          </button>
        )}
      </div>

      {/* Credits headline: a single bar with the spent / total ratio
          across all three counters (labelling, upload, storage). Sets
          the scale users actually pay against; the rows below break
          out the individual counters. */}
      <CreditsHeadline
        consumed={creditsConsumed}
        total={creditsTotal}
        ready={!!usage}
      />

      <div className="mt-5 grid sm:grid-cols-2 gap-x-5 gap-y-3">
        <UsageRow
          label="Auto-labelled images"
          value={labelled}
          ready={!!usage}
          hint={`${CREDIT_COSTS.labelledImagesPerCredit} images / credit`}
        />
        <UsageRow
          label="Uploads this period"
          value={uploaded}
          ready={!!usage}
          hint={`${CREDIT_COSTS.uploadedImagesPerCredit} images / credit`}
        />
        <UsageRow
          label="Images stored"
          value={stored}
          ready={!!usage}
          hint={`${CREDIT_COSTS.storedImagesPerCreditPerMonth} stored / credit / month`}
        />
        <UsageRow
          label="Projects"
          value={projects}
          limit={projectLimit}
          ready={!!usage}
          exceeded={projectExceeded}
        />
      </div>

      {usage && !usage.backendUp && (
        <p className="mt-3 text-[11px] text-amber-300/90">
          Couldn&rsquo;t reach the labelling backend, usage figures may be stale.
        </p>
      )}
    </div>
  );
}

function CreditsHeadline({
  consumed,
  total,
  ready,
}: {
  consumed: number;
  total: number;
  ready: boolean;
}) {
  const pct = total > 0 ? Math.min(100, (consumed / total) * 100) : 0;
  const exceeded = total > 0 && consumed >= total;
  const colour = exceeded
    ? "bg-rose-400/80"
    : pct > 80
    ? "bg-amber-300/80"
    : "bg-foreground/55";
  // Round only for display so a half-credit still surfaces. Hide
  // decimals once the number is over 10 so the readout stays compact.
  const consumedDisplay = consumed >= 10 ? Math.round(consumed) : Math.round(consumed * 10) / 10;
  return (
    <div className="grid gap-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-foreground/75">Credits used</span>
        <span className={["tabular-nums font-mono text-xs", exceeded ? "text-rose-300" : "text-foreground/70"].join(" ")}>
          {ready
            ? total > 0
              ? `${consumedDisplay.toLocaleString()} / ${total.toLocaleString()}`
              : consumedDisplay.toLocaleString()
            : ","}
        </span>
      </div>
      <div className="h-2 rounded-full bg-foreground/[0.06] overflow-hidden">
        <div
          className={["h-full transition-all duration-500", colour].join(" ")}
          style={{ width: `${pct}%` }}
        />
      </div>
      {exceeded && (
        <p className="text-[11px] text-rose-300/90">Credit limit reached, upgrade to keep going.</p>
      )}
    </div>
  );
}

function UsageRow({
  label,
  value,
  limit,
  ready,
  exceeded,
  hint,
}: {
  label: string;
  value: number;
  limit?: number;
  ready: boolean;
  exceeded?: boolean;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5 border-b border-foreground/[0.06] last:border-b-0">
      <div className="min-w-0">
        <div className="text-sm text-foreground/75">{label}</div>
        {hint && <div className="text-[10px] text-foreground/40 mt-0.5">{hint}</div>}
      </div>
      <span
        className={[
          "tabular-nums font-mono text-sm shrink-0",
          exceeded ? "text-rose-300" : "text-foreground/85",
        ].join(" ")}
      >
        {ready
          ? limit != null
            ? `${value.toLocaleString()} / ${limit.toLocaleString()}`
            : value.toLocaleString()
          : ","}
      </span>
    </div>
  );
}

// "3 May 2026" or "Tomorrow" if reset is within 24h, "today" if even
// closer. Reads the supplied ISO end-of-cycle string.
function formatResetDate(iso: string): string {
  try {
    const end = new Date(iso);
    const now = new Date();
    const ms = end.getTime() - now.getTime();
    const oneDay = 24 * 60 * 60 * 1000;
    if (ms <= 0) return "today";
    if (ms < oneDay) return "tomorrow";
    return `on ${end.toLocaleDateString("en-GB", {
      day: "numeric",
      month: "long",
      year: "numeric",
    })}`;
  } catch {
    return iso;
  }
}

// ---------- account settings ----------

type AccountSnapshot = {
  name: string | null;
  username: string | null;
  email: string | null;
  shareTrainingData: boolean;
  hasPassword: boolean;
  linkedAccounts: { provider: string; hint: string }[];
};

type PendingEmailChange = {
  pendingEmail: string;
  expiresAt: string;
  canResendInMs: number;
};

function SettingsPanel({
  plan,
  subscriptionStatus,
  betaExpiresAt,
}: {
  plan: PlanId;
  subscriptionStatus: string | null;
  betaExpiresAt: string | null;
}) {
  const { update: refreshSession } = useSession();
  const router = useRouter();
  const [account, setAccount] = useState<AccountSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<"name" | "username" | "email" | "password" | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pendingEmail, setPendingEmail] = useState<PendingEmailChange | null>(null);
  const [reloading, setReloading] = useState(false);

  const reload = useCallback(async () => {
    try {
      const r = await fetch("/api/users/me", { cache: "no-store" });
      if (r.ok) setAccount((await r.json()) as AccountSnapshot);
    } finally {
      setLoading(false);
    }
  }, []);

  const reloadPendingEmail = useCallback(async () => {
    try {
      const r = await fetch("/api/auth/verify-status", { cache: "no-store" });
      if (!r.ok) return;
      const body = (await r.json()) as
        | { pending: false }
        | { pending: true; pendingEmail: string; expiresAt: string; canResendInMs: number };
      if (body.pending) {
        setPendingEmail({
          pendingEmail: body.pendingEmail,
          expiresAt: body.expiresAt,
          canResendInMs: body.canResendInMs,
        });
      } else {
        setPendingEmail(null);
      }
    } catch {
      /* leave previous state alone */
    }
  }, []);

  useEffect(() => {
    void reload();
    void reloadPendingEmail();
  }, [reload, reloadPendingEmail]);

  const onSaved = async () => {
    setEditing(null);
    await reload();
    // Show a frosted blur over the page BEFORE we navigate so the
    // user sees a smooth dim-and-blur instead of a white flash when
    // the hard reload happens. The overlay survives until the new
    // page mounts because it lives in the current document.
    setReloading(true);
    // next-auth v5 beta's update() is unreliable about re-issuing the
    // JWT cookie, it sometimes resolves before the Set-Cookie lands,
    // or skips the jwt() refresh entirely. The only deterministic
    // way to push the new username/name/email through every consumer
    // (TopNav, plan pill, project owner queries, new-project owner
    // string, etc.) is a hard navigation. We send the user back to
    // /app?profile=1 so /app/page.tsx re-opens the profile pane on
    // mount instead of dropping them on Workspace.
    await refreshSession({});
    router.refresh();
    // Two RAFs so the overlay's opacity transition has actually
    // started painting before the navigation tears down the page.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        window.location.href = "/app?profile=1";
      });
    });
  };

  const toggleTrainingShare = async (next: boolean) => {
    if (!account) return;
    setAccount({ ...account, shareTrainingData: next });
    try {
      await fetch("/api/users/me", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shareTrainingData: next }),
      });
    } catch {
      // Roll back the optimistic flip if the request fails.
      setAccount((prev) => (prev ? { ...prev, shareTrainingData: !next } : prev));
    }
  };

  return (
    <div className="grid gap-3">
      {reloading && <ReloadOverlay />}
      <h2 className="text-xs uppercase tracking-wider text-foreground/45 mt-2">Settings</h2>

      <div className="rounded-2xl border border-foreground/10 bg-[var(--surface)] shadow-[var(--shadow-soft)] divide-y divide-foreground/[0.06]">
        <SettingsRow
          label="Display name"
          value={account?.name || ","}
          onEdit={() => setEditing("name")}
          loading={loading}
        />
        <SettingsRow
          label="Username"
          value={account?.username ? `@${account.username}` : ","}
          mono
          onEdit={() => setEditing("username")}
          loading={loading}
        />
        <SettingsRow
          label="Email"
          value={account?.email || ","}
          onEdit={() => setEditing("email")}
          loading={loading}
          subline={
            pendingEmail ? (
              <PendingEmailNotice
                pendingEmail={pendingEmail}
                onCancelled={async () => {
                  setPendingEmail(null);
                  await reloadPendingEmail();
                }}
              />
            ) : null
          }
        />
        <SettingsRow
          label="Password"
          value={account?.hasPassword ? "••••••••" : "Not set"}
          mono
          onEdit={() => setEditing("password")}
          loading={loading}
          editLabel={account?.hasPassword ? "Change" : "Set password"}
        />
        {(account?.linkedAccounts ?? []).map((a) => (
          <LinkedAccountRow
            key={a.provider}
            provider={a.provider}
            hint={a.hint}
            // Allow unlinking when there's another way back in:
            // either a password is set OR another provider is linked.
            canUnlink={
              !!account?.hasPassword ||
              (account?.linkedAccounts.length ?? 0) > 1
            }
            onUnlinked={async () => {
              await reload();
            }}
          />
        ))}
        <ToggleRow
          label="Share training data"
          description="Help improve Pixel Kit's auto-labelling by sharing selected images, labels, predictions, crops, embeddings and metadata."
          hint={
            <>
              By enabling, you agree to our{" "}
              <a
                href="/model-improvement"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground/60 hover:text-foreground underline underline-offset-2"
              >
                Model Improvement policy
              </a>
              .
            </>
          }
          checked={!!account?.shareTrainingData}
          onChange={toggleTrainingShare}
          disabled={!account}
        />
        <BetaCodeRow plan={plan} betaExpiresAt={betaExpiresAt} />
        <BillingRow plan={plan} subscriptionStatus={subscriptionStatus} />
      </div>

      {/* Danger-zone card. Light mode uses deeper rose tones so
          the destructive copy + CTA stay legible against a near-
          white profile surface; dark mode keeps the existing
          soft-rose accent. */}
      <div className="rounded-2xl border border-rose-500/30 bg-rose-500/[0.05] dark:border-rose-300/15 dark:bg-rose-300/[0.02] p-5 flex flex-wrap items-start gap-4 justify-between">
        <div className="min-w-0">
          <div className="text-xs uppercase tracking-wider text-rose-800 dark:text-rose-200/85 font-mono">
            Danger zone
          </div>
          <p className="mt-1 text-sm text-rose-900/80 dark:text-foreground/70 leading-relaxed">
            Permanently delete your account, every workspace project you own, and all uploaded images. This cannot be undone.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setConfirmDelete(true)}
          className="rounded-full border border-rose-600/45 bg-rose-500/[0.12] hover:border-rose-600/70 hover:bg-rose-500/[0.22] text-rose-800 dark:border-rose-300/30 dark:bg-rose-300/[0.06] dark:hover:border-rose-300/55 dark:hover:bg-rose-300/[0.12] dark:text-rose-100 px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors shrink-0"
        >
          Delete account
        </button>
      </div>

      {editing === "name" && account && (
        <FieldEditor
          title="Display name"
          field="name"
          initial={account.name ?? ""}
          placeholder="Your name"
          onClose={() => setEditing(null)}
          onSaved={onSaved}
        />
      )}
      {editing === "username" && account && (
        <FieldEditor
          title="Username"
          field="username"
          initial={account.username ?? ""}
          placeholder="3–32 chars · letters / digits / _ -"
          mono
          lowercase
          onClose={() => setEditing(null)}
          onSaved={onSaved}
        />
      )}
      {editing === "email" && account && (
        <FieldEditor
          title="Email"
          field="email"
          initial={account.email ?? ""}
          placeholder="you@example.com"
          inputType="email"
          lowercase
          requireCurrentPassword={!!account.hasPassword}
          onClose={() => setEditing(null)}
          onSaved={onSaved}
        />
      )}
      {editing === "password" && (
        <PasswordEditor
          hasPassword={!!account?.hasPassword}
          onClose={() => setEditing(null)}
          onSaved={async () => {
            setEditing(null);
            await reload();
          }}
        />
      )}
      {confirmDelete && (
        <DeleteAccountModal
          onClose={() => setConfirmDelete(false)}
          username={account?.username ?? ""}
        />
      )}
    </div>
  );
}

function SettingsRow({
  label,
  value,
  onEdit,
  mono = false,
  loading,
  editLabel = "Edit",
  subline,
}: {
  label: string;
  value: string;
  onEdit: () => void;
  mono?: boolean;
  loading?: boolean;
  editLabel?: string;
  subline?: React.ReactNode;
}) {
  return (
    <div className="px-5 py-3.5">
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-wider text-foreground/40">{label}</div>
          <div
            className={[
              "mt-0.5 text-sm truncate",
              mono ? "font-mono" : "",
              loading ? "text-foreground/30" : "text-foreground/85",
            ].join(" ")}
          >
            {loading ? "…" : value}
          </div>
        </div>
        <button
          type="button"
          onClick={onEdit}
          disabled={loading}
          className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/75 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors shrink-0 disabled:opacity-40"
        >
          {editLabel}
        </button>
      </div>
      {subline && <div className="mt-2">{subline}</div>}
    </div>
  );
}

// Banner that appears under the Email row when there's an in-flight
// email-change verification. Shows the target address, a Resend
// button (rate-limited to once a minute), and a Cancel button that
// drops the verification without changing the email.
function PendingEmailNotice({
  pendingEmail,
  onCancelled,
}: {
  pendingEmail: PendingEmailChange;
  onCancelled: () => Promise<void> | void;
}) {
  const [cooldown, setCooldown] = useState<number>(
    Math.ceil((pendingEmail.canResendInMs ?? 0) / 1000),
  );
  const [resending, setResending] = useState(false);
  const [resendError, setResendError] = useState<string | null>(null);
  const [resentAt, setResentAt] = useState<number | null>(null);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = window.setTimeout(() => setCooldown((s) => Math.max(0, s - 1)), 1000);
    return () => window.clearTimeout(t);
  }, [cooldown]);

  const resend = async () => {
    if (resending || cooldown > 0) return;
    setResending(true);
    setResendError(null);
    try {
      const r = await fetch("/api/auth/verify-resend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ purpose: "email_change" }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      setResentAt(Date.now());
      setCooldown(60);
    } catch (e) {
      setResendError(e instanceof Error ? e.message : String(e));
    } finally {
      setResending(false);
    }
  };

  const cancel = async () => {
    if (cancelling) return;
    setCancelling(true);
    try {
      await fetch("/api/auth/verify-cancel", { method: "POST" });
      await onCancelled();
    } finally {
      setCancelling(false);
    }
  };

  return (
    <div className="rounded-xl border border-amber-300/20 bg-amber-300/[0.04] px-3.5 py-2.5">
      <div className="flex items-start gap-2.5">
        <span aria-hidden className="mt-1.5 h-1 w-1 rounded-full bg-amber-300/80 shrink-0" />
        <div className="min-w-0 flex-1">
          <div className="text-[11px] uppercase tracking-wider text-amber-200/85 font-mono">
            Verification sent
          </div>
          <p className="mt-0.5 text-[12px] text-foreground/70 leading-relaxed">
            We emailed a confirmation link to{" "}
            <span className="font-mono text-foreground/90">{pendingEmail.pendingEmail}</span>.
            Click it to switch your account email.
          </p>
          {resentAt && !resendError && (
            <p className="mt-1 text-[11px] text-emerald-300/85">Sent again, check your inbox.</p>
          )}
          {resendError && <p className="mt-1 text-[11px] text-rose-300/90">{resendError}</p>}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={resend}
              disabled={cooldown > 0 || resending}
              className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/75 hover:text-foreground px-2.5 py-0.5 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {resending ? "Sending…" : cooldown > 0 ? `Resend in ${cooldown}s` : "Resend"}
            </button>
            <button
              type="button"
              onClick={cancel}
              disabled={cancelling}
              className="rounded-full border border-rose-300/25 bg-rose-300/[0.04] hover:border-rose-300/55 hover:bg-rose-300/[0.10] text-rose-100 px-2.5 py-0.5 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors disabled:opacity-40"
            >
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Row for a linked OAuth provider (GitHub / Google).
// Shows the provider, a small hint of the account id, and an Unlink
// button that goes through /api/users/me/unlink. The unlink endpoint
// refuses if removing this provider would lock the user out.
function LinkedAccountRow({
  provider,
  hint,
  canUnlink,
  onUnlinked,
}: {
  provider: string;
  hint: string;
  canUnlink: boolean;
  onUnlinked: () => Promise<void> | void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const niceName =
    provider === "github" ? "GitHub" : provider === "google" ? "Google" : provider;

  const unlink = async () => {
    if (busy || !canUnlink) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/users/me/unlink", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      await onUnlinked();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="px-5 py-3.5">
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-wider text-foreground/40">
            Connected · {niceName}
          </div>
          <div className="mt-0.5 text-sm text-foreground/85">
            <span className="text-foreground/55 font-mono text-xs">…{hint}</span>
          </div>
        </div>
        <Tooltip
          side="top"
          align="end"
          label={
            canUnlink
              ? `Unlink ${niceName}`
              : "Set a password first, this is your only way to sign in"
          }
          className="shrink-0"
        >
          <button
            type="button"
            onClick={unlink}
            disabled={busy || !canUnlink}
            className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/75 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Unlinking…" : "Unlink"}
          </button>
        </Tooltip>
      </div>
      {error && <p className="mt-1.5 text-[11px] text-rose-300/90">{error}</p>}
    </div>
  );
}

function ToggleRow({
  label,
  description,
  hint,
  checked,
  onChange,
  disabled = false,
}: {
  label: string;
  description: React.ReactNode;
  // Optional smaller line below the description (e.g. policy link).
  hint?: React.ReactNode;
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-4 px-5 py-3.5">
      <div className="min-w-0 max-w-prose">
        <div className="text-[11px] uppercase tracking-wider text-foreground/40">{label}</div>
        <p className="mt-0.5 text-sm text-foreground/70 leading-relaxed">{description}</p>
        {hint && <p className="mt-1 text-[11px] text-foreground/40 leading-relaxed">{hint}</p>}
      </div>
      <label
        className={[
          "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full transition-colors",
          checked ? "bg-foreground/30" : "bg-foreground/[0.06]",
          disabled ? "opacity-40 cursor-not-allowed" : "",
        ].join(" ")}
      >
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only"
        />
        <span
          className={[
            "absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform",
            checked ? "translate-x-5" : "translate-x-0.5",
          ].join(" ")}
        />
      </label>
    </div>
  );
}

// Always-visible billing row. Three modes:
//   - User has been to Stripe checkout (has customer record): "Manage"
//     opens the Stripe Customer Portal, update card, change plan,
//     cancel, view invoices.
//   - User has never paid: "View plans" navigates to /pricing.
//   - User is on an admin-granted Mega plan with no Stripe customer:
//     same "View plans" fallback (Mega isn't billed through Stripe).
// Profile-row twin to BillingRow. When the user is on Free, surface a
// "redeem code" input that flips them to Beta. When the user is
// actively on Beta, surface the remaining window so they know when it
// ends. Hidden for Pro/Mega since those plans already cover the same
// limits and a beta code would be a no-op.
function BetaCodeRow({
  plan,
  betaExpiresAt,
}: {
  plan: PlanId;
  betaExpiresAt: string | null;
}) {
  const { update: refreshSession } = useSession();
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Mega is admin-granted and outranks Beta, so the redeem control is
  // pointless there. Pro accounts still see the input so they can
  // stack Beta on top of their existing subscription (e.g. to unlock
  // the feedback bar without losing their Pro tier).
  if (plan === "mega") return null;

  // Show the "active redemption" row when there's a beta window in
  // the future, regardless of whether the effective tier reads as
  // "beta" or "pro_50" — NEURON6-BACKER-PK grants the Pro tier but
  // still uses betaExpiresAt as the clock.
  const hasActiveWindow = !!betaExpiresAt
    && new Date(betaExpiresAt).getTime() > Date.now();
  if (plan === "beta" || hasActiveWindow) {
    return (
      <ActiveBetaRow betaExpiresAt={betaExpiresAt} onLeft={() => {
        refreshPlan();
        void refreshSession({});
      }} />
    );
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    const trimmed = code.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/billing/redeem-beta", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: trimmed }),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body?.error || `http ${r.status}`);
      // Refresh the plan cache so the pill + this row pick up the
      // new tier instantly, AND trigger a session.update() so the
      // next page reload sees the JWT-baked plan without a stale-cookie
      // flash back to Free.
      refreshPlan();
      void refreshSession({});
      setCode("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="flex items-center justify-between gap-4 px-5 py-3.5">
      <div className="min-w-0 flex-1">
        <div className="text-[11px] uppercase tracking-wider text-foreground/40">Redeem code</div>
        <p className="mt-0.5 text-sm text-foreground/70 leading-relaxed">
          Have a code? Redeem it for Pro-level access.
        </p>
        {error && (
          <p className="mt-1 text-[11px] text-red-700 dark:text-red-300">{error}</p>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <input
          value={code}
          onChange={(e) => setCode(e.target.value)}
          aria-label="Redeem code"
          spellCheck={false}
          autoComplete="off"
          className="font-mono text-xs rounded-full border border-foreground/12 bg-foreground/[0.03] px-3 py-1.5 w-40 placeholder:text-foreground/30 focus:outline-none focus:border-foreground/30 transition-colors"
        />
        <button
          type="submit"
          disabled={busy || !code.trim()}
          className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/75 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {busy ? "Redeeming…" : "Redeem"}
        </button>
      </div>
    </form>
  );
}

// Renders the "Active until X · N days left" copy for a beta user
// plus a Leave-beta confirm flow. Two clicks to leave (Leave →
// Confirm) so the action isn't a one-click footgun. On success the
// parent's onLeft() refreshes the plan cache so the row swaps back
// to the redeem input without a page reload.
function ActiveBetaRow({
  betaExpiresAt,
  onLeft,
}: {
  betaExpiresAt: string | null;
  onLeft: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const expiry = betaExpiresAt ? new Date(betaExpiresAt) : null;
  const valid = expiry && Number.isFinite(expiry.getTime());
  const expiryText = valid
    ? expiry!.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : "soon";
  const msLeft = valid ? expiry!.getTime() - Date.now() : 0;
  const daysLeft = Math.max(0, Math.ceil(msLeft / (24 * 60 * 60 * 1000)));

  const leave = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/billing/redeem-beta", { method: "DELETE" });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body?.error || `http ${r.status}`);
      setConfirming(false);
      onLeft();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-start justify-between gap-4 px-5 py-3.5">
      <div className="min-w-0">
        <div className="text-[11px] uppercase tracking-wider text-foreground/40">Beta access</div>
        <p className="mt-0.5 text-sm text-foreground/70 leading-relaxed">
          Active until <span className="text-[var(--foreground)] font-medium">{expiryText}</span>
          {valid && daysLeft > 0 && (
            <span className="text-foreground/45"> · {daysLeft} day{daysLeft === 1 ? "" : "s"} left</span>
          )}
          .
        </p>
        {confirming && (
          <p className="mt-1 text-[11px] text-foreground/55 leading-relaxed">
            Leaving drops you back to Free immediately. You can&rsquo;t redeem the same code twice.
          </p>
        )}
        {error && (
          <p className="mt-1 text-[11px] text-red-700 dark:text-red-300">{error}</p>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {confirming ? (
          <>
            <button
              type="button"
              disabled={busy}
              onClick={() => { setConfirming(false); setError(null); }}
              className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/70 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors disabled:opacity-40"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void leave()}
              className="rounded-full bg-red-600 text-white hover:opacity-90 px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {busy ? "Leaving…" : "Confirm leave"}
            </button>
          </>
        ) : (
          <>
            <span className="rounded-full border border-sky-500/40 bg-sky-500/[0.08] text-sky-800 dark:border-sky-300/30 dark:bg-sky-300/[0.06] dark:text-sky-100/90 px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono">
              Active
            </span>
            <button
              type="button"
              onClick={() => setConfirming(true)}
              className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/70 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors"
              title="Leave the beta program"
            >
              Leave beta
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function BillingRow({
  plan,
  subscriptionStatus,
}: {
  plan: PlanId;
  subscriptionStatus: string | null;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Heuristic: anyone who has a stripe-side status field has been
  // through checkout once and therefore has a customer record. Mega
  // plans are admin-granted and don't have a status, so they fall
  // through to the "View plans" path even if the column says "mega".
  const hasStripe = !!subscriptionStatus;

  const onClick = async () => {
    if (busy) return;
    if (!hasStripe) {
      navigateAppTo("pricing");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/billing/portal", { method: "POST" });
      if (!r.ok) {
        // Most likely cause: server has no stripeCustomerId for this
        // user (rare race, they're flagged as having a status but
        // the customer column is null). Fall back to /pricing rather
        // than showing a dead-end error.
        if (r.status === 400) {
          navigateAppTo("pricing");
          return;
        }
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      const body = await r.json();
      if (body.url) {
        window.location.href = body.url;
        return;
      }
      throw new Error("no portal url");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  const description =
    plan === "mega"
      ? "Mega plan, admin-granted, not billed through Stripe."
      : plan === "beta"
      ? "Beta access, time-boxed, no card required. Upgrade to Pro to continue after expiry."
      : hasStripe
      ? "Manage your subscription, update card, view invoices."
      : "View plans and pricing.";
  const buttonLabel = busy
    ? "Opening…"
    : hasStripe
    ? "Manage"
    : "View plans";

  return (
    <div className="flex items-center justify-between gap-4 px-5 py-3.5">
      <div className="min-w-0">
        <div className="text-[11px] uppercase tracking-wider text-foreground/40">Billing</div>
        <p className="mt-0.5 text-sm text-foreground/70 leading-relaxed">{description}</p>
        {error && <p className="mt-1 text-[11px] text-rose-300/90">{error}</p>}
      </div>
      <button
        type="button"
        onClick={onClick}
        disabled={busy}
        className="rounded-full border border-foreground/12 bg-foreground/[0.03] hover:border-foreground/30 hover:bg-foreground/[0.07] text-foreground/75 hover:text-foreground px-3 py-1 text-[10px] uppercase tracking-[0.18em] font-mono transition-colors shrink-0 disabled:opacity-40"
      >
        {buttonLabel}
      </button>
    </div>
  );
}

function FieldEditor({
  title,
  field,
  initial,
  placeholder,
  inputType = "text",
  mono = false,
  lowercase = false,
  requireCurrentPassword = false,
  onClose,
  onSaved,
}: {
  title: string;
  field: "name" | "username" | "email";
  initial: string;
  placeholder?: string;
  inputType?: string;
  mono?: boolean;
  lowercase?: boolean;
  // Email changes re-authenticate: the server rejects an email PATCH
  // without the current password (when one is set), so a hijacked
  // session can't quietly move the account-recovery anchor.
  requireCurrentPassword?: boolean;
  onClose: () => void;
  onSaved: () => Promise<void> | void;
}) {
  const [value, setValue] = useState(initial);
  const [currentPassword, setCurrentPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const v = lowercase ? value.trim().toLowerCase() : value.trim();
    try {
      const r = await fetch("/api/users/me", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          requireCurrentPassword ? { [field]: v, currentPassword } : { [field]: v },
        ),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        throw new Error(body?.error || `http ${r.status}`);
      }
      // Email changes don't apply directly, the server sent a
      // verification link to the new address. Stay on the profile
      // page (we don't blur the whole UI mid-session) so the user
      // can hit Cancel any time before clicking the link, or just
      // ignore the email and the change quietly expires after 24h.
      // The pending banner under the Email row handles the rest.
      await onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <ModalShell title={title} onClose={onClose}>
      <form onSubmit={submit} className="grid gap-3">
        <input
          autoFocus
          type={inputType}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          className={[
            "rounded-lg border border-foreground/10 bg-foreground/[0.03] px-3 py-2.5 text-sm focus:outline-none focus:border-foreground/30 placeholder:text-foreground/35 transition-colors",
            mono ? "font-mono" : "",
          ].join(" ")}
        />
        {requireCurrentPassword && (
          <label className="grid gap-1.5">
            <span className="text-xs text-foreground/55">Current password</span>
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              placeholder="Confirm it's you"
              autoComplete="current-password"
              className="rounded-lg border border-foreground/10 bg-foreground/[0.03] px-3 py-2.5 text-sm focus:outline-none focus:border-foreground/30 placeholder:text-foreground/35 transition-colors"
            />
          </label>
        )}
        {error && <p className="text-[11px] text-rose-300/90">{error}</p>}
        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-full border border-foreground/15 bg-foreground/[0.04] hover:border-foreground/30 hover:bg-foreground/[0.08] text-foreground/75 hover:text-foreground px-3 py-1.5 text-xs transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={
              busy ||
              value.trim().length === 0 ||
              value.trim() === initial.trim() ||
              (requireCurrentPassword && currentPassword.length === 0)
            }
            className="rounded-full bg-foreground text-background px-4 py-1.5 text-xs font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Saving…" : "Save"}
          </button>
        </div>
      </form>
    </ModalShell>
  );
}

function PasswordEditor({
  hasPassword,
  onClose,
  onSaved,
}: {
  hasPassword: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/users/me/password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          hasPassword
            ? { currentPassword: current, newPassword: next }
            : { newPassword: next },
        ),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <ModalShell title={hasPassword ? "Change password" : "Set password"} onClose={onClose}>
      <form onSubmit={submit} className="grid gap-3">
        {!hasPassword && (
          <p className="text-xs text-foreground/55 leading-relaxed">
            Add a password so you can sign in directly, useful if you ever want to unlink your social account.
          </p>
        )}
        {hasPassword && (
          <label className="grid gap-1">
            <span className="text-xs text-foreground/55">Current password</span>
            <input
              autoFocus
              type="password"
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
              className="rounded-lg border border-foreground/10 bg-foreground/[0.03] px-3 py-2.5 text-sm focus:outline-none focus:border-foreground/30"
            />
          </label>
        )}
        <label className="grid gap-1">
          <span className="text-xs text-foreground/55">{hasPassword ? "New password" : "Password"}</span>
          <input
            autoFocus={!hasPassword}
            type="password"
            minLength={8}
            value={next}
            onChange={(e) => setNext(e.target.value)}
            className="rounded-lg border border-foreground/10 bg-foreground/[0.03] px-3 py-2.5 text-sm focus:outline-none focus:border-foreground/30"
          />
          <span className="text-[10px] text-foreground/40">Minimum 8 characters.</span>
        </label>
        {error && <p className="text-[11px] text-rose-300/90">{error}</p>}
        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-full border border-foreground/15 bg-foreground/[0.04] hover:border-foreground/30 hover:bg-foreground/[0.08] text-foreground/75 hover:text-foreground px-3 py-1.5 text-xs transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy || (hasPassword && !current) || next.length < 8}
            className="rounded-full bg-foreground text-background px-4 py-1.5 text-xs font-medium hover:bg-zinc-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Saving…" : hasPassword ? "Update password" : "Set password"}
          </button>
        </div>
      </form>
    </ModalShell>
  );
}

function DeleteAccountModal({ onClose, username }: { onClose: () => void; username: string }) {
  const [confirmText, setConfirmText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const target = username || "delete";
  const canDelete = confirmText.trim() === target && !busy;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canDelete) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/users/me", { method: "DELETE" });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      // Successful delete → sign out so the session cookie clears,
      // then redirect home. The user no longer exists in Postgres,
      // so leaving the cookie around would lead to broken /api/auth
      // round-trips.
      await signOut({ callbackUrl: "/" });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <ModalShell title="Delete account" onClose={busy ? () => {} : onClose} tone="danger">
      <form onSubmit={submit} className="grid gap-3">
        <p className="text-sm text-foreground/75 leading-relaxed">
          This permanently removes your account, every workspace project you own,
          and all uploaded images. There&rsquo;s no undo.
        </p>
        <label className="grid gap-1">
          <span className="text-xs text-foreground/55">
            Type <span className="font-mono text-rose-200/85">{target}</span> to confirm
          </span>
          <input
            autoFocus
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            className="rounded-lg border border-rose-300/25 bg-rose-300/[0.04] px-3 py-2.5 text-sm font-mono focus:outline-none focus:border-rose-300/55"
          />
        </label>
        {error && <p className="text-[11px] text-rose-300/90">{error}</p>}
        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="rounded-full border border-foreground/15 bg-foreground/[0.04] hover:border-foreground/30 hover:bg-foreground/[0.08] text-foreground/75 hover:text-foreground px-3 py-1.5 text-xs transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!canDelete}
            className="rounded-full bg-rose-300/[0.10] border border-rose-300/40 text-rose-100 hover:bg-rose-300/[0.18] hover:border-rose-300/70 px-4 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? "Deleting…" : "Delete forever"}
          </button>
        </div>
      </form>
    </ModalShell>
  );
}

function ModalShell({
  title,
  children,
  onClose,
  tone = "default",
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
  tone?: "default" | "danger";
}) {
  return (
    <div
      // Theme-aware backdrop. White-tinted in light mode, black-
      // tinted in dark mode, both with blur. Pure bg-black/60 was
      // dark-only and read as a grey wash over the light theme.
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-white/70 dark:bg-black/60 backdrop-blur-md"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className={[
          "w-full max-w-md rounded-2xl border bg-[var(--surface)] p-6 grid gap-4",
          tone === "danger" ? "border-rose-300/25" : "border-foreground/10",
        ].join(" ")}
        style={{ boxShadow: "var(--shadow-strong)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <h3 className="text-lg font-medium tracking-tight">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-foreground/45 hover:text-foreground text-lg leading-none"
          >
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// Frosted full-screen overlay shown for the brief moment between
// "save succeeded" and the hard navigation to /app?profile=1. Lets
// the page fade behind a blur instead of flashing white when the
// browser tears down the document.
function ReloadOverlay() {
  return (
    <div
      aria-hidden
      // Theme-aware backdrop using the page's background token + a
      // blur. Hard rgba(10,10,12,0.55) was dark-only.
      className="fixed inset-0 z-[300] pointer-events-auto bg-[rgb(var(--background-rgb)/0.55)] backdrop-blur-[10px]"
      style={{
        WebkitBackdropFilter: "blur(10px) saturate(120%)",
        // Animate in across one frame so React's first paint already
        // has a non-zero opacity and the user sees a smooth fade
        // rather than a sudden snap.
        animation: "pixelkit-reload-fade-in 220ms ease-out forwards",
      }}
    >
      <style>{`
        @keyframes pixelkit-reload-fade-in {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
      `}</style>
      <div className="h-full w-full grid place-items-center">
        <div className="flex items-center gap-3 text-[11px] uppercase tracking-[0.2em] font-mono text-foreground/65">
          <span
            aria-hidden
            className="h-1.5 w-1.5 rounded-full bg-foreground/70"
            style={{ animation: "pulse 1.4s ease-in-out infinite" }}
          />
          Saving…
        </div>
      </div>
    </div>
  );
}
