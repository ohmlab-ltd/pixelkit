"use client";

import { useEffect, useMemo, useState } from "react";

import { Footer } from "./Footer";
import { PLANS, PRO_TIER_IDS, CREDIT_COSTS, isProPlan, type ProTierId } from "../lib/plans";
import { usePlan } from "./PlanPill";

// Reassurance strip beneath the pricing cards. Static copy, no logic.
const TRUST_ITEMS: { title: string; desc: string; icon: React.ReactNode }[] = [
  {
    title: "Cancel any time",
    desc: "No lock-in, no contract.",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <circle cx="12" cy="12" r="9" />
        <path d="M9 12h6" />
      </svg>
    ),
  },
  {
    title: "Secure by default",
    desc: "Encrypted storage, private projects on Pro.",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <rect x="4" y="11" width="16" height="10" rx="2" />
        <path d="M8 11V7a4 4 0 0 1 8 0v4" />
      </svg>
    ),
  },
  {
    title: "Built for scale",
    desc: "GPU-backed training and labelling.",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M3 17l6-6 4 4 8-8" />
        <path d="M14 7h7v7" />
      </svg>
    ),
  },
  {
    title: "Friendly support",
    desc: "Real humans, quick replies.",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
      </svg>
    ),
  },
];

export function PricingView() {
  // The plan chip in the profile / nav routes here. Reading usePlan
  // means we can mark the current tier as "Current plan" instead of
  // showing a buy CTA the user has already taken.
  const usage = usePlan();
  const currentPlan = usage?.plan ?? null;
  const currentProTier: ProTierId | null = currentPlan && isProPlan(currentPlan)
    ? (currentPlan as ProTierId)
    : null;

  // Default selected Pro tier:
  //   • Pro user → land them on their own tier so "Current plan" is
  //     the first thing they see.
  //   • Everyone else → the middle (150) tier, the upgrade most
  //     people land on once they outgrow Free.
  const [proTier, setProTier] = useState<ProTierId>("pro_150");
  useEffect(() => {
    if (currentProTier) setProTier(currentProTier);
  }, [currentProTier]);

  return (
    <main className="min-h-screen bg-[var(--background)]">
      <section className="mx-auto max-w-6xl px-6 pt-16 pb-10 flex items-start justify-between gap-6 flex-wrap">
        <div className="min-w-0">
          <h1 className="text-5xl md:text-6xl font-medium tracking-tight">Pricing</h1>
          <p className="mt-4 max-w-xl text-foreground/50 text-lg">
            Pick a credit allowance that fits how much data you label, upload,
            and store. Upgrade or cancel any time.
          </p>
        </div>
        {/* Reassurance note, top-right, mirrors the mockup. */}
        <div className="flex items-start gap-2.5 rounded-2xl border border-foreground/10 bg-[var(--surface)] px-4 py-3 max-w-xs shadow-[var(--shadow-soft)]">
          <svg viewBox="0 0 24 24" className="h-4 w-4 mt-0.5 shrink-0 text-[var(--accent-orange)]" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M12 2 4 5v6c0 5 3.4 8.5 8 10 4.6-1.5 8-5 8-10V5l-8-3z" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          <p className="text-[12px] leading-relaxed text-foreground/55">
            All plans include secure storage, 99.9% uptime and friendly support.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-10">
        <div className="mb-4 flex items-center gap-2.5">
          <span className="pk-accent-bar" aria-hidden />
          <h2 className="pk-section-title text-xl">Plans</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <FreeCard isCurrent={currentPlan === "free"} />
          <ProCard tier={proTier} onTierChange={setProTier} currentProTier={currentProTier} />
          <EnterpriseCard isCurrent={currentPlan === "enterprise"} />
        </div>
      </section>

      <CreditUsageTable />

      <section className="mx-auto max-w-6xl px-6 pb-12 pt-2">
        <ContactCard />
      </section>

      {/* Trust row, mirrors the reassurance strip in the mockup. */}
      <section className="mx-auto max-w-6xl px-6 pb-16">
        <div className="mb-4 flex items-center gap-2.5">
          <span className="pk-accent-bar" aria-hidden />
          <h2 className="pk-section-title text-xl">Every plan includes</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {TRUST_ITEMS.map((t) => (
            <div key={t.title} className="pk-card rounded-2xl p-4 flex items-start gap-3">
              <span className="grid h-9 w-9 place-items-center rounded-xl bg-foreground/[0.05] text-foreground/60 shrink-0">
                {t.icon}
              </span>
              <div className="min-w-0">
                <div className="text-sm font-medium text-[var(--foreground)]">{t.title}</div>
                <div className="text-[12px] text-foreground/45 leading-snug">{t.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <Footer />
    </main>
  );
}

// ─── Free card ────────────────────────────────────────────────────

function FreeCard({ isCurrent }: { isCurrent: boolean }) {
  const meta = PLANS.free;
  return (
    <article className="pk-card pk-card-hover relative rounded-2xl p-7 flex flex-col gap-5">
      <header className="grid gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-emerald-500/12 text-emerald-600 dark:text-emerald-400">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M11 20A7 7 0 0 1 4 13c0-5 4-9 9-9a7 7 0 0 1 7 7c0 5-4 9-9 9z" />
            <path d="M11 20c0-5 2.5-9 6.5-11" />
          </svg>
        </span>
        <div className="grid gap-1.5">
          <h2 className="text-lg font-medium tracking-tight">{meta.name}</h2>
          <p className="text-sm text-foreground/50">Get started with auto-labelling.</p>
        </div>
      </header>

      <div className="flex items-baseline gap-1.5">
        <span className="text-5xl font-thin tracking-tight tabular-nums">£0</span>
        <span className="text-sm text-foreground/40">/ month</span>
      </div>

      <ul className="flex-1 flex flex-col gap-3 text-sm">
        <FeatureLi>{meta.limits.creditsPerMonth} credits / month</FeatureLi>
        <FeatureLi>
          {meta.limits.creditsPerMonth * CREDIT_COSTS.labelledImagesPerCredit} labels ·{" "}
          {meta.limits.creditsPerMonth * CREDIT_COSTS.uploadedImagesPerCredit} uploads ·{" "}
          {meta.limits.creditsPerMonth * CREDIT_COSTS.storedImagesPerCreditPerMonth} stored
        </FeatureLi>
        <FeatureLi>{meta.limits.projects} public projects</FeatureLi>
        <FeatureLi>Community projects access</FeatureLi>
      </ul>

      <button
        type="button"
        disabled
        className="mt-2 rounded-full px-4 py-2.5 text-sm font-medium border border-foreground/15 bg-foreground/5 text-foreground/80 opacity-60 cursor-not-allowed"
      >
        {isCurrent ? "Current plan" : "Free"}
      </button>
    </article>
  );
}

// ─── Pro card with tier selector ──────────────────────────────────

function ProCard({
  tier,
  onTierChange,
  currentProTier,
}: {
  tier: ProTierId;
  onTierChange: (next: ProTierId) => void;
  currentProTier: ProTierId | null;
}) {
  const isCurrent = currentProTier === tier;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const meta = PLANS[tier];

  const features = useMemo(() => {
    const c = meta.limits.creditsPerMonth;
    return [
      `${c} credits / month`,
      `${(c * CREDIT_COSTS.labelledImagesPerCredit).toLocaleString()} labels · ${(c * CREDIT_COSTS.uploadedImagesPerCredit).toLocaleString()} uploads · ${(c * CREDIT_COSTS.storedImagesPerCreditPerMonth).toLocaleString()} stored`,
      `${meta.limits.projects} private or public projects`,
      "Priority labelling queue",
    ];
  }, [meta]);

  const onClick = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    const { url, error: err } = await startCheckout(tier);
    if (err) {
      setError(err);
      setBusy(false);
      return;
    }
    if (url) window.location.href = url;
  };

  return (
    <article className="relative flex flex-col gap-5 rounded-2xl border border-[rgb(var(--accent-orange-rgb)/0.45)] bg-[rgb(var(--accent-orange-rgb)/0.05)] p-7">
      {/* Top-centre "Most popular" tag, overlapping the orange border. Flat to
          match the rest of the app (no glow). */}
      <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[10px] rounded-full bg-[var(--accent-orange)] text-black px-3 py-1 uppercase tracking-wider font-semibold">
        Most popular
      </span>
      <header className="grid gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-orange-500/15 text-[var(--accent-orange)]">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden>
            <path d="M3 7l4.5 3.5L12 4l4.5 6.5L21 7l-1.6 11H4.6L3 7z" />
          </svg>
        </span>
        <div className="grid gap-1.5">
          <h2 className="text-lg font-medium tracking-tight">Pro</h2>
          <p className="text-sm text-foreground/50">For serious dataset builders.</p>
        </div>
      </header>

      <div className="flex items-baseline gap-1.5">
        {/* Tabular nums + a fixed-width container so the digit count
            change between £29/£69/£179 doesn't shove the cadence to
            the right by a few pixels each click. */}
        <span className="text-5xl font-thin tracking-tight tabular-nums transition-all duration-200">
          £{meta.priceGbp}
        </span>
        <span className="text-sm text-foreground/40">/ month</span>
      </div>

      {/* Tier selector, three pill buttons. Drives the visible
          price + features + the Stripe checkout payload. */}
      <div className="rounded-full border border-foreground/15 bg-foreground/[0.04] p-1 flex">
        {PRO_TIER_IDS.map((id) => {
          const m = PLANS[id];
          const active = id === tier;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onTierChange(id)}
              className={[
                "flex-1 rounded-full px-2 py-1.5 text-[12px] font-mono uppercase tracking-wider transition-colors",
                active
                  ? "bg-foreground text-background"
                  : "text-foreground/65 hover:text-foreground",
              ].join(" ")}
            >
              {m.limits.creditsPerMonth}
            </button>
          );
        })}
      </div>

      <ul className="flex-1 flex flex-col gap-3 text-sm">
        {features.map((f) => (
          <FeatureLi key={f}>{f}</FeatureLi>
        ))}
      </ul>

      <button
        type="button"
        onClick={onClick}
        disabled={busy || isCurrent}
        className={[
          "mt-2 rounded-full px-4 py-2.5 text-sm font-medium transition-colors",
          isCurrent
            ? "border border-foreground/15 bg-foreground/5 text-foreground/80 cursor-not-allowed"
            : "bg-foreground text-background hover:opacity-90",
          busy ? "opacity-60 cursor-not-allowed" : "",
        ].join(" ")}
      >
        {isCurrent
          ? "Current plan"
          : busy
          ? "Redirecting…"
          : currentProTier
          ? `Switch to ${meta.limits.creditsPerMonth} credits`
          : `Upgrade to Pro · ${meta.limits.creditsPerMonth} credits`}
      </button>
      {error && <p className="text-[11px] text-rose-600 dark:text-rose-300/90">{error}</p>}
    </article>
  );
}

// ─── Enterprise card ──────────────────────────────────────────────

function EnterpriseCard({ isCurrent }: { isCurrent: boolean }) {
  return (
    <article className="pk-card pk-card-hover relative rounded-2xl p-7 flex flex-col gap-5">
      <header className="grid gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-violet-500/12 text-violet-600 dark:text-violet-400">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M3 21h18M5 21V5a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v16M19 21V11a1 1 0 0 0-1-1h-2" />
            <path d="M8 7h2M8 11h2M8 15h2" />
          </svg>
        </span>
        <div className="grid gap-1.5">
          <h2 className="text-lg font-medium tracking-tight">Enterprise</h2>
          <p className="text-sm text-foreground/50">
            Volume credits, on-prem, SSO, and dedicated support.
          </p>
        </div>
      </header>

      <div className="flex items-baseline gap-1.5">
        <span className="text-5xl font-thin tracking-tight">Custom</span>
      </div>

      <ul className="flex-1 flex flex-col gap-3 text-sm">
        <FeatureLi>Custom credit allowance</FeatureLi>
        <FeatureLi>Unlimited projects, public or private</FeatureLi>
        <FeatureLi>SSO + audit log + on-prem deployment</FeatureLi>
        <FeatureLi>Dedicated support + SLAs</FeatureLi>
      </ul>

      {isCurrent ? (
        <button
          type="button"
          disabled
          className="mt-2 rounded-full px-4 py-2.5 text-sm font-medium border border-foreground/15 bg-foreground/5 text-foreground/80 opacity-60 cursor-not-allowed"
        >
          Current plan
        </button>
      ) : (
        <a
          href="mailto:info@ohmlab.co.uk?subject=PixelKit%20Enterprise"
          className="mt-2 inline-flex items-center justify-center rounded-full px-4 py-2.5 text-sm font-medium border border-foreground/15 bg-foreground/5 text-foreground/85 hover:bg-foreground/10 hover:text-foreground transition-colors"
        >
          Contact sales
        </a>
      )}
    </article>
  );
}

// ─── Credit usage table ───────────────────────────────────────────

function CreditUsageTable() {
  const rows: Array<{ label: string; cost: string }> = [
    {
      label: `${CREDIT_COSTS.labelledImagesPerCredit} labelled images`,
      cost: "1 credit",
    },
    {
      label: `${CREDIT_COSTS.uploadedImagesPerCredit} uploaded images`,
      cost: "1 credit",
    },
    {
      label: `${CREDIT_COSTS.storedImagesPerCreditPerMonth} stored images`,
      cost: "1 credit / month",
    },
  ];
  return (
    <section className="mx-auto max-w-6xl px-6 pb-10">
      <div className="mb-4 flex items-center gap-2.5">
        <span className="pk-accent-bar" aria-hidden />
        <h2 className="pk-section-title text-xl">Credit costs</h2>
      </div>
      <div className="pk-card rounded-2xl overflow-hidden">
        <header className="px-7 py-5 border-b border-foreground/[0.07] flex items-baseline justify-between gap-3 flex-wrap">
          <div>
            <p className="text-[13px] text-foreground/55">
              One credit covers any of the rates below. Combine freely
              within your monthly allowance.
            </p>
          </div>
          <span className="text-[10px] uppercase tracking-[0.18em] font-mono text-foreground/45">
            Usage&nbsp;type → credit cost
          </span>
        </header>
        <table className="w-full text-sm">
          <tbody>
            {rows.map((r, i) => (
              <tr
                key={r.label}
                className={[
                  "border-foreground/[0.06]",
                  i > 0 ? "border-t" : "",
                ].join(" ")}
              >
                <td className="px-7 py-4 text-[var(--foreground)]">{r.label}</td>
                <td className="px-7 py-4 text-right font-mono tabular-nums text-foreground/75">
                  {r.cost}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ─── Shared helpers ───────────────────────────────────────────────

function FeatureLi({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2.5 text-foreground/80">
      <svg
        viewBox="0 0 24 24"
        className="h-4 w-4 mt-0.5 shrink-0 text-foreground/60"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <polyline points="20 6 9 17 4 12" />
      </svg>
      <span>{children}</span>
    </li>
  );
}

async function startCheckout(plan: ProTierId): Promise<{ url?: string; error?: string }> {
  const r = await fetch("/api/billing/checkout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan }),
  });
  if (r.status === 401) {
    window.location.href = `/login?callbackUrl=${encodeURIComponent("/app?tab=pricing")}`;
    return {};
  }
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    return { error: body.error || `Checkout failed (${r.status})` };
  }
  const body = await r.json();
  return { url: body.url };
}

// ─── Contact card (general inbound, kept for non-Enterprise leads) ─

function ContactCard() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (state === "sending") return;
    if (!message.trim()) return;
    setState("sending");
    setError(null);
    try {
      const r = await fetch("/api/contact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message.trim(), email: email.trim() || undefined }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      setState("sent");
      setMessage("");
      setEmail("");
      window.setTimeout(() => setState("idle"), 4000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("error");
      window.setTimeout(() => setState("idle"), 4000);
    }
  };

  return (
    <article className="pk-card pk-card-hover relative rounded-2xl p-7 flex flex-col gap-5">
      <header className="grid gap-1.5">
        <h2 className="text-lg font-medium tracking-tight">Got a question?</h2>
        <p className="text-sm text-foreground/50">
          Not sure which tier fits, or need something custom? Drop us a line.
        </p>
      </header>

      <form onSubmit={submit} className="flex-1 flex flex-col gap-2.5">
        {/* Email pill, matches the bug-box style in TopNav. Required
            because the contact endpoint must have someone to reply to. */}
        <div className="w-full flex items-center gap-2 rounded-full bg-foreground/[0.04] border border-foreground/10 hover:border-foreground/20 focus-within:border-foreground/30 transition-colors pl-3 pr-3 py-1">
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-foreground/40 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <path d="m3 7 9 6 9-6" />
          </svg>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Your email"
            disabled={state === "sending"}
            className="flex-1 bg-transparent outline-none text-sm py-1 placeholder:text-foreground/40"
          />
        </div>

        {/* Message pill, same affordance as the Found-a-bug search
            box: state-driven placeholder, embedded Send button on the
            right. Single line of input matches the bug-box rhythm. */}
        <div
          className={[
            "w-full flex items-center gap-2 rounded-full bg-foreground/[0.04] border transition-colors pl-3 pr-1 py-1",
            state === "sent"
              ? "border-emerald-500/40"
              : state === "error"
              ? "border-rose-500/40"
              : "border-foreground/10 hover:border-foreground/20 focus-within:border-foreground/30",
          ].join(" ")}
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-foreground/40 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            maxLength={2000}
            disabled={state === "sending"}
            placeholder={
              state === "sent"
                ? "Thanks. We'll be in touch."
                : state === "error"
                ? `Couldn't send${error ? `: ${error}` : ""}`
                : "What do you need? Volume, deployment, team size…"
            }
            className={[
              "flex-1 bg-transparent outline-none text-sm py-1 placeholder:text-foreground/40",
              state === "sent" ? "text-emerald-700 dark:text-emerald-300 placeholder:text-emerald-700/80 dark:placeholder:text-emerald-300/80" : "",
              state === "error" ? "text-rose-700 dark:text-rose-300 placeholder:text-rose-700/80 dark:placeholder:text-rose-300/80" : "",
            ].join(" ")}
          />
          <button
            type="submit"
            disabled={!message.trim() || state === "sending"}
            className={[
              "rounded-full px-3 py-1 text-xs font-medium transition-colors shrink-0",
              message.trim() && state !== "sending"
                ? "bg-foreground text-background hover:opacity-90"
                : "bg-foreground/10 text-foreground/40 cursor-not-allowed",
            ].join(" ")}
            aria-label="Send message"
          >
            {state === "sending" ? "Sending…" : "Send"}
          </button>
        </div>

        <p className="text-[11px] text-foreground/35">
          Or email{" "}
          <a
            href="mailto:info@ohmlab.co.uk"
            className="text-foreground/55 hover:text-foreground underline underline-offset-2"
          >
            info@ohmlab.co.uk
          </a>
          .
        </p>
      </form>
    </article>
  );
}
