"use client";

import { useEffect, useState } from "react";

import type { PlanId, PlanLimits } from "@/lib/plans";
import { isProPlan } from "@/lib/plans";
import { navigateAppTo } from "@/lib/appNav";

export type PlanData = {
  plan: PlanId;
  planName: string;
  subscriptionStatus: string | null;
  // ISO timestamp of when a cancelled-at-period-end subscription
  // will terminate. Null when not cancelling.
  subscriptionCancelAt: string | null;
  // ISO timestamp of when a redeemed beta code lapses. Null when the
  // user has never redeemed one.
  betaExpiresAt: string | null;
  limits: PlanLimits;
  usage: {
    projects: number;
    imagesLabelledThisMonth: number;
    imagesLabelledThisPeriod: number;
    imagesUploadedThisPeriod: number;
    imagesStoredNow: number;
  };
  cycle: {
    start: string; // ISO
    end: string;   // ISO, the day quota resets
  };
  over: {
    images: boolean;
    projects: boolean;
    credits: boolean;
    anyLabelLimit: boolean;
  };
  backendUp: boolean;
};

// Module-level cache + subscriber list. Multiple <PlanPill /> mounts
// (header + profile page) share a single in-flight request and a
// single cached result so we don't fan out duplicate /api/users/usage
// fetches on every page transition.
let cached: PlanData | null = null;
let inflight: Promise<PlanData | null> | null = null;
const subscribers = new Set<(p: PlanData | null) => void>();

async function fetchPlan(): Promise<PlanData | null> {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const r = await fetch("/api/users/usage", { cache: "no-store" });
      if (!r.ok) return null;
      const data = (await r.json()) as PlanData;
      cached = data;
      for (const cb of subscribers) cb(data);
      return data;
    } catch {
      return null;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

export function usePlan(): PlanData | null {
  const [data, setData] = useState<PlanData | null>(cached);
  useEffect(() => {
    subscribers.add(setData);
    if (!cached) void fetchPlan();
    return () => {
      subscribers.delete(setData);
    };
  }, []);
  return data;
}

// Force a refresh, call after a successful checkout or admin action
// so the pill reflects the new plan without a full page reload.
export function refreshPlan() {
  cached = null;
  void fetchPlan();
}

// Subtle, glass-style pill that matches the floating header capsule.
// Free reads as a quiet status chip; Pro adds a low-chroma orange
// accent so the upgrade is recognisable but not garish.
export function PlanPill({
  plan,
  planName,
  size = "sm",
  asLink = true,
  className = "",
}: {
  plan: PlanId;
  planName: string;
  size?: "xs" | "sm" | "lg";
  asLink?: boolean;
  className?: string;
}) {
  // All three Pro tiers share one chip palette/copy, the tier
  // suffix (50 / 150 / 500) is reflected in `planName`, not in the
  // colour.
  const isPro = isProPlan(plan);
  const sizeClasses =
    size === "xs"
      ? "px-2 py-0.5 text-[9px] tracking-[0.18em]"
      : size === "lg"
      ? "px-4 py-2 text-[11px] tracking-[0.18em]"
      : "px-2.5 py-1 text-[10px] tracking-[0.18em]";

  // Each plan gets a different colour treatment. Mega cranks the
  // orange right up, strong fill + bright text, to make it read
  // as a step beyond Pro. Beta uses a sky-cyan tint so it's
  // distinct from Pro/Mega but still reads as a premium tier.
  // Free stays neutral white-on-glass. Light-mode pivots use deeper
  // tones so chip text doesn't wash out on a near-white card.
  const palette =
    plan === "mega"
      ? "border-orange-500/60 bg-orange-500/[0.16] text-orange-800 dark:border-orange-400/55 dark:bg-orange-400/[0.10] dark:text-orange-50 hover:border-orange-500/80 hover:bg-orange-500/[0.22] dark:hover:border-orange-400/75 dark:hover:bg-orange-400/[0.16]"
      : isPro
      ? "border-orange-500/50 bg-orange-500/[0.10] text-orange-800 dark:border-orange-300/30 dark:bg-orange-300/[0.05] dark:text-orange-100/90 hover:border-orange-500/70 hover:bg-orange-500/[0.16] dark:hover:border-orange-300/50 dark:hover:bg-orange-300/[0.10]"
      : plan === "beta"
      ? "border-sky-500/55 bg-sky-500/[0.10] text-sky-800 dark:border-sky-300/40 dark:bg-sky-300/[0.07] dark:text-sky-100/90 hover:border-sky-500/75 hover:bg-sky-500/[0.16] dark:hover:border-sky-300/60 dark:hover:bg-sky-300/[0.12]"
      : "border-foreground/25 bg-foreground/[0.05] text-[var(--foreground)] hover:border-foreground/40 hover:bg-foreground/[0.08]";

  const inner = (
    <>
      {plan === "mega" ? (
        <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-orange-300" />
      ) : isPro ? (
        <span aria-hidden className="h-1 w-1 rounded-full bg-orange-300/80" />
      ) : plan === "beta" ? (
        <span aria-hidden className="h-1 w-1 rounded-full bg-sky-300" />
      ) : (
        <span aria-hidden className="h-1 w-1 rounded-full bg-foreground/40" />
      )}
      {planName}
    </>
  );

  // Glow intensifies with plan tier. Mega is brightest, twin-stop
  // box-shadow halo *plus* a text-shadow so the lettering itself
  // looks lit. Pro gets a soft single-stop glow. Beta gets a cyan
  // halo to match its palette. Free is unglowed.
  const glow =
    plan === "mega"
      ? {
          boxShadow:
            "0 0 18px -2px rgba(249,115,22,0.85), 0 0 38px -8px rgba(249,115,22,0.55)",
          textShadow:
            "0 0 6px rgba(249,180,80,0.95), 0 0 14px rgba(249,115,22,0.7)",
        }
      : isPro
      ? { boxShadow: "0 0 14px -2px rgba(249,115,22,0.45)" }
      : plan === "beta"
      ? { boxShadow: "0 0 14px -2px rgba(56,189,248,0.45)" }
      : undefined;

  const baseClass = [
    "inline-flex items-center gap-1.5 rounded-full border font-mono uppercase transition-colors",
    sizeClasses,
    palette,
    className,
  ].join(" ");

  if (!asLink) {
    return (
      <span className={baseClass} title={`Plan: ${planName}`} style={glow}>
        {inner}
      </span>
    );
  }

  // Programmatic in-app navigation rather than a URL change. /app is
  // a single SPA route, pushing a different ?tab= via <Link> doesn't
  // reliably re-render the parent page, so we fire an event the
  // parent listens for instead.
  return (
    <button
      type="button"
      onClick={() => navigateAppTo("pricing")}
      aria-label={`${planName} plan, view pricing`}
      title={`Plan: ${planName}`}
      className={baseClass}
      style={glow}
    >
      {inner}
    </button>
  );
}
