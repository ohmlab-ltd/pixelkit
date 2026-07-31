// Single source of truth for plan limits + credit allocations. Both
// the profile page and the create/label gates read from here so the
// displayed quotas and the enforced quotas can't drift apart.

// Plan IDs. Three Pro tiers replace the single old Pro plan so the
// user can pick a price / credit allocation that matches their
// volume. `beta` and `mega` stay for backward compatibility with
// existing accounts; new signups can't reach them via Stripe
// checkout (beta = code redemption; mega = admin-granted).
//
// `enterprise` is the contact-sales destination, no Stripe product
// is created for it.
export type PlanId =
  | "free"
  | "beta"
  | "pro_50"
  | "pro_150"
  | "pro_500"
  | "mega"
  | "enterprise";

// Pro tier IDs. Used by the FE pricing-card selector + checkout
// payload validation. Order matters: array index drives the default
// "selected" tier on the pricing card (mid-tier).
export type ProTierId = "pro_50" | "pro_150" | "pro_500";
export const PRO_TIER_IDS: ProTierId[] = ["pro_50", "pro_150", "pro_500"];

export type PlanLimits = {
  // Credits per calendar month. One credit covers ~100 labelled
  // images, ~800 uploaded images, or ~400 stored images for the
  // month. See CREDIT_COSTS below.
  creditsPerMonth: number;
  // Hard cap on auto-labelled images per calendar month. Derived
  // from creditsPerMonth × CREDIT_COSTS.labelledImagesPerCredit so
  // existing call-sites that gate on a flat image cap keep working
  // during the credit-system rollout. Once everything reads from
  // `creditsPerMonth` directly we can drop this.
  imagesLabelledPerMonth: number;
  // Hard cap on projects the user can own (public + private combined).
  projects: number;
  // Whether the plan permits private projects. Free is public-only.
  allowsPrivateProjects: boolean;
};

export type PlanMeta = {
  name: string;
  // Display label, "Pro" for all three pro tiers, distinct names
  // for free/beta/mega/enterprise.
  shortName: string;
  // Monthly GBP. null for plans that don't have a fixed price
  // (mega = admin grant, enterprise = custom, beta = free).
  priceGbp: number | null;
  limits: PlanLimits;
};

// Build a PlanLimits from just credits + projects + private flag.
// Derives the legacy imagesLabelledPerMonth field so older call
// sites still resolve a number without us repeating the arithmetic
// at every entry. Kept private to plans.ts.
function makeLimits(opts: {
  credits: number;
  projects: number;
  privateProjects: boolean;
}): PlanLimits {
  return {
    creditsPerMonth: opts.credits,
    imagesLabelledPerMonth: opts.credits * 100, // 1 credit = 100 labels
    projects: opts.projects,
    allowsPrivateProjects: opts.privateProjects,
  };
}

export const PLANS: Record<PlanId, PlanMeta> = {
  free: {
    name: "Free",
    shortName: "Free",
    priceGbp: 0,
    limits: makeLimits({ credits: 5, projects: 5, privateProjects: false }),
  },
  // Free-of-charge time-boxed tier handed out via beta codes. Matches
  // the entry Pro tier so beta testers can stress the product;
  // auto-downgrades to Free 30 days after redemption.
  beta: {
    name: "Beta",
    shortName: "Beta",
    priceGbp: 0,
    limits: makeLimits({ credits: 50, projects: 200, privateProjects: true }),
  },
  pro_50: {
    name: "Pro",
    shortName: "Pro · 50",
    priceGbp: 29,
    limits: makeLimits({ credits: 50, projects: 200, privateProjects: true }),
  },
  pro_150: {
    name: "Pro",
    shortName: "Pro · 150",
    priceGbp: 69,
    limits: makeLimits({ credits: 150, projects: 200, privateProjects: true }),
  },
  pro_500: {
    name: "Pro",
    shortName: "Pro · 500",
    priceGbp: 179,
    limits: makeLimits({ credits: 500, projects: 200, privateProjects: true }),
  },
  // Admin-granted only (no Stripe checkout). For internal users +
  // hand-picked partners. Effectively-unlimited limits.
  mega: {
    name: "Mega",
    shortName: "Mega",
    priceGbp: null,
    limits: makeLimits({ credits: 10_000, projects: 10_000, privateProjects: true }),
  },
  // Contact-sales tier. Limits are placeholders, actual quota is
  // negotiated per customer and applied via mega plus a separate
  // ledger entry.
  enterprise: {
    name: "Enterprise",
    shortName: "Enterprise",
    priceGbp: null,
    limits: makeLimits({ credits: 10_000, projects: 10_000, privateProjects: true }),
  },
};

// Credit cost table. The pricing page's usage table renders straight
// from this object so the doc and the gate can't disagree.
export const CREDIT_COSTS = {
  labelledImagesPerCredit: 100,
  uploadedImagesPerCredit: 800,
  storedImagesPerCreditPerMonth: 400,
} as const;

// Compute credits used in a billing window from raw counters. Used
// by both the profile page and the backend's usage endpoint so the
// display + the gate share the same arithmetic.
//
// Storage is rounded UP because a partial-month occupation still
// burns a credit-month, the user occupies that slot.
export function creditsUsed(opts: {
  labelledThisMonth: number;
  uploadedThisMonth: number;
  storedNow: number;
}): number {
  const labelled = opts.labelledThisMonth / CREDIT_COSTS.labelledImagesPerCredit;
  const uploaded = opts.uploadedThisMonth / CREDIT_COSTS.uploadedImagesPerCredit;
  const stored = Math.ceil(opts.storedNow / CREDIT_COSTS.storedImagesPerCreditPerMonth);
  return labelled + uploaded + stored;
}

// Resolve a stored plan-id string into the canonical PlanId enum.
// Tolerates the legacy "pro" string by mapping it to the entry tier
// so accounts created before the three-tier split don't lose their
// subscription.
export function planFor(planId: string | null | undefined): PlanId {
  if (planId === "mega") return "mega";
  if (planId === "enterprise") return "enterprise";
  if (planId === "pro_50") return "pro_50";
  if (planId === "pro_150") return "pro_150";
  if (planId === "pro_500") return "pro_500";
  if (planId === "pro") return "pro_50"; // legacy pre-split → entry tier
  if (planId === "beta") return "beta";
  return "free";
}

export function isProPlan(id: PlanId): id is ProTierId {
  return id === "pro_50" || id === "pro_150" || id === "pro_500";
}

export function limitsFor(planId: string | null | undefined): PlanLimits {
  return PLANS[planFor(planId)].limits;
}

// Per-user limit override. Resolves the plan's normal limits, then
// applies any account-level grants on top. Used by the /api/users/usage
// route so the FE displays the same allowance the backend honours.
//
// Founder / internal-account credit overrides (username.lower() → credits/mo).
// Mirror of the backend's _CREDIT_OVERRIDES in plans.py — keeping it on both
// sides means a deploy of either can't silently lose the override.
//
// Account overrides currently:
//   hamish - founder account (10,000 credits/month)
//   mukund - founder account (1,000,000 credits/month)
const CREDIT_OVERRIDES: Record<string, number> = {
  hamish: 10_000,
  mukund: 1_000_000,
};

export function limitsForUser(
  planId: string | null | undefined,
  username: string | null | undefined,
): PlanLimits {
  const base = limitsFor(planId);
  const credits = CREDIT_OVERRIDES[(username ?? "").trim().toLowerCase()];
  if (credits != null) {
    return {
      ...base,
      creditsPerMonth: credits,
      imagesLabelledPerMonth: credits * CREDIT_COSTS.labelledImagesPerCredit,
    };
  }
  return base;
}

export function planName(planId: string | null | undefined): string {
  return PLANS[planFor(planId)].name;
}

// Stripe statuses that count as a paid subscription. Everything else
// (past_due, unpaid, canceled, incomplete, incomplete_expired, paused)
// drops the user back to the free quota until they fix their payment.
const PAID_STATUSES = new Set(["active", "trialing"]);

// Compute the start + end of the current usage cycle for a user.
//
//   - Pro user with a known `subscriptionCurrentPeriodEnd`: the cycle
//     ends on that date (the day Stripe will charge them next), and
//     starts one month earlier. This way a Pro user who paid on the
//     15th sees their quota reset on the 15th, not on the 1st.
//   - Everyone else (Free, or a Pro user we don't yet have a period
//     end for): anchor on the day-of-month they signed up. The cycle
//     ends on the next signup-day-anniversary in the future and
//     starts one month before that.
//
// Returns ISO timestamps (truncated to the second) suitable for
// passing straight to the FastAPI usage endpoint.
export function cycleWindow(opts: {
  effectivePlan: PlanId;
  subscriptionCurrentPeriodEnd: Date | null;
  signupDate: Date;
  now?: Date;
}): { start: string; end: string } {
  const now = opts.now ?? new Date();
  let end: Date;
  if (isProPlan(opts.effectivePlan) && opts.subscriptionCurrentPeriodEnd) {
    end = new Date(opts.subscriptionCurrentPeriodEnd);
  } else {
    // Free, Beta, Mega and Enterprise all anchor on the signup
    // day-of-month, Beta doesn't need a Stripe-style sliding
    // window because its lifecycle is driven by `betaExpiresAt`
    // instead.
    end = nextAnchorDate(opts.signupDate, now);
  }
  const start = shiftMonths(end, -1);
  return {
    start: start.toISOString().replace(/\.\d{3}Z$/, "Z"),
    end: end.toISOString().replace(/\.\d{3}Z$/, "Z"),
  };
}

// Roll a date forward in 1-month steps until it's strictly after `now`.
// Handles month-length differences by clamping to the last day of the
// month (e.g. signed up Jan 31 → cycles end Feb 28/29, then Mar 31).
function nextAnchorDate(anchor: Date, now: Date): Date {
  let d = new Date(anchor.getTime());
  while (d <= now) {
    d = shiftMonths(d, 1);
  }
  return d;
}

function shiftMonths(d: Date, delta: number): Date {
  const day = d.getUTCDate();
  const target = new Date(Date.UTC(
    d.getUTCFullYear(),
    d.getUTCMonth() + delta,
    1,
    d.getUTCHours(),
    d.getUTCMinutes(),
    d.getUTCSeconds(),
    0,
  ));
  const lastDayOfTarget = new Date(Date.UTC(
    target.getUTCFullYear(),
    target.getUTCMonth() + 1,
    0,
  )).getUTCDate();
  target.setUTCDate(Math.min(day, lastDayOfTarget));
  return target;
}

// What the user is *effectively* on right now, taking subscription
// health into account. `subscriptionPlan` records what they paid for
// most recently; this function is what the rest of the app should
// read when deciding limits. A failed renewal therefore re-engages
// the free-tier quota gates without us touching the DB column.
//
// Beta is time-boxed via `betaExpiresAt` and sits *over* whatever
// plan the user already has (except Mega/Enterprise): a Pro user who
// redeems a beta code reads as Beta for 30 days, then drops back to
// Pro when the window lapses.
export function effectivePlanFor(
  planId: string | null | undefined,
  status: string | null | undefined,
  betaExpiresAt?: string | Date | null,
): PlanId {
  const stored = planFor(planId);

  // Mega + Enterprise beat every other tier, admin-granted, no clock.
  if (stored === "mega") return "mega";
  if (stored === "enterprise") return "enterprise";

  // Active beta window overrides the stored tier. A user's stored
  // plan during the window depends on the code they redeemed: the
  // backer/Pro codes set subscriptionPlan="pro_50" and read AS Pro,
  // the legacy beta code keeps subscriptionPlan="beta" and reads AS
  // Beta. After expiry the stored plan takes over again (Free for
  // code-only users; the real Stripe tier for paying users).
  if (betaExpiresAt) {
    const expiry = betaExpiresAt instanceof Date ? betaExpiresAt : new Date(betaExpiresAt);
    if (Number.isFinite(expiry.getTime()) && expiry.getTime() > Date.now()) {
      // Stored Pro tier (from a Pro-grade redeem code OR a Stripe
      // subscription) wins over "beta" because Pro is the strictly
      // higher tier — beta has identical limits to pro_50 but a
      // lower-rank label, and we don't want to downgrade a Pro
      // user's badge during an active beta window.
      if (isProPlan(stored)) return stored;
      return "beta";
    }
  }

  if (stored === "free") return "free";
  // A stored "beta" with a lapsed (or missing) timestamp drops to Free.
  if (stored === "beta") return "free";
  // Pro tiers need a healthy Stripe subscription.
  return status && PAID_STATUSES.has(status) ? stored : "free";
}
