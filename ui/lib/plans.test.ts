import { describe, expect, it } from "vitest";

import {
  creditsUsed,
  effectivePlanFor,
  isProPlan,
  limitsFor,
  limitsForUser,
  planFor,
  planName,
} from "./plans";

// Pure plan/authorisation logic. effectivePlanFor() is the gate the rest of
// the app reads to decide a user's limits — a bug here means unpaid Pro
// access or paying users dropped to Free, so it's worth pinning hard.

describe("planFor (stored string -> canonical PlanId)", () => {
  it("passes known tiers through", () => {
    expect(planFor("mega")).toBe("mega");
    expect(planFor("enterprise")).toBe("enterprise");
    expect(planFor("pro_50")).toBe("pro_50");
    expect(planFor("pro_150")).toBe("pro_150");
    expect(planFor("pro_500")).toBe("pro_500");
    expect(planFor("beta")).toBe("beta");
  });

  it("maps the legacy 'pro' alias to the entry tier", () => {
    expect(planFor("pro")).toBe("pro_50");
  });

  it("defaults unknown / null / undefined to free", () => {
    expect(planFor(null)).toBe("free");
    expect(planFor(undefined)).toBe("free");
    expect(planFor("")).toBe("free");
    expect(planFor("totally-made-up")).toBe("free");
  });
});

describe("isProPlan", () => {
  it("is true only for the three pro tiers", () => {
    expect(isProPlan("pro_50")).toBe(true);
    expect(isProPlan("pro_150")).toBe(true);
    expect(isProPlan("pro_500")).toBe(true);
    expect(isProPlan("free")).toBe(false);
    expect(isProPlan("beta")).toBe(false);
    expect(isProPlan("mega")).toBe(false);
    expect(isProPlan("enterprise")).toBe(false);
  });
});

describe("effectivePlanFor (the authorisation gate)", () => {
  const future = new Date(Date.now() + 60_000);
  const past = new Date(Date.now() - 60_000);

  it("lets mega / enterprise beat every status and clock", () => {
    expect(effectivePlanFor("mega", null)).toBe("mega");
    expect(effectivePlanFor("mega", "past_due")).toBe("mega");
    expect(effectivePlanFor("enterprise", null)).toBe("enterprise");
  });

  it("applies an active beta window, with stored Pro winning over beta", () => {
    expect(effectivePlanFor("pro_150", "active", future)).toBe("pro_150");
    expect(effectivePlanFor("free", null, future)).toBe("beta");
    expect(effectivePlanFor("beta", null, future)).toBe("beta");
    // string ISO timestamps are accepted too
    expect(effectivePlanFor("free", null, future.toISOString())).toBe("beta");
  });

  it("drops to the stored tier once the beta window has lapsed", () => {
    expect(effectivePlanFor("free", null, past)).toBe("free");
    expect(effectivePlanFor("beta", null, past)).toBe("free");
    expect(effectivePlanFor("beta", null)).toBe("free");
    expect(effectivePlanFor("pro_50", "active", past)).toBe("pro_50");
    expect(effectivePlanFor("pro_50", "past_due", past)).toBe("free");
  });

  it("requires a healthy Stripe status for pro tiers (no beta window)", () => {
    expect(effectivePlanFor("pro_500", "active")).toBe("pro_500");
    expect(effectivePlanFor("pro_500", "trialing")).toBe("pro_500");
    expect(effectivePlanFor("pro_500", "past_due")).toBe("free");
    expect(effectivePlanFor("pro_500", "canceled")).toBe("free");
    expect(effectivePlanFor("pro_500", null)).toBe("free");
  });

  it("ignores an unparseable beta timestamp", () => {
    expect(effectivePlanFor("pro_50", "active", "not-a-date")).toBe("pro_50");
  });
});

describe("limitsForUser (founder credit overrides)", () => {
  it("applies the founder override regardless of stored plan", () => {
    expect(limitsForUser("free", "hamish").creditsPerMonth).toBe(10_000);
    expect(limitsForUser("free", "mukund").creditsPerMonth).toBe(1_000_000);
  });

  it("matches the override case-insensitively and trims", () => {
    expect(limitsForUser("free", "HAMISH").creditsPerMonth).toBe(10_000);
    expect(limitsForUser("free", "  mukund  ").creditsPerMonth).toBe(1_000_000);
  });

  it("derives imagesLabelledPerMonth from the overridden credits", () => {
    expect(limitsForUser("free", "hamish").imagesLabelledPerMonth).toBe(10_000 * 100);
  });

  it("falls back to the plan's base limits for everyone else", () => {
    expect(limitsForUser("free", "randomuser").creditsPerMonth).toBe(
      limitsFor("free").creditsPerMonth,
    );
    expect(limitsForUser("pro_50", null).creditsPerMonth).toBe(50);
  });
});

describe("creditsUsed", () => {
  it("sums labelled/uploaded fractions and rounds storage up", () => {
    expect(creditsUsed({ labelledThisMonth: 100, uploadedThisMonth: 800, storedNow: 400 })).toBe(3);
    expect(creditsUsed({ labelledThisMonth: 0, uploadedThisMonth: 0, storedNow: 1 })).toBe(1);
    expect(creditsUsed({ labelledThisMonth: 50, uploadedThisMonth: 0, storedNow: 0 })).toBe(0.5);
  });
});

describe("planName", () => {
  it("returns the display name for the resolved tier", () => {
    expect(planName("pro_150")).toBe("Pro");
    expect(planName("pro")).toBe("Pro"); // legacy alias
    expect(planName("mega")).toBe("Mega");
    expect(planName(null)).toBe("Free");
  });
});
