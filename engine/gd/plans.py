"""Server-side mirror of the frontend's `lib/plans.ts`.

The credit ceiling for every AI-spending operation is enforced HERE,
not (only) on the frontend. The FE was treated as the single line of
defence for a while; any user who could send a bearer token directly
to FastAPI could bypass it. This module + `enforce_credits()` close
that gap.

Keep this file in lockstep with `frontend/lib/plans.ts`. The rules
(plan list, credit allowance per tier, credit costs per action,
effective-plan resolution, hamish founder override) must match exactly
so the FE display + the server gate never disagree.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Iterable, Literal, TypedDict


# ─── Plan identity ────────────────────────────────────────────────────────────

PlanId = Literal[
    "free", "beta", "pro_50", "pro_150", "pro_500", "mega", "enterprise",
]

PRO_TIER_IDS = ("pro_50", "pro_150", "pro_500")
# Stripe subscription statuses that count as paid. Anything else
# (past_due, unpaid, canceled, incomplete, incomplete_expired, paused)
# drops the user back to free until the payment recovers.
PAID_STATUSES = frozenset({"active", "trialing"})


def plan_for(plan_id: str | None) -> PlanId:
    """Resolve a stored plan-id string into the canonical PlanId.
    Tolerates the legacy "pro" string by mapping to the entry tier."""
    p = (plan_id or "").strip().lower()
    if p == "mega":
        return "mega"
    if p == "enterprise":
        return "enterprise"
    if p in PRO_TIER_IDS:
        return p  # type: ignore[return-value]
    if p == "pro":
        return "pro_50"  # legacy pre-split → entry tier
    if p == "beta":
        return "beta"
    return "free"


def is_pro_plan(plan: PlanId) -> bool:
    return plan in PRO_TIER_IDS


# ─── Credit costs ─────────────────────────────────────────────────────────────

# How many of each "thing" one credit covers per calendar cycle. Mirror
# of the FE's CREDIT_COSTS exactly.
LABELLED_IMAGES_PER_CREDIT = 100
UPLOADED_IMAGES_PER_CREDIT = 800
STORED_IMAGES_PER_CREDIT_PER_MONTH = 400


def credits_used(
    *,
    labelled_this_month: int,
    uploaded_this_month: int,
    stored_now: int,
    training_credits_this_period: float = 0.0,
) -> float:
    """Convert raw counters into credit-equivalents. Storage rounds
    UP because a partial-month occupation still burns a credit-month;
    the user is occupying that slot.

    `training_credits_this_period` is already in credit units (the ML
    job billing charges 1 credit per completed 15-min active-training
    block — see ml_jobs.BLOCK_SECONDS), so it adds in directly. Training
    therefore draws from the SAME monthly credit pool as labelling /
    uploads / storage, which is why it's summed here rather than gated
    separately."""
    labelled = labelled_this_month / LABELLED_IMAGES_PER_CREDIT
    uploaded = uploaded_this_month / UPLOADED_IMAGES_PER_CREDIT
    stored = math.ceil(stored_now / STORED_IMAGES_PER_CREDIT_PER_MONTH)
    return labelled + uploaded + stored + max(0.0, float(training_credits_this_period))


# ─── Plan limits ──────────────────────────────────────────────────────────────

class PlanLimits(TypedDict):
    creditsPerMonth: int
    imagesLabelledPerMonth: int
    projects: int
    allowsPrivateProjects: bool


def _build_limits(*, credits: int, projects: int, private_projects: bool) -> PlanLimits:
    return {
        "creditsPerMonth": credits,
        # Legacy flat cap derived from credits so older call sites
        # (FE display) keep getting a number, in lockstep with the FE.
        "imagesLabelledPerMonth": credits * LABELLED_IMAGES_PER_CREDIT,
        "projects": projects,
        "allowsPrivateProjects": private_projects,
    }


PLAN_LIMITS: dict[PlanId, PlanLimits] = {
    "free":       _build_limits(credits=5,      projects=5,      private_projects=False),
    "beta":       _build_limits(credits=50,     projects=200,    private_projects=True),
    "pro_50":     _build_limits(credits=50,     projects=200,    private_projects=True),
    "pro_150":    _build_limits(credits=150,    projects=200,    private_projects=True),
    "pro_500":    _build_limits(credits=500,    projects=200,    private_projects=True),
    "mega":       _build_limits(credits=10_000, projects=10_000, private_projects=True),
    "enterprise": _build_limits(credits=10_000, projects=10_000, private_projects=True),
}


def limits_for(plan_id: str | None) -> PlanLimits:
    return PLAN_LIMITS[plan_for(plan_id)]


# Founder / internal-account credit overrides (username.lower() → credits/mo).
# Mirror of the FE's limitsForUser. Keeping the override on the SERVER too
# means a future deploy of the FE can't lose it without the backend
# immediately re-capping the account.
_CREDIT_OVERRIDES: dict[str, int] = {
    "hamish": 10_000,
    "mukund": 1_000_000,
}


def limits_for_user(plan_id: str | None, username: str | None) -> PlanLimits:
    base = limits_for(plan_id)
    credits = _CREDIT_OVERRIDES.get((username or "").strip().lower())
    if credits is not None:
        return {
            **base,
            "creditsPerMonth": credits,
            "imagesLabelledPerMonth": credits * LABELLED_IMAGES_PER_CREDIT,
        }
    return base


# ─── Effective plan resolution ────────────────────────────────────────────────

def _parse_beta_expires(beta_expires_at: str | None) -> float | None:
    """ISO-8601 string (with or without ms / Z) → unix-seconds. Returns
    None if absent / unparseable so the caller can treat that as "no
    beta override"."""
    if not beta_expires_at:
        return None
    s = beta_expires_at.strip()
    if not s:
        return None
    # datetime.fromisoformat accepts the "...Z" suffix from Py 3.11+;
    # for older interpreters strip it explicitly to keep parity.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def effective_plan_for(
    plan_id: str | None,
    status: str | None,
    beta_expires_at: str | None = None,
) -> PlanId:
    """What the user is *effectively* on right now, taking subscription
    health into account. Mirror of the FE's effectivePlanFor.

      - Mega / Enterprise beat every other tier (admin-granted, no clock)
      - Active beta window overrides the stored tier
      - A stored Pro tier with a non-paid Stripe status drops to Free
      - A stored Beta with a lapsed/missing window drops to Free
    """
    stored = plan_for(plan_id)
    if stored == "mega":
        return "mega"
    if stored == "enterprise":
        return "enterprise"
    beta_exp = _parse_beta_expires(beta_expires_at)
    if beta_exp is not None and beta_exp > time.time():
        return "beta"
    if stored == "free":
        return "free"
    if stored == "beta":
        return "free"  # lapsed beta without active timestamp
    return stored if (status or "").lower() in PAID_STATUSES else "free"


# ─── Cycle window ─────────────────────────────────────────────────────────────

def _shift_months(d: datetime, delta: int) -> datetime:
    """Add `delta` months to `d`, clamping the day-of-month to the
    target month's length (e.g. Jan 31 + 1 → Feb 28/29)."""
    target_year = d.year + (d.month - 1 + delta) // 12
    target_month = (d.month - 1 + delta) % 12 + 1
    # Last day of target month.
    if target_month == 12:
        last = 31
    else:
        next_first = datetime(target_year, target_month + 1, 1, tzinfo=d.tzinfo)
        last = (next_first.replace(day=1) - _one_day()).day
    day = min(d.day, last)
    return d.replace(year=target_year, month=target_month, day=day)


def _one_day():
    from datetime import timedelta
    return timedelta(days=1)


def _next_anchor_date(anchor: datetime, now: datetime) -> datetime:
    d = anchor
    while d <= now:
        d = _shift_months(d, 1)
    return d


def cycle_window(
    *,
    effective_plan: PlanId,
    subscription_current_period_end: datetime | None,
    signup_date: datetime,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Resolve the current billing cycle's [start, end). Pro users
    anchor on subscriptionCurrentPeriodEnd (Stripe's authoritative
    next-charge time); everyone else anchors on their signup day-of-
    month. Same rules as the FE's cycleWindow."""
    now = now or datetime.now(timezone.utc)
    if is_pro_plan(effective_plan) and subscription_current_period_end is not None:
        end = subscription_current_period_end
    else:
        end = _next_anchor_date(signup_date, now)
    start = _shift_months(end, -1)
    return start, end


# ─── Convenience aggregation ──────────────────────────────────────────────────

class CreditState(TypedDict):
    plan: PlanId
    limits: PlanLimits
    credits_used: float
    credits_remaining: float
    over_credits: bool


def evaluate_credits(
    *,
    username: str | None,
    plan_id: str | None,
    status: str | None,
    beta_expires_at: str | None,
    labelled_this_month: int,
    uploaded_this_month: int,
    stored_now: int,
    training_blocks_this_period: int = 0,
) -> CreditState:
    """One-shot resolution: take the JWT claims + usage counters,
    return the effective plan + computed credit ledger + over-cap flag.
    The HTTP gate (`enforce_credits_or_402`) reads this to decide
    whether to allow an AI spend.

    `training_blocks_this_period` is the count of 15-min active-training
    blocks already billed this cycle (1 block = 1 credit)."""
    eff = effective_plan_for(plan_id, status, beta_expires_at)
    limits = limits_for_user(eff, username)
    used = credits_used(
        labelled_this_month=labelled_this_month,
        uploaded_this_month=uploaded_this_month,
        stored_now=stored_now,
        training_credits_this_period=float(training_blocks_this_period),
    )
    allowed = float(limits["creditsPerMonth"])
    return {
        "plan": eff,
        "limits": limits,
        "credits_used": used,
        "credits_remaining": max(0.0, allowed - used),
        "over_credits": allowed > 0 and used >= allowed,
    }


__all__: Iterable[str] = (
    "PlanId",
    "PRO_TIER_IDS",
    "PAID_STATUSES",
    "PlanLimits",
    "PLAN_LIMITS",
    "LABELLED_IMAGES_PER_CREDIT",
    "UPLOADED_IMAGES_PER_CREDIT",
    "STORED_IMAGES_PER_CREDIT_PER_MONTH",
    "plan_for",
    "is_pro_plan",
    "credits_used",
    "limits_for",
    "limits_for_user",
    "effective_plan_for",
    "cycle_window",
    "evaluate_credits",
    "CreditState",
)
