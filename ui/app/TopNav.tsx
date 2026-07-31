"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { useSession } from "next-auth/react";

import { effectivePlanFor, isProPlan, planName as planNameFor } from "@/lib/plans";
import { usePlan } from "./PlanPill";
import { ThemeToggle } from "./ThemeToggle";

export type NavTab = "workspaces" | "projects" | "guide" | "pricing" | "terminal";

type Props = {
  current: NavTab;
  onNavigate: (tab: NavTab) => void;
  onProfile: () => void;
  onHome: () => void;
  user: { name: string; username?: string; email: string; image?: string | null };
  /** When false the right side shows Login + Sign up buttons instead of the
      profile chip. */
  loggedIn?: boolean;
  /** Only the admin sees the Terminal link. The page itself is also gated by
      a token, but hiding the link keeps it out of plain sight. */
  showTerminal?: boolean;
};

export function TopNav({ current, onNavigate, onProfile, onHome, user, loggedIn = true, showTerminal = false }: Props) {
  const plan = usePlan();
  const { data: session } = useSession();
  // Fall back to the plan baked into the JWT when the /api/users/usage
  // round-trip hasn't returned yet. Without this the plan chip pops in
  // 500–1000 ms after the rest of the user card on every refresh.
  const sessionPlan = effectivePlanFor(
    session?.user?.subscriptionPlan ?? null,
    session?.user?.subscriptionStatus ?? null,
    session?.user?.betaExpiresAt ?? null,
  );
  const planForChip = plan
    ? { plan: plan.plan, name: plan.planName }
    : loggedIn
    ? { plan: sessionPlan, name: planNameFor(sessionPlan) }
    : null;
  const initials = user.name
    .split(/\s+/)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .filter(Boolean)
    .slice(0, 2)
    .join("") || (user.email[0]?.toUpperCase() ?? "?");

  return (
    // Floating capsule: sticks under the viewport top with a small offset and
    // is inset from the side edges. Same translucent backdrop as before.
    <header className="sticky top-3 z-30 mx-3 md:mx-6 mt-3">
      <div className="mx-auto max-w-7xl rounded-full ring-1 backdrop-blur-md bg-[rgb(var(--background-rgb)/0.72)] text-[var(--foreground)] ring-foreground/[0.08] shadow-[0_8px_24px_-12px_rgb(var(--shadow-rgb)/0.45)]">
        <div className="px-5 sm:px-7 h-14 flex items-center gap-6">
          <div className="flex items-center gap-6 min-w-0 shrink-0">
            <button onClick={onHome} className="flex items-center gap-3 shrink-0" aria-label="Home">
              {/* Both logo variants render. CSS picks the right one
                  from the `.dark` class set pre-hydration in
                  layout.tsx, so the logo can't lag the theme. */}
              <Image
                src="/pixelkit2.svg"
                alt="PixelKit"
                width={120}
                height={24}
                priority
                className="h-6 w-auto hidden dark:block"
              />
              <Image
                src="/pixelkit-light.svg"
                alt="PixelKit"
                width={120}
                height={24}
                priority
                className="h-6 w-auto block dark:hidden"
              />
            </button>
            <nav className="hidden md:flex items-center gap-1 ml-2">
              {loggedIn && (
                <NavLink active={current === "workspaces"} onClick={() => onNavigate("workspaces")}>
                  Workspace
                </NavLink>
              )}
              <NavLink active={current === "projects"} onClick={() => onNavigate("projects")}>
                Community
              </NavLink>
              <NavLink active={current === "guide"} onClick={() => onNavigate("guide")}>
                Guide
              </NavLink>
              <NavLink active={current === "pricing"} onClick={() => onNavigate("pricing")}>
                Pricing
              </NavLink>
            </nav>
          </div>

          {/* Centred feedback bar, shown to anyone with an active
              redeemed code — beta testers (effective plan "beta") AND
              backers (NEURON6-BACKER-PK redeems as Pro tier but keeps
              the same betaExpiresAt clock). Free / paying Pro / Mega
              users see an empty slot in its place so the left and
              right groups keep the same horizontal alignment. Hidden
              on small screens regardless to keep the capsule readable. */}
          {loggedIn && (
            plan?.plan === "beta"
            || (plan?.betaExpiresAt && new Date(plan.betaExpiresAt).getTime() > Date.now())
          ) ? (
            <div className="hidden lg:flex flex-1 justify-center px-4">
              <FeedbackInput user={user} loggedIn={loggedIn} />
            </div>
          ) : (
            <div className="hidden lg:block flex-1" aria-hidden />
          )}

          <div className="flex items-center gap-3 shrink-0">
            <nav className="md:hidden flex items-center gap-1">
              {loggedIn && (
                <NavLink active={current === "workspaces"} onClick={() => onNavigate("workspaces")}>
                  Workspace
                </NavLink>
              )}
              <NavLink active={current === "projects"} onClick={() => onNavigate("projects")}>
                Community
              </NavLink>
              <NavLink active={current === "guide"} onClick={() => onNavigate("guide")}>
                Guide
              </NavLink>
              <NavLink active={current === "pricing"} onClick={() => onNavigate("pricing")}>
                Pricing
              </NavLink>
            </nav>
            <ThemeToggle />
            {showTerminal && (
              <button
                onClick={() => onNavigate("terminal")}
                className={[
                  "px-3 py-1.5 rounded-full text-sm font-medium transition-colors border",
                  current === "terminal"
                    ? "bg-orange-500 text-black border-orange-400 shadow-[0_0_14px_2px_rgba(249,115,22,0.55)]"
                    : "text-orange-300 border-orange-400/50 hover:bg-orange-500/15 hover:text-orange-100",
                ].join(" ")}
                title="Operator terminal"
              >
                Terminal
              </button>
            )}
            {loggedIn ? (
              <button
                onClick={onProfile}
                className="group flex items-center gap-2 rounded-full pl-1 pr-3 py-1 hover:bg-foreground/[0.04] transition-colors"
                title={`Signed in as ${user.email}`}
                aria-label="Open profile"
              >
                {user.image ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={user.image}
                    alt={user.name}
                    className="h-8 w-8 rounded-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                ) : (
                  <span
                    className="h-8 w-8 rounded-full grid place-items-center text-[11px] font-semibold tracking-wide text-[var(--foreground)]"
                    style={{
                      backgroundImage: "linear-gradient(135deg, #6366f1 0%, #ec4899 60%, #f59e0b 100%)",
                    }}
                  >
                    {initials}
                  </span>
                )}
                <span className="hidden sm:flex flex-col items-start leading-tight">
                  <span className="text-sm">{user.name}</span>
                  {planForChip && (
                    <span
                      className={[
                        // Stronger contrast for plan label: deeper
                        // orange/foreground tones in light mode so
                        // the subscription tier sits legibly under
                        // the username on a near-white nav pill.
                        "text-[9px] uppercase tracking-[0.18em] font-mono",
                        planForChip.plan === "mega"
                          ? "text-orange-600 dark:text-orange-300"
                          : isProPlan(planForChip.plan)
                          ? "text-orange-700 dark:text-orange-200/90"
                          : planForChip.plan === "beta"
                          ? "text-sky-700 dark:text-sky-200/90"
                          : "text-foreground/65",
                      ].join(" ")}
                      style={
                        planForChip.plan === "mega"
                          ? {
                              textShadow:
                                "0 0 6px rgba(249,180,80,0.95), 0 0 14px rgba(249,115,22,0.7)",
                            }
                          : undefined
                      }
                    >
                      {planForChip.name}
                    </span>
                  )}
                </span>
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <Link
                  href="/login"
                  className="text-sm px-3 py-1.5 rounded-full text-[var(--muted)] hover:text-foreground hover:bg-foreground/[0.04] transition-colors"
                >
                  Login
                </Link>
                <Link
                  href="/signup"
                  className="text-sm px-3 py-1.5 rounded-full bg-foreground text-background hover:opacity-90 transition-opacity"
                >
                  Sign up
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

function NavLink({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "px-3 py-1.5 rounded-full text-sm transition-colors",
        // Active tab pill inverts on theme: black-on-white in light
        // mode, white-on-black in dark. Using the themable
        // foreground/background pair keeps both modes legible.
        active
          ? "bg-foreground text-background"
          : "text-[var(--muted)] hover:text-foreground hover:bg-foreground/[0.04]",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

function FeedbackInput({
  user,
  loggedIn,
}: {
  user: { username?: string; email: string };
  loggedIn: boolean;
}) {
  const [value, setValue] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    const message = value.trim();
    if (!message || state === "sending") return;
    setState("sending");
    setError(null);
    try {
      const r = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          username: loggedIn ? user.username ?? null : null,
          email: loggedIn ? user.email ?? null : null,
        }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body?.error || `http ${r.status}`);
      }
      setState("sent");
      setValue("");
      // Auto-clear the "Thanks!" state after a few seconds so the input
      // is ready for the next note.
      window.setTimeout(() => setState("idle"), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("error");
      window.setTimeout(() => setState("idle"), 3500);
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="w-full max-w-md flex items-center gap-2 rounded-full bg-foreground/[0.04] border border-foreground/10 hover:border-foreground/20 focus-within:border-foreground/30 transition-colors pl-3 pr-1 py-1"
    >
      <svg
        viewBox="0 0 24 24"
        className="h-4 w-4 text-foreground/40 shrink-0"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 2v4" />
        <path d="m6.34 6.34-2.83-2.83" />
        <path d="m17.66 6.34 2.83-2.83" />
        <path d="M4 12H2" />
        <path d="M22 12h-2" />
        <path d="M12 18a4 4 0 0 0 4-4H8a4 4 0 0 0 4 4Z" />
        <path d="M12 22v-4" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        maxLength={500}
        placeholder={
          state === "sent"
            ? "Thanks! Got your note."
            : state === "error"
            ? `Couldn't send${error ? `: ${error}` : ""}`
            : "Found a bug or idea? Drop a note for the team…"
        }
        disabled={state === "sending"}
        className={[
          "flex-1 bg-transparent outline-none text-sm py-1 placeholder:text-foreground/40",
          state === "sent" ? "text-emerald-300 placeholder:text-emerald-300/80" : "",
          state === "error" ? "text-red-300 placeholder:text-red-300/80" : "",
        ].join(" ")}
      />
      <button
        type="submit"
        disabled={!value.trim() || state === "sending"}
        className={[
          "rounded-full px-3 py-1 text-xs font-medium transition-colors shrink-0",
          value.trim() && state !== "sending"
            ? "bg-foreground text-background hover:bg-zinc-200"
            : "bg-foreground/10 text-foreground/40 cursor-not-allowed",
        ].join(" ")}
        aria-label="Send feedback"
      >
        {state === "sending" ? "Sending…" : "Send"}
      </button>
    </form>
  );
}
