"use client";

import Image from "next/image";

import { ThemeToggle } from "./ThemeToggle";

export type NavTab = "workspaces" | "guide";

type Props = {
  current: NavTab;
  onNavigate: (tab: NavTab) => void;
  onProfile: () => void;
  onHome: () => void;
  user: { name: string; username?: string; email: string; image?: string | null };
};

export function TopNav({ current, onNavigate, onProfile, onHome, user }: Props) {
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
              <NavLink active={current === "workspaces"} onClick={() => onNavigate("workspaces")}>
                Workspace
              </NavLink>
              <NavLink active={current === "guide"} onClick={() => onNavigate("guide")}>
                Guide
              </NavLink>
            </nav>
          </div>

          {/* Spacer keeps the left and right groups pinned to the capsule
              edges now that the centre feedback slot is gone. */}
          <div className="flex-1" aria-hidden />

          <div className="flex items-center gap-3 shrink-0">
            <nav className="md:hidden flex items-center gap-1">
              <NavLink active={current === "workspaces"} onClick={() => onNavigate("workspaces")}>
                Workspace
              </NavLink>
              <NavLink active={current === "guide"} onClick={() => onNavigate("guide")}>
                Guide
              </NavLink>
            </nav>
            <ThemeToggle />
            <button
              onClick={onProfile}
              className="group flex items-center gap-2 rounded-full pl-1 pr-3 py-1 hover:bg-foreground/[0.04] transition-colors"
              title="Settings"
              aria-label="Open settings"
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
              <span className="hidden sm:inline text-sm">Settings</span>
            </button>
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
