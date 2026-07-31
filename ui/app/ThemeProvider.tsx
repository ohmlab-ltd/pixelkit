"use client";

// Site-wide theme management — permanently DARK in this build.
//
//   • The `.dark` class on <html> is still the single source of truth
//     for all Tailwind `dark:` variants + CSS custom properties in
//     globals.css; that mechanism is untouched so nothing downstream
//     breaks. Only the choice is pinned: the provider always applies
//     dark, `setTheme`/`toggle` are inert no-ops, and any stored
//     'pixelkit-theme' preference from older builds is ignored.
//   • A small script in <head> (layout.tsx) adds the class BEFORE
//     React hydrates so the first paint is dark with no flash.

import { createContext, useContext, useEffect } from "react";

export type Theme = "light" | "dark";

type Ctx = {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggle: () => void;
};

const ThemeContext = createContext<Ctx | null>(null);

const PINNED: Ctx = {
  theme: "dark",
  setTheme: () => {},
  toggle: () => {},
};

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Re-assert dark on mount. The layout's inline script already did
  // this pre-paint; this covers isolated mounts (tests, storybook)
  // where that script never ran. Idempotent and cheap.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    root.classList.add("dark");
    root.style.colorScheme = "dark";
  }, []);

  return (
    <ThemeContext.Provider value={PINNED}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): Ctx {
  // Dark everywhere — components outside the provider get the same
  // pinned context, so nothing can flip the theme.
  return useContext(ThemeContext) ?? PINNED;
}
