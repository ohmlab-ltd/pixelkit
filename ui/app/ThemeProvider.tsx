"use client";

// Site-wide light/dark theme management.
//
//   • Default is dark (the brand look), new visitors land on dark
//     mode with no flash. Once they toggle into light it persists
//     until they toggle back.
//   • The current theme is stored in `localStorage("pixelkit-theme")`
//     so a user's preference survives navigation and reloads.
//   • The `.dark` class on <html> is the single source of truth for
//     all Tailwind `dark:` variants + CSS custom properties in
//     globals.css. We mutate it imperatively here so the page chrome
//     responds the instant the toggle fires (React rerender plus a
//     CSS variable swap with a 240 ms ease, see globals.css).
//   • A small script in <head> sets the class BEFORE React hydrates
//     to avoid a flash-of-incorrect-theme on first paint. See
//     layout.tsx for the matching inline <script>.

import { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";

export type Theme = "light" | "dark";

const STORAGE_KEY = "pixelkit-theme";

type Ctx = {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggle: () => void;
};

const ThemeContext = createContext<Ctx | null>(null);

// Read the persisted theme without throwing in SSR. The matching
// inline script in layout.tsx applies the class for first paint ,
// this hook then mirrors that into React state on hydration.
function readStored(): Theme {
  if (typeof window === "undefined") return "dark";
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    if (v === "light" || v === "dark") return v;
  } catch { /* localStorage blocked, fall through */ }
  return "dark";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // Initialise from localStorage on the client; default to dark on
  // the server so the SSR markup matches the inline-script-applied
  // class on first paint.
  const [theme, setThemeState] = useState<Theme>(() => readStored());

  // Keep <html class> + localStorage in sync whenever the theme
  // changes. Idempotent: re-applying the same class is cheap.
  //
  // While the swap is happening we stamp `.theme-changing` onto
  // <html> for ~260 ms. globals.css uses that class to force every
  // element onto the same 220 ms cubic-bezier transition so all
  // text / borders / backgrounds cross-fade together, without it
  // each component would run its own hover/transition-colors clock
  // and the swap would feel staggered.
  const initialMountRef = useRef(true);
  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    const apply = () => {
      if (theme === "dark") root.classList.add("dark");
      else root.classList.remove("dark");
      root.style.colorScheme = theme;
    };
    if (initialMountRef.current) {
      // First render, class is already applied by the inline
      // script in <head>, no transition needed. Mirror state then
      // bail.
      apply();
      initialMountRef.current = false;
    } else {
      root.classList.add("theme-changing");
      // Flip the class inside a requestAnimationFrame so the
      // browser has a chance to acknowledge `.theme-changing` as
      // the starting style before the colours actually change.
      window.requestAnimationFrame(() => {
        apply();
        window.setTimeout(() => {
          root.classList.remove("theme-changing");
        }, 260);
      });
    }
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch { /* ignore, preference just won't persist */ }
  }, [theme]);

  // Cross-tab sync, if the user toggles in one tab, mirror to others.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const onStorage = (e: StorageEvent) => {
      if (e.key !== STORAGE_KEY) return;
      const next = e.newValue;
      if (next === "light" || next === "dark") setThemeState(next);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const setTheme = useCallback((t: Theme) => setThemeState(t), []);
  const toggle = useCallback(
    () => setThemeState((cur) => (cur === "dark" ? "light" : "dark")),
    [],
  );

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): Ctx {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    // Render-time fallback so components that mount briefly outside
    // the provider (storybook, isolated tests) don't crash. The toggle
    // still works on the live app where the provider is mounted.
    return {
      theme: "dark",
      setTheme: () => {},
      toggle: () => {},
    };
  }
  return ctx;
}
