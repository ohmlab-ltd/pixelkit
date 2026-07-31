"use client";

// Theme toggle button, sits to the left of the profile chip in the
// TopNav. Animates between two glyphs:
//   • Dark mode  → solid moon (filled circle with a crescent crop)
//   • Light mode → minimalist sun (filled circle + 8 short rays)
//
// The morph is driven entirely by CSS variables + transforms so the
// transition feels fluid (rays fan out, crescent retracts, fill
// rotates ~60°). No keyframe animation library needed.

import { useTheme } from "./ThemeProvider";

export function ThemeToggle({ className }: { className?: string }) {
  const { theme, toggle } = useTheme();
  const isDark = theme === "dark";

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className={[
        "relative h-9 w-9 rounded-full grid place-items-center",
        // Border tints the same colour as the glyph (foreground)
        // at low opacity, so dark mode shows a white ring around
        // the moon and light mode shows a black ring around the
        // sun, matching the icon's own tone instead of inverting it.
        "border border-foreground/20 hover:border-foreground/40",
        "text-[var(--foreground)]/85 hover:text-[var(--foreground)]",
        "bg-transparent hover:bg-foreground/[0.06]",
        "transition-colors",
        className ?? "",
      ].join(" ")}
    >
      <svg
        viewBox="0 0 24 24"
        width={18}
        height={18}
        aria-hidden
        // Full icon rotates softly between modes, dark settles
        // upright, light gets a quarter-turn so the rays land at the
        // cardinal angles when fully shown.
        style={{
          transform: isDark ? "rotate(0deg)" : "rotate(45deg)",
          transition: "transform 480ms cubic-bezier(0.34, 1.56, 0.64, 1)",
        }}
      >
        {/* Sun/moon body, a single filled circle. In dark mode a
            sibling circle masks out a crescent to make the moon. */}
        <defs>
          <mask id="moon-mask">
            {/* White = visible, black = hidden. The big white circle
                covers the full disc; the offset black circle carves
                the crescent. Mask shrinks to zero in light mode so
                the disc reveals as a full sun. */}
            <rect x="0" y="0" width="24" height="24" fill="white" />
            <circle
              cx={isDark ? 16 : 12}
              cy={isDark ? 8 : 12}
              r={isDark ? 7 : 0}
              fill="black"
              style={{
                transition:
                  "cx 480ms cubic-bezier(0.4, 0, 0.2, 1), cy 480ms cubic-bezier(0.4, 0, 0.2, 1), r 480ms cubic-bezier(0.4, 0, 0.2, 1)",
              }}
            />
          </mask>
        </defs>

        <circle
          cx="12"
          cy="12"
          r={isDark ? 8 : 4.5}
          fill="currentColor"
          mask="url(#moon-mask)"
          style={{
            transition: "r 480ms cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />

        {/* Eight short rays around the sun. Each ray fades in +
            extends from the circle in light mode and retracts +
            fades out in dark mode. Stroke widths are constant; the
            animated length comes from y1/y2. */}
        {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
          <line
            key={angle}
            x1="12"
            x2="12"
            // y1/y2 drive ray extension from centre (12,12). In dark
            // mode both endpoints sit on the centre (invisible);
            // in light mode they spread outward so the ray is a
            // short segment from r≈4 to r≈6.5 from centre. CSS
            // transitions on the wrapping <g> handle smoothing
            // where the engine supports SMIL-less attr animation.
            y1={isDark ? 12 : 4}
            y2={isDark ? 12 : 6.5}
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            transform={`rotate(${angle} 12 12)`}
            style={{
              opacity: isDark ? 0 : 1,
              transition: "opacity 320ms ease",
            }}
          />
        ))}
      </svg>
    </button>
  );
}
