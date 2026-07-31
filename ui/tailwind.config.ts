import type { Config } from "tailwindcss";

const config: Config = {
  // Class-based dark mode so the ThemeProvider can flip the `.dark`
  // class on <html> without depending on the OS preference.
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Themable colour tokens. Each maps to an RGB triplet
        // variable (`--x-rgb`) so Tailwind's alpha-value syntax can
        // splice opacity in without losing the theme. Use these via:
        //   bg-foreground, text-foreground/55, border-foreground/[0.04]
        background: "rgb(var(--background-rgb) / <alpha-value>)",
        foreground: "rgb(var(--foreground-rgb) / <alpha-value>)",
        muted: "rgb(var(--muted-rgb) / <alpha-value>)",
        border: "rgb(var(--border-rgb) / <alpha-value>)",
        accent: "rgb(var(--accent-rgb) / <alpha-value>)",
        surface: "rgb(var(--surface-rgb) / <alpha-value>)",
        "surface-2": "rgb(var(--surface-2-rgb) / <alpha-value>)",
      },
    },
  },
  plugins: [],
};
export default config;
