import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// `.mts` (ESM) because the repo is CommonJS (no "type":"module") and
// @vitejs/plugin-react is ESM-only — mirrors next.config.mjs.
//
// Two test environments under one config:
//   • node    — route handlers (app/api/**) and lib/* helpers (default)
//   • jsdom   — React components (opt in per file with
//               `// @vitest-environment jsdom` at the top)
// The `@/` alias mirrors tsconfig.json ("@/*": ["./*"]).
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "."),
    },
  },
  test: {
    environment: "node",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["app/**/*.test.{ts,tsx}", "lib/**/*.test.{ts,tsx}", "auth.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "text-summary"],
      include: ["app/**", "lib/**", "auth.ts"],
      // Scaffolding PR: no threshold gate yet. Later sessions ratchet
      // coverage thresholds up as the exhaustive suites land.
    },
  },
});
