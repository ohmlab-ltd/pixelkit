"use client";

import Image from "next/image";

// Portable build footer: logo, a GitHub link, and the copyright line.
// The SaaS footer's marketing / policies / contact links pointed at
// routes that don't exist in this build.
export function Footer() {
  return (
    <footer className="border-t border-[var(--border)] mt-16">
      <div className="mx-auto max-w-6xl px-6 py-10 flex flex-wrap items-center justify-between gap-6">
        {/* Both logo variants render; CSS hides the wrong one based
            on the `.dark` class. The inline script in layout.tsx
            applies that class BEFORE React hydrates, so the right
            logo shows from first paint. */}
        <div style={{ width: 120, height: 24 }} className="relative">
          <Image
            src="/pixelkit2.svg"
            alt="PixelKit"
            fill
            sizes="120px"
            style={{ objectFit: "contain", objectPosition: "left" }}
            className="hidden dark:block"
          />
          <Image
            src="/pixelkit-light.svg"
            alt="PixelKit"
            fill
            sizes="120px"
            style={{ objectFit: "contain", objectPosition: "left" }}
            className="block dark:hidden"
          />
        </div>
        <a
          href="https://github.com/ohmlab-ltd/pixelkit"
          target="_blank"
          rel="noreferrer"
          className="text-sm text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
        >
          GitHub
        </a>
      </div>

      <div className="border-t border-[var(--border)]">
        <div className="mx-auto max-w-6xl px-6 py-5 text-xs text-[var(--muted)] flex flex-wrap items-center justify-between gap-2">
          <span>
            © 2026 Ohm Lab Ltd · Company no. 16699752 · Registered in England &amp; Wales
          </span>
          <span className="font-mono">pixelkit.ai</span>
        </div>
      </div>
    </footer>
  );
}
