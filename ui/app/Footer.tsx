"use client";

import Image from "next/image";
import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-[var(--border)] mt-16">
      <div className="mx-auto max-w-6xl px-6 py-12 grid gap-10 md:grid-cols-12">
        <div className="md:col-span-4">
          {/* Both logo variants render; CSS hides the wrong one based
              on the `.dark` class. The inline script in layout.tsx
              applies that class BEFORE React hydrates, so the right
              logo shows from first paint; no flash where a dark page
              briefly displays the light logo (or vice versa) while
              the ThemeProvider's state catches up to localStorage. */}
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
          <p className="mt-3 text-sm text-[var(--muted)] leading-snug">
            Build, train, and deploy vision AI faster.
          </p>
          <div className="mt-2 flex gap-3 text-[var(--muted)]">
            <a
              href="https://www.instagram.com/ohmlab.co.uk"
              target="_blank"
              rel="noreferrer"
              aria-label="Instagram"
              className="hover:text-[var(--foreground)] transition-colors"
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6">
                <rect x="3" y="3" width="18" height="18" rx="5" ry="5" />
                <circle cx="12" cy="12" r="4" />
                <circle cx="17.5" cy="6.5" r="0.7" fill="currentColor" />
              </svg>
            </a>
            <a
              href="https://www.tiktok.com/@ohmlab.co.uk"
              target="_blank"
              rel="noreferrer"
              aria-label="TikTok"
              className="hover:text-[var(--foreground)] transition-colors"
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor">
                <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5.8 20.1a6.34 6.34 0 0 0 10.86-4.43V8.69a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1.84-.12z" />
              </svg>
            </a>
            <a
              href="https://www.youtube.com/@OhmLab-ltd"
              target="_blank"
              rel="noreferrer"
              aria-label="YouTube"
              className="hover:text-[var(--foreground)] transition-colors"
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
              </svg>
            </a>
          </div>
        </div>

        <div className="md:col-span-5">
          <div className="text-xs uppercase tracking-wider text-[var(--muted)] mb-3">PixelKit</div>
          <ul className="space-y-2 text-sm">
            <li>
              <a href="https://www.pixelkit.ai/" target="_blank" rel="noreferrer" className="hover:underline underline-offset-4">
                Home
              </a>
            </li>
            <li>
              <a href="https://ohmlab.co.uk" target="_blank" rel="noreferrer" className="hover:underline underline-offset-4">
                About
              </a>
            </li>
            <li>
              <Link href="/policies" className="hover:underline underline-offset-4">
                Policies
              </Link>
            </li>
          </ul>
        </div>

        <div className="md:col-span-3">
          <div className="text-xs uppercase tracking-wider text-[var(--muted)] mb-3">Contact</div>
          <ul className="space-y-1.5 text-sm text-[var(--muted)]">
            <li>
              <a href="mailto:info@ohmlab.co.uk" className="hover:text-foreground transition-colors">
                info@ohmlab.co.uk
              </a>
            </li>
            <li>84 Lakewood Road</li>
            <li>Chandler&apos;s Ford</li>
            <li>SO53 5AA</li>
            <li>United Kingdom</li>
          </ul>
        </div>
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
