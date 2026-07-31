"use client";

import Image from "next/image";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

// Pixel Kit's editor needs precision input (drag-handles, scroll, multiselect)
// that doesn't translate well to a phone. Block phones inside /app only ,
// the marketing pages render fine on a phone and we don't want to scare
// away first-time visitors before they've seen what the product is.
export function MobileBlock() {
  const [isPhone, setIsPhone] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    if (typeof navigator === "undefined") return;
    const ua = navigator.userAgent || "";
    // iPhone / iPod block unconditionally. Android UA covers both phones
    // and tablets, so gate Android by viewport width (<768px) to leave
    // tablets alone. iPads (which report "Macintosh" on iPadOS 13+) and
    // desktops are unaffected either way.
    const isiPhone = /iPhone|iPod/i.test(ua);
    const isAndroidPhone = /Android/i.test(ua) && window.innerWidth < 768;
    setIsPhone(isiPhone || isAndroidPhone);
  }, []);

  // Only block inside the app, landing, login, signup, pricing all render
  // fine on a phone.
  const inApp = pathname?.startsWith("/app") ?? false;
  if (!isPhone || !inApp) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center px-6 bg-[var(--background)] text-[var(--foreground)]"
      role="dialog"
      aria-modal="true"
    >
      <div className="max-w-sm w-full text-center rounded-3xl border border-[var(--border)] bg-[var(--surface-2)] px-6 py-8">
        <div className="flex justify-center mb-5">
          {/* Both variants render; CSS picks the right one from the
              `.dark` class set by layout.tsx's pre-hydration script. */}
          <Image src="/pixelkit2.svg" alt="Pixel Kit" width={132} height={26} priority className="hidden dark:block" />
          <Image src="/pixelkit-light.svg" alt="Pixel Kit" width={132} height={26} priority className="block dark:hidden" />
        </div>
        <div className="text-xs uppercase tracking-wider text-[var(--muted)] mb-2">Desktop required</div>
        <h2 className="text-2xl font-light tracking-tight leading-snug">
          Please switch to a desktop to build models using Pixel Kit
        </h2>
      </div>
    </div>
  );
}
