"use client";

// Shared user avatar: renders the real profile picture when we have one,
// otherwise a stable hashed-gradient monogram. Falls back to the monogram on a
// real image load error too. `className` controls size / rounding / text size
// (e.g. "h-7 w-7 rounded-full text-[11px] font-bold").
import { useState } from "react";

function hueFor(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

export function Avatar({
  name,
  src,
  className = "",
}: {
  name: string;
  src?: string | null;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  if (src && !failed) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt=""
        referrerPolicy="no-referrer"
        onError={() => setFailed(true)}
        className={`${className} object-cover`}
      />
    );
  }
  const h = hueFor(name || "?");
  return (
    <span
      className={`${className} grid place-items-center text-white`}
      style={{ background: `linear-gradient(135deg, hsl(${h},72%,55%), hsl(${(h + 50) % 360},72%,55%))` }}
      aria-hidden
    >
      {(name || "?").charAt(0).toUpperCase()}
    </span>
  );
}
