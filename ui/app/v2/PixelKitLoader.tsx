"use client";

// Pixelkit loading indicator. The "P" body is static; the 10 small
// particle squares scattered around it pulse + drift outward and back
// like dust disintegrating off the letterform. Used wherever we're
// waiting on Claude (dataset-type preview, label expansion, etc).
//
// All animation is pure CSS, no JS animation loop, no requestAnimationFrame
//, so the loader is cheap to mount/unmount.

type ParticleRect = {
  x: number;
  y: number;
  w: number;
  h: number;
  delay: number;   // stagger so particles don't pulse in unison
  vx: number;      // outward drift unit-vector x (away from P centre)
  vy: number;      // outward drift unit-vector y
};

// Centre of the P (rough centroid of the static "P" rectangles).
const CX = 170;
const CY = 245;

// Helper, given a rectangle's (x, y, w, h) compute the unit drift
// vector pointing from the P's centre to the rectangle's centre, so
// the particle moves AWAY from the P during its pulse.
function buildParticle(x: number, y: number, w: number, h: number, delay: number): ParticleRect {
  const px = x + w / 2;
  const py = y + h / 2;
  const dx = px - CX;
  const dy = py - CY;
  const len = Math.max(1, Math.hypot(dx, dy));
  return { x, y, w, h, delay, vx: dx / len, vy: dy / len };
}

// Particles taken from pixelkit-favicon.svg (the 10 small floating
// squares, NOT the 6 rectangles that make up the P body itself).
const PARTICLES: ParticleRect[] = [
  buildParticle(283.03, 157.35, 40.54, 40.54, 0.0),
  buildParticle(323.57, 197.89, 37.84, 37.84, 0.15),
  buildParticle(285.73, 270.86, 45.95, 45.95, 0.30),
  buildParticle(253.30, 346.54, 37.84, 37.84, 0.45),
  buildParticle(331.68, 335.73, 29.73, 29.73, 0.60),
  buildParticle(374.92, 262.76, 24.32, 24.32, 0.75),
  buildParticle(369.51, 162.76, 29.73, 29.73, 0.90),
  buildParticle(431.68, 57.35, 24.32, 24.32, 1.05),
  buildParticle(404.65, 384.38, 21.62, 21.62, 1.20),
  buildParticle(339.78, 424.92, 24.32, 24.32, 1.35),
];

export function PixelKitLoader({
  size = 96,
  message,
}: {
  size?: number;
  message?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-3">
      <svg
        width={size}
        height={size}
        viewBox="0 0 512 512"
        aria-label="Loading"
        role="img"
        style={{ overflow: "visible" }}
      >
        {/* Static P, exactly the same six rectangles as the favicon. */}
        <g fill="#ff7900">
          <rect x={56} y={119.51} width={75.68} height={335.14} />
          <rect x={56} y={119.51} width={227.03} height={75.68} />
          <rect x={207.35} y={119.51} width={75.68} height={75.68} />
          <rect x={207.35} y={195.19} width={75.68} height={75.68} />
          <rect x={131.68} y={270.86} width={75.68} height={75.68} />
          <rect x={56} y={270.86} width={151.35} height={75.68} />
        </g>

        {/* Animated particle squares, each one drifts outward + fades
            on a 1.8 s cycle, staggered by `delay` so the cloud reads
            as a continuous shimmer rather than a single pulse. */}
        <g fill="#ff7900">
          {PARTICLES.map((p, i) => (
            <rect
              key={i}
              x={p.x}
              y={p.y}
              width={p.w}
              height={p.h}
              style={{
                transformOrigin: `${p.x + p.w / 2}px ${p.y + p.h / 2}px`,
                animation: `pkx-particle-${i} 1.8s ease-in-out ${p.delay}s infinite`,
              }}
            />
          ))}
        </g>

        {/* Per-particle keyframes embedded in <style> so the unique
            translate vector lands inside the SVG without polluting
            the global CSS. Each keyframe pulses the particle outward
            along its drift unit-vector by ~32 px and fades it. */}
        <style>{PARTICLES.map((p, i) => keyframesFor(i, p)).join("\n")}</style>
      </svg>
      {message ? (
        <p className="text-[12px] tracking-wider uppercase font-mono text-foreground/55 text-center">
          {message}
        </p>
      ) : null}
    </div>
  );
}

function keyframesFor(idx: number, p: ParticleRect): string {
  const drift = 32; // px outward from the P centre at the peak
  const dx = (p.vx * drift).toFixed(2);
  const dy = (p.vy * drift).toFixed(2);
  return `
    @keyframes pkx-particle-${idx} {
      0%, 100% {
        transform: translate(0, 0) scale(1);
        opacity: 1;
      }
      50% {
        transform: translate(${dx}px, ${dy}px) scale(0.6);
        opacity: 0.35;
      }
    }
  `;
}
