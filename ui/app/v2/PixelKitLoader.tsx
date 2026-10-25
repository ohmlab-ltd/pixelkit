"use client";

// Loading indicator: the favicon "P" with ten particles pulsing outward.

type ParticleRect = {
  x: number;
  y: number;
  w: number;
  h: number;
  delay: number;   // s
  vx: number;      // unit drift vector, away from the P centre
  vy: number;
};

// Approximate centre of the P, in viewBox units.
const CX = 170;
const CY = 245;

function buildParticle(x: number, y: number, w: number, h: number, delay: number): ParticleRect {
  const px = x + w / 2;
  const py = y + h / 2;
  const dx = px - CX;
  const dy = py - CY;
  const len = Math.max(1, Math.hypot(dx, dy));
  return { x, y, w, h, delay, vx: dx / len, vy: dy / len };
}

// The ten particle squares of pixelkit-favicon.svg.
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
        {/* The favicon's six P rectangles. */}
        <g fill="#ff7900">
          <rect x={56} y={119.51} width={75.68} height={335.14} />
          <rect x={56} y={119.51} width={227.03} height={75.68} />
          <rect x={207.35} y={119.51} width={75.68} height={75.68} />
          <rect x={207.35} y={195.19} width={75.68} height={75.68} />
          <rect x={131.68} y={270.86} width={75.68} height={75.68} />
          <rect x={56} y={270.86} width={151.35} height={75.68} />
        </g>

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
