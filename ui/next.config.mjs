// Security headers. CSP + HSTS apply in production only — dev needs
// websockets/eval for fast refresh, and HSTS on localhost is a trap.
// The CSP is deliberately moderate: Next's hydration needs inline
// scripts (no nonce plumbing here yet) and msgpackr's fast paths use
// Function codegen, so 'unsafe-inline'/'unsafe-eval' stay. The value
// is in the allowlists: external scripts/connects/frames are pinned to
// self + Turnstile + Stripe + our API, and framing is blocked outright.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";
const POSTHOG_HOST = process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://eu.i.posthog.com";

const csp = [
  "default-src 'self'",
  // Turnstile widget + Stripe.js.
  "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://challenges.cloudflare.com https://js.stripe.com",
  "style-src 'self' 'unsafe-inline'",
  // Project images 302-redirect from our API to per-account R2
  // presigned URLs; https: keeps avatars + R2 + any future CDN working.
  "img-src 'self' data: blob: https:",
  `connect-src 'self' ${API_URL} ${POSTHOG_HOST} https://api.stripe.com https://challenges.cloudflare.com https://*.r2.cloudflarestorage.com`,
  "font-src 'self' data:",
  "frame-src https://challenges.cloudflare.com https://js.stripe.com https://hooks.stripe.com",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  ...(process.env.NODE_ENV === "production"
    ? [
        { key: "Content-Security-Policy", value: csp },
        {
          key: "Strict-Transport-Security",
          value: "max-age=63072000; includeSubDomains",
        },
      ]
    : []),
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
