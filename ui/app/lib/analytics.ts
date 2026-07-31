// Analytics stub for the portable build. The SaaS version wrapped
// posthog-js here; portable PixelKit ships no telemetry, so every helper
// is a no-op. The typed AnalyticsEvent union is kept so call sites keep
// compiling until they're removed in the frontend-slimming phase.

export function markAnalyticsReady() {}
export function analyticsReady() {
  return false;
}

export type AnalyticsEvent =
  | "demo_upload"
  | "demo_run"
  | "hit_demo_cap"
  | "demo_quota_reached"
  | "cta_signup_click"
  | "cta_contact_click"
  | "contact_submitted"
  | "signup"
  | "project_create"
  | "dataset_export"
  | "dataset_import";

export function capture(
  event: AnalyticsEvent,
  properties?: Record<string, unknown>,
) {
  void event;
  void properties;
}

export function identifyUser(
  distinctId: string,
  properties?: Record<string, unknown>,
) {
  void distinctId;
  void properties;
}

export function resetAnalytics() {}
