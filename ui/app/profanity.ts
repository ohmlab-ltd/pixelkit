// Profanity check — stubbed out in the portable build. The SaaS app
// pre-screened names/labels headed for shared public storage; the
// portable app is a single local user writing into their own workspace,
// so everything passes. Call sites keep their shape.
export function containsProfanity(text: string): string | null {
  void text; // keeps the call-site signature; the portable build moderates nothing
  return null;
}
