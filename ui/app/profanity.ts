// Shared profanity check for the frontend. Mirrors the curated core
// set used by `backend/gd/profanity.py` so a label that's blocked
// client-side will also be blocked server-side, and vice-versa for
// anything that slips past this list, the backend is still the
// final authority.
//
// Kept small (~150 terms) on purpose: we want instant inline
// feedback on input, not a 5,000-term JSON blob shipped to every
// session. The backend extends with the full dsojevic list when
// `profanity.txt` is present.

const CORE = new Set<string>([
  // Vulgar / sexual
  "fuck", "fucker", "fuckers", "fucking", "fucked", "motherfucker",
  "motherfucking", "fck", "fuk", "shit", "shitty", "shitting",
  "shitter", "bullshit", "horseshit", "crap", "piss", "pissed",
  "pissing", "ass", "asshole", "assholes", "asshat", "arse", "arsehole",
  "dick", "dickhead", "dicks", "dickface", "cock", "cocks", "cocksucker",
  "prick", "pricks", "bollocks", "bollox", "bullocks", "knob", "knobhead",
  "twat", "twats", "wanker", "wankers", "wank", "tosser", "git",
  "bastard", "bastards", "bitch", "bitches", "bitching", "biatch",
  "cunt", "cunts", "kunt", "pussy", "pussies", "douche", "douchebag",
  "jerk", "jackass", "scumbag",
  // Sexual acts / parts
  "blowjob", "handjob", "rimjob", "boner", "tits", "boobs", "boob",
  "nipple", "nipples", "vagina", "penis", "scrotum", "ballsack",
  "anus", "rectum", "clit", "clitoris",
  // Slurs
  "nigger", "nigga", "niggers", "niggas", "chink", "chinks", "spic",
  "spics", "kike", "kikes", "wetback", "wetbacks", "gook", "gooks",
  "wog", "wogs", "paki", "pakis", "raghead", "ragheads", "towelhead",
  "redskin", "redskins", "honky", "cracker", "fag", "faggot", "fags",
  "faggots", "dyke", "dykes", "tranny", "trannies", "shemale",
  "retard", "retards", "retarded", "spaz", "spastic",
  // Other common abuse
  "whore", "whores", "slut", "sluts", "skank", "skanks", "hoe", "hoes",
  "thot", "thots", "cuck", "cucks", "simp", "simps", "incel", "incels",
  // Religion / blasphemy used as profanity
  "goddamn", "goddamned", "damn", "damned", "hell",
]);

const LEET: Record<string, string> = {
  "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
  "@": "a", "$": "s", "!": "i",
};

function normalise(s: string): string {
  let out = "";
  for (const ch of s.toLowerCase()) {
    if (ch in LEET) out += LEET[ch];
    else if ((ch >= "a" && ch <= "z") || ch === " ") out += ch;
    else out += " ";
  }
  return out.replace(/(.)\1{2,}/g, "$1");
}

/**
 * Returns the offending term if `text` contains profanity, else null.
 * Matches whole words only, Scunthorpe, Penistone, Cockermouth pass.
 */
export function containsProfanity(text: string): string | null {
  if (!text) return null;
  const norm = normalise(text);
  for (const token of norm.split(/\s+/)) {
    if (token && CORE.has(token)) return token;
  }
  return null;
}
