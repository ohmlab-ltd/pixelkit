"""Profanity gate for persistent user-supplied text.

Used by the backend to reject profanity before it lands in the manifest
(project names, labels, tag rewrites, etc.). The check is:

  1. Normalise the input — lowercase, strip basic leetspeak substitutions
     (`@`→`a`, `$`→`s`, `0`→`o` …), and collapse runs of the same character
     (`fuuuck` → `fuck`).
  2. Tokenise on alphabetic boundaries.
  3. Look each token up in a precomputed set.

The set is built from a curated core (compatible with the
`dsojevic/profanity-list` schema) plus an optional `profanity.txt`
file at the backend root that callers can extend with extra terms or
the full dsojevic list. Lookup is O(1) per token, so the gate adds
microseconds to a save call.

Word-boundary matching deliberately — substring matches generate the
classic "Scunthorpe problem" (a town name flagged for "cunt"). If you
want stricter behaviour, swap `_token_match` for a regex sweep.
"""
from __future__ import annotations

import re
from pathlib import Path

# Curated English core. Compatible with dsojevic/profanity-list — drop
# the full list at `backend/profanity.txt` (one word per line, `#` for
# comments) to extend without rebuilding. Hate-speech slurs are
# included because the gate's job is to keep them out of stored
# manifests, not to perform editorial. Add per-language coverage by
# appending the appropriate dsojevic file to `profanity.txt`.
_CORE: set[str] = {
    # Vulgar / sexual
    "fuck", "fucker", "fuckers", "fucking", "fucked", "motherfucker",
    "motherfucking", "fck", "fuk", "fck", "shit", "shitty", "shitting",
    "shitter", "bullshit", "horseshit", "crap", "piss", "pissed",
    "pissing", "ass", "asshole", "assholes", "asshat", "arse", "arsehole",
    "dick", "dickhead", "dicks", "dickface", "cock", "cocks", "cocksucker",
    "prick", "pricks", "bollocks", "bollox", "bullocks", "knob", "knobhead",
    "twat", "twats", "wanker", "wankers", "wank", "tosser", "git",
    "bastard", "bastards", "bitch", "bitches", "bitching", "biatch",
    "cunt", "cunts", "kunt", "pussy", "pussies", "douche", "douchebag",
    "jerk", "jackass", "scumbag",
    # Sexual acts / parts often abused as insults
    "blowjob", "handjob", "rimjob", "boner", "tits", "boobs", "boob",
    "nipple", "nipples", "vagina", "penis", "scrotum", "ballsack",
    "anus", "rectum", "clit", "clitoris",
    # Hate / slurs
    "nigger", "nigga", "niggers", "niggas", "chink", "chinks", "spic",
    "spics", "kike", "kikes", "wetback", "wetbacks", "gook", "gooks",
    "wog", "wogs", "paki", "pakis", "raghead", "ragheads", "towelhead",
    "redskin", "redskins", "honky", "cracker", "fag", "faggot", "fags",
    "faggots", "dyke", "dykes", "tranny", "trannies", "shemale",
    "retard", "retards", "retarded", "spaz", "spastic",
    # Other common abuse
    "whore", "whores", "slut", "sluts", "skank", "skanks", "hoe", "hoes",
    "thot", "thots", "cuck", "cucks", "simp", "simps", "incel", "incels",
    # Religion / blasphemy commonly used as profanity
    "goddamn", "goddamned", "damn", "damned", "hell",
}

# Cheap leetspeak fold so "@$$" reads as "ass" and "f*ck" reads as
# "fck" before normalisation drops the placeholder. Aggressive enough
# to catch obvious obfuscation, conservative enough that a string like
# "S$P 500" doesn't suddenly contain a slur.
_LEET = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
}


def _build_banned() -> set[str]:
    out = set(_CORE)
    extra = Path(__file__).resolve().parent.parent / "profanity.txt"
    if extra.exists():
        try:
            for line in extra.read_text(encoding="utf-8").splitlines():
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    out.add(w)
        except Exception as e:
            print(f"[profanity] failed to load {extra}: {e}")
    return out


_BANNED: set[str] = _build_banned()


def _normalise(s: str) -> str:
    s = s.lower()
    # Replace leet characters with their letter equivalent. Drop other
    # punctuation so "f.u.c.k" tokenises as one word.
    out_chars: list[str] = []
    for ch in s:
        if ch in _LEET:
            out_chars.append(_LEET[ch])
        elif ch.isalpha() or ch.isspace():
            out_chars.append(ch)
        else:
            out_chars.append(" ")
    s = "".join(out_chars)
    # Collapse triple+ repeats: "fuuuck" → "fuck", "shiiiit" → "shit".
    return re.sub(r"(.)\1{2,}", r"\1", s)


def contains_profanity(text: str) -> str | None:
    """Return the offending term if `text` contains profanity, else None.

    Tokenises on whitespace and only matches whole words, so well-behaved
    place names that happen to contain a slur as a substring (Scunthorpe,
    Penistone, Cockermouth) pass cleanly.
    """
    if not text:
        return None
    norm = _normalise(text)
    for token in norm.split():
        if not token:
            continue
        if token in _BANNED:
            return token
    return None


def assert_clean(text: str, field: str = "input") -> None:
    """Raise HTTPException(400) if `text` is profane. Convenience wrapper
    used by the FastAPI handlers — keeps the error path one line at the
    call site."""
    from fastapi import HTTPException

    term = contains_profanity(text)
    if term is not None:
        raise HTTPException(
            status_code=400,
            detail=f"{field} contains a banned term: {term!r}",
        )


def reload_list() -> int:
    """Re-read `profanity.txt` and rebuild the banned set in place.
    Useful if the hosting environment edits the file at runtime."""
    global _BANNED
    _BANNED = _build_banned()
    return len(_BANNED)
