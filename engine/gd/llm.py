"""Claude-API helpers used by the auto-label pipeline.

Exposes two functions:

  * `expand_tag(tag)` — single-tag synonym lookup (used in legacy
    paths and tests).
  * `expand_tags_batch(tags)` — preferred API. Asks Haiku once for
    synonyms across the project's full tag list with explicit
    instructions to avoid synonyms that visually overlap any
    sibling tag. Single round-trip regardless of project size, and
    the model can disambiguate ("car" + "truck" no longer both pull
    in "vehicle").

Reads the API key from `ANTHROPIC_API_KEY` (also accepts
`CLAUDE_API_KEY`) and silently degrades to "no expansion" when the
key isn't set or the request fails — auto-label keeps working with
the bare user tags. Never blocks the pipeline.
"""
from __future__ import annotations

import json
import os
import re

LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")

_CLIENT = None


def _api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")


def is_configured() -> bool:
    return _api_key() is not None


def _client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not _api_key():
        return None
    try:
        from anthropic import Anthropic

        _CLIENT = Anthropic(api_key=_api_key())
        return _CLIENT
    except Exception as e:
        print(f"[llm] failed to construct Anthropic client: {e}")
        return None


_PROMPT = """You are helping prepare prompts for an open-vocabulary object \
detector. The user wants to detect: "{tag}".

Return up to {n} short synonyms or near-synonyms that describe the SAME \
visual object — variations that the detector might match better than the \
exact word the user typed. Prefer concrete nouns the model would have seen \
in caption training data.

Rules:
- Lowercase, 1–3 words each.
- Concrete, visual things — no abstract descriptors ("dangerous", "important").
- No brand names. No unsafe / illegal content.
- Don't repeat the input tag.
- If you can't think of good variants, return an empty list.

Return ONLY JSON, no commentary:
{{
  "synonyms": []
}}"""


_BATCH_PROMPT = """You are helping prepare prompts for an open-vocabulary \
object detector. The user wants to detect a SET of distinct visual \
objects in their images:

{tags_block}

DEFAULT POSTURE: be CONSERVATIVE. The user has already typed the \
exact word they want detected; synonyms are an optional recall boost, \
not a recall obligation. Your goal is precision, not coverage. A short \
or empty list is the right answer most of the time. Every wrong \
synonym creates false positives the user has to delete by hand —
prefer omitting a borderline candidate over including it.

STEP 1 — for EACH tag, silently classify it as either:
  * GENERIC (broad everyday category that demonstrably has a small \
    set of universally-known true synonyms — e.g.
    "car" ≈ "automobile", "dog" ≈ "puppy" only when subtype matters,
    "person" ≈ "human"), OR
  * SPECIFIC (any narrow visual concept, technical term, or anything \
    you're not 100% sure has well-known synonyms — e.g. "pothole",
    "stop sign", "traffic cone", "manhole cover", "crosswalk",
    "license plate", "drone", "forklift", "wheelbarrow").

When in doubt between the two, default to SPECIFIC.

STEP 2 — return alternative phrasings PER TAG, calibrated to the \
tag's specificity. Hard cap of {n} per tag, but typically you should \
return FEWER than the cap — only include candidates you're highly \
confident about.

  GENERIC tags
    Up to 4 close synonyms, common subtypes the user clearly \
wants captured under the umbrella, or directly-equivalent terms. \
Example: "car" → ["automobile", "sedan", "hatchback", "saloon car"] \
is good; "car" → ["automobile"] alone is too few — give the \
user real recall benefit when the tag genuinely is a broad \
category. Stay away from broader UMBRELLA terms that would \
also catch sibling tags (e.g. "vehicle" catches trucks too — \
omit). Don't list every body style; pick the strongest 3–4.

  SPECIFIC tags
    Empty list is usually the right answer. Only include a \
candidate if it's a literal alternate spelling, regional variant, \
or word-for-word synonym for the EXACT same visual object. \
Example: "pothole" → [] is correct; ["road pothole"] is \
acceptable; ["road crack", "road damage", "asphalt defect", \
"sinkhole", "depression"] is WRONG because those are visually \
different things, not synonyms. If you can't think of an obvious \
near-spelling, return [].

CROSS-TAG RULE — synonyms for one tag must NOT visually overlap any \
of the OTHER tags in the list. Example: if the list contains both \
"car" and "truck", do NOT return "vehicle" as a synonym for "car" \
because the detector would also match trucks. Pick distinct concrete \
nouns specific to each tag. If two tags would naturally share a \
synonym, omit it from BOTH.

Rules per synonym:
- Lowercase, 1–3 words.
- Concrete, visual things — no abstract descriptors ("dangerous", "important").
- No brand names. No unsafe / illegal content.
- Don't repeat the tag itself.
- If the synonym is more general OR more specific than the tag, omit it.
- When in doubt, omit.

Return ONLY JSON, no commentary, with the original tag spellings as \
keys:
{{
  "synonyms": {{
{example_keys}
  }}
}}"""


def _parse_synonyms(raw: str) -> list[str]:
    """Pull the synonyms array out of a Claude response. Handles
    fenced code blocks and stray prose around the JSON."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else raw
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    syns = obj.get("synonyms") if isinstance(obj, dict) else None
    if not isinstance(syns, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for s in syns:
        if not isinstance(s, str):
            continue
        t = s.strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _parse_batch_synonyms(raw: str, tags: list[str]) -> dict[str, list[str]]:
    """Pull a {tag → synonyms[]} dict out of the batch response and
    coerce it to lowercase / dedup. Unknown tag keys are dropped;
    missing tag keys map to an empty list."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else raw
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return {t: [] for t in tags}
    inner = obj.get("synonyms") if isinstance(obj, dict) else None
    if not isinstance(inner, dict):
        return {t: [] for t in tags}
    # Build a case-insensitive lookup so the model's casing of keys
    # doesn't have to match ours exactly.
    by_lower = {str(k).strip().lower(): v for k, v in inner.items()}
    out: dict[str, list[str]] = {}
    for t in tags:
        raw_list = by_lower.get(t.strip().lower())
        if not isinstance(raw_list, list):
            out[t] = []
            continue
        seen: set[str] = set()
        cleaned: list[str] = []
        for s in raw_list:
            if not isinstance(s, str):
                continue
            v = s.strip().lower()
            if not v or v in seen:
                continue
            seen.add(v)
            cleaned.append(v)
        out[t] = cleaned
    return out


_DATASET_TYPE_PROMPT = """Classify CV labels: {labels_csv}

Decide by appearance: would an open-vocabulary text-prompt detector reliably tell these apart, or do they look so alike it needs reference images?
GENERAL = visually distinct; a text prompt alone separates them. e.g. "person, helmet, glove, vest"; "pothole, road, car".
SPECIFIC = easily-confused look-alikes split only by fine detail — material, sub-type, breed, pose or variant — even when they're technically different objects. e.g. "hare, rabbit"; "glass cup, plastic cup"; "screw, threaded standoff"; "UK plate, EU plate"; "pawn, knight".

JSON: {{"type": "general"|"specific", "reason": "<=12 words"}}"""


def classify_dataset_type(labels: list[str]) -> tuple[str, str]:
    """Ask Claude whether a label set is general (distinct categories)
    or specific (fine-grained variants of one concept). Returns
    ``(type, reason)`` where ``type`` is "general" or "specific".
    Falls back to ``("general", "<reason>")`` on missing API key /
    errors so callers never have to handle exceptions.
    """
    cleaned = [str(t).strip() for t in labels if t and str(t).strip()]
    if not cleaned:
        return "general", "no labels yet"
    cli = _client()
    if cli is None:
        return "general", "claude unavailable, defaulting to general"
    prompt = _DATASET_TYPE_PROMPT.format(
        labels_csv=", ".join(f'"{t}"' for t in cleaned)
    )
    try:
        msg = cli.messages.create(
            model=LLM_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[llm] classify_dataset_type({cleaned!r}) failed: {e}")
        return "general", "claude error, defaulting to general"
    raw = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else raw
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return "general", "couldn't parse claude response"
    type_ = str(obj.get("type", "general")).strip().lower()
    if type_ not in ("general", "specific"):
        type_ = "general"
    reason = str(obj.get("reason", "")).strip()[:200]
    print(f"[llm] classify_dataset_type({cleaned!r}) -> {type_} · {reason}")
    return type_, reason


def dataset_insight(summary: dict) -> dict | None:
    """One smart, dataset-specific coaching insight from a COMPACT stats summary
    (numbers only — no images, no manifest). Returns
    ``{"headline": str, "detail": str, "tone": "good"|"warn"|"info"}`` or None on
    missing key / error so the caller can fall back to rule-based insights.

    Token-frugal by design: a tiny JSON in, ~60 tokens out, on Haiku — and the
    caller caches the result by a coarse signature so this only runs when the
    dataset materially changes.
    """
    cli = _client()
    if cli is None:
        return None
    payload = json.dumps(summary, separators=(",", ":"))[:900]
    prompt = (
        "You are a computer-vision dataset coach helping someone build a strong "
        "object-detection training set. Given this dataset summary as JSON, return "
        "ONE specific, actionable insight that references the actual numbers and "
        "tells them the single most valuable next step. Avoid generic advice.\n"
        "IMPORTANT: The label set is the user's deliberate choice. NEVER suggest "
        "adding more label classes or that they 'need more labels' — a single "
        "class is a valid, common, intentional setup. Focus instead on: more "
        "example IMAGES per existing label, image variety/quality, removing "
        "near-duplicates, labelling images that have no detections, and (only "
        "when there are 2+ classes) balancing examples across classes.\n"
        "Reply with ONLY compact JSON, no prose, no code fence: "
        '{"headline":"max 6 words","detail":"max 24 words, concrete","tone":"good|warn|info"}. '
        "Use tone 'good' when the set looks healthy, 'warn' when something needs "
        "fixing, 'info' otherwise.\n"
        f"Summary: {payload}"
    )
    try:
        msg = cli.messages.create(
            model=LLM_MODEL,
            max_tokens=140,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[llm] dataset_insight failed: {e}")
        return None
    raw = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else raw
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    headline = str(obj.get("headline", "")).strip()[:60]
    detail = str(obj.get("detail", "")).strip()[:160]
    tone = str(obj.get("tone", "info")).strip().lower()
    if tone not in ("good", "warn", "info"):
        tone = "info"
    if not headline and not detail:
        return None
    return {"headline": headline, "detail": detail, "tone": tone}


def expand_tags_batch(tags: list[str], n: int = 4) -> dict[str, list[str]]:
    """Generate synonyms for a list of tags in ONE Claude call, with
    every tag's siblings visible to the model so it can avoid
    suggesting overlapping variants.

    Returns a dict keyed by the *input* tag spelling (preserving the
    case the caller passed in) → list of synonyms. Tags that get no
    response or fail validation come back as []. Always returns a
    dict — never raises. Empty input → {}.
    """
    cleaned_inputs: list[str] = []
    seen: set[str] = set()
    for t in tags:
        s = (t or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_inputs.append(s)
    if not cleaned_inputs:
        return {}
    cli = _client()
    if cli is None:
        return {t: [] for t in cleaned_inputs}
    tags_block = "\n".join(f'  - "{t}"' for t in cleaned_inputs)
    example_keys = ",\n".join(f'    "{t}": []' for t in cleaned_inputs)
    prompt = _BATCH_PROMPT.format(
        tags_block=tags_block,
        example_keys=example_keys,
        n=n,
    )
    try:
        msg = cli.messages.create(
            model=LLM_MODEL,
            # ~70 tokens per tag is plenty for "5 lowercase 1-3 word
            # variants" plus brackets and commas.
            max_tokens=max(300, 80 * len(cleaned_inputs)),
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[llm] expand_tags_batch({cleaned_inputs!r}) request failed: {e}")
        return {t: [] for t in cleaned_inputs}
    raw = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
    parsed = _parse_batch_synonyms(raw, cleaned_inputs)
    # Strip the canonical tag (case-insensitive plurals included) and
    # any duplicates already present in another tag's list — first
    # listed tag claims a shared synonym, the others lose it.
    claimed: set[str] = set()
    out: dict[str, list[str]] = {}
    for t in cleaned_inputs:
        canonical_forms = {t.lower(), t.lower().rstrip("s"), t.lower() + "s"}
        kept: list[str] = []
        for s in parsed.get(t, []):
            if s in canonical_forms or s in claimed:
                continue
            claimed.add(s)
            kept.append(s)
            if len(kept) >= n:
                break
        out[t] = kept
    print(f"[llm] expand_tags_batch -> {out}")
    return out


def expand_tag(tag: str, n: int = 5) -> list[str]:
    """Return up to `n` synonyms for `tag`, deduped and lowercased.

    Always returns a list — never raises. On any failure (no API key,
    network blip, malformed response) returns []. Callers should
    treat that as "no expansion" and fall back to the bare tag.
    """
    clean = (tag or "").strip().lower()
    if not clean:
        return []
    cli = _client()
    if cli is None:
        return []
    try:
        msg = cli.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": _PROMPT.format(tag=clean, n=n)}],
        )
    except Exception as e:
        print(f"[llm] expand_tag({clean!r}) request failed: {e}")
        return []
    raw = ""
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
    syns = _parse_synonyms(raw)
    # Strip the canonical tag (case-insensitive, also against simple
    # plural variants) so the expanded prompt doesn't repeat it.
    seen = {clean, clean.rstrip("s"), clean + "s"}
    out: list[str] = []
    for s in syns:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n:
            break
    print(f"[llm] expand_tag({clean!r}) -> {out}")
    return out
