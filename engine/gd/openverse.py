"""Openverse image search + GD scoring.

Openverse aggregates Creative Commons-licensed images from sources like
Flickr, Wikimedia, and various stock libraries. The free API needs no
auth for basic queries - we just hit it with the user's word and pull
back the first N results, surfacing the URL, thumbnail, and licence
metadata so the frontend can show previews + attribution.

Uses urllib so we don't pull in another HTTP dep just for one GET.
"""
from __future__ import annotations

import io
import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as PILImage

OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
OPENVERSE_UA = "PixelKit/1.0 (+https://pixel.kit)"


# SSRF guard. Openverse aggregates many CC sources (Flickr, Wikimedia,
# Smithsonian, ...) so a strict allowlist is impractical. Instead we:
#   1. Require https
#   2. Resolve the hostname and reject any non-global IP
# That blocks 127.0.0.1, 10/8, 172.16/12, 192.168/16, 169.254.169.254
# (cloud metadata), ::1, fc00::/7, fe80::/10 - i.e. everything an
# attacker would need to pivot to internal services if Openverse ever
# returned a malicious URL.
def _url_is_safe_to_fetch(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("https",):
        return False
    host = parsed.hostname
    if not host:
        return False
    # Resolve every A + AAAA record so an IPv6-only host (or a dual-
    # stack host whose AAAA points at a link-local / loopback address)
    # can't sneak past a v4-only resolver. getaddrinfo with AF_UNSPEC
    # returns both families.
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror):
        return False
    if not infos:
        return False
    for family, _socktype, _proto, _canonname, sockaddr in infos:
        if family == socket.AF_INET:
            addr_str = sockaddr[0]
        elif family == socket.AF_INET6:
            addr_str = sockaddr[0].split("%", 1)[0]  # strip zone-id
        else:
            return False
        try:
            ip = ipaddress.ip_address(addr_str)
        except ValueError:
            return False
        if not ip.is_global:
            # Catches loopback, private, link-local, multicast, reserved
            # (both v4 + v6 paths share the same ipaddress check).
            return False
    return True


def search_images(query: str, count: int = 5, commercial: bool = False) -> list[dict]:
    """Hit Openverse's image search endpoint and normalise the result
    shape down to what the frontend actually needs. Raises on network
    or HTTP errors so the caller can surface the failure to the user.

    Openverse caps `page_size` at 20 per request, so for `count > 20`
    we paginate and concatenate. Stops early if a page comes back
    empty or short - both signal we've hit the end of what the
    upstream sources have for this query.

    `commercial=True` adds Openverse's `license_type=commercial` filter,
    which restricts results to licences that permit commercial use
    (CC0, BY, BY-SA, PDM) and excludes the NC variants."""
    clean = (query or "").strip()
    if not clean:
        return []
    target = max(1, min(250, count))

    PAGE_SIZE = 20  # Openverse hard cap per request
    results: list[dict] = []
    page = 1
    while len(results) < target:
        remaining = target - len(results)
        page_size = min(PAGE_SIZE, remaining)
        params: dict[str, str | int] = {"q": clean, "page_size": page_size, "page": page}
        if commercial:
            params["license_type"] = "commercial"
        qs = urllib.parse.urlencode(params)
        url = f"{OPENVERSE_ENDPOINT}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": OPENVERSE_UA})
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            payload = json.loads(resp.read())

        batch = payload.get("results", []) or []
        if not batch:
            # No more results upstream - return what we've got.
            break

        for item in batch:
            if not isinstance(item, dict):
                continue
            results.append({
                "id": item.get("id"),
                "url": item.get("url"),
                "thumbnail": item.get("thumbnail") or item.get("url"),
                "title": item.get("title"),
                "creator": item.get("creator"),
                "license": item.get("license"),
                "license_version": item.get("license_version"),
                "source": item.get("source"),
                "foreign_landing_url": item.get("foreign_landing_url"),
                "width": item.get("width"),
                "height": item.get("height"),
            })
            if len(results) >= target:
                break

        # Short page = upstream has nothing more for this query.
        if len(batch) < page_size:
            break
        page += 1

    return results


def _probe_thumbnail(url: str, timeout: float = 2.5) -> bool:
    """Lightweight reachability check for one thumbnail URL.

    Tries HEAD first. If that succeeds we trust the headers - image
    MIME and Content-Length above a small floor (1.5 KB; anything
    smaller is almost certainly a placeholder, not a real photo).
    Some upstream hosts reject HEAD with 405, in which case we fall
    through to a tiny GET that reads the first 4 KB to inspect the
    Content-Type and the leading bytes.

    Returns True only when we're confident the URL serves a real
    image. Network errors / timeouts → False so the search results
    don't carry dead links into the preview grid.
    """
    if not url or not _url_is_safe_to_fetch(url):
        return False
    headers = {"User-Agent": OPENVERSE_UA}
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status not in (200, 206):
                return False
            ctype = (resp.headers.get("Content-Type") or "").lower()
            cl = resp.headers.get("Content-Length")
            size = int(cl) if cl and cl.isdigit() else None
            if not ctype.startswith("image/"):
                return False
            if size is not None and size < 1500:
                return False
            return True
    except urllib.error.HTTPError as e:
        # 405 / 403 on HEAD → fall through to a tiny GET. Other HTTP
        # errors (404 / 410 / 5xx) mean the URL is dead.
        if e.code not in (405, 403):
            return False
    except Exception:
        # Connection error, DNS failure, timeout - fall through to GET.
        pass
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status not in (200, 206):
                return False
            ctype = (resp.headers.get("Content-Type") or "").lower()
            data = resp.read(4096)
        if not ctype.startswith("image/"):
            return False
        if len(data) < 1500:
            return False
        return True
    except Exception:
        return False


def validate_thumbnails(items: list[dict], max_workers: int = 32) -> list[dict]:
    """Filter `items` to those whose thumbnail URL responds with a
    real image. Order is preserved. Items are probed in parallel -
    on a typical 50-item search this completes in under 3 s.

    The probe runs server-side so we don't depend on the user's
    browser-side CORS / cache state for filtering, which is what was
    letting broken thumbnails through to the search preview grid
    even after the frontend probe.
    """
    if not items:
        return items
    targets = [it.get("thumbnail") or it.get("url") for it in items]

    def probe(idx_url: tuple[int, str | None]) -> tuple[int, bool]:
        idx, url = idx_url
        return idx, _probe_thumbnail(url) if url else False

    valid_idx: set[int] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, ok in pool.map(probe, list(enumerate(targets))):
            if ok:
                valid_idx.add(idx)
    return [items[i] for i in range(len(items)) if i in valid_idx]


def _download_image(url: str, timeout: float = 15.0, max_side: int = 800) -> PILImage.Image | None:
    """Pull a single image from `url` and decode to RGB. Resize the
    longest side to `max_side` so the GD scoring pass stays fast even
    when Openverse hands us a 4K stock photo. Returns None on failure
    so a single bad URL doesn't kill the whole batch."""
    if not _url_is_safe_to_fetch(url):
        print(f"[openverse] refused unsafe URL for download: {url}")
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": OPENVERSE_UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.BILINEAR)
        return img
    except Exception as e:
        print(f"[openverse] download failed ({url}): {e}")
        return None


def download_images(urls: list[str], max_workers: int = 5) -> list[PILImage.Image | None]:
    """Concurrent download - 5 images come back in ~1s on a normal
    connection vs 3-4s sequential. Returns a list aligned with `urls`
    so callers can map URL → image directly."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_download_image, urls))


def _download_bytes(url: str, timeout: float = 30.0, max_bytes: int = 25_000_000) -> tuple[bytes | None, str | None]:
    """Pull raw image bytes for dataset import - full resolution, no
    PIL roundtrip. Returns (data, content_type) or (None, None) on
    failure. `max_bytes` caps the largest image we'll store so a bad
    URL can't drop a 200 MB blob in the project bucket."""
    if not _url_is_safe_to_fetch(url):
        print(f"[openverse] refused unsafe URL for dataset download: {url}")
        return None, None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": OPENVERSE_UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                print(f"[openverse] dataset download too large ({url}): >{max_bytes}B")
                return None, None
            ctype = resp.headers.get("Content-Type")
        return data, ctype
    except Exception as e:
        print(f"[openverse] dataset download failed ({url}): {e}")
        return None, None


def download_images_bytes(urls: list[str], max_workers: int = 8) -> list[tuple[bytes | None, str | None]]:
    """Bulk byte download for the URL-import flow. Returns a list
    aligned with `urls` so the caller can pair each (data, ctype) with
    its source URL."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_download_bytes, urls))
