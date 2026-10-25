"""Openverse image search and image download."""
from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OPENVERSE_ENDPOINT = "https://api.openverse.org/v1/images/"
OPENVERSE_UA = "PixelKit (+https://github.com/ohmlab-ltd/pixelkit)"


def _public_addrs(host: str, port: int | None) -> list[tuple]:
    """(family, type, proto, sockaddr) for each address of `host`, or [] when
    it does not resolve or any address is not global."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror):
        return []
    out = []
    for family, socktype, proto, _canonname, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            return []
        try:
            ip = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])  # IPv6 zone id
        except ValueError:
            return []
        if not ip.is_global:
            return []
        out.append((family, socktype, proto, sockaddr))
    return out


# SSRF guard: https only, and every resolved address must be global.
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
    return bool(host) and bool(_public_addrs(host, None))


def _connect_public(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    """socket.create_connection to an address the guard checked in this same
    lookup, so a name that resolves differently the second time (DNS
    rebinding) cannot reach a private address."""
    host, port = address
    addrs = _public_addrs(host, port)
    if not addrs:
        raise OSError(f"{host} does not resolve to public addresses only")
    err: OSError | None = None
    for family, socktype, proto, sockaddr in addrs:
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            err = e
            sock.close()
    raise err if err is not None else OSError(f"cannot connect to {host}")


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to checked addresses only. Through a proxy, the proxy resolves."""

    def connect(self):
        if not self._tunnel_host:
            self._create_connection = _connect_public
        super().connect()


class _PublicHTTPS(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PublicHTTPSConnection, req, context=self._context)


class _SafeRedirects(urllib.request.HTTPRedirectHandler):
    """Applies the SSRF guard to every redirect target too."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _url_is_safe_to_fetch(newurl):
            raise urllib.error.HTTPError(newurl, code, "redirect to a non-public address", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirects, _PublicHTTPS)


def search_images(query: str, count: int = 5, commercial: bool = False) -> list[dict]:
    """Search Openverse; raises on network or HTTP errors.
    `commercial` keeps only licences that permit commercial use."""
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
        with _OPENER.open(req, timeout=20.0) as resp:
            payload = json.loads(resp.read())

        batch = payload.get("results", []) or []
        if not batch:
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
                "license_url": item.get("license_url"),
                "attribution": item.get("attribution"),
                "source": item.get("source"),
                "foreign_landing_url": item.get("foreign_landing_url"),
                "width": item.get("width"),
                "height": item.get("height"),
            })
            if len(results) >= target:
                break

        if len(batch) < page_size:
            break
        page += 1

    return results


def _probe_thumbnail(url: str, timeout: float = 2.5) -> bool:
    """True when `url` serves an image; under 1500 bytes counts as a placeholder."""
    if not url or not _url_is_safe_to_fetch(url):
        return False
    headers = {"User-Agent": OPENVERSE_UA}
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with _OPENER.open(req, timeout=timeout) as resp:
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
        # Some hosts refuse HEAD with 405 or 403; those get a GET.
        if e.code not in (405, 403):
            return False
    except Exception:
        pass
    try:
        req = urllib.request.Request(url, headers=headers)
        with _OPENER.open(req, timeout=timeout) as resp:
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
    """The `items` whose thumbnail serves an image, in order."""
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


def _download_bytes(url: str, timeout: float = 30.0, max_bytes: int = 25_000_000) -> tuple[bytes | None, str | None]:
    """(data, content_type) of `url`, or (None, None) on failure or above `max_bytes`."""
    if not _url_is_safe_to_fetch(url):
        print(f"[openverse] refused unsafe URL for dataset download: {url}")
        return None, None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": OPENVERSE_UA})
        with _OPENER.open(req, timeout=timeout) as resp:
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
    """_download_bytes for each of `urls` in parallel; the result is aligned with `urls`."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_download_bytes, urls))
