"""Regression tests for the CORS allow-origin regex (finding S3).

The middleware layers a regex on top of the exact localhost allow-list so
the FE can call the API from any `pixelkit.ai` subdomain (www, app, dev,
staging) and any `*.vercel.app` preview without enumerating each one. The
security-relevant property S3 verified: the **required literal dot before
`pixelkit`** means a look-alike apex such as `evil-pixelkit.ai` is NOT a
`pixelkit.ai` subdomain and must be rejected.

Black-box test through Starlette's CORSMiddleware: a CORS preflight from an
allowed origin echoes `access-control-allow-origin`; a disallowed origin
gets no such header. The heavy ML stack is stubbed by conftest.py so
`server.py` imports CPU-only.
"""
import pytest

# Origins the regex must accept.
ALLOWED = [
    "https://pixelkit.ai",            # bare apex
    "https://www.pixelkit.ai",        # subdomain
    "https://app.pixelkit.ai",
    "https://staging.pixelkit.ai",
    "https://pr-123-preview.vercel.app",  # Vercel preview
]

# Origins the regex must reject. The first three are the S3 attack shapes.
DISALLOWED = [
    "https://evil-pixelkit.ai",            # look-alike apex (no dot boundary)
    "https://notpixelkit.ai",              # suffix without a dot boundary
    "https://pixelkit.ai.attacker.com",    # pixelkit.ai as a left-label
    "http://pixelkit.ai",                  # plaintext, not https
    "https://pixelkit-ai.com",             # different TLD
    "https://vercel.app",                  # bare apex, no preview subdomain
    "https://evil.vercel.app.attacker.com",  # vercel.app as a left-label
]


def _preflight(client, origin):
    return client.options(
        "/",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


@pytest.mark.parametrize("origin", ALLOWED)
def test_cors_allows_pixelkit_subdomains_and_vercel_previews(client, origin):
    r = _preflight(client, origin)
    assert r.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", DISALLOWED)
def test_cors_blocks_lookalike_and_foreign_origins(client, origin):
    r = _preflight(client, origin)
    assert "access-control-allow-origin" not in r.headers


if __name__ == "__main__":  # allow running as a plain script too
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
