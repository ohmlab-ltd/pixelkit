"""Image fetches from Openverse results refuse non-public addresses, redirects included."""
import urllib.error
import urllib.request

import pytest

import openverse


def _redirect(target):
    req = urllib.request.Request("https://images.example/a.jpg")
    return openverse._SafeRedirects().redirect_request(req, None, 302, "Found", {}, target)


@pytest.mark.parametrize("target", ["https://127.0.0.1/x.jpg", "http://images.example/x.jpg",
                                    "https://169.254.169.254/latest/meta-data/"])
def test_a_redirect_to_a_non_public_or_plain_http_address_is_refused(target):
    with pytest.raises(urllib.error.HTTPError):
        _redirect(target)


def test_a_redirect_to_a_public_address_is_followed(monkeypatch):
    monkeypatch.setattr(openverse, "_url_is_safe_to_fetch", lambda url: True)
    follow = _redirect("https://cdn.example/x.jpg")
    assert follow is not None and follow.full_url == "https://cdn.example/x.jpg"


def test_the_user_agent_names_the_project():
    assert "pixel.kit" not in openverse.OPENVERSE_UA and "github.com" in openverse.OPENVERSE_UA
