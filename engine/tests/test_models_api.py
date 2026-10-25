"""Model manager and settings API, without network access."""


def test_models_status_shape(client):
    r = client.get("/api/models/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["models"].keys()) == {"sam3", "dinov2"}
    sam3 = body["models"]["sam3"]
    assert sam3["gated"] is True and sam3["required"] is True
    # empty test cache -> nothing downloaded, nothing loaded
    assert sam3["downloaded"] is False and sam3["loaded"] is False
    assert body["hfTokenConfigured"] is False
    assert body["freeDiskGb"] > 0


def test_unknown_model_404(client):
    assert client.post("/api/models/yolo9000/load").status_code == 404


def test_load_before_download_conflicts(client):
    r = client.post("/api/models/dinov2/load")
    assert r.status_code == 409
    assert "not downloaded" in r.json()["detail"]


def test_weights_download_at_startup_or_with_a_token_not_on_request(client):
    for action in ("download", "unload"):
        assert client.post(f"/api/models/sam3/{action}").status_code == 405


def test_token_set_validates_and_persists(client, app_module, monkeypatch):
    calls = {}

    def fake_validate(token=None):
        calls["token"] = token
        return {"configured": True, "valid": True, "username": "tester",
                "sam3Access": True, "detail": None}

    monkeypatch.setattr(app_module.models_mgr, "validate_token", fake_validate)
    r = client.post("/api/settings/hf-token", json={"token": "hf_test_123"})
    assert r.status_code == 200, r.text
    assert r.json()["username"] == "tester"
    assert calls["token"] == "hf_test_123"

    # Persisted and reported by settings.
    import workspace
    assert workspace.load_config().get("hf_token") == "hf_test_123"
    assert client.get("/api/settings").json()["hfTokenConfigured"] is True

    # cleared
    assert client.delete("/api/settings/hf-token").status_code == 200
    assert workspace.load_config().get("hf_token") is None
    assert client.get("/api/settings").json()["hfTokenConfigured"] is False


def test_a_token_brings_sam3_up_when_the_models_run(client, app_module, monkeypatch):
    """As at startup: on a GPU, or on the CPU when chosen, a saved token
    brings SAM 3 up; with the models off it only saves the token."""
    import time

    monkeypatch.setattr(app_module.models_mgr, "validate_token",
                        lambda token=None: {"configured": True, "valid": True, "username": "u",
                                            "sam3Access": True, "detail": None})
    monkeypatch.setattr(app_module.models_mgr, "is_downloaded", lambda name: True)
    loads = []

    async def load():
        loads.append(1)

    monkeypatch.setattr(app_module, "_bg_load_charlie", load)
    monkeypatch.setitem(app_module.state, "charlie", None)
    monkeypatch.setitem(app_module.state, "device", "cpu")
    for off, expected in ((True, 0), (False, 1)):
        loads.clear()
        monkeypatch.setitem(app_module.state, "sam3_off", off)
        assert client.post("/api/settings/hf-token", json={"token": "hf_x"}).status_code == 200
        deadline = time.time() + 5
        while len(loads) != expected and time.time() < deadline:
            time.sleep(0.05)
        assert len(loads) == expected, off
    client.delete("/api/settings/hf-token")


def test_token_set_rejects_invalid(client, app_module, monkeypatch):
    monkeypatch.setattr(
        app_module.models_mgr, "validate_token",
        lambda token=None: {"configured": True, "valid": False,
                            "sam3Access": None, "username": None,
                            "detail": "token rejected by Hugging Face: 401"},
    )
    r = client.post("/api/settings/hf-token", json={"token": "hf_bad"})
    assert r.status_code == 400
    assert "rejected" in r.json()["detail"]


def test_settings_and_workspace_change(client, workspace_dir, tmp_path):
    s = client.get("/api/settings").json()
    assert s["workspace"] == str(workspace_dir)
    assert s["device"] in ("cuda", "mps", "cpu")

    r = client.post("/api/settings/workspace", json={"path": str(tmp_path / "new-ws")})
    assert r.status_code == 200
    assert r.json()["restartRequired"] is True
    # the live process keeps its workspace; only the config changed
    assert client.get("/api/settings").json()["workspace"] == str(workspace_dir)
    import workspace
    assert workspace.load_config()["workspace"] == str(tmp_path / "new-ws")
    # Restore the config for later tests.
    cfg = workspace.load_config()
    cfg.pop("workspace", None)
    workspace.save_config(cfg)


def test_download_progress_counts_each_byte_once():
    import models

    progress = {"downloaded_bytes": 0}
    cls = models._make_tqdm(progress)
    # huggingface_hub reports a download on two byte bars: received and written.
    received = cls(total=100, unit="B", desc="Downloading bytes", disable=True)
    written = cls(total=100, unit="B", desc="Reconstructing (incomplete total...)", disable=True)
    received.update(60)
    written.update(60)
    files = cls(total=3, unit="it", desc="Fetching 3 files", disable=True)
    files.update(1)
    assert progress["downloaded_bytes"] == 60
