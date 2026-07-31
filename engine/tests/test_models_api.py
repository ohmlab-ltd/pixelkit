"""Model manager + settings API — no network, config isolated by conftest."""


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


def test_sam3_download_requires_token(client):
    r = client.post("/api/models/sam3/download")
    assert r.status_code == 400
    assert "token" in r.json()["detail"].lower()


def test_unknown_model_404(client):
    assert client.post("/api/models/yolo9000/download").status_code == 404
    assert client.post("/api/models/yolo9000/load").status_code == 404


def test_load_before_download_conflicts(client):
    r = client.post("/api/models/dinov2/load")
    assert r.status_code == 409
    assert "not downloaded" in r.json()["detail"]


def test_unload_unsupported(client):
    assert client.post("/api/models/dinov2/unload").status_code == 400


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

    # persisted (isolated config dir) + reported by settings
    import workspace
    assert workspace.load_config().get("hf_token") == "hf_test_123"
    assert client.get("/api/settings").json()["hfTokenConfigured"] is True

    # cleared
    assert client.delete("/api/settings/hf-token").status_code == 200
    assert workspace.load_config().get("hf_token") is None
    assert client.get("/api/settings").json()["hfTokenConfigured"] is False


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
    # put the config back so later tests aren't affected on rerun
    cfg = workspace.load_config()
    cfg.pop("workspace", None)
    workspace.save_config(cfg)
