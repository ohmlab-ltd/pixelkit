"""Device-preference settings API; PK_DEVICE overrides the stored choice."""
import pytest


@pytest.fixture(autouse=True)
def _clean_device_config():
    """Each test starts and ends with no device key in the app config."""
    import workspace

    def _clear():
        cfg = workspace.load_config()
        if "device" in cfg:
            del cfg["device"]
            workspace.save_config(cfg)

    _clear()
    yield
    _clear()


def test_settings_report_auto_by_default(client):
    body = client.get("/api/settings").json()
    assert body["devicePreference"] == "auto"
    assert body["deviceEnvOverride"] is None
    assert isinstance(body["gpuAvailable"], bool)


def test_set_device_cpu_persists_and_requires_restart(client):
    import workspace

    r = client.post("/api/settings/device", json={"device": "cpu"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "devicePreference": "cpu", "restartRequired": True}
    assert workspace.load_config()["device"] == "cpu"
    assert client.get("/api/settings").json()["devicePreference"] == "cpu"


def test_set_device_auto_clears_the_key(client):
    import workspace

    client.post("/api/settings/device", json={"device": "cpu"})
    r = client.post("/api/settings/device", json={"device": "auto"})
    assert r.status_code == 200
    assert "device" not in workspace.load_config()
    assert client.get("/api/settings").json()["devicePreference"] == "auto"


def test_set_device_rejects_unknown_values(client):
    r = client.post("/api/settings/device", json={"device": "tpu"})
    assert r.status_code == 400


def test_config_choice_reaches_the_resolver(client, app_module):
    client.post("/api/settings/device", json={"device": "cpu"})
    assert app_module._configured_device() == "cpu"
    # cpu is always available, so the resolver honours the choice outright.
    assert app_module._resolve_device() == "cpu"


def test_env_var_beats_config(client, app_module, monkeypatch):
    client.post("/api/settings/device", json={"device": "cpu"})
    monkeypatch.setenv("PK_DEVICE", "mps")
    assert app_module._configured_device() == "mps"
    assert client.get("/api/settings").json()["deviceEnvOverride"] == "mps"
    monkeypatch.delenv("PK_DEVICE")
    assert app_module._configured_device() == "cpu"


def test_concurrent_config_updates_keep_every_change(tmp_path, monkeypatch):
    import threading

    import workspace

    monkeypatch.setenv("PIXELKIT_CONFIG_DIR", str(tmp_path))
    workspace.save_config({"hf_token": "keep-me"})
    threads = [threading.Thread(target=workspace.update_config,
                                args=(lambda c, i=i: c.__setitem__(f"k{i}", i),))
               for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cfg = workspace.load_config()
    assert cfg["hf_token"] == "keep-me"
    assert all(cfg[f"k{i}"] == i for i in range(24))
    assert not list(tmp_path.glob("*.tmp"))
