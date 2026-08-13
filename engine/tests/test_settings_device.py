"""Device-preference settings — the Settings "Compute" picker's API.

The choice persists in the app config (never the workspace) and takes
effect on the next boot; PK_DEVICE env always beats the config.
"""
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
