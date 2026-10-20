import asyncio, json
import pytest
from simready_bench.harnesses import BenchmarkPi
from simready_bench.capabilities import normalize_native_harness


def test_manifest_selects_exactly_one_retry_owner():
    h = {
        "id": "pi",
        "adapter": "pi",
        "kwargs": {"version": "0.85.1", "retry_owner": "relay"},
    }
    assert normalize_native_harness(h)["kwargs"]["retry_owner"] == "relay"
    for value in ("automatic", "both", True):
        with pytest.raises(ValueError):
            normalize_native_harness({**h, "kwargs": {"retry_owner": value}})


def test_relay_owner_writes_pi_retry_disabled_without_replacing_agent_loop(monkeypatch):
    from simready_bench.engine.pi import PiAgent

    obj = object.__new__(BenchmarkPi)
    obj._retry_owner = "relay"
    written = []

    async def ensure(self, environment):
        pass

    async def upload(environment, **kwargs):
        written.append(kwargs)

    async def execute(environment, **kwargs):
        pass

    monkeypatch.setattr(PiAgent, "_ensure_config_dir", ensure)
    obj._upload_config_text = upload
    obj.exec_as_agent = execute
    asyncio.run(obj._ensure_config_dir(object()))
    assert len(written) == 1
    assert json.loads(written[0]["content"]) == {
        "retry": {"enabled": False, "maxRetries": 0, "provider": {"maxRetries": 0}}
    }
    assert written[0]["remote_path"].endswith("/settings.json")
    assert BenchmarkPi.run is PiAgent.run


def test_native_harness_keeps_its_retry_configuration(monkeypatch):
    from simready_bench.engine.pi import PiAgent

    obj = object.__new__(BenchmarkPi)
    obj._retry_owner = "harness"
    called = []

    async def ensure(self, environment):
        called.append("ensure")

    async def upload(*a, **k):
        raise AssertionError("Native settings must not be rewritten")

    monkeypatch.setattr(PiAgent, "_ensure_config_dir", ensure)
    obj._upload_config_text = upload
    asyncio.run(obj._ensure_config_dir(object()))
    assert called == ["ensure"]
