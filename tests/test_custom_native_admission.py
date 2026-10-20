import time
import pytest


def spec():
    return {
        "schema": "simready-operator-model-v1",
        "model": "custom-vision-model",
        "api_style": "openai-completions",
        "base_url": "https://models.example.org/v1",
        "context_window": 131072,
        "maximum_output_tokens": None,
        "pricing": {"input_per_million_usd": "1", "output_per_million_usd": "2"},
        "pricing_source": "https://models.example.org/pricing",
        "checked": "2026-10-19",
    }


def test_custom_model_native_opencode_resolves_without_network_or_credentials(setup):
    from simready_bench.config import resolve_experiment

    root, m, p = setup
    m["defaults"]["generation"].pop("reasoning_effort", None)
    m["authoring_protocol"] = "authoring-32-native-output-v1"
    m["harnesses"] = [
        {
            "id": "opencode",
            "adapter": "harbor",
            "agent": "opencode",
            "kwargs": {"version": "1.19.3"},
        }
    ]
    m["models"] = [
        {
            "id": "custom",
            "adapter": "openai-completions",
            "provider_ref": "custom",
            "model_id": "custom-vision-model",
            "metadata": {
                "contextWindow": 131072,
                "maxTokens": 4096,
                "input": ["text", "image"],
                "reasoning": False,
            },
        }
    ]
    p["providers"] = {
        "custom": {
            "kind": "openai",
            "base_url": spec()["base_url"],
            "api_key_env": "TEST_UNUSED_KEY",
            "model_spec": spec(),
        }
    }
    p["limits"].update(
        max_input_tokens=None, max_output_tokens=None, inference_requests=32
    )
    r = resolve_experiment(m, base_dir=root, policy=p)
    c = r["conditions"][0]
    assert (
        c["harness"]["native_agent"] == "opencode"
        and c["settings"]["budgets"]["max_output_tokens"] is None
    )
    assert (
        c["model"]["native_output_facts"]["reservation_basis"]
        == "operator-declared-provider-capacity"
    )
    assert c["settings"]["budgets"]["inference_requests"] == 32
    assert list(root.rglob("*.sqlite3")) == []


def test_custom_cost_reservation_is_explicit_route_bound_and_precedes_dispatch(
    tmp_path,
):
    from simready_bench.engine.inference_gateway import GatewayConfig, ProjectBudget
    from simready_bench.engine.custom_model import reserve_custom_cost
    from simready_bench.engine.request_cost import CostLedger, CostDenied

    auth = ProjectBudget(
        project="simready-bench",
        authorization_id="fixture-only",
        approved=True,
        upstream_origin="https://models.example.org",
        model=spec()["model"],
        expires_at=time.time() + 100,
        maximum_requests=32,
        maximum_spend_usd=1,
        pricing_status="known",
        external_spend_cap_verified=False,
    )
    cfg = GatewayConfig(
        spec()["base_url"],
        spec()["model"],
        "fixture-key",
        api_style="openai-completions",
        budget=auth,
        native_output_limits=True,
        max_output_tokens=None,
        max_input_tokens=131072,
        model_spec=spec(),
        strip_native_output_fields=True,
    )
    payload = {"model": cfg.model, "messages": [{"role": "user", "content": "fixture"}]}
    ledger = CostLedger(tmp_path)
    r = reserve_custom_cost(cfg, payload, ledger, "first", "actor")
    assert (
        r["reserved_upper_cost_micro_usd"] == 393216
        and r["actual_cost_usd"] is None
        and (r["input_tokens"] is None)
    )
    assert r["input_token_upper_bound"] == 131072
    reserve_custom_cost(cfg, payload, ledger, "second-valid", "actor")
    with pytest.raises(CostDenied):
        reserve_custom_cost(cfg, payload, ledger, "third-over-budget", "actor")
    with pytest.raises(CostDenied):
        reserve_custom_cost(
            cfg, {**payload, "model": "wrong"}, ledger, "second", "actor"
        )
    with pytest.raises(ValueError):
        GatewayConfig(
            "https://wrong.example/v1",
            cfg.model,
            "fixture",
            api_style=cfg.api_style,
            budget=auth,
            native_output_limits=True,
            max_output_tokens=None,
            max_input_tokens=131072,
            model_spec=spec(),
        )


def test_opencode_native_config_has_no_output_cap():
    from simready_bench.harbor_agents import _opencode_config

    v = _opencode_config(
        "custom",
        "openai-completions",
        {"contextWindow": 131072},
        {"max_input_tokens": 131072, "max_output_tokens": None},
        "",
        {},
    )
    assert v["provider"]["openai"]["models"]["custom"]["limit"] == {"context": 131072}


@pytest.mark.parametrize("native", [False, True])
def test_gateway_preserves_custom_declaration_in_both_protocols(monkeypatch, native):
    from simready_bench.local import _gateway
    from simready_bench.engine.inference_gateway import ProjectBudget
    from dataclasses import asdict

    value = spec()
    auth = ProjectBudget(
        project="simready-bench",
        authorization_id="fixture-only",
        approved=True,
        upstream_origin="https://models.example.org",
        model=value["model"],
        expires_at=time.time() + 100,
        maximum_requests=32,
        maximum_spend_usd=1,
        pricing_status="known",
        external_spend_cap_verified=False,
    )
    monkeypatch.setenv("TEST_UNUSED_KEY", "synthetic-fixture-only")
    c = {
        "condition_id": "fixture",
        "asset_id": "fixture",
        "model": {
            "adapter": "openai-completions",
            "model_id": value["model"],
            "provider_ref": "custom",
        },
        "harness": {"adapter": "harbor"},
        "settings": {
            "generation": {},
            "budgets": {
                "maximum_spend_usd": 1,
                "inference_requests": 32,
                "max_output_tokens": None if native else 4096,
                "max_input_tokens": 131072,
                "agent_seconds": 60,
            },
        },
    }
    if native:
        c["authoring_protocol"] = "authoring-32-native-output-v1"
    resolved = {
        "operator_policy": {
            "providers": {
                "custom": {
                    "kind": "openai",
                    "api_key_env": "TEST_UNUSED_KEY",
                    "base_url": value["base_url"],
                    "model_spec": value,
                    "project_budget": asdict(auth),
                }
            }
        }
    }
    result = _gateway(c, resolved)
    assert result.model_spec == value and result.native_output_limits == native
