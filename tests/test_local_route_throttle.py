from simready_bench.local_route_throttle import local_route_throttles
from simready_bench.route_throttle import RouteThrottleClient


def resolved(tmp_path):
    return {
        "operator_policy": {
            "runtime": {"authority_state_dir": str(tmp_path)},
            "providers": {"hf": {"kind": "huggingface", "base_url": "https://router.huggingface.co/v1",
                "api_key_env": "LOCAL_THROTTLE_TEST_KEY", "project_budget": {"model": "fixture"}}},
        },
        "conditions": [{"model": {"provider_ref": "hf", "model_id": "fixture"}}],
    }


def test_local_coordinators_share_durable_route_capacity_and_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_THROTTLE_TEST_KEY", "private-test-credential")
    config = resolved(tmp_path)
    with local_route_throttles(config) as first, local_route_throttles(config) as second:
        one, two = RouteThrottleClient(**first[("hf", "fixture")]), RouteThrottleClient(**second[("hf", "fixture")])
        assert one.url != two.url
        leases = [one.acquire(timeout=5) for _ in range(4)]
        try:
            denied = two._post("/acquire", {"request_id": "a" * 32})
            assert denied["admitted"] is False
        finally:
            for lease in leases:
                lease.finish(status_code=200)
        with two.acquire(timeout=5) as lease:
            lease.finish(status_code=429, headers={"Retry-After": "60"})
        assert one._post("/acquire", {"request_id": "b" * 32})["admitted"] is False


def test_other_routes_do_not_start_an_unrequested_governor(tmp_path):
    config = resolved(tmp_path)
    config["operator_policy"]["providers"]["hf"].update(kind="openai", base_url="https://api.openai.com/v1")
    with local_route_throttles(config) as descriptors:
        assert descriptors == {}
    assert not (tmp_path / "route-throttles").exists()


def test_models_using_the_same_provider_have_independent_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_THROTTLE_TEST_KEY", "private-test-credential")
    config = resolved(tmp_path)
    config["conditions"].append({"model": {"provider_ref": "hf", "model_id": "another"}})
    with local_route_throttles(config) as descriptors:
        assert len(descriptors) == 2
        assert descriptors[("hf", "fixture")]["route_key"] != descriptors[("hf", "another")]["route_key"]


def test_explicit_route_limit_is_shared_and_does_not_change_other_models(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCAL_THROTTLE_TEST_KEY', 'private-test-credential')
    config = resolved(tmp_path)
    config['operator_policy']['runtime']['inference_route_concurrency'] = {'fixture': 12}
    config['conditions'].append({'model': {'provider_ref': 'hf', 'model_id': 'another'}})
    with local_route_throttles(config) as first, local_route_throttles(config) as second:
        one = RouteThrottleClient(**first[('hf', 'fixture')])
        two = RouteThrottleClient(**second[('hf', 'fixture')])
        leases = [one.acquire(timeout=5) for _ in range(12)]
        try:
            assert two._post('/acquire', {'request_id': 'c' * 32})['admitted'] is False
        finally:
            for lease in leases:
                lease.finish(status_code=200)
        other = RouteThrottleClient(**second[('hf', 'another')])
        leases = [other.acquire(timeout=5) for _ in range(4)]
        try:
            assert other._post('/acquire', {'request_id': 'd' * 32})['admitted'] is False
        finally:
            for lease in leases:
                lease.finish(status_code=200)
