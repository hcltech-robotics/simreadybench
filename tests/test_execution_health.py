from simready_bench.execution_health import ExecutionHealth


def test_repeated_failures_open_a_persistent_circuit(tmp_path):
    health = ExecutionHealth(tmp_path, now=lambda: 100)
    failures = [{'attempt_id': str(n), 'status': 'infrastructure_error'} for n in range(5)]
    assert health.observe(failures, active=True)
    assert health.state['reason'] == 'repeated_infrastructure_failures'
    assert ExecutionHealth(tmp_path, now=lambda: 101).opened
    assert health.state['failures_since_completion'] == 5


def test_repeated_observation_does_not_spend_the_failure_budget_twice(tmp_path):
    health = ExecutionHealth(tmp_path, now=lambda: 100)
    failed = [{'attempt_id': 'worker/condition', 'status': 'infrastructure_error'}]
    for _ in range(20):
        assert not health.observe(failed, active=True)
    assert health.state['failures_since_completion'] == 1


def test_only_a_new_completed_result_resets_progress(tmp_path):
    clock = [100]
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    assert not health.observe([{'attempt_id': '1/c', 'status': 'infrastructure_error'}], active=True)
    clock[0] = 119
    assert not health.observe([{'attempt_id': '2/c', 'status': 'completed'}], active=True)
    clock[0] = 125
    assert not health.observe([{'attempt_id': '3/c', 'status': 'infrastructure_error'}], active=True)
    clock[0] = 140
    assert health.observe([{'attempt_id': '2/c', 'status': 'completed'}], active=True)
    assert health.state['reason'] == 'no_completed_result_progress'


def test_a_long_healthy_generation_is_not_a_failure(tmp_path):
    clock = [100]
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    clock[0] = 1000
    assert not health.observe([], active=True)
