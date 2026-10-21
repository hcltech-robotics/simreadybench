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


def test_duplicate_completion_cannot_extend_the_failure_window(tmp_path):
    clock = [100]
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    assert not health.observe([{'attempt_id': '1/c', 'status': 'infrastructure_error'}], active=True)
    clock[0] = 119
    assert not health.observe([{'attempt_id': '2/c', 'status': 'completed'}], active=True)
    clock[0] = 125
    assert not health.observe([{'attempt_id': '3/c', 'status': 'infrastructure_error'}], active=True)
    clock[0] = 140
    assert not health.observe([{'attempt_id': '2/c', 'status': 'completed'}], active=True)
    clock[0] = 145
    assert health.observe([{'attempt_id': '2/c', 'status': 'completed'}], active=True)
    assert health.state['reason'] == 'no_completed_result_progress'


def test_a_long_healthy_generation_is_not_a_failure(tmp_path):
    clock = [100]
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    clock[0] = 1000
    assert not health.observe([], active=True)


def test_late_first_failure_does_not_immediately_stop_a_healthy_peer(tmp_path):
    clock = [100]
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    clock[0] = 1000
    failed = [{'attempt_id': 'late/c', 'status': 'infrastructure_error'}]
    assert not health.observe(failed, active=True)
    assert health.state['failure_window_started_at'] == 1000
    clock[0] = 1019
    assert not health.observe(failed, active=True)
    clock[0] = 1020
    assert health.observe(failed, active=True)
    assert health.state['reason'] == 'no_completed_result_progress'


def test_failure_window_survives_restart_and_is_not_extended_by_more_failures(tmp_path):
    clock = [100]
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    assert not health.observe([{'attempt_id': 'first/c', 'status': 'infrastructure_error'}], active=True)
    clock[0] = 119
    health = ExecutionHealth(tmp_path, progress_seconds=20, now=lambda: clock[0])
    assert not health.observe([{'attempt_id': 'second/c', 'status': 'infrastructure_error'}], active=True)
    assert health.state['failure_window_started_at'] == 100
    clock[0] = 120
    assert health.observe([], active=True)
