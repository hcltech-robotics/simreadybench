from simready_bench.engine.controller import Controller
from simready_bench.engine.verifier import scoring_policy


def candidate_fixture(tmp_path):
    actor = tmp_path / "actor"
    (actor / "output").mkdir(parents=True)
    (actor / "output/asset.usd").write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    controller = Controller(tmp_path / "state")
    controller.register_episode(
        "ep",
        actor,
        task_id="task",
        trial_id="trial",
        inference_limit=32,
        deadline_seconds=60,
    )
    controller.start_episode("ep")
    return (
        actor,
        controller,
        {"task_id": "task", "validator_scoring": scoring_policy()},
        None,
        None,
    )
