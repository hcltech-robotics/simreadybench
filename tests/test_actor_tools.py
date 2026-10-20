from PIL import Image

from helpers import candidate_fixture
from simready_bench.engine.tools import ActorTools, _result


def actor_tools(tmp_path):
    _, controller, _, _, _ = candidate_fixture(tmp_path)
    thumbnail = tmp_path / "thumbnail.png"
    Image.new("RGB", (2, 2), "white").save(thumbnail)

    class ValidationTools(ActorTools):
        calls = 0

        def _validation_job(self, *, binding, job, payload, arguments):
            self.calls += 1
            return _result({**payload, "status": "completed"})

    tools = ValidationTools(controller)
    tools.bind_episode(
        "ep", "episode-scoped-test-token", thumbnail, "Reconstruct the reference asset."
    )
    return tools, controller


def test_actor_roster_contains_only_available_operations(tmp_path):
    tools, _ = actor_tools(tmp_path)
    definitions = tools.describe("episode-scoped-test-token").structured_content[
        "tools"
    ]
    assert [tool["name"] for tool in definitions] == [
        "get_task_thumbnail",
        "validate_candidate",
        "submit_asset",
    ]
    validation = definitions[1]["inputSchema"]
    assert set(validation["properties"]) == {"request_id"}
    assert validation["required"] == ["request_id"]
    assert validation["additionalProperties"] is False
    thumbnail = tools.invoke("episode-scoped-test-token", "get_task_thumbnail", {})
    assert thumbnail.structured_content["thumbnail"]["width"] == 2
    assert thumbnail.content[1].type == "image"


def test_unavailable_preview_and_visual_arguments_reserve_no_validation(tmp_path):
    tools, controller = actor_tools(tmp_path)
    preview = tools.invoke(
        "episode-scoped-test-token",
        "render_candidate",
        {"request_id": "one", "views": ["+X"]},
    )
    visual = tools.invoke(
        "episode-scoped-test-token",
        "validate_candidate",
        {"request_id": "one", "visual_feedback": False},
    )
    assert preview.structured_content["status"] == "unknown_operation"
    assert visual.structured_content["status"] == "invalid_arguments"
    assert preview.is_error and visual.is_error
    assert controller.get_episode("ep").validation_calls_used == 0
    assert tools.calls == 0


def test_validation_retry_returns_same_frozen_result_without_another_call(tmp_path):
    tools, controller = actor_tools(tmp_path)
    first = tools.invoke(
        "episode-scoped-test-token", "validate_candidate", {"request_id": "one"}
    )
    retry = tools.invoke(
        "episode-scoped-test-token", "validate_candidate", {"request_id": "one"}
    )
    assert first.structured_content["status"] == "completed"
    assert first.structured_content["snapshot_sha256"]
    assert retry == first
    assert tools.calls == controller.get_episode("ep").validation_calls_used == 1
