from copy import deepcopy
import hashlib, json
import pytest
from simready_bench.engine.scoring import public_report
from simready_bench.engine.validation_reports import ReportStore


def test_original_messages_suggestions_and_every_candidate_part_are_preserved(tmp_path):
    raw = {
        "asset_path": str(tmp_path / "candidate/asset.usd"),
        "passed": False,
        "issues": [
            {
                "rule_id": "com.nvidia.usd.VG.014",
                "severity": "failure",
                "prim_path": "/Asset/Part_" + str(i),
                "message": "NVIDIA original detailed diagnostic " + str(i),
                "suggestions": [
                    {
                        "message": "Original NVIDIA repair suggestion",
                        "location": "/Asset/Part_" + str(i),
                    }
                ],
            }
            for i in range(100)
        ],
    }
    before = deepcopy(raw)
    public = public_report(raw, candidate_path=tmp_path / "candidate")
    assert (
        public["issues"] == raw["issues"]
        and len(public["issues"]) == 100
        and raw == before
    )
    assert public["asset_path"] == "/output/asset.usd"
    store = ReportStore(tmp_path / "reports")
    d = store.register("ep", public)
    assert (
        d["delivery"] == "agent_file"
        and d["report"] is None
        and d["issue_counts"]["issues"] == 100
    )
    assert d["file_path"].endswith("/" + d["sha256"] + ".json")


def test_large_report_pages_reconstruct_exact_json_and_are_episode_scoped(tmp_path):
    report = {
        "issues": [
            {"prim_path": "/Asset/P" + str(i), "message": "Original message 😀 " * 200}
            for i in range(100)
        ]
    }
    store = ReportStore(tmp_path / "reports")
    d = store.register("one", report)
    assert (
        d["delivery"] == "agent_file"
        and d["complete_report_available"] is True
        and d["report"] is None
    )
    text = ""
    offset = 0
    while offset is not None:
        page = store.read(
            "one", {"report_id": d["report_id"], "offset": offset, "limit": 203}
        )
        text += page["text"]
        offset = page["next_offset"]
    assert (
        json.loads(text) == report
        and hashlib.sha256(text.encode()).hexdigest() == d["sha256"]
    )
    assert len(json.loads(text)["issues"]) == 100
    with pytest.raises(ValueError):
        store.read("other", {"report_id": d["report_id"]})
    with pytest.raises(ValueError):
        store.read("one", {"report_id": "../etc/passwd"})


def test_redaction_removes_host_and_credential_data_without_replacing_USD_paths():
    report = {
        "issues": [
            {
                "prim_path": "/Asset/Collision",
                "message": "Failure at /Asset/Collision in /home/operator/private/run/asset.usd",
                "suggestions": [{"message": "Set token[] labels on /Asset/Geometry"}],
            }
        ],
        "logs": [{"message": "Bearer SECRET_EXAMPLE_VALUE"}],
        "inventory": {"profile_manifest_path": "/home/operator/sdk/profile.toml"},
    }
    public = public_report(report)
    encoded = json.dumps(public)
    assert "SECRET_EXAMPLE_VALUE" not in encoded and "/home/operator" not in encoded
    assert public["issues"][0]["prim_path"] == "/Asset/Collision"
    assert public["issues"][0]["suggestions"] == report["issues"][0]["suggestions"]


def test_actual_reference_tool_delivers_original_report_and_paginated_tool(tmp_path):
    from types import SimpleNamespace
    from simready_bench.engine.harbor_services import ReferenceActorTools

    tools = object.__new__(ReferenceActorTools)
    tools.validator = object()
    report = {
        "schema_version": "simready-evaluation-validator-v1",
        "nvidia_validator_report": {
            "issues": [
                {
                    "rule_id": "com.nvidia.usd.VG.014",
                    "message": "Original NVIDIA message",
                    "prim_path": "/Asset/First",
                },
                {
                    "rule_id": "com.nvidia.usd.VG.014",
                    "message": "Original second message",
                    "prim_path": "/Asset/Second",
                },
            ]
        },
        "feedback": {
            "schema_version": "simready-validator-feedback-v1",
            "simready": {"checks": []},
        },
    }
    tools.services = SimpleNamespace(representation_wrappers={})
    tools._adapter = lambda *args: lambda job: report
    saved = []
    tools._private = lambda *args: saved.append(args)
    tools.validator_reports = ReportStore(tmp_path / "reports")
    result = tools._validation_job(
        binding=SimpleNamespace(episode_id="ep"),
        job=SimpleNamespace(job_id="job", episode_id="ep"),
        payload={"status": "completed"},
        arguments={},
    )
    data = result.structured_content
    received = data["feedback"]["nvidia_validator_report"]
    assert (
        received["report"] is None
        and received["delivery"] == "agent_file"
        and received["issue_counts"]["issues"] == 2
    )
    assert "Original NVIDIA message" not in result.model_dump_json()
    tools._binding = lambda token: (
        SimpleNamespace(episode_id="ep") if token == "authorized" else None
    )
    page = tools.invoke(
        "authorized", "get_validation_report", {"report_id": received["report_id"]}
    )
    assert (
        json.loads(page.structured_content["text"]) == report["nvidia_validator_report"]
    )
    assert (
        tools.invoke(
            "unauthorized",
            "get_validation_report",
            {"report_id": received["report_id"]},
        ).is_error
        is True
    )


def test_candidate_USD_sites_with_host_like_names_are_preserved():
    sites = ["/private/Geometry", "/tmp/Part", "/home/Asset"]
    raw = {
        "issues": [
            {
                "prim_path": site,
                "message": "Failure on "
                + site
                + " at /home/operator/private/job/asset.usd",
                "suggestions": [{"location": site, "message": "Inspect " + site}],
            }
            for site in sites
        ]
    }
    projected = public_report(raw)
    for site, issue in zip(sites, projected["issues"]):
        assert (
            issue["prim_path"] == site and issue["suggestions"][0]["location"] == site
        )
        assert (
            site in issue["message"]
            and "/home/operator/private/job" not in issue["message"]
        )


def test_credential_keys_are_redacted_without_changing_semantic_token_values():
    projected = public_report(
        {
            "api_key": "SECRET_FIXTURE",
            "headers": {
                "Authorization": "Bearer TOKEN_FIXTURE",
                "x-api-key": "SECOND_FIXTURE",
            },
            "issues": [{"message": "A token[] value is required", "token": "Q123"}],
        }
    )
    text = json.dumps(projected)
    assert (
        "SECRET_FIXTURE" not in text
        and "TOKEN_FIXTURE" not in text
        and "SECOND_FIXTURE" not in text
    )
    assert projected["issues"] == [
        {"message": "A token[] value is required", "token": "Q123"}
    ]


def test_report_over_64_mib_reconstructs_every_finding_without_validation_cost(
    tmp_path,
):
    message = "Original NVIDIA diagnostic 😀 " + ("x" * 65536)
    report = {
        "issues": [
            {
                "prim_path": "/Asset/Part_" + str(i),
                "message": message,
                "suggestions": [{"message": "Original repair suggestion"}],
            }
            for i in range(1024)
        ]
    }
    store = ReportStore(tmp_path / "reports")
    descriptor = store.register("episode", report)
    assert descriptor["bytes"] > 64 * 1024**2 and descriptor["report"] is None
    digest = hashlib.sha256()
    offset = 0
    pages = 0
    reconstructed = tmp_path / "reconstructed.json"
    with reconstructed.open("wb") as output:
        while offset is not None:
            page = store.read(
                "episode",
                {
                    "report_id": descriptor["report_id"],
                    "offset": offset,
                    "limit": 65536,
                },
            )
            assert (
                page["validation_calls_consumed"] == 0 and not page["report_truncated"]
            )
            data = page["text"].encode()
            output.write(data)
            digest.update(data)
            offset = page["next_offset"]
            pages += 1
    assert pages > 1024 and page["complete"] is True
    assert digest.hexdigest() == descriptor["sha256"]
    with reconstructed.open() as stream:
        assert json.load(stream) == report


@pytest.mark.parametrize("change", ["content", "replace", "symlink", "hardlink"])
def test_changed_or_linked_report_is_unavailable(tmp_path, change):
    import os

    store = ReportStore(tmp_path / "reports")
    descriptor = store.register("episode", {"issues": []})
    path = tmp_path / "reports" / (descriptor["report_id"] + ".json")
    if change == "content":
        path.write_text("changed")
    elif change == "replace":
        path.unlink()
        path.write_text('{"issues": []}\n')
    elif change == "symlink":
        target = tmp_path / "outside"
        target.write_text("private")
        path.unlink()
        path.symlink_to(target)
    else:
        os.link(path, tmp_path / "linked")
    with pytest.raises(ValueError):
        store.read("episode", {"report_id": descriptor["report_id"]})


def test_actor_proxy_saves_large_complete_file_without_returning_its_text(
    tmp_path, monkeypatch
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from simready_bench.engine import actor_mcp

    message = "Original diagnostic with Unicode 😀 " + ("x" * 65536)
    report = {
        "issues": [
            {
                "rule_id": "com.nvidia.usd.VG.014",
                "prim_path": "/Asset/Part_" + str(i),
                "message": message,
                "suggestions": [{"message": "Repair the original issue"}],
            }
            for i in range(1024)
        ]
    }
    store = ReportStore(tmp_path / "private-reports")
    descriptor = store.register("episode", report)
    assert descriptor["bytes"] > 64 * 1024**2

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert (
                self.path == "/v1/tools/get_validation_report"
                and self.headers["Authorization"] == "Bearer episode-token"
            )
            arguments = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            raw = json.dumps(
                {"structuredContent": store.read("episode", arguments)}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(actor_mcp, "REPORT_ROOT", tmp_path / "agent-reports")
    try:
        result = actor_mcp.materialize_validation_report(
            {
                "structuredContent": {
                    "feedback": {"nvidia_validator_report": descriptor}
                },
                "content": [],
            },
            "episode-token",
            "http://127.0.0.1:" + str(server.server_port),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    received = result["structuredContent"]["feedback"]["nvidia_validator_report"]
    path = __import__("pathlib").Path(received["file_path"])
    assert (
        path.stat().st_size == descriptor["bytes"] and path.stat().st_mode & 0o222 == 0
    )
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == descriptor["sha256"]
    with path.open() as stream:
        assert json.load(stream) == report
    assert received["file_available"] and received["validation_calls_consumed"] == 0
    assert len(json.dumps(result)) < 4096 and message not in json.dumps(result)
