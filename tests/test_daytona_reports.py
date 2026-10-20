import copy
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from types import SimpleNamespace

import pytest

from simready_bench.controller_runner import ControllerReporter, verify_reconstruction
from simready_bench.daytona import _collect_live_reports
from test_controller_runner import manifest, measured


def remote_reports(tmp_path):
    work, output = tmp_path / "work", tmp_path / "remote-output"
    work.mkdir()
    output.mkdir()
    reporter = ControllerReporter(output, manifest(), stream=io.StringIO())
    row = measured(work, 0)
    files = work / "c0/artifacts/submission/files"
    files.mkdir(parents=True)
    asset = b'#usda 1.0\ndef Xform "Box" {}\n'
    (files / "asset.usd").write_bytes(asset)
    roster = [
        {
            "relative_path": "asset.usd",
            "size_bytes": len(asset),
            "sha256": hashlib.sha256(asset).hexdigest(),
        }
    ]
    snapshot = hashlib.sha256(
        json.dumps(
            {"entrypoint": "asset.usd", "files": roster},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    receipt = files.parent / "snapshot-manifest.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "simready-bench-submission-v1",
                "entrypoint": "asset.usd",
                "files": roster,
                "snapshot_sha256": snapshot,
                "size_bytes": len(asset),
                "copied_independently": True,
            }
        )
    )
    row.update(
        snapshot_sha256=snapshot,
        submission={
            "directory": str(files),
            "manifest": str(receipt),
            "snapshot_sha256": snapshot,
            "copied_independently": True,
        },
    )
    image = work / "candidate.png"
    image.write_bytes(b"captured candidate image")
    row["illustrations"] = {
        "images": [
            {
                "file": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "view": "candidate / front",
            }
        ]
    }
    reporter(work, {"results": [row]})
    progress = json.loads((output / "reconstructions/results.json").read_text())
    return output, progress


class Remote:
    def __init__(self, root, *, corrupt=False):
        self.root, self.corrupt, self.downloads, self.archives = root, corrupt, 0, {}
        self.process = SimpleNamespace(exec=self.execute)
        self.fs = SimpleNamespace(download_file=self.download)

    def execute(self, command, timeout):
        if "sha256sum" not in command:
            assert "rm -f" in command
            return SimpleNamespace(exit_code=0, result="")
        identity = re.search(r"condition-[a-z0-9-]+", command).group()
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            directory = self.root / "reconstructions" / identity
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                if self.corrupt is True and path.name == "asset.usd":
                    raw += b"changed"
                member = tarfile.TarInfo(
                    identity + "/" + path.relative_to(directory).as_posix()
                )
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
            if self.corrupt in {
                "traversal",
                "absolute",
                "unexpected",
                "symlink",
                "duplicate",
            }:
                name = {
                    "traversal": identity + "/../outside",
                    "absolute": "/tmp/outside",
                    "unexpected": identity + "/private/operator.json",
                    "symlink": identity + "/evidence/link",
                    "duplicate": identity + "/results.json",
                }[self.corrupt]
                member = tarfile.TarInfo(name)
                if self.corrupt == "symlink":
                    member.type, member.linkname = tarfile.SYMTYPE, "/tmp/outside"
                    archive.addfile(member)
                else:
                    member.size = 7
                    archive.addfile(member, io.BytesIO(b"outside"))
        raw = data.getvalue()
        remote = re.search(
            r"/tmp/simready-reconstruction-[a-z0-9-]+\.tar\.gz", command
        ).group()
        self.archives[remote] = raw
        return SimpleNamespace(
            exit_code=0, result=hashlib.sha256(raw).hexdigest() + " " + remote
        )

    def download(self, remote, destination, timeout):
        assert timeout == 300
        self.downloads += 1
        Path(destination).write_bytes(self.archives[remote])


def test_completed_item_collects_its_entire_verified_closure_and_callback_once(
    tmp_path,
):
    remote, progress = remote_reports(tmp_path)
    sandbox = Remote(remote)
    output = tmp_path / "collected"
    output.mkdir()
    record = {"status": "running"}
    callbacks = []

    def published(report, *, final=False):
        portable = json.loads(Path(report["results"]).read_text())
        row = portable["results"][0]
        root = Path(report["directory"])
        assert (root / row["candidate"]["entrypoint"]).is_file()
        assert (root / "evidence" / row["images"][0]["path"]).is_file()
        assert (root / "evidence" / row["traces"][0]["path"]).is_file()
        callbacks.append((report, final))

    _collect_live_reports(
        sandbox, output, progress, record, save=lambda: None, on_report=published
    )
    assert sandbox.downloads == 1 and len(callbacks) == 1
    assert callbacks[0][1] is False and record["status"] == "running"
    identity = progress["results"][0]["condition_id"]
    assert record["collected_reconstruction_ids"] == [identity]
    assert len(record["collected_reconstruction_manifests"][identity]) == 64
    verify_reconstruction(
        output / "reconstructions" / identity, identity, progress["results"][0]
    )
    _collect_live_reports(
        sandbox, output, progress, record, save=lambda: None, on_report=published
    )
    assert sandbox.downloads == 1 and len(callbacks) == 1


def test_changed_candidate_is_not_installed_or_published(tmp_path):
    remote, progress = remote_reports(tmp_path)
    output = tmp_path / "collected"
    output.mkdir()
    record, callbacks = {"status": "running"}, []
    _collect_live_reports(
        Remote(remote, corrupt=True),
        output,
        progress,
        record,
        save=lambda: None,
        on_report=lambda *args, **kwargs: callbacks.append(args),
    )
    assert record["reconstruction_collection_error"] == "ValueError"
    assert record["status"] == "running" and not record.get(
        "collected_reconstruction_ids"
    )
    assert callbacks == []
    assert not (output / "reconstructions/condition-0").exists()


@pytest.mark.parametrize(
    "corrupt", ["traversal", "absolute", "unexpected", "symlink", "duplicate"]
)
def test_unsafe_or_repeated_archive_members_are_rejected(tmp_path, corrupt):
    remote, progress = remote_reports(tmp_path)
    output = tmp_path / "collected"
    output.mkdir()
    record, callbacks = {"status": "running"}, []
    _collect_live_reports(
        Remote(remote, corrupt=corrupt),
        output,
        progress,
        record,
        save=lambda: None,
        on_report=lambda *args, **kwargs: callbacks.append(args),
    )
    assert record.get("reconstruction_collection_error") in {
        "ValueError",
        "FileExistsError",
    }
    assert record["status"] == "running" and callbacks == []
    assert not (output / "reconstructions/condition-0").exists()
    assert not (output / "outside").exists()


def test_publication_failure_retries_only_publication_not_collection(tmp_path):
    remote, progress = remote_reports(tmp_path)
    output = tmp_path / "collected"
    output.mkdir()
    sandbox, record, callbacks = Remote(remote), {"status": "running"}, []

    def unavailable(report, *, final=False):
        raise ConnectionError("repository unavailable")

    _collect_live_reports(
        sandbox, output, progress, record, save=lambda: None, on_report=unavailable
    )
    assert record["report_callback_error"] == "ConnectionError"
    assert record["status"] == "running" and sandbox.downloads == 1
    _collect_live_reports(
        sandbox,
        output,
        progress,
        record,
        save=lambda: None,
        on_report=lambda report, **kwargs: callbacks.append(report),
    )
    assert len(callbacks) == 1 and sandbox.downloads == 1
    assert "report_callback_error" not in record


def test_collection_has_a_per_poll_limit(tmp_path):
    remote, progress = remote_reports(tmp_path)
    output = tmp_path / "collected"
    output.mkdir()
    row = copy.deepcopy(progress["results"][0])
    row["condition_id"] = "condition-1"
    progress["results"].append(row)
    sandbox, record = Remote(remote), {"status": "running"}
    _collect_live_reports(
        sandbox, output, progress, record, save=lambda: None, maximum_items=1
    )
    assert sandbox.downloads == 1
    assert record["collected_reconstruction_ids"] == ["condition-0"]


def test_final_collection_refreshes_index_and_preserves_immutable_item(tmp_path):
    import shutil
    from simready_bench.daytona import _collect_portable

    remote, progress = remote_reports(tmp_path)
    output = tmp_path / "collected"
    output.mkdir()
    _collect_live_reports(Remote(remote), output, progress, {}, save=lambda: None)
    before = (output / "reconstructions/condition-0/artifacts.json").read_bytes()
    (output / "reconstructions/results.json").write_text('{"results":[]}')
    final = tmp_path / "final"
    final.mkdir()
    shutil.copytree(remote / "reconstructions", final / "reconstructions")
    _collect_portable(final, output)
    assert (
        json.loads((output / "reconstructions/results.json").read_text())[
            "completed_conditions"
        ]
        == 1
    )
    assert (
        output / "reconstructions/condition-0/artifacts.json"
    ).read_bytes() == before


def test_reconnect_retries_final_publication_without_recreating_compute(tmp_path):
    from daytona.common.errors import DaytonaNotFoundError
    from simready_bench.daytona import DaytonaReportError, reconnect_daytona

    configuration = manifest()
    rows = [
        {
            "asset_id": "box",
            "repetition": index,
            "status": "completed",
            "simready_reward": 100,
        }
        for index in range(4)
    ]
    remote = tmp_path / "daytona-executor"
    remote.mkdir()
    (remote / "run-result.json").write_text(
        json.dumps(
            {"planned_conditions": 4, "completed_conditions": 4, "results": rows}
        )
    )
    (tmp_path / "report.html").write_text("completed reconstructions")
    (tmp_path / "results.json").write_text(json.dumps({"results": rows}))
    (tmp_path / "daytona-delivery.json").write_text(
        json.dumps(
            {
                "sandbox_id": "deleted-executor",
                "controller_exit": 0,
                "terminal_confirmed": True,
                "collection": "completed",
                "status": "completed",
            }
        )
    )

    class Deleted:
        def get(self, identity):
            assert identity == "deleted-executor"
            raise DaytonaNotFoundError("deleted")

        def create(self, *args, **kwargs):
            raise AssertionError("Publication must never recreate compute")

    def unavailable(report, *, final=False):
        assert final
        raise ConnectionError("repository unavailable")

    with pytest.raises(DaytonaReportError) as captured:
        reconnect_daytona(
            configuration,
            policy={},
            output_dir=tmp_path,
            client=Deleted(),
            on_report=unavailable,
        )
    assert captured.value.result["completed_conditions"] == 4
    saved = json.loads((tmp_path / "daytona-delivery.json").read_text())
    assert saved["status"] == "completed" and saved["cleanup_confirmed"] is True
    callbacks = []
    result = reconnect_daytona(
        configuration,
        policy={},
        output_dir=tmp_path,
        client=Deleted(),
        on_report=lambda report, **kwargs: callbacks.append((report, kwargs)),
    )
    assert result["completed_conditions"] == 4
    assert len(callbacks) == 1 and callbacks[0][1] == {"final": True}
