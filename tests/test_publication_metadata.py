from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json

import pytest

from simready_bench.publishing import HuggingFacePublisher, PublicationError
from test_publishing import publication as publication


def with_comparison(row):
    alignment = {
        "scale": 1,
        "yaw_degrees": 90,
        "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "translation_m": [0.01, 0.02, 0.03],
    }
    row["fidelity"]["alignment"] = alignment
    rig = {
        "schema": "simready-illustration-rig-v1",
        "basis": "reference bounds in metres/Z-up",
        "center": [0, 0, 0],
        "span": 0.2,
        "projection": "orthographic",
        "aperture": 0.33,
        "lighting": "Storm fixed camera light, source lights disabled",
        "width": 512,
        "candidate_alignment": deepcopy(alignment),
        "views": {
            name: {
                "camera_to_world": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "clipping_range": [0.0002, 20],
            }
            for name in ("isometric", "front", "side")
        },
    }
    row["illustrations"] = {
        "rig": rig,
        "rig_sha256": rig_hash(rig),
    }
    return row


def rig_hash(rig):
    return hashlib.sha256(
        json.dumps(rig, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def old_publication(hub):
    path = "runs/metadata/results.json"
    old = json.loads(hub.files[path])
    for row in old["results"]:
        row.pop("illustrations", None)
        row["simready"].pop("score_lower_bound", None)
        row["simready"].pop("score_upper_bound", None)
        row["mandatory_lower_bound_percent"] = 95
        row["mandatory_upper_bound_percent"] = 100
    hub.files[path] = json.dumps(old).encode()


def test_existing_publication_accepts_only_canonical_derived_metadata_enrichment(
    publication,
):
    manifest, report, hub, row, save = publication
    source = with_comparison(row())
    source["simready"].update(
        score_lower_bound=95, score_upper_bound=100, completion_bonus=5
    )
    source["completion_bonus"] = 5
    publisher = HuggingFacePublisher("team/results", "metadata", manifest, api=hub)
    publisher(save([source]))
    old_publication(hub)
    callback = save([source])
    original = (report / "results.json").read_bytes()
    publisher(callback)
    public = json.loads(hub.files["runs/metadata/results.json"])["results"][0]
    assert public["simready"]["score_percent"] == 100
    assert public["simready"]["score_lower_bound"] == 100
    assert public["simready"]["score_upper_bound"] == 100
    assert public["illustrations"] == source["illustrations"]
    assert public["fidelity"]["alignment"] == source["fidelity"]["alignment"]
    scores = next(
        csv.DictReader(StringIO(hub.files["runs/metadata/scores.csv"].decode()))
    )
    assert float(scores["mandatory_lower_bound_percent"]) == 100
    assert float(scores["mandatory_upper_bound_percent"]) == 100
    assert scores["completion_bonus"] == "5"
    publisher(callback)
    assert (report / "results.json").read_bytes() == original


@pytest.mark.parametrize(
    "damage", ["rig", "hash", "remove_rig", "remove_hash", "remove_both", "alignment"]
)
def test_published_comparison_metadata_cannot_be_replaced_or_removed(
    publication, damage
):
    manifest, _, hub, row, save = publication
    source = with_comparison(row())
    publisher = HuggingFacePublisher("team/results", "metadata", manifest, api=hub)
    publisher(save([source]))
    changed = deepcopy(source)
    if damage == "rig":
        changed["illustrations"]["rig"]["center"][0] = 0.3
        changed["illustrations"]["rig_sha256"] = rig_hash(
            changed["illustrations"]["rig"]
        )
    elif damage == "hash":
        changed["illustrations"]["rig_sha256"] = "a" * 64
    elif damage == "remove_rig":
        changed["illustrations"].pop("rig")
    elif damage == "remove_hash":
        changed["illustrations"].pop("rig_sha256")
    elif damage == "remove_both":
        changed.pop("illustrations")
    else:
        changed["illustrations"]["rig"]["candidate_alignment"]["yaw_degrees"] = 91
        changed["illustrations"]["rig_sha256"] = rig_hash(
            changed["illustrations"]["rig"]
        )
    uploads = len(hub.uploads)
    with pytest.raises(PublicationError):
        publisher(save([changed]))
    assert len(hub.uploads) == uploads


@pytest.mark.parametrize(
    "damage", ["empty", "minimal", "schema", "camera", "views", "aperture", "nonfinite"]
)
def test_noncanonical_comparison_rig_cannot_be_added_to_a_publication(
    publication, damage
):
    manifest, _, hub, row, save = publication
    source = with_comparison(row())
    publisher = HuggingFacePublisher("team/results", "metadata", manifest, api=hub)
    publisher(save([source]))
    old_publication(hub)
    changed = deepcopy(source)
    rig = changed["illustrations"]["rig"]
    if damage == "empty":
        changed["fidelity"].pop("alignment")
        rig.clear()
    elif damage == "minimal":
        rig = {"candidate_alignment": rig["candidate_alignment"]}
        changed["illustrations"]["rig"] = rig
    elif damage == "schema":
        rig["schema"] = "other"
    elif damage == "camera":
        rig["views"]["front"]["camera_to_world"][0].pop()
    elif damage == "views":
        rig["views"] = {}
    elif damage == "aperture":
        rig["aperture"] = 0.4
    else:
        rig["center"][0] = float("nan")
    changed["illustrations"]["rig_sha256"] = rig_hash(rig)
    uploads = len(hub.uploads)
    with pytest.raises(PublicationError):
        publisher(save([changed]))
    assert len(hub.uploads) == uploads


@pytest.mark.parametrize("missing", ["rig", "rig_sha256"])
def test_incomplete_prior_comparison_metadata_is_not_treated_as_absent(
    publication, missing
):
    manifest, _, hub, row, save = publication
    source = with_comparison(row())
    publisher = HuggingFacePublisher("team/results", "metadata", manifest, api=hub)
    publisher(save([source]))
    previous = json.loads(hub.files["runs/metadata/results.json"])
    previous["results"][0]["illustrations"].pop(missing)
    hub.files["runs/metadata/results.json"] = json.dumps(previous).encode()
    uploads = len(hub.uploads)
    with pytest.raises(PublicationError):
        publisher(save([source]))
    assert len(hub.uploads) == uploads


@pytest.mark.parametrize(
    "damage",
    [
        "measurement",
        "score",
        "bonus",
        "candidate",
        "image",
        "trace",
        "validator",
        "alignment",
    ],
)
def test_additive_metadata_never_authorises_changed_measurements_or_evidence(
    publication, damage
):
    manifest, _, hub, row, save = publication
    source = with_comparison(row())
    source["simready"]["completion_bonus"] = 5
    publisher = HuggingFacePublisher("team/results", "metadata", manifest, api=hub)
    publisher(save([source]))
    old_publication(hub)
    previous = json.loads(hub.files["runs/metadata/results.json"])
    old = previous["results"][0]
    if damage == "measurement":
        old["fidelity"]["components"]["shape"]["score_percent"] = 5
    elif damage == "score":
        old["simready"]["score_percent"] = old["simready_reward"] = 0
        old["simready"]["passed_count"] = 0
    elif damage == "bonus":
        old["simready"]["completion_bonus"] = 0
    elif damage == "candidate":
        old["candidate"]["files"][0]["sha256"] = "a" * 64
    elif damage == "image":
        old["images"][0]["view"] = "candidate / front"
    elif damage == "trace":
        old["traces"][0]["sha256"] = "a" * 64
    elif damage == "validator":
        old["nvidia_validator_report_file"]["sha256"] = "a" * 64
    else:
        old["fidelity"]["alignment"]["translation_m"][0] += 1
    hub.files["runs/metadata/results.json"] = json.dumps(previous).encode()
    uploads = len(hub.uploads)
    with pytest.raises(PublicationError):
        publisher(save([source]))
    assert len(hub.uploads) == uploads
