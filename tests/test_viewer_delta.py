"""Mesh differences preserve physical scale and measure both triangle surfaces."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_viewer_delta", ROOT / "scripts/build_viewer_delta.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def square(scale=1, z=0):
    return trimesh.Trimesh(vertices=[[-scale, -scale, z], [scale, -scale, z],
                                      [scale, scale, z], [-scale, scale, z]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)


def test_identical_surface_is_zero_and_sampling_is_reproducible():
    mesh = trimesh.creation.box(extents=[2, 4, 6])
    first = builder.surface_samples(mesh, 1024, 37)
    assert np.array_equal(first, builder.surface_samples(mesh, 1024, 37))
    assert not np.array_equal(first, builder.surface_samples(mesh, 1024, 38))
    rows = builder.delta_rows(mesh, mesh, first, builder.surface_samples(mesh, 1024, 38))
    assert rows.dtype == np.dtype("<f4")
    assert rows.shape == (2048, 4)
    assert np.max(rows[:, 3]) < 1e-12


def test_translated_parallel_planes_preserve_translation_and_distance_both_ways():
    reference, candidate = square(z=0), square(z=2)
    a = builder.surface_samples(reference, 500, 1)
    b = builder.surface_samples(candidate, 500, 2)
    rows = builder.delta_rows(reference, candidate, a, b)
    np.testing.assert_array_equal(rows[:500, 2], 0)
    np.testing.assert_array_equal(rows[500:, 2], 2)
    np.testing.assert_allclose(rows[:, 3], 2, atol=1e-12)


def test_scale_difference_survives_and_bidirectional_distance_detects_excess_geometry():
    reference, candidate = square(1), square(2)
    a = builder.surface_samples(reference, 1024, 9)
    b = builder.surface_samples(candidate, 1024, 10)
    rows = builder.delta_rows(reference, candidate, a, b)
    assert np.max(rows[:1024, 3]) < 1e-12
    assert np.count_nonzero(rows[1024:, 3] > .1) > 600
    assert rows[1024:, :2].max() > 1.9
    # Analytic distance from a point to the smaller square in the same plane.
    expected = np.linalg.norm(np.maximum(np.abs(b[:, :2]) - 1, 0), axis=1)
    np.testing.assert_allclose(rows[1024:, 3], expected, rtol=1e-6, atol=1e-7)


def test_distance_is_to_triangle_interior_not_nearest_vertex_or_sample():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [4, 0, 0], [0, 4, 0]],
                           faces=[[0, 1, 2]], process=False)
    distances = builder.triangle_distances(mesh, np.array([[1, 1, 2], [3, 3, 0]]))
    np.testing.assert_allclose(distances, [2, np.sqrt(2)], atol=1e-12)


@pytest.mark.parametrize("scale", [1, .001, .000001])
def test_triangle_projection_preserves_small_surface_features(scale):
    triangle = np.array([[[0, 0, 0], [4, 0, 0], [0, 4, 0]]]) * scale
    point = np.array([[1, 1, .2]]) * scale
    np.testing.assert_allclose(builder.point_triangle_squared(triangle, point),
                               [.04 * scale ** 2], rtol=1e-12, atol=0)
    point[:, 2] = 0
    np.testing.assert_array_equal(builder.point_triangle_squared(triangle, point), [0])


def test_area_weighting_tracks_surface_area_and_queries_match_brute_force():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                      [10, 0, 0], [12, 0, 0], [10, 2, 0]],
                           faces=[[0, 1, 2], [3, 4, 5]], process=False)
    points = builder.surface_samples(mesh, 20_000, 123)
    assert .79 < np.mean(points[:, 0] > 5) < .81
    queries = np.random.default_rng(42).uniform([-2, -2, -2], [14, 4, 2], size=(100, 3))
    expected = trimesh.proximity.closest_point_naive(mesh, queries)[1]
    np.testing.assert_allclose(builder.triangle_distances(mesh, queries, batch_size=7), expected, atol=1e-12)


def test_invalid_points_counts_and_escaping_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        builder.surface_samples(square(), 0, 1)
    with pytest.raises(ValueError, match="finite 3D"):
        builder.triangle_distances(square(), np.array([[np.nan, 0, 0]]))
    with pytest.raises(ValueError, match="escapes"):
        builder.safe_path(tmp_path, "../outside.glb")


def test_published_clouds_keep_hashes_counts_and_frames():
    site = ROOT / "site"
    source_bytes = (site / "assets/explorer/manifest.json").read_bytes()
    source = json.loads(source_bytes)
    manifest = json.loads((site / "assets/explorer/delta-manifest.json").read_text())
    assert manifest["source_manifest_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert manifest["samples_per_direction"] == 12_000
    variants = [(obj, variant) for obj in source['objects'] for variant in obj['variants']]
    assert len(manifest['pairs']) == len(variants)
    assert sum(p['src'] is not None for p in manifest['pairs'].values()) == sum(v.get('src') is not None for _,v in variants)
    if any(str(obj['reference']['src']).startswith('https://') for obj in source['objects']):
        # Hosted geometry is not vendored into a source checkout. Check the
        # content-addressed joins here; distance math is exercised above.
        receipt_bytes = (site / 'assets/explorer/geometry-provenance.json').read_bytes()
        receipts = json.loads(receipt_bytes)
        assert manifest['source_geometry_provenance_sha256'] == hashlib.sha256(receipt_bytes).hexdigest()
        prefix = 'https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/'
        for obj,variant in variants:
            pair = manifest['pairs'][f"{obj['id']}/{variant['id']}"]
            assert obj['reference']['src'].startswith(prefix)
            assert variant['run_id'] == source['job_id']
            assert pair['source_reference_sha256'] == obj['reference']['sha256']
            assert pair['source_candidate_sha256'] == variant['sha256']
            if variant.get('src') is None:
                assert pair['src'] is None
                continue
            assert variant['src'].startswith(prefix) and pair['src'].startswith(prefix)
            assert pair['point_count'] == 24_000 and pair['reference_point_count'] == 12_000
            assert 0 <= pair['p95_distance_m'] <= pair['max_distance_m']
            for identity,item in [(f"{obj['id']}/reference",obj['reference']),(f"{obj['id']}/{variant['id']}",variant)]:
                assert receipts[identity]['glb_sha256'] == item['sha256']
                assert receipts[identity]['coordinates'] == 'metres/Y-up'
            assert pair['alignment_status'] == ('recorded' if receipts[f"{obj['id']}/{variant['id']}"]['alignment_m_z_up'] else 'authored')
        return
    for obj in source["objects"]:
        reference = builder.load_mesh(site, obj["reference"])
        previous_reference_points = None
        for variant in obj["variants"]:
            pair = manifest["pairs"][f"{obj['id']}/{variant['id']}"]
            if not variant["src"]:
                assert pair["src"] is None
                assert pair["max_distance_m"] is None
                assert pair["reason"] == "Candidate geometry unavailable"
                continue
            candidate = builder.load_mesh(site, variant)
            payload = (site / pair["src"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == pair["sha256"]
            assert pair["source_reference_sha256"] == obj["reference"]["sha256"]
            assert pair["source_candidate_sha256"] == variant["sha256"]
            assert pair["alignment_status"] == ("authored" if obj["id"] == "food_beverage_coffeeCup_a01"
                                                 and variant["id"] == "nemotron-pi" else "recorded")
            rows = np.frombuffer(payload, dtype="<f4").reshape(-1, 4)
            split = pair["reference_point_count"]
            assert len(rows) == pair["point_count"] == 24_000
            assert split == 12_000
            assert np.isfinite(rows).all() and (rows[:, 3] >= 0).all()
            assert float(rows[:, 3].max()) == pair["max_distance_m"]
            assert float(np.percentile(rows[:, 3].astype(np.float64), 95)) == pair["p95_distance_m"]
            assert pair["reference_diagonal_m"] == pytest.approx(np.linalg.norm(reference.extents))
            if previous_reference_points is not None:
                np.testing.assert_array_equal(rows[:split, :3], previous_reference_points)
            previous_reference_points = rows[:split, :3]
            # Stratified checks across the file recompute true surface distances
            # from the stored float32 positions in both query directions.
            for selected, target, own in ((rows[:split:997], candidate, reference),
                                          (rows[split::997], reference, candidate)):
                np.testing.assert_allclose(builder.triangle_distances(target, selected[:, :3]),
                                           selected[:, 3], rtol=1e-5, atol=2e-7)
                assert builder.triangle_distances(own, selected[:, :3]).max() < 2e-7
