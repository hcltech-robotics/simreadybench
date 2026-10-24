#!/usr/bin/env python3
"""Build deterministic, bidirectional mesh-surface distance point clouds.

The input GLBs already contain the recorded rigid evaluation alignment in
metres/Y-up. No additional centring, fitting or scaling is performed here.
Rows are little-endian float32 x, y, z, unsigned distance in metres; reference
samples precede candidate samples. Distances are to triangle surfaces, not to
the other sample cloud. This display diagnostic does not change benchmark scores.

Build dependencies: numpy==2.4.1, scipy==1.17.1, trimesh==4.11.5, rtree==1.4.1.
Run ``python scripts/build_viewer_delta.py site``; add ``--check`` to recompute
and verify all bytes without writing. No benchmark or private inputs are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import rtree
import scipy
import trimesh


DEFAULT_SAMPLES = 12_000
METHOD = "bidirectional-area-weighted-surface-to-triangle-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seed_from_hash(value: str) -> int:
    """Use the pinned GLB content identity rather than process hash state."""
    return int(value[:16], 16)


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes the site root")
    return path


def load_mesh(site: Path, item: dict) -> trimesh.Trimesh:
    payload = safe_path(site, item["src"]).read_bytes()
    if sha256(payload) != item["sha256"]:
        raise ValueError("Geometry differs from its pinned GLB hash")
    mesh = trimesh.load(safe_path(site, item["src"]), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Geometry has no triangle surface")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Nonfinite mesh geometry")
    # Float32 export can collapse an otherwise tiny triangle. Zero-area faces
    # have no surface to sample and cannot affect the nearest surface distance.
    valid = mesh.area_faces > 0
    if not valid.all():
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[valid], process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Geometry has no finite positive-area surface")
    if not np.allclose(mesh.bounds, item["bounds_m"], rtol=0, atol=1e-12):
        raise ValueError("Loaded geometry changed the recorded shared coordinate frame")
    return mesh


def surface_samples(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Uniform surface-area sampling with an explicit reproducible PCG64 RNG."""
    if count < 1:
        raise ValueError("Sample count must be positive")
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Mesh has no finite positive-area surface")
    rng = np.random.Generator(np.random.PCG64(seed))
    cdf = np.cumsum(areas)
    face_ids = np.searchsorted(cdf, rng.random(count) * cdf[-1], side="right")
    triangles = mesh.triangles[face_ids]
    root = np.sqrt(rng.random(count))
    second = rng.random(count)
    weights = np.column_stack((1 - root, root * (1 - second), root * second))
    return np.einsum("ni,nij->nj", weights, triangles)


def point_triangle_squared(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact point-to-triangle squared distance without an absolute size epsilon.

    The closest point is either the orthogonal plane projection when it lies
    inside the triangle or a projection onto one of its three closed segments.
    Cross-product barycentrics avoid the cancellation of Gram determinants for
    narrow triangles. No mesh-relative scale transformation is introduced.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    if np.any(normal2 <= 0):
        raise ValueError("Degenerate target triangle")
    weight_b = np.einsum("ij,ij->i", np.cross(ap, ac), normal) / normal2
    weight_c = np.einsum("ij,ij->i", np.cross(ab, ap), normal) / normal2
    inside = (weight_b >= 0) & (weight_c >= 0) & (weight_b + weight_c <= 1)
    plane_dot = np.einsum("ij,ij->i", ap, normal)
    squared = np.where(inside, plane_dot ** 2 / normal2, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        edge2 = np.einsum("ij,ij->i", edge, edge)
        parameter = np.clip(np.einsum("ij,ij->i", points - start, edge) / edge2, 0, 1)
        vector = points - (start + parameter[:, None] * edge)
        squared = np.minimum(squared, np.einsum("ij,ij->i", vector, vector))
    return squared


def triangle_distances(mesh: trimesh.Trimesh, points: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Unsigned minimum Euclidean distance to all target triangle surfaces.

    nearby_faces uses an R-tree and a nearest-vertex upper bound to retain every
    potentially closest triangle. The final query is an exact point/triangle
    projection and minimum, never a nearest-vertex or sample-neighbour distance.
    Batching limits temporary point/triangle arrays for highly detailed meshes.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Query points must be finite 3D positions")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    triangles = np.asarray(mesh.triangles)
    distances = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        faces = trimesh.proximity.nearby_faces(mesh, batch)
        counts = np.fromiter((len(ids) for ids in faces), dtype=np.int64)
        if np.any(counts == 0):
            raise ValueError("Nearest-surface query produced no candidate triangle")
        queries = np.repeat(batch, counts, axis=0)
        squared = point_triangle_squared(triangles[np.concatenate(faces)], queries)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        distances[start:start + len(batch)] = np.sqrt(np.minimum.reduceat(squared, starts))
    if not np.isfinite(distances).all():
        raise ValueError("Nearest-surface query produced nonfinite distances")
    return distances


def delta_rows(reference: trimesh.Trimesh, candidate: trimesh.Trimesh,
               reference_points: np.ndarray, candidate_points: np.ndarray) -> np.ndarray:
    reference_rows = np.column_stack((reference_points, triangle_distances(candidate, reference_points)))
    candidate_rows = np.column_stack((candidate_points, triangle_distances(reference, candidate_points)))
    return np.concatenate((reference_rows, candidate_rows)).astype("<f4")


def write_or_check(path: Path, payload: bytes, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != payload:
            raise ValueError(f"Recomputed output differs: {path.name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def build(site: Path, *, samples: int = DEFAULT_SAMPLES, check: bool = False) -> dict:
    site = site.resolve()
    source_path = site / "assets/explorer/manifest.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    receipts_path = site / "assets/explorer/geometry-provenance.json"
    receipt_bytes = receipts_path.read_bytes()
    receipts = json.loads(receipt_bytes)
    pairs = {}
    result = {
        "schema_version": 1,
        "method": METHOD,
        "units": "metres",
        "coordinates": "metres/Y-up; recorded rigid evaluation alignment when available; no additional fitting or scaling",
        "distance": "Unsigned nearest triangle-surface distance, sampled in both directions",
        "sampling": "Uniform surface area; PCG64 seeded by first 64 bits of each pinned GLB SHA-256",
        "samples_per_direction": samples,
        "format": {"dtype": "float32", "byte_order": "little", "stride_bytes": 16,
                   "columns": ["x", "y", "z", "distance_m"], "order": ["reference", "candidate"]},
        "source_manifest_sha256": sha256(source_bytes),
        "source_geometry_provenance_sha256": sha256(receipt_bytes),
        "build_dependencies": {"numpy": np.__version__, "scipy": scipy.__version__,
                               "trimesh": trimesh.__version__, "rtree": rtree.__version__},
        "limits": "Distances describe the triangulated neutral geometry previews, not materials, physics, animation or benchmark fidelity scores. Finite surface samples can miss small features. Maxima and percentiles describe these samples, not continuous Hausdorff distance.",
        "pairs": pairs,
    }
    for obj in source["objects"]:
        object_id = obj["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", object_id):
            raise ValueError("Invalid object identifier")
        ref_info = obj["reference"]
        reference = load_mesh(site, ref_info)
        reference_points = surface_samples(reference, samples, seed_from_hash(ref_info["sha256"]))
        diagonal = float(np.linalg.norm(np.diff(reference.bounds, axis=0)))
        for variant in obj["variants"]:
            variant_id = variant["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", variant_id):
                raise ValueError("Invalid variant identifier")
            key = f"{object_id}/{variant_id}"
            entry = {
                "src": None, "sha256": None, "point_count": 0, "reference_point_count": 0,
                "max_distance_m": None, "p95_distance_m": None,
                "reference_diagonal_m": diagonal,
                "source_reference_sha256": ref_info["sha256"],
                "source_candidate_sha256": variant.get("sha256"),
                "alignment_status": ("recorded" if receipts.get(key, {}).get("alignment_m_z_up")
                                     else "authored" if variant.get("src") else None),
            }
            pairs[key] = entry
            if not variant.get("src"):
                entry["reason"] = "Candidate geometry unavailable"
                continue
            for identity, info in ((f"{object_id}/reference", ref_info), (key, variant)):
                receipt = receipts[identity]
                if receipt["glb_sha256"] != info["sha256"] or receipt["coordinates"] != "metres/Y-up":
                    raise ValueError("Geometry receipt identity or coordinate frame mismatch")
                alignment = receipt.get("alignment_m_z_up")
                if alignment and alignment.get("scale", 1) != 1:
                    raise ValueError("Scaled alignment is not supported")
            candidate = load_mesh(site, variant)
            candidate_points = surface_samples(candidate, samples, seed_from_hash(variant["sha256"]))
            rows = delta_rows(reference, candidate, reference_points, candidate_points)
            payload = rows.tobytes(order="C")
            relative = f"assets/explorer/delta/{object_id}/{variant_id}.bin"
            entry.update({
                "src": relative, "sha256": sha256(payload), "point_count": len(rows),
                "reference_point_count": len(reference_points),
                "max_distance_m": float(rows[:, 3].max()),
                "p95_distance_m": float(np.percentile(rows[:, 3].astype(np.float64), 95)),
            })
            write_or_check(safe_path(site, relative), payload, check)
            print(f"{'Verified' if check else 'Built'} {key}: {len(rows):,} points", flush=True)
    manifest_bytes = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    write_or_check(site / "assets/explorer/delta-manifest.json", manifest_bytes, check)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "site")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.site, samples=args.samples, check=args.check)
    available = sum(entry["src"] is not None for entry in result["pairs"].values())
    print(f"{'Verified' if args.check else 'Built'} {available} delta clouds; {len(result['pairs']) - available} unavailable")


if __name__ == "__main__":
    main()
