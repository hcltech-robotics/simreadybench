"""Fit one scale-preserving yaw and translation to an entire USD observation."""

from __future__ import annotations

import math
import numpy as np

SAMPLES = 128
YAW_STARTS = 32
REFINED_STARTS = 2
MAX_ITERATIONS = 8


def _rotation(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], float)


def _centre(triangles):
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return np.average(triangles.mean(axis=1), axis=0, weights=areas)


def _cloud_cost(reference, candidate):
    squared = np.sum((reference[:, None] - candidate[None]) ** 2, axis=2)
    return float((squared.min(axis=1).mean() + squared.min(axis=0).mean()) / 2)


def _fit_yaw(source, target, previous):
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    own, other = source[:, :2] - source_centre[:2], target[:, :2] - target_centre[:2]
    cross = float(np.sum(own[:, 0] * other[:, 1] - own[:, 1] * other[:, 0]))
    dot = float(np.sum(own * other))
    rotation = (
        _rotation(math.atan2(cross, dot)) if abs(cross) + abs(dot) > 1e-30 else previous
    )
    return rotation, target_centre - source_centre @ rotation.T


def transform_observation(observation, rotation, translation):
    """Apply one rigid transform to geometry and every world-space physics field."""
    result = {
        **observation,
        "triangles": np.asarray(observation["triangles"]) @ rotation.T + translation,
    }
    result["bodies"] = {
        name: {
            **record,
            **(
                {
                    "position": (
                        np.asarray(record["position"]) @ rotation.T + translation
                    ).tolist()
                }
                if "position" in record
                else {}
            ),
        }
        for name, record in observation["bodies"].items()
    }
    result["joints"] = []
    for joint in observation["joints"]:
        aligned = dict(joint)
        for field in ("anchors", "axes", "frames"):
            if joint.get(field):
                transformed = np.asarray(joint[field], float) @ rotation.T
                if field == "anchors":
                    transformed += translation
                aligned[field] = transformed.tolist()
        result["joints"].append(aligned)
    return result


def align_observation(reference, candidate):
    """Minimise symmetric surface distance over a single Z-axis yaw and translation."""
    from simready_eval.correspondence import _nearest_bruteforce, _TriangleIndex, sample

    rt, ct = (
        np.asarray(reference["triangles"], float),
        np.asarray(candidate["triangles"], float),
    )
    rp, _, _ = sample(rt, SAMPLES, 142)
    cp, _, _ = sample(ct, SAMPLES, 143)
    reference_centre, candidate_centre = _centre(rt), _centre(ct)

    def index(triangles):
        if len(triangles) >= 4096:
            return _TriangleIndex(triangles).query
        return lambda points: _nearest_bruteforce(points, triangles)

    reference_nearest, candidate_nearest = index(rt), index(ct)

    def evaluate(rotation, translation):
        local_reference = (rp - translation) @ rotation
        rd, ri, rb = candidate_nearest(local_reference)
        candidate_closest = np.einsum("ni,nij->nj", rb, ct[ri])
        world_candidate = cp @ rotation.T + translation
        cd, ci, cb = reference_nearest(world_candidate)
        reference_closest = np.einsum("ni,nij->nj", cb, rt[ci])
        cost = float((np.mean(rd**2) + np.mean(cd**2)) / 2)
        return (
            cost,
            np.concatenate([cp, candidate_closest]),
            np.concatenate([reference_closest, rp]),
        )

    initial = [(np.eye(3), np.zeros(3))]
    for index in range(YAW_STARTS):
        rotation = _rotation(2 * math.pi * index / YAW_STARTS)
        translation = reference_centre - candidate_centre @ rotation.T
        initial.append((rotation, translation))
    ranked = sorted(
        enumerate(initial),
        key=lambda row: (_cloud_cost(rp, cp @ row[1][0].T + row[1][1]), row[0]),
    )
    best_rotation, best_translation = initial[0]
    raw_cost, _, _ = evaluate(best_rotation, best_translation)
    best_cost = raw_cost
    iterations = 0
    scale = max(float(np.linalg.norm(np.ptp(rt.reshape(-1, 3), axis=0))), 1e-9)
    tolerance = scale**2 * 1e-12
    converged = raw_cost <= scale**2 * 1e-20
    for _, (rotation, translation) in [] if converged else ranked[:REFINED_STARTS]:
        for step in range(MAX_ITERATIONS + 1):
            cost, source, target = evaluate(rotation, translation)
            if cost < best_cost - tolerance:
                best_cost, best_rotation, best_translation = cost, rotation, translation
                converged = False
            updated_rotation, updated_translation = _fit_yaw(source, target, rotation)
            movement = np.linalg.norm(
                updated_translation - translation
            ) + scale * np.linalg.norm(updated_rotation - rotation)
            if movement <= scale * 1e-7:
                if np.array_equal(rotation, best_rotation) and np.array_equal(
                    translation, best_translation
                ):
                    converged = True
                break
            if step == MAX_ITERATIONS:
                break
            rotation, translation = updated_rotation, updated_translation
            iterations += 1
    yaw = math.degrees(math.atan2(best_rotation[1, 0], best_rotation[0, 0]))
    return transform_observation(candidate, best_rotation, best_translation), {
        "method": "global_z_yaw_translation_surface_icp",
        "objective": "symmetric_mean_squared_point_to_surface_distance",
        "rotation": best_rotation.tolist(),
        "yaw_degrees": yaw,
        "translation_m": best_translation.tolist(),
        "scale": 1.0,
        "samples_each_direction": SAMPLES,
        "yaw_starts": YAW_STARTS,
        "refined_starts": REFINED_STARTS,
        "maximum_iterations_per_start": MAX_ITERATIONS,
        "iterations": iterations,
        "converged": converged,
        "unaligned_rms_m": math.sqrt(raw_cost),
        "aligned_rms_m": math.sqrt(best_cost),
    }
