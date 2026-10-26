"""Freeze one task-seeded observation camera while preserving comparison views.

This module uses only Python's standard library and can run as a standalone
script. Matrices store rows, act on column vectors in metres/Z-up, and have
right/up/backward/position columns. Cameras look along local -Z.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

PROTOCOL_SALT = 'simreadybench-observation-camera-v1'
MARGIN = 1.15


def _identity(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _copy_manifest(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict):
        raise ValueError('Camera manifest must be an object')
    return json.loads(json.dumps(value, allow_nan=False))


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _camera_matrix(view):
    if not isinstance(view, dict) or not isinstance(view.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', view['id']):
        raise ValueError('Safe camera identifiers required')
    matrix = view.get('camera_to_world')
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ValueError('Camera matrix must be 4x4')
    matrix = [[float(value) for value in row] for row in matrix]
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError('Finite camera matrix required')
    if any(abs(a-b) > 1e-9 for a, b in zip(matrix[3], (0, 0, 0, 1))):
        raise ValueError('Invalid homogeneous camera matrix')
    columns = [tuple(matrix[row][column] for row in range(3)) for column in range(3)]
    if any(abs(_dot(columns[a], columns[b]) - (a == b)) > 1e-7 for a in range(3) for b in range(3)) or abs(_dot(_cross(columns[0], columns[1]), columns[2])-1) > 1e-7:
        raise ValueError('Rigid right-handed camera required')
    near, far = float(view.get('clip_start', 0)), float(view.get('clip_end', 0))
    if not math.isfinite(near+far) or not 0 < near < far:
        raise ValueError('Positive ordered clipping planes required')
    fields = {'orthographic': ('orthographic_scale',), 'perspective': ('focal_length_mm', 'sensor_width_mm')}.get(view.get('projection'))
    if fields is None or any(not math.isfinite(float(view.get(key, 0))) or float(view.get(key, 0)) <= 0 for key in fields):
        raise ValueError('Valid camera projection required')
    return matrix


def _same_matrix(a, b):
    return all(abs(x-y) <= 1e-9 for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def _comparison_matrices(manifest):
    if manifest.get('schema') != 'simready-cameras-v1':
        raise ValueError('Expected simready-cameras-v1')
    size = manifest.get('resolution')
    if not isinstance(size, list) or len(size) != 2 or any(type(value) is not int or not 11 <= value <= 4096 for value in size):
        raise ValueError('Resolution requires two integers in [11,4096]')
    views = manifest.get('views')
    if not isinstance(views, list) or len(views) != 4:
        raise ValueError('Exactly four frozen comparison cameras required')
    matrices = [_camera_matrix(view) for view in views]
    if len({view['id'] for view in views}) != 4 or any(_same_matrix(matrix, other) for index, matrix in enumerate(matrices) for other in matrices[index+1:]):
        raise ValueError('Comparison cameras must have distinct ids and matrices')
    return matrices


def comparison_identity(cameras_manifest):
    """Hash canonical JSON of exactly {schema,resolution,views}; ignore actor view."""
    manifest = _copy_manifest(cameras_manifest)
    _comparison_matrices(manifest)
    return _identity({key: manifest[key] for key in ('schema', 'resolution', 'views')})


def _bounds(value):
    if isinstance(value, dict):
        value = value.get('reference_bounds_m_Z_up')
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in value):
        raise ValueError('Reference bounds require two 3-vectors in metres/Z-up')
    bounds = [[float(component) for component in row] for row in value]
    if not all(math.isfinite(component) for row in bounds for component in row):
        raise ValueError('Finite reference bounds required')
    spans = [high-low for low, high in zip(*bounds)]
    diagonal = math.hypot(*spans)
    if any(span < 0 for span in spans) or not math.isfinite(3*diagonal) or diagonal <= 0:
        raise ValueError('Ordered nondegenerate reference bounds required')
    return bounds, spans, diagonal


def _angles(seed, index):
    draw = seed if index == 0 else hashlib.sha256(seed + index.to_bytes(4, 'big')).digest()
    # Explicit 53-bit fractions avoid Python RNG/hash state and the rounding of
    # a maximum uint64 to 1.0. Azimuth is uniform on [0,360); elevation on [15,70].
    azimuth = 360 * ((int.from_bytes(draw[:8], 'big') >> 11) / 2**53)
    elevation = 15 + 55 * ((int.from_bytes(draw[8:16], 'big') >> 11) / 2**53)
    return azimuth, elevation


def _frame(bounds, spans, diagonal, resolution, identifier, azimuth, elevation):
    center = [low + span/2 for low, span in zip(bounds[0], spans)]
    azimuth, elevation = math.radians(azimuth), math.radians(elevation)
    right = (-math.sin(azimuth), math.cos(azimuth), 0.0)
    back = (math.cos(elevation)*math.cos(azimuth), math.cos(elevation)*math.sin(azimuth), math.sin(elevation))
    upward = _cross(back, right)
    distance = 3*diagonal
    position = [component + distance*axis for component, axis in zip(center, back)]
    matrix = [[right[row], upward[row], back[row], position[row]] for row in range(3)] + [[0, 0, 0, 1]]
    corners = list(itertools.product(*zip(*bounds)))
    local = [[_dot([point-origin for point, origin in zip(corner, position)], axis) for axis in (right, upward, back)] for corner in corners]
    aspect = resolution[0]/resolution[1]
    vertical_span = 2*MARGIN*max(max(abs(row[1]) for row in local), max(abs(row[0]) for row in local)/aspect)
    depths = [-row[2] for row in local]
    near, far = min(depths)-.15*diagonal, max(depths)+.15*diagonal
    if not all(math.isfinite(value) for value in (*position, vertical_span, near, far)) or not 0 < near < min(depths) <= max(depths) < far or vertical_span <= 0:
        raise ValueError('Reference bounds cannot be framed with finite positive clipping')
    return {'id': identifier, 'projection': 'orthographic', 'camera_to_world': matrix,
            'orthographic_scale': vertical_span, 'clip_start': near, 'clip_end': far}, distance


def freeze_observation(cameras_manifest, reference_bounds_m_Z_up, task_id, *, candidate_index=0, reselection_reason=None):
    """Copy a frozen rig and replace only observation plus observation_policy.

    Sampling depends on the protocol salt, exact UTF-8 task id and candidate
    index, never roster order. Comparison dicts, value types and resolution survive
    unchanged. Reapplying this function with the same bounds/task is idempotent.
    """
    manifest = _copy_manifest(cameras_manifest)
    matrices = _comparison_matrices(manifest)
    observation = manifest.get('observation')
    previous_matrix = _camera_matrix(observation)
    if manifest.get('observed_camera_id') != observation['id'] or observation['id'] in {view['id'] for view in manifest['views']} or any(_same_matrix(previous_matrix, matrix) for matrix in matrices):
        raise ValueError('A distinct separate observation camera is required')
    if not isinstance(task_id, str) or not task_id or len(task_id.encode('utf-8')) > 1024:
        raise ValueError('A nonempty task id of at most 1024 UTF-8 bytes is required')
    if type(candidate_index) is not int or not 0 <= candidate_index < 2**32:
        raise ValueError('candidate_index must be an unsigned 32-bit integer')
    if reselection_reason is None:
        if candidate_index > 0:
            raise ValueError('A nonzero candidate_index requires reselection_reason')
    elif not isinstance(reselection_reason, str) or not reselection_reason.strip() or len(reselection_reason) > 1024 or not all(character.isprintable() for character in reselection_reason):
        raise ValueError('reselection_reason must be nonblank printable text of at most 1024 characters')
    bounds, spans, diagonal = _bounds(reference_bounds_m_Z_up)
    seed = hashlib.sha256(PROTOCOL_SALT.encode('utf-8') + b'\0' + task_id.encode('utf-8') + b'\0' + candidate_index.to_bytes(4, 'big')).digest()
    for index in range(32):
        azimuth, elevation = _angles(seed, index)
        camera, distance = _frame(bounds, spans, diagonal, manifest['resolution'], observation['id'], azimuth, elevation)
        if not any(_same_matrix(camera['camera_to_world'], matrix) for matrix in matrices):
            break
    else:
        raise ValueError('Unable to freeze a distinct observation camera')
    manifest['observation'] = camera
    manifest['observation_policy'] = {
        'schema': 'simready-observation-policy-v1', 'protocol_salt': PROTOCOL_SALT,
        'task_id': task_id, 'candidate_index': candidate_index,
        'reselection_reason': reselection_reason,
        'seed_sha256': seed.hex(), 'sample_index': index,
        'seed_derivation': 'SHA256(UTF8(salt) + NUL + UTF8(task_id) + NUL + uint32be(candidate_index))',
        'azimuth_degrees': azimuth, 'elevation_degrees': elevation,
        'azimuth_range_degrees': [0, 360], 'elevation_range_degrees': [15, 70],
        'reference_bounds_m_Z_up': bounds, 'framing_margin': MARGIN,
        'camera_distance_m': distance, 'comparison_sha256': comparison_identity(manifest),
        'comparison_hash_scope': 'canonical JSON {schema,resolution,views}',
        'framing': 'orthographic; project all eight reference AABB corners; fixed center; no candidate refit',
    }
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', required=True, type=Path)
    parser.add_argument('--bounds', required=True, type=Path, help='JSON two 3-vectors or metadata with reference_bounds_m_Z_up')
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--candidate-index', type=int, default=0, help='Frozen visual reselection index, uint32; default 0')
    parser.add_argument('--reselection-reason', help='Required printable review note when candidate-index is nonzero')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = freeze_observation(args.cameras, json.loads(args.bounds.read_text()), args.task_id, candidate_index=args.candidate_index, reselection_reason=args.reselection_reason)
        if args.output.resolve() == args.cameras.resolve():
            raise ValueError('Output must be separate from the original frozen cameras')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
        receipt = {'status': 'frozen', 'output': str(args.output), 'camera_sha256': _identity(manifest),
                   'comparison_sha256': comparison_identity(manifest), 'observation_policy': manifest['observation_policy']}
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(json.dumps({'status': 'unknown', 'reason': 'invalid_observation_input', 'error': str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
