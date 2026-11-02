"""Versioned area-weighted surface-cloud similarity in canonical metres.

Alignment remains the existing exact yaw/translation protocol. Only the final
surface-distance measurements use point neighbours. Reference eligibility and
material sample rosters retain their separately frozen legacy protocol.
"""
from __future__ import annotations

import numpy as np

VERSION = 'simready-area-cloud-similarity-v1'
SAMPLES = 65536
SEED = 142
BAND_FRACTION = .02
PROTOCOL = {
    'schema': VERSION,
    'samples_each_direction': SAMPLES,
    'seed': SEED,
    'sampling': 'fixed area-weighted authored triangle surface samples',
    'distance': 'symmetric Euclidean nearest sampled surface point',
    'distance_band_fraction': BAND_FRACTION,
    'normalization': 'reference-only 1st-to-99th-percentile cloud bounding-box diagonal',
    'directional_credit': 'mean(1/(1+(distance/band)^2))',
    'aggregation': 'harmonic mean of directional credits',
    'alignment': 'unchanged global_z_yaw_translation_surface_icp; no scale fitting',
    'query_workers': 1,
    'nearest_neighbor_eps': 0,
}


class SurfaceCloud:
    def __init__(self, triangles):
        from scipy.spatial import cKDTree
        from .correspondence import sample

        self.points, self.indices, self.bary = sample(triangles, SAMPLES, SEED)
        self.tree = cKDTree(self.points, leafsize=32, balanced_tree=True)

    def nearest(self, points):
        return self.tree.query(points, k=1, eps=0, workers=1)


def surface(reference, candidate, cfg, material_band):
    from .correspondence import sample

    own, other = SurfaceCloud(reference), SurfaceCloud(candidate)
    low, high = np.quantile(own.points, [.01, .99], axis=0)
    diagonal = max(float(np.linalg.norm(high - low)), 1e-9)
    band = BAND_FRACTION * diagonal
    rd, _ = other.nearest(own.points)
    cd, _ = own.nearest(other.points)
    recall = float(np.mean(1 / (1 + (rd / band) ** 2)))
    precision = float(np.mean(1 / (1 + (cd / band) ** 2)))
    score = 2 * recall * precision / (recall + precision) if recall + precision else 0.
    result = {
        'schema': VERSION,
        'score_percent': 100 * score,
        'recall': recall,
        'precision': precision,
        'rms_m': float(np.sqrt(np.mean(np.r_[rd, cd] ** 2))),
        'p95_m': float(np.quantile(np.r_[rd, cd], .95)),
        'symmetric_mean_distance_m': float((rd.mean() + cd.mean()) / 2),
        'normalized_symmetric_mean_distance': float((rd.mean() + cd.mean()) / (2 * diagonal)),
        'reference_robust_diagonal_m': diagonal,
        'band_m': band,
        'samples_each_direction': SAMPLES,
        'protocol': dict(PROTOCOL),
    }
    # Preserve the frozen reference material roster and its fixed denominator.
    # Candidate values come from the nearest cloud point's original triangle
    # and barycentric coordinates; these remain diagnostics, not image scores.
    points, ri, rb = sample(reference, cfg['samples'])
    distance, index = other.nearest(points)
    return result, (ri, rb, other.indices[index], other.bary[index], distance)
