"""Exact BVH query and fixed-XYZ arithmetic regression coverage."""
import numpy as np
import pytest
from simready_eval.correspondence import _TriangleIndex, _nearest_bruteforce, _cross3


@pytest.mark.parametrize('count', [0, 1, 129, 257])
def test_packets_match_exhaustive_triangle_queries(count):
    rng = np.random.default_rng(19)
    triangles = rng.normal(size=(96, 3, 3))
    # Include collapsed triangles, coincident surfaces and narrow triangles.
    triangles[0] = 0
    triangles[1] = triangles[2]
    triangles[3] = [[0, 0, 0], [1, 0, 0], [.5, 1e-12, 0]]
    points = rng.normal(size=(count, 3))
    actual = _TriangleIndex(triangles, leaf_size=4).query(points)
    expected = _nearest_bruteforce(points, triangles)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)


def test_ties_select_lowest_original_triangle_across_leaves():
    triangle = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    triangles = np.tile(triangle, (128, 1, 1))
    distances, indices, weights = _TriangleIndex(triangles, leaf_size=4).query(
        [[.25, .25, 0], [0, 0, 0], [2, 2, 2]]
    )
    assert indices.tolist() == [0, 0, 0]
    for actual, expected in zip((distances, indices, weights),
                                _nearest_bruteforce([[.25,.25,0],[0,0,0],[2,2,2]],triangles)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shapes', [((3,), (3,)), ((3,3), (3,3)),
                                    ((4,32,3), (32,3)), ((32,3), (4,32,3))])
def test_xyz_cross_preserves_numpy_bits(shapes):
    rng = np.random.default_rng(314)
    a, b = (rng.normal(size=shape) for shape in shapes)
    assert _cross3(a,b).tobytes() == np.cross(a,b).tobytes()
    a.fill(-0.)
    b.fill(0.)
    assert _cross3(a,b).tobytes() == np.cross(a,b).tobytes()
