"""Direct authored-USD correspondence; CPU measurements, no image or aggregate reward."""

from __future__ import annotations
import collections, hashlib, heapq, json, math
import numpy as np

VERSION = "source-usd-fidelity-v4"
DEFAULTS = {
    "enabled": False,
    "samples": 2048,
    "seconds": 300,
    "distance_m": 0.001,
    "distance_relative": 0.01,
    "joint_position_m": 0.001,
    "joint_position_relative": 0.01,
    "joint_angle_degrees": 3.0,
    "range_degrees": 5.0,
    "range_m": 0.005,
    "range_relative": 0.05,
    "material_absolute": 0.05,
    "material_relative": 0.05,
    "joint_existence_type": True,
}


def settings(value=None):
    if value is None:
        value = {}
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError("Known fidelity settings required")
    result = {**DEFAULTS, **value}
    for key in ("enabled", "joint_existence_type"):
        if type(result[key]) is not bool:
            raise ValueError("Boolean fidelity " + key + " required")
    for key, low, high in [("samples", 128, 8192), ("seconds", 10, 900)]:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError("Bounded fidelity " + key + " required")
    for key in set(DEFAULTS) - {
        "enabled",
        "joint_existence_type",
        "samples",
        "seconds",
    }:
        v = result[key]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 < v <= 100
        ):
            raise ValueError("Positive finite fidelity " + key + " required")
    return result


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sample(triangles, count, seed=142):
    t = np.asarray(triangles, float)
    areas = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    if not len(t) or not np.isfinite(t).all() or areas.sum() <= 0:
        raise ValueError("Finite nonempty surface required")
    rng = np.random.default_rng(seed)
    index = rng.choice(len(t), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    u = np.sqrt(uv[:, 0])
    bary = np.c_[1 - u, u * (1 - uv[:, 1]), u * uv[:, 1]]
    return np.einsum("ni,nij->nj", bary, t[index]), index, bary


def _nearest_bruteforce(points, triangles, *, squared=False):
    """Exact nearest point on polygon triangles, in bounded CPU/memory batches."""
    p = np.asarray(points, float)
    t = np.asarray(triangles, float)
    if not len(t):
        return np.full(len(p), np.inf), np.zeros(len(p), int), np.zeros((len(p), 3))
    best = np.full(len(p), np.inf)
    indices = np.zeros(len(p), int)
    weights = np.zeros((len(p), 3))
    for start in range(0, len(p), 64):
        q = p[start : start + 64]
        own = np.full(len(q), np.inf)
        for ts in range(0, len(t), 512):
            tri = t[ts : ts + 512]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            n2 = np.einsum("ij,ij->i", normal, normal)
            if np.any(n2 <= 0):
                raise ValueError("Nondegenerate triangles required")
            ap = q[:, None, :] - a
            dot = np.einsum("ntj,tj->nt", ap, normal)
            plane = ap - dot[:, :, None] / n2[None, :, None] * normal
            # Cross-product barycentrics avoid cancellation in the Gram
            # determinant for narrow, nonzero-area authored triangles.
            u = np.einsum("ntj,tj->nt", np.cross(plane, ac), normal) / n2
            v = np.einsum("ntj,tj->nt", np.cross(ab, plane), normal) / n2
            bary = np.stack((1 - u - v, u, v), axis=2)
            inside = (u >= 0) & (v >= 0) & (u + v <= 1)
            distances = np.where(inside, dot * dot / n2, np.inf)
            for first, second in ((0, 1), (1, 2), (2, 0)):
                origin = tri[:, first]
                edge = tri[:, second] - origin
                delta = q[:, None, :] - origin
                alpha = np.clip(
                    np.einsum("ntj,tj->nt", delta, edge)
                    / np.einsum("tj,tj->t", edge, edge),
                    0,
                    1,
                )
                residual = delta - alpha[:, :, None] * edge
                distance = np.einsum("ntj,ntj->nt", residual, residual)
                use = distance < distances
                distances = np.minimum(distance, distances)
                edge_bary = np.zeros_like(bary)
                edge_bary[:, :, first] = 1 - alpha
                edge_bary[:, :, second] = alpha
                bary = np.where(use[:, :, None], edge_bary, bary)
            chosen = np.argmin(distances, axis=1)
            rows = np.arange(len(q))
            values = distances[rows, chosen]
            use = values < own
            own[use] = values[use]
            indices[start + rows[use]] = ts + chosen[use]
            weights[start + rows[use]] = bary[rows[use], chosen[use]]
        best[start : start + len(q)] = own
    return (best if squared else np.sqrt(best)), indices, weights


class _TriangleIndex:
    """Exact AABB branch-and-bound; never substitutes sampled target points."""

    def __init__(self, triangles, leaf_size=32):
        self.triangles = np.asarray(triangles, float)
        self.nodes = []
        self.triangle_tests = 0
        low = self.triangles.min(axis=1)
        high = self.triangles.max(axis=1)
        centres = (low + high) * 0.5

        def build(indices):
            lo = low[indices].min(axis=0)
            hi = high[indices].max(axis=0)
            number = len(self.nodes)
            self.nodes.append(None)
            if len(indices) <= leaf_size:
                self.nodes[number] = (lo, hi, None, None, np.sort(indices))
            else:
                axis = int(np.argmax(hi - lo))
                order = np.argpartition(centres[indices, axis], len(indices) // 2)
                middle = len(indices) // 2
                left = build(indices[order[:middle]])
                right = build(indices[order[middle:]])
                self.nodes[number] = (lo, hi, left, right, None)
            return number

        if len(self.triangles):
            build(np.arange(len(self.triangles)))

    def query(self, points):
        points = np.asarray(points, float)
        distances = np.full(len(points), np.inf)
        chosen = np.zeros(len(points), int)
        bary = np.zeros((len(points), 3))
        for row, point in enumerate(points):

            def bound(node):
                lo, hi, _, _, _ = self.nodes[node]
                delta = np.maximum(np.nextafter(lo, -np.inf) - point, 0) + np.maximum(
                    point - np.nextafter(hi, np.inf), 0
                )
                # Conservative rounding allowance only for pruning, not scoring.
                return max(0.0, float(delta @ delta) * (1 - 32 * np.finfo(float).eps))

            queue = [(bound(0), 0)] if self.nodes else []
            best = np.inf
            best_index = len(self.triangles)
            while queue:
                lower, node = heapq.heappop(queue)
                if lower > best:
                    continue
                _, _, left, right, indices = self.nodes[node]
                if indices is not None:
                    self.triangle_tests += len(indices)
                    value, index, weights = _nearest_bruteforce(
                        point[None], self.triangles[indices], squared=True
                    )
                    actual = int(indices[index[0]])
                    if value[0] < best or value[0] == best and actual < best_index:
                        best = float(value[0])
                        best_index = actual
                        bary[row] = weights[0]
                else:
                    for child in (left, right):
                        lower = bound(child)
                        if lower <= best:
                            heapq.heappush(queue, (lower, child))
            distances[row] = best
            if best_index < len(self.triangles):
                chosen[row] = best_index
        return np.sqrt(distances), chosen, bary


def nearest(points, triangles):
    triangles = np.asarray(triangles, float)
    if len(triangles) < 4096:
        return _nearest_bruteforce(points, triangles)
    return _TriangleIndex(triangles).query(points)


def surface(reference, candidate, cfg, band):
    rp, ri, rb = sample(reference, cfg["samples"])
    cp, _, _ = sample(candidate, cfg["samples"], 143)
    rd, ci, cb = nearest(rp, candidate)
    cd, _, _ = nearest(cp, reference)
    recall = float(np.mean(rd <= band))
    precision = float(np.mean(cd <= band))
    f = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "score_percent": 100 * f,
        "recall": recall,
        "precision": precision,
        "rms_m": float(np.sqrt(np.mean(np.r_[rd, cd] ** 2))),
        "p95_m": float(np.quantile(np.r_[rd, cd], 0.95)),
        "band_m": band,
        "samples_each_direction": cfg["samples"],
    }, (ri, rb, ci, cb, rd)


def assignment(cost):
    """Rectangular minimum-cost assignment (Hungarian), no names or topology IDs."""
    cost = np.asarray(cost, float)
    n, m = cost.shape
    if not n or not m:
        return {}
    size = max(n, m)
    a = np.full((size, size), 1e6)
    a[:n, :m] = cost
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, int)
    way = np.zeros(size + 1, int)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j] = cur
                        way[j] = j0
                    if minimum[j] < delta:
                        delta = minimum[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return {p[j] - 1: j - 1 for j in range(1, size + 1) if 0 < p[j] <= n and j <= m}


def _body_match(reference, candidate, scale):
    def body_features(o):
        result = []
        for name, record in o["bodies"].items():
            triangles = (
                o["triangles"]
                if len(o["bodies"]) == 1
                else o["triangles"][
                    np.asarray(o["triangle_bodies"], dtype=object) == name
                ]
            )
            points = triangles.reshape(-1, 3)
            center = (
                (points.min(0) + points.max(0)) / 2
                if len(points)
                else np.asarray(record.get("position", [0, 0, 0]))
            )
            extent = np.ptp(points, axis=0) if len(points) else np.zeros(3)
            degree = sum(name in j["bodies"] for j in o["joints"])
            result.append((name, center, extent, degree))
        return result

    r = body_features(reference)
    c = body_features(candidate)
    if not r or not c:
        return {}, {"ambiguous_reference_bodies": [], "costs": []}
    costs = [
        [
            float(np.linalg.norm(x[1] - y[1]) + np.linalg.norm(x[2] - y[2])) / scale
            + 0.05 * abs(x[3] - y[3])
            for y in c
        ]
        for x in r
    ]
    ambiguous = [
        r[i][0]
        for i, row in enumerate(costs)
        if sum(abs(v - min(row)) < 1e-9 for v in row) > 1
    ]
    return {r[i][0]: c[j][0] for i, j in assignment(costs).items()}, {
        "ambiguous_reference_bodies": ambiguous,
        "costs": costs,
        "reference_order": [x[0] for x in r],
        "candidate_order": [x[0] for x in c],
    }


def _value(material, key, bary):
    value = material.get(key)
    if not isinstance(value, dict):
        return value
    if "texture" not in value:
        return None
    uv = np.asarray(bary) @ np.asarray(value["uv"])
    image = value["texture"]
    h, w = image.shape[:2]
    for axis, mode in enumerate(value["wrap"]):
        uv[axis] = np.clip(uv[axis], 0, 1) if mode == "clamp" else uv[axis] % 1
    x = min(int(uv[0] * w), w - 1)
    y = min(int((1 - uv[1]) * h), h - 1)
    colour = np.asarray(image[y, x], float)
    if np.asarray(image).dtype == np.uint8:
        colour /= 255
    if value.get("sRGB"):
        colour[:3] = np.where(
            colour[:3] <= 0.04045,
            colour[:3] / 12.92,
            ((colour[:3] + 0.055) / 1.055) ** 2.4,
        )
    channels = value.get("channels", [0, 1, 2])
    colour = np.atleast_1d(colour)[channels]
    result = colour * np.asarray(value.get("scale", 1)) + np.asarray(
        value.get("bias", 0)
    )
    return float(result[0]) if len(result) == 1 else result.tolist()


def _equal(a, b, cfg, absolute=None):
    if a is None or b is None:
        return False
    if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return a == b
    try:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return bool(
            a.shape == b.shape
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.all(
                np.abs(a - b)
                <= (cfg["material_absolute"] if absolute is None else absolute)
                + cfg["material_relative"] * np.abs(a)
            )
        )
    except (TypeError, ValueError):
        return False


def _component(rows):
    eligible = len(rows)
    return {
        "status": "measured" if eligible else "not_applicable",
        "score_percent": 100 * sum(r["passed"] for r in rows) / eligible
        if eligible
        else None,
        "earned": sum(r["passed"] for r in rows),
        "eligible": eligible,
        "measured": eligible,
        "coverage_fraction": 1 if eligible else 0,
        "checks": rows,
    }


def _appearance_component(rows):
    """Retain equal-area reference samples with unmeasured shader appearance."""
    assigned = {row["sample"] for row in rows if row["feature"] == "visual:assignment"}
    observed = collections.defaultdict(list)
    for row in rows:
        if (
            row["feature"].startswith("visual:")
            and row["feature"] != "visual:assignment"
        ):
            observed[row["sample"]].append(row["passed"])
    eligible = len(assigned | observed.keys())
    measured = len(observed)
    earned = sum(sum(checks) / len(checks) for checks in observed.values())
    known = measured == eligible
    return {
        "status": "not_applicable"
        if not eligible
        else "measured"
        if known
        else "unknown",
        "score_percent": 100 * earned / eligible if eligible and known else None,
        "earned": earned,
        "eligible": eligible,
        "measured": measured,
        "unknown": eligible - measured,
        "coverage_fraction": measured / eligible if eligible else 0,
        "coverage_basis": "reference surface samples with assigned visual materials",
        "lower_bound_percent": 100 * earned / eligible if eligible else None,
        "upper_bound_percent": 100 * (earned + eligible - measured) / eligible
        if eligible
        else None,
        "aggregation": "mean per-sample appearance-property match fraction",
    }


def _joint_checks(reference, candidate, mapping, cfg, scale):
    rows = []
    used = set()
    rj = reference["joints"]
    cj = (candidate or {}).get("joints", [])

    def add(i, feature, passed):
        rows.append({"reference_joint": i, "feature": feature, "passed": bool(passed)})

    for i, r in enumerate(rj):
        expected = [mapping.get(b) if b is not None else None for b in r["bodies"]]
        matches = []
        for j, c in enumerate(cj):
            if j in used:
                continue
            if any(b is not None and b not in mapping for b in r["bodies"]):
                continue
            for swapped in (False, True):
                bodies = list(reversed(c["bodies"])) if swapped else c["bodies"]
                if expected == bodies:
                    anchors = list(reversed(c["anchors"])) if swapped else c["anchors"]
                    error = sum(
                        np.linalg.norm(np.asarray(a) - b)
                        for a, b in zip(r["anchors"], anchors)
                    )
                    matches.append(
                        (int(c["type"] != r["type"]), float(error), j, swapped)
                    )
        chosen = min(matches) if matches else None
        c = cj[chosen[2]] if chosen else None
        if chosen:
            used.add(chosen[2])
        if cfg["joint_existence_type"]:
            add(
                i,
                "existence_type_connectivity",
                c is not None and c["type"] == r["type"],
            )
        if r.get("enabled") is not None:
            add(i, "joint_enabled", c is not None and c.get("enabled") == r["enabled"])
        for endpoint, body in enumerate(r["bodies"]):
            if body is None:
                continue
            for field in ("enabled", "kinematic"):
                target = reference["bodies"].get(body, {}).get(field)
                if target is not None:
                    actual = (
                        (candidate or {})
                        .get("bodies", {})
                        .get(mapping.get(body), {})
                        .get(field)
                    )
                    add(
                        i,
                        "body" + str(endpoint) + "_" + field,
                        c is not None and actual == target,
                    )
        anchors = (
            list(reversed(c["anchors"]))
            if c and chosen[3]
            else c["anchors"]
            if c
            else None
        )
        band = cfg["joint_position_m"] + cfg["joint_position_relative"] * scale
        add(
            i,
            "anchors",
            anchors is not None
            and all(
                np.linalg.norm(np.asarray(a) - b) <= band
                for a, b in zip(r["anchors"], anchors)
            ),
        )
        sign = -1 if chosen and chosen[3] else 1
        constrained_spherical = r["type"] == "PhysicsSphericalJoint" and any(
            b.get("state") == "finite"
            for b in r.get("limits", {}).get("bounds", {}).values()
        )
        if r.get("axes") and (
            r["type"] != "PhysicsSphericalJoint" or constrained_spherical
        ):
            axes = (
                list(reversed(c.get("axes") or []))
                if c and chosen[3]
                else (c or {}).get("axes")
            )
            dots = [
                float(np.clip(np.dot(a, b), -1, 1))
                for a, b in zip(r["axes"], axes or [])
            ]
            add(
                i,
                "axes",
                len(dots) == 2
                and all(
                    math.degrees(math.acos(abs(d))) <= cfg["joint_angle_degrees"]
                    for d in dots
                ),
            )
            if dots and dots[0] < 0:
                sign *= -1
        if r.get("frames") and (
            r["type"]
            in ("PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
            or constrained_spherical
        ):
            frames = (
                list(reversed(c.get("frames") or []))
                if c and chosen[3]
                else (c or {}).get("frames", [])
            )

            def angle(a, b):
                return math.degrees(
                    math.acos(float(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)))
                )

            relative = np.asarray(r["frames"][0]).T @ np.asarray(r["frames"][1])
            measured = (
                np.asarray(frames[0]).T @ np.asarray(frames[1])
                if len(frames) == 2
                else None
            )
            add(
                i,
                "relative_orientation",
                measured is not None
                and angle(relative, measured) <= cfg["joint_angle_degrees"],
            )
            if r["type"] == "PhysicsSphericalJoint":
                angles = [
                    angle(np.asarray(a), np.asarray(b))
                    for a, b in zip(r["frames"], frames)
                ]
                add(
                    i,
                    "cone_orientation",
                    len(angles) == 2 and max(angles) <= cfg["joint_angle_degrees"],
                )
        limits = r.get("limits", {})
        target = limits.get("state")
        if target not in ("not_applicable", "unknown", "invalid", None):
            actual = (c or {}).get("limits", {})
            passed = actual.get("state") == target
            for field, bound in limits.get("bounds", {}).items():
                other = field
                if sign < 0 and field in ("physics:lowerLimit", "physics:upperLimit"):
                    other = (
                        "physics:upperLimit"
                        if field.endswith("lowerLimit")
                        else "physics:lowerLimit"
                    )
                measured = actual.get("bounds", {}).get(other, {})
                if target == "locked":
                    continue
                passed = passed and measured.get("state") == bound["state"]
                if bound["state"] == "finite":
                    value = measured.get("value")
                    value = (
                        value * sign
                        if isinstance(value, (int, float))
                        and field in ("physics:lowerLimit", "physics:upperLimit")
                        else value
                    )
                    absolute = (
                        math.radians(cfg["range_degrees"])
                        if limits.get("units") == "radians"
                        else cfg["range_m"]
                    )
                    passed = (
                        passed
                        and value is not None
                        and abs(value - bound["value"])
                        <= absolute + cfg["range_relative"] * abs(bound["value"])
                    )
                elif bound["state"] == "unbounded" and bound.get("direction"):
                    direction = measured.get("direction")
                    if sign < 0:
                        direction = {
                            "positive": "negative",
                            "negative": "positive",
                        }.get(direction)
                    passed = passed and direction == bound["direction"]
            add(i, "limits", passed)
    if rj and cfg["joint_existence_type"]:
        add("all", "joint_count", candidate is not None and len(cj) == len(rj))
    result = _component(rows)
    result.update(
        reference_joint_count=len(rj),
        candidate_joint_count=len(cj),
        extra_joint_count=max(0, len(cj) - len(rj)),
    )
    possible = int(bool(rj) and cfg["joint_existence_type"])
    for joint in rj:
        kind = joint["type"]
        spherical = kind == "PhysicsSphericalJoint" and any(
            b.get("state") == "finite"
            for b in joint.get("limits", {}).get("bounds", {}).values()
        )
        possible += (
            int(cfg["joint_existence_type"])
            + 2
            + 2 * sum(b is not None for b in joint["bodies"])
        )
        possible += int(
            kind in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint") or spherical
        )
        possible += int(
            kind
            in ("PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
            or spherical
        ) + int(spherical)
        possible += int(kind != "PhysicsFixedJoint")
    result["reference_possible_checks"] = possible
    result["coverage_fraction"] = len(rows) / possible if possible else 0
    return result


def reference_profile(reference, cfg):
    cfg = settings(cfg)
    rt = np.asarray(reference["triangles"], float)
    _, ri, _ = sample(rt, cfg["samples"]) if len(rt) else ([], [], [])
    material_roster = [
        (int(i), str(key))
        for i in ri
        for key in sorted(reference["materials"][i])
        if not key.startswith("_")
    ]
    return {
        "version": VERSION,
        "settings": cfg,
        "source": reference.get("eligibility", {}),
        "joint_count": len(reference["joints"]),
        "joint_fields": [
            {k: j.get(k) for k in ("type", "bodies", "limits", "enabled")}
            for j in reference["joints"]
        ],
        "body_states": {
            name: {field: record.get(field) for field in ("enabled", "kinematic")}
            for name, record in reference["bodies"].items()
        },
        "material_roster_sha256": digest(material_roster),
        "material_eligible_fields": len(material_roster),
        "material_features": dict(
            collections.Counter(key for _, key in material_roster)
        ),
        "shape_triangle_count": len(rt),
        "shape_digest": hashlib.sha256(rt.tobytes()).hexdigest(),
        "part_roster": sorted(set(reference["triangle_bodies"])),
    }


def compare(reference, candidate, cfg, *, candidate_invalid=False):
    """Reference-only eligibility is computed before inspecting candidate observations."""
    cfg = settings(cfg)
    rt = np.asarray(reference["triangles"], float)
    rpoints = rt.reshape(-1, 3)
    scale = (
        max(float(np.linalg.norm(np.ptp(rpoints, axis=0))), 1e-9)
        if len(rpoints)
        else 1.0
    )
    band = cfg["distance_m"] + cfg["distance_relative"] * scale
    _, ri, rb = sample(rt, cfg["samples"]) if len(rt) else ([], [], [])
    eligibility = reference_profile(reference, cfg)
    valid = (
        candidate is not None
        and not candidate_invalid
        and len(candidate.get("triangles", [])) > 0
    )
    alignment = None
    if valid and len(rt):
        from simready_bench.geometry_alignment import align_observation

        candidate, alignment = align_observation(reference, candidate)
    source_unsupported = sum(
        bool(x.get("geometry"))
        for x in reference.get("eligibility", {}).get("unsupported", [])
    )
    candidate_unsupported = sum(
        bool(x.get("geometry"))
        for x in (candidate or {}).get("eligibility", {}).get("unsupported", [])
    )
    unsupported_candidate = candidate_unsupported > source_unsupported
    mapping, matching = (
        _body_match(reference, candidate, scale)
        if valid
        else ({}, {"ambiguous_reference_bodies": [], "costs": []})
    )
    if len(rt):
        if valid:
            shape, nearest_data = surface(
                rt, np.asarray(candidate["triangles"]), cfg, band
            )
        else:
            shape = {
                "score_percent": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "rms_m": None,
                "p95_m": None,
                "band_m": band,
                "samples_each_direction": cfg["samples"],
            }
            nearest_data = None
        shape.update(
            status="measured",
            earned=shape["score_percent"] / 100,
            eligible=1,
            measured=1,
            coverage_fraction=reference["eligibility"].get("supported_prims", 0)
            / max(1, reference["eligibility"].get("visible_prims", 1)),
        )
        shape.update(
            coverage_basis="fixed source supported-visible-prim fraction",
            full_surface_correspondence="partial"
            if source_unsupported or candidate_unsupported
            else "supported_authored_surfaces",
        )
        if unsupported_candidate:
            shape.update(
                score_percent=0.0,
                earned=0.0,
                reason="Candidate contains unsupported/invalid visible geometry; full correspondence not established",
            )
    else:
        shape = {
            "status": "not_applicable",
            "score_percent": None,
            "earned": 0,
            "eligible": 0,
            "measured": 0,
            "coverage_fraction": 0,
        }
        nearest_data = None
    parts = []
    reference_parts = sorted(set(reference["triangle_bodies"]))
    candidate_parts = set(candidate["triangle_bodies"]) if valid else set()
    for body in reference_parts:
        own = (
            rt
            if len(reference_parts) == 1
            else rt[np.asarray(reference["triangle_bodies"], dtype=object) == body]
        )
        other = mapping.get(body)
        ct = (
            (
                candidate["triangles"]
                if candidate_parts == {other}
                else np.asarray(candidate["triangles"])[
                    np.asarray(candidate["triangle_bodies"], dtype=object) == other
                ]
            )
            if valid and other
            else []
        )
        detail, _ = (
            (dict(shape), None)
            if valid
            and len(reference_parts) == 1
            and len(candidate_parts) == 1
            and len(ct)
            else surface(own, ct, cfg, band)
            if len(ct)
            else ({"score_percent": 0.0, "recall": 0.0, "precision": 0.0}, None)
        )
        if unsupported_candidate:
            detail["score_percent"] = 0.0
        parts.append({"reference_body": body, "candidate_body": other, **detail})
    part_result = {
        "status": "measured" if parts else "not_applicable",
        "score_percent": float(np.mean([p["score_percent"] for p in parts]))
        if parts
        else None,
        "eligible": len(parts),
        "earned": sum(p["score_percent"] / 100 for p in parts),
        "measured": len(parts),
        "coverage_fraction": len(parts) / max(1, len(reference["bodies"])),
        "parts": parts,
        "missing_parts": sum(p["candidate_body"] is None for p in parts),
        "extra_bodies": max(
            0, len((candidate or {}).get("bodies", {})) - len(reference["bodies"])
        ),
    }
    structural = [
        {
            "feature": "body_count",
            "passed": candidate is not None
            and len(candidate["bodies"]) == len(reference["bodies"]),
        }
    ]
    if cfg["joint_existence_type"]:
        structural.append(
            {
                "feature": "joint_inventory_count",
                "passed": candidate is not None
                and len(candidate["joints"]) == len(reference["joints"]),
            }
        )
    for body, record in reference["bodies"].items():
        for field in ("enabled", "kinematic"):
            if record.get(field) is not None:
                actual = (
                    (candidate or {})
                    .get("bodies", {})
                    .get(mapping.get(body), {})
                    .get(field)
                )
                structural.append(
                    {
                        "reference_body": body,
                        "feature": "body_" + field,
                        "passed": actual == record[field],
                    }
                )
    if parts:
        part_result["geometry_score_percent"] = part_result["score_percent"]
        part_result["structure"] = _component(structural)
        part_result["score_percent"] = min(
            part_result["score_percent"], part_result["structure"]["score_percent"]
        )
        part_result["earned"] = part_result["score_percent"] / 100 * len(parts)
        part_result["treatment"] = (
            "minimum of reference-body mean surface F-score and fixed rigid-structure check fraction; no cross-component weighting"
        )
    materials = []
    for sample_index, (triangle, bary) in enumerate(zip(ri, rb)):
        target = reference["materials"][triangle]
        if valid:
            distance, ci, cb = (
                nearest_data[4][sample_index],
                nearest_data[2][sample_index],
                nearest_data[3][sample_index],
            )
            measured = candidate["materials"][ci]
            other_bary = cb
        else:
            distance = math.inf
            measured = {}
            other_bary = [0, 0, 0]
        for key in sorted(k for k in target if not k.startswith("_")):
            a = _value(target, key, bary)
            b = _value(measured, key, other_bary)
            passed = bool(distance <= band and _equal(a, b, cfg))
            materials.append({"sample": sample_index, "feature": key, "passed": passed})
    material_result = _component(materials)
    material_result["appearance"] = _appearance_component(materials)
    # Compact per-feature evidence retains the fixed denominator without thousands of rows.
    material_result["features"] = {
        k: {
            "eligible": sum(r["feature"] == k for r in materials),
            "earned": sum(r["passed"] for r in materials if r["feature"] == k),
        }
        for k in sorted(set(r["feature"] for r in materials))
    }
    material_result.pop("checks")
    material_result["coverage_basis"] = (
        "reference area samples with supported effective material properties"
    )
    material_result["reference_possible_fields"] = cfg["samples"] * 12
    material_result["coverage_fraction"] = (
        material_result["eligible"] / material_result["reference_possible_fields"]
    )
    joints = _joint_checks(reference, candidate if valid else None, mapping, cfg, scale)
    from simready_bench.kinematics import compare_kinematics

    kinematics = compare_kinematics(
        reference,
        candidate,
        mapping,
        cfg,
        scale,
        ambiguous_bodies=matching["ambiguous_reference_bodies"],
    )
    if any(
        body in matching["ambiguous_reference_bodies"]
        for j in reference["joints"]
        for body in j["bodies"]
        if body is not None
    ):
        joints.update(
            status="measured",
            score_percent=0.0,
            earned=0.0,
            lower_bound_percent=0.0,
            upper_bound_percent=0.0,
            reason="candidate_ambiguous_correspondence",
        )
    unobserved = [
        {"role": role, **joint}
        for role, observation in (("reference", reference), ("candidate", candidate or {}))
        for joint in observation.get("kinematic_observation", {}).get("unobserved_joints", [])
    ]
    reference_unobserved = [joint for joint in unobserved if joint["role"] == "reference"]
    candidate_unobserved = [joint for joint in unobserved if joint["role"] == "candidate"]
    if reference_unobserved:
        reason = "unsupported_reference_joint_connectivity"
        unknown = {
            "status": "unknown", "score_percent": None, "earned": None,
            "eligible": None, "coverage_fraction": None,
            "lower_bound_percent": 0, "upper_bound_percent": 100, "reason": reason,
        }
        joints.update(unknown, unobserved_joints=unobserved)
        kinematics.update(unknown, unobserved_joints=unobserved)
        affected = {
            "fixed_constraints" if joint["type"] == "PhysicsFixedJoint" else "moving_joints"
            for joint in reference_unobserved
        }
        for name in affected:
            kinematics["groups"][name].update(unknown)
    elif candidate_unobserved:
        failed = {
            "status": "measured", "score_percent": 0.0, "earned": 0.0,
            "coverage_fraction": 1.0, "lower_bound_percent": 0.0,
            "upper_bound_percent": 0.0, "reason": "candidate_malformed_joint_connectivity",
        }
        joints.update(failed, unobserved_joints=unobserved)
        kinematics["unobserved_joints"] = unobserved
        kinematics["candidate_joint_count"] += len(candidate_unobserved)
        kinematics["extra_joint_count"] += len(candidate_unobserved)
        affected = {
            "fixed_constraints" if joint["type"] == "PhysicsFixedJoint" else "moving_joints"
            for joint in candidate_unobserved
        }
        for name in affected:
            group = kinematics["groups"][name]
            count = sum(
                (joint["type"] == "PhysicsFixedJoint") == (name == "fixed_constraints")
                for joint in candidate_unobserved
            )
            group["candidate_joint_count"] += count
            group["eligible"] += count
            group.update(failed, measured=group["eligible"])
        if kinematics["status"] != "unknown":
            kinematics.update(failed)
            active = [group for group in kinematics["groups"].values() if group["eligible"]]
            kinematics["eligible"] = kinematics["measured"] = len(active)
    complete_reference = (
        not source_unsupported
        and reference["eligibility"].get("visible_prims", 0) > 0
        and reference["eligibility"].get("supported_prims")
        == reference["eligibility"].get("visible_prims")
    )
    complete_candidate = (
        valid
        and not candidate_unsupported
        and candidate["eligibility"].get("supported_prims")
        == candidate["eligibility"].get("visible_prims")
    )
    return {
        "reference_geometry_status": "complete" if complete_reference else "partial",
        "complete_geometry_comparison": bool(complete_reference and complete_candidate),
        "complete_reconstruction_success": None,
        "schema": VERSION,
        "status": "measured",
        "aggregate_score_percent": None,
        "compliance_weight": None,
        "illustrations_scored": False,
        "reference_eligibility_sha256": digest(eligibility),
        "reference_eligibility": eligibility,
        "settings": cfg,
        "alignment": alignment,
        "components": {
            "shape": shape,
            "rigid_parts": part_result,
            "joints": joints,
            "materials": material_result,
            "kinematics": kinematics,
        },
        "body_correspondence": mapping,
        "body_matching": matching,
        "reference_limitations": reference["eligibility"].get("unsupported", []),
        "candidate_limitations": (candidate or {})
        .get("eligibility", {})
        .get("unsupported", []),
        "scope": "default-time authored polygon/analytic surfaces; source correspondence, not independent real-world physical truth",
        "tolerance_status": "documented defaults; controlled-fixture validated, not empirical calibration",
        "acceptance": "component measurements only; compliance acceptance remains separate",
    }
