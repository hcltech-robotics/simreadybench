# Frozen evaluator 2.2.0

This source reconciles the native evaluator with the already published NV100
2.2.0 release. The runner package keeps its independent version. The evaluator
is CPU-only; the release retains five formal feedback admissions, 64 authoring
turns, the separate offline verifier and actor/reference role separation.

| Identity | Frozen value |
| --- | --- |
| Accepted evaluator wheel | `simreadybench_eval-2.2.0-py3-none-any.whl` |
| Wheel SHA-256 | `caf759ff7d9b392922530567b22678da4556de366fa879ef6395299425627d52` |
| Geometry protocol | `simready-area-cloud-similarity-v1` |
| Export schema | `simready-harbor-release-v2-evaluator-2.2.0` |

All 31 Python/profile package files were compared byte for byte with the accepted
wheel and its frozen source snapshot. Their SHA-256 inventory is committed in
[`native/frozen-2.2.0.json`](../native/frozen-2.2.0.json). A rebuilt wheel has the
same package bytes, dependency declarations, Python requirements and entry
points. Archive hashes can differ because build timestamps and public README
metadata differ. The accepted archive hash above remains the release identity.

Run the verification from a fresh checkout after building:

```sh
python -m build --wheel --sdist native
python native/tools/verify_frozen_wheel.py \
  --built-wheel native/dist/simreadybench_eval-2.2.0-py3-none-any.whl
```

Add `--frozen-wheel /path/to/accepted.whl` to verify its archive hash and compare
both distributions' dependency and entry-point metadata. The regression suite verifies source members against the frozen inventory.
The command above also verifies a newly built wheel.

## Scoring and qualification

Geometry scoring changed intentionally between 2.1.0 and 2.2.0. It uses 65,536
fixed area-weighted samples in each direction and graded distance similarity;
reports carry the scoring version. SciPy 1.18.0 belongs only to the verifier
runtime. Formal checks, alignment and kinematic measurements retain their
frozen protocols. Renderer compatibility admits the exact historical reference
seal through the qualified adapter and rejects altered seals.

The retained CPU replay compared the accepted 2.1.0 and 2.2.0 evaluators with four
CPU cores and 16 GiB memory. Formal rule statuses, alignment, full kinematic
evidence, render-scene identity and appearance score were identical. All eight
comparison RGB/mask files were byte-identical. Geometry and derived visual
scores changed under the declared new protocol. This is evidence for the
intentional version transition; equality to the accepted 2.2.0 runtime follows
from the package-byte comparison above. No private reference or result payload
is included in this repository.
