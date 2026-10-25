# NV100 candidate package size recommendation

Recommendation only: the implemented candidate limit remains 256 MiB. No live
dataset, account, task allowance or reference content was changed for this note.

The limit is inherited from the original repository, not newly introduced by the
native port. Commit `1148527c2b3cc40c7b366856acf21c20721b8ace` (20 October 2026,
initial SimReadyBench 1.0.0 release) already sets it in
`engine/controller.py` (`max_package_bytes`, a configurable constructor default),
`engine/package_checks.py` (`LIMIT_BYTES`) and `candidate_export.py` (`MAX_BYTES`,
explicitly described as the controller default). It remains in source baseline
`ad38697277f897833ee9a19e329f99063827de1f`. No explicit user selection of this
number appears in the supplied protocol history or original public README.
The native plan retained the existing defensive value; that does not establish
a scientific reason for retaining it when the selected references exceed it.

The asset owner supplied the completed revised 100-asset runtime-closure counts;
they were reused without another asset audit. Exactly five exceed 256 MiB:

| Asset | Required runtime bytes | MiB |
|---|---:|---:|
| storage_toolbox_foldout_hardCase_a01 | 503,323,398 | 480.01 |
| bed_hospital_standard_a01 | 452,060,328 | 431.12 |
| robotics_small_bot_a01 | 411,827,612 | 392.75 |
| electrical_main_terminal_a01 | 364,169,524 | 347.30 |
| light_surgical_ceiling_arm_b01 | 289,041,323 | 275.65 |

All 100 fit below 512 MiB. That cap leaves only 33,547,514 bytes (6.7%) above
the largest closure. A 640 MiB cap would leave 33.3% headroom; recommend the
simpler **1 GiB bounded candidate/expanded-package allowance**, leaving
570,418,426 bytes (113.3%) above the largest current closure. Zero current
runtime closures exceed the proposed limit. Keep reference acquisition separately
bounded at 2 GiB; do not force candidates to reduce texture quality to satisfy
an inherited byte default.

This is a small coordinated implementation change, not a config-only flip:

- Share one candidate-byte limit across `package.py`, final freezing in `cli.py`,
  formal worker file-size bounds, feedback extraction and public instructions.
- Separate the plain-TAR transport allowance from expanded candidate bytes.
  Bound header/PAX overhead from the entry and UTF-8 path limits; it must not
  increase the expanded-byte limit or silently reduce usable candidate bytes.
- Increase feedback scratch from 512 MiB to 2 GiB and bound admission before
  reading large request bodies: at most one upload/evaluation in flight, with
  explicit retryable busy responses before quota reservation. Preserve the
  five-call ledger, same-ID/digest replay and sixth-call rejection. This avoids
  multiplying GiB request buffers and validator processes under the 8 GiB
  sidecar limit.
- Retain all existing path/link/archive checks, 10,000 entries, per-image and
  geometry/decoded-allocation guards, worker address-space bounds, container
  memory/CPU limits, final 600s deadline and Blender 180s child limit. No score
  formula or unknown/failure classification changes.
- Regenerate the public instruction bytes and their manifests before the HF
  freeze. No image rerender is caused by a package-limit change. Run focused
  boundary, TAR-overhead, admission/quota and container checks when implemented.

Package capacity is only one eligibility condition. Increasing it does not
prove that an exact full-resolution reference clone fits the 600s runtime.
The 102.73 MiB texture-reduced bed witness has a real 0.70-point visual loss;
it is not exact-identity/Oracle evidence. Keep runtime qualification separate
and use STC for eventual published-package validation. Do not repeat accepted
smoke/render evidence merely to refresh labels.

Read-only Harbor auth status in this executor returned `Not authenticated`.
No login, key creation, organization change or publication was attempted.
Publishing rights for `hcltech-robotics` need verification in an already
authorized publishing environment. HF freeze precedes any Hub publication.
