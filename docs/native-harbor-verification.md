# Verify native Harbor packages

Native task verification checks evaluator behavior, role isolation, packaging and
the Harbor lifecycle. Use the exact evaluator wheel, dependency locks, renderer
and asset revision declared by the release.

## Source and package checks

Install the test dependencies and native package in Python 3.12. The
[`Execution and recovery` workflow](../.github/workflows/ci.yml) defines the
dependency and Blender versions used by CI.

```sh
python -m pip install '.[test,daytona,publish]' './native[validator,fetch]' build
python -m pytest -q -p no:cacheprovider tests/harbor_native
python -m build --wheel --sdist native
```

The tests cover mandatory and optional requirement accounting, invalid and
unknown results, reference-scale geometry, appearance, kinematics, feedback
quotas, selective acquisition and exported task inventories. They also check
that the native wheel is independent of the benchmark controller and agent
implementations.

Container, licensed and paid test tiers require their respective explicit
selection and opt-in flags. See [execution testing](execution.md#licensed-and-runtime-gates).
Unit and fixture tests do not establish that every reference workload fits the
release's resource limits.

## Export validation

Use the [publishing guide](../native/tools/README.md) to prepare a fresh
export outside the source checkout. Check that:

- The release contains exactly the declared task roster and dataset files.
- Every task pins its immutable asset revision and per-file checksums.
- The actor receives one observation image and its instruction, with no reference
  USD, comparison images or result files.
- The verifier receives only the selected task's reference and comparison closure.
- The candidate entry point, dependency rules, limits and feedback instructions
  agree across task metadata and runtime instructions.
- The exported evaluator wheel, role manifests and package digests match the
  inspected build.

Role acquisition must reject missing, extra or modified files. Final evaluation
runs offline against the declared submission and frozen verifier inputs.

## Runtime verification

Use a Linux Docker/Compose host that supports Harbor 0.24.0 network enforcement.
Check the exported package with a clean cache and no dependency on the source
checkout. A lifecycle test should exercise candidate submission, artifact
transfer, the separate verifier, reward collection and cleanup.

Feedback tests must verify five distinct admissions, rejection of a sixth call,
same-ID/digest replay, rejection of an ID reused with different bytes, and quota
persistence across service restart. Candidate parsing must not expose the
service's control ledger or verifier reference files.

Inspect collected rewards, `evaluation_complete`, `exception_info` and the
verifier report. A successful command exit alone does not establish that Harbor
collected a complete evaluation. A no-output agent checks startup and known
invalid-submission handling; it does not measure reconstruction quality.

## Reference qualification

Qualify each reference with the release's geometry, material, camera and resource
settings. Preserve original asset bytes and the actual rendering identities.
Reference self-comparison checks geometry, appearance and kinematic agreement;
formal conformance must still be measured against the profile.

Large textures, supported-but-expensive geometry and concurrent render workers
can exhaust a fixed runtime envelope. A package passing the byte limit or a
render-only probe does not establish full-evaluation eligibility. Measure the
complete evaluation with candidate headroom. Unsupported reference content and
unavailable measurements retain unknown scores and bounds. Infrastructure
failures remain distinct from invalid candidate output.

The [native evaluator guide](../native/README.md) documents support and scoring;
the [resource guide](native-harbor-size-policy.md) defines package and runtime
bounds. Retain the evaluator version, asset and camera checksums, container
identities and unmodified result reports with each published measurement.
