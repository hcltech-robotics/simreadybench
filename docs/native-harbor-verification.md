# Native Harbor port: local verification

The implementation starts at source commit
`ad38697277f897833ee9a19e329f99063827de1f` on isolated branch
`feat/native-harbor-nv100`. Nothing in this work publishes a repository, changes
live HF data, provisions credentials or invokes paid inference.

The clean package is under `native/`; it contains its own evaluator, renderer,
candidate-only feedback service, selective fetcher and allowlisted exporter.
It does not import the working benchmark harness or private application.
`native/README.md` documents the asset and camera interfaces.

## Runtime evidence

The two opt-in Docker tests in `tests/harbor_native/test_native_runtime.py`
passed on actual containers in 50.62 seconds:

- Genuine NVIDIA formal validation admits five distinct requests, rejects the
  sixth, replays an identical request after restart and retains the exhausted
  ledger. The parsing worker cannot read the controller's ledger. Actor tools
  contain neither Blender nor the NVIDIA validation SDK.
- A fresh verifier with `--network=none`, a read-only root, no capabilities and
  only the declared submission plus frozen fixtures produces formal evidence
  and geometry/appearance/visual/kinematic scores of 100 for an original neutral
  articulated fixture. This is an integration fixture, not NV100 qualification.

Container build RUN steps on this host cannot reach public package servers
(`Errno 101: Network unreachable`). For these local runtime checks, the pinned
public Python base, public dependency wheels and public Debian Blender files
were preloaded. No private image or host renderer was injected into scoring.
Production recipes still perform checksum-pinned role-specific HF acquisition
during trusted setup and run final scoring offline. A cold setup fetch against
the new benchmark revision remains unverified until that revision exists.

A normal Harbor 0.24.0 Oracle trial was attempted with the generated task,
ordinary Compose sidecar and separate verifier. It was blocked before execution:
Harbor's daemon-kernel probe returned false for `CONFIG_NFT_FIB_INET`, and Harbor
rejected `network_mode='allowlist'` as unsupported. The probe's actual
`/proc/config.gz` output has neither NFT FIB IPv4 nor IPv6 enabled. We did not
patch Harbor or relax the declared network policy. Full native Harbor lifecycle
acceptance requires a compatible Docker host.

## Review and score contract

Independent review findings were corrected with regression coverage: public
fixture permissions for actor UID 1000, canonical `asset.usd`
entrypoint consistency between feedback and final scoring, and a known zero
geometry score becoming an unavailable visual score; the later renderer pass
also fixed constant fallbacks requiring UVs, valid nested sibling textures being
rejected, and the measured 335 MiB heat-gun scene exceeding final-worker file
limits. Only render workers receive the larger bounded 4 GiB file allowance;
other workers retain 128 MiB. A fully measured zero
factor now proves visual zero while retaining appearance uncertainty and partial
measurement coverage in the detailed report.

The standard Compose test also passed (51.94 seconds), exercising the actual
exported feedback definition and actor client with genuine validation in a shared
offline namespace. The namespace anchor substitutes only for the host-unavailable
Harbor network controller; this does not claim Harbor's phase policy executed.
Final scoring now rejects missing/mismatched bridge-code and Blender-binary
identities as well as recipe/camera changes before candidate work.

The corrected renderer traverses effective instance-proxy shaders, translates
active emission/IOR, shares equivalent UV data without changing numeric values,
and releases Python geometry after Blender has copied it. A current-version
barcode-printer probe completes all five RGBs and masks. The 1,931,457-triangle
locker remains unqualified: it exceeded the original address-space bound, then
decoder duplication, and finally the 180-second Blender diagnostic window.
Partial images are not admitted. The user selected replacement of the largest
reference outliers; no additional locker-specific work gates the release.
The largest previously completed real full-five-camera probe is the
305,896-triangle heat gun; current-version resource qualification is separate.
Render workers now have a bounded 12 GiB address space within the declared
16 GiB container. No geometry simplification or score-definition change occurs.

Formal uses `95 * passed / 46`, plus 5 only when all 46 mandatory requirements
pass; the optional 51 remain diagnostic. Visual uses
`sqrt(full_3D_geometry * mean(four_foreground_SSIM_percentages))`. Four views have
equal fixed weight and use the union of reference/candidate opacity-independent
foreground masks. Kinematics retains the geometric mean of applicable moving
and fixed constraint recovery groups. There is no overall F/V/K weighting.
`reward.json` always emits separate known-credit, known, coverage and lower/upper
fields; standalone `metric.py` aggregates each dimension with its own known
denominator and explicit unavailable counts and bounds.

## Baseline failures

A broad repository run returned 1866 passed, 9 skipped and 15 failed while some
new renderer/preparation files were still being edited. Five port failures from
that intermediate run are superseded by focused final verification; the other
ten were reproduced in the untouched baseline checkout with the same command
and interpreter (10 failed, 2 passed, 2.08 seconds):

- `test_benchmark_command.py::test_sequential_execution_keeps_repetitions_budget_identities_and_exported_captures`
- `test_coordinator_contract.py::test_collection_failure_is_bounded_and_persisted[exception0]`
- `test_coordinator_contract.py::test_collection_failure_is_bounded_and_persisted[exception1]`
- `test_coordinator_contract.py::test_saved_dispatch_resumes_its_original_batch`
- `test_coordinator_contract.py::test_measured_batch_without_delivery_cannot_author_again`
- `test_coordinator_contract.py::test_failure_before_delivery_retries_the_same_owned_batch`
- `test_coordinator_contract.py::test_held_data_can_retry_cleanup_without_reauthoring[False]`
- `test_coordinator_contract.py::test_held_data_can_retry_cleanup_without_reauthoring[True]`
- `test_orchestration_retries.py::test_unstarted_tail_membership_does_not_consume_transport_retries`
- `test_publishing.py::test_invalid_submission_is_published_and_does_not_block_later_reconstructions`

Eight coordinator/orchestration cases fail with `KeyError: 'providers'` in
unchanged route-throttling code. The benchmark case fails its retained-capture
assertion; the publishing case expects 51 optional unknowns where baseline
scoring returns zero. These unrelated baseline failures were not changed.

## Release gates

No export is certified publish-ready merely by parsing successfully. The
separate asset owner must supply the immutable clean HF revision, complete
roster and five-frame render qualification. All task manifests pin that revision
and per-file checksums; public packages never fetch `results/`.

Historical migration is additive and local-only: dry-run/apply receipts record
source/destination hashes, preserve all eleven parquet shards and missing legacy
references, and provide rollback evidence without deleting source paths.
Live copy, reader/writer cutover and root benchmark card changes remain pending.
The optional per-set bucket manifest is only a mirror of a pinned dataset;
no bucket has been created or copied.
