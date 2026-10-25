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
The largest completed real full-five-camera probe is the 305,896-triangle heat
gun; its current-version evaluation evidence is below.
Render workers now have a bounded 12 GiB address space within the declared
16 GiB container. No geometry simplification or score-definition change occurs.

The current heat gun is now qualified through actual five-camera preparation
and the complete evaluator. Its runtime dependency inventory selects 20 files /
44,287,839 bytes from 46 originals / 80,853,193 bytes, without changing originals.
The five-frame render took 115.80s. Full evaluation took 179.40s under the
unchanged 600s limit: formal 88.8043, geometry/appearance/visual/kinematic 100,
all measured at coverage 1. Maximum individual process and child RSS was
1,299,416 KiB; this is not an aggregate process-tree peak. All 46 original file
hashes were checked again after evaluation. The candidate retained full geometry.
The source formal result is reported as measured, not inflated to 100.

Runtime inventories now walk only reachable authored dependencies and notices,
retain a referenced archive, exclude unrelated originals, and canonicalize
relative UDIM tile paths. They do not copy or publish data. Exported recipes
also pin the same OMP/OpenBLAS/USD thread limits used by these measurements.
Nominal XYZ dimensions remain public as required by the existing input contract.
Reference preparation no longer accidentally applies the candidate's 256 MiB
package bound: trusted references have a separate 2 GiB acquisition bound, while
candidate and feedback package limits remain unchanged. Reference/compare file
limits also permit the bounded USDZ expansion. Final formal and feedback workers
permit candidate archive members up to the declared 256 MiB package limit;
feedback JSON is still independently capped at 8 MiB. The targeted package, CLI,
feedback and material-diagnostic SDK checks passed (82 tests). This is a preparation fix, not evidence that an
asset exceeding the candidate limit is eligible; a faithful bounded candidate
representation must still be demonstrated.
The material-property sampler now leaves PNG16 properties unmeasured instead of
reporting Pillow's truncated 8-bit values. This diagnostic limit is separate from
the corrected renderer, which streams lossless files, shares content descriptors
and preserves original encoded PNG16 for native Blender decoding. Nine targeted
tests pass, including actual native low-bit samples and canonical ICC removal.
The final bridge's canonical files match all 31 active 8-bit inputs from the
printer and heat gun exactly. Actual five-RGB/five-mask probes match the previous
baseline byte for byte (40.58s / 451412 KiB child RSS for the printer, 115.74s /
1254900 KiB for the heat gun). Their scientific scenes, cameras and Blender
binary also match. These equivalence results preserve the original receipts;
they do not claim that new code originally produced historical images.
Bed/toolbox full-precision render probes now complete: bed 853847 triangles,
390.16s and 3518760 KiB maximum child RSS; toolbox 742798 triangles, 297.42s and
3116868 KiB. Each generated five RGBs/five masks at camera hashes matching the
asset owner's exact frozen rigs. All 47/82 original source hashes stayed equal.
Bed executed the pre-ICC-clear compiler; a separate 22-file equivalence proof
records four native PNG16 copies and 18 unchanged canonical PNG8 outputs under
the final compiler. Toolbox executed the final compiler directly. The bed
bridge's pixel-buffer estimate peaked at 176 MiB and retained no decoded bank.
Neither probe establishes candidate/full-grader eligibility: original source
texture bytes exceed the candidate package allowance and all evaluation phases
must fit the existing 600s deadline. The original extra outer 180s render-worker
cap was incompatible with these measured workloads; its correction is below.
One additional targeted test confirms resource-bound MemoryError is a typed
unknown with no images, rather than unsupported content or model-quality zero.

Portable deliverables are saved through Library: corrected neutral wheel/lock
bundle `libfile_10c62858c51c8191b0d253a305d03ebf` version 2 (ZIP SHA256
`94e54828a26e8a36622009b1ada957ac429311c60251e8e08e6dd9f460b75510`),
clean original Oracle fixture `libfile_6f035bcc9dbc81919c715d9ccf15fc14` version 0
(SHA256 `49b105454eed52e8b7611283d7f178978cf5d259338a38a707f18b4c848f6b54`),
and real-render evidence `libfile_cb87d231c4a08191b527fe9928faedda` version 0
(SHA256 `a72404eb196b8169dbfb6f0cf520171fda53d6d746290bb746b6795e071ae1ec`).
The bundle and Oracle fixture use exact implementation commit
`55be78b25a72c59762fc4f38450b9b816dfa46b1`, native wheel SHA256
`2f2cfcc9948b93d5991ea495d1eab0fc089e4b10638d949ddd1c0ea06dc6322d`.
No working checkout, source binaries or historical results entered those code
handoffs. The render evidence contains derived images, attribution and receipts.

The first exact portable Oracle trial executed on STC under official Harbor
0.24.0 from 19:12:40Z to 19:15:45Z. All five feedback calls completed, sixth-call
rejection and actor isolation assertions passed, candidate artifacts transferred,
and the separate offline verifier produced complete scores (formal 82.6087;
G/R/V/K 100). The trial still errored when Harbor read `reward.json`: atomic
temporary outputs retained mode 0600. Later cleanup copied readable host-owned
files, so their final ownership does not describe access at the failure point.
CLI exit zero is not proof of a successful Harbor trial.

The output writer now chmods completed temporary report/reward files to 0644
before atomic replacement. A restrictive-umask regression reproduced 0600 before
the fix; 35 contract/CLI tests pass after it. A real network-none, read-only,
cap-drop Docker process wrote the patched outputs as a different mapped UID;
the ordinary host collector could read both 0644 files. No score, renderer,
network, UID or capability policy changed. The original failed trial is preserved in Library
`libfile_5b6bd07d66ac81919e1516dc1bfde698` version 0, ZIP SHA256
`1e1abcd26a45a61760c59e133b7a574258d0b6a847e05dd67c7d8b4e1a0211f1`.

The corrected full standard Harbor trial passed on STC from
2026-10-25T19:24:38.549424Z to 19:26:17.894890Z, trial
`nv100-neutral-hinge__XTToxvF`, with `exception_info=null` and collected
`verifier_result.rewards`: `evaluation_complete=1`, formal credit
0.826086956521739, visual and kinematic credit 1. Five feedback calls completed
and the sixth was rejected. The live read-only publication monitor observed
root-owned reward mode 0644 readable by host UID1000 at 19:26:04.765Z before
cleanup, followed by normal cleanup ownership change. There was no manual
chmod/chown or isolation change. The passing clean bundle is Oracle Library
`libfile_6f035bcc9dbc81919c715d9ccf15fc14` version 1, ZIP SHA256
`f19cdfa92fac557e6738ab3eeb85c640e664123dd6155b3456c2333fe085d433`,
implementation commit `e52d48b7bdb64949a4e00e418b06b15d49659fb3`.
The matching evaluator bundle is Library `libfile_10c62858c51c8191b0d253a305d03ebf`
version 3, ZIP SHA256
`1b62fbb732a77126910446e26ce00c346bbb6f90c46994328bc7912ce81d2b22`,
native wheel SHA256
`54661461b5231f433b6b677ae340c8bf790b7871172bf210ed91f44f74ce6a45`.
This is a native lifecycle pass for the original neutral fixture, not published
NV100 asset-fetch acceptance or whole-cohort eligibility.
The consolidated STC evidence is Library
`libfile_5b6bd07d66ac81919e1516dc1bfde698` version 1, 719334 bytes / 77 files,
SHA256 `40fbb8a86194772be447cbf405b1c9c1b8e2d4933f5b2a5637dfbbbad4d36649`.
It preserves both original failed and corrected runs, the verified corrected
bundle, live output-publication trace and host pre/postflight. The reported
corrected job has one completed trial, zero errors and no exception. The same
20 service containers and existing Python process remained untouched.

Existing bed timestamps/logs establish that texture extraction was still active
at approximately 190.55s, before Blender; comparison RGB/mask render calls total
69.85s and the separate observation pair adds 17.47s. Subtracting the observation
pair gives a 372.69s four-view whole-job estimate, not a separately executed
four-view measurement. Thus the extra 180s whole render-worker cap fails even
before view count matters. The worker now receives remaining time from the
existing final deadline, while Blender still has 180s and render address space
stays 12 GiB. Overall wall time remains 600s; candidate package/geometry/texture
rules, image recipes and unknown scoring semantics are unchanged. The new
genuine SDK comparison/dispatch regression reproduced 180 before the change;
36 contract/CLI tests pass with the corrected allocation. No accepted STC run
or reference render was repeated for this diagnosis.

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
