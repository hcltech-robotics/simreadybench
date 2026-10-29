# Publish NV100 task packages

Build Harbor packages from an immutable asset release and its matching native
evaluator wheel. Package preparation writes local files; publication is a
separate Harbor command. Keep credentials outside the exported directory.

## Release inputs

The canonical input to `simready_eval.export` contains `schema`, `repo_id`,
`revision`, `source_commit`, `source_collection` and `tasks`. The
[native evaluator guide](../README.md#native-harbor-export-and-migration)
describes the versioned asset and runtime interfaces.

- `repo_id` is `hcltech-robotics/simreadybench`; `revision` is a full immutable
  Hugging Face commit.
- `source_commit` identifies the source used to build the evaluator wheel.
  `source_collection` identifies the NVIDIA collection and revision.
- `tasks` contains the 100 NV100 tasks. Each has a stable `id`, readable
  `object_name`, positive `nominal_dimensions_m` in canonical XYZ metres,
  `actor`, `verifier`, `reference`, `entrypoint`, `views` and `budgets`.
- Actor and verifier role manifests use `simready-role-assets-v1` and enumerate
  each permitted file's HF path, destination, byte size and SHA-256. Actor inputs
  contain one observation and the public instruction. Verifier inputs contain
  the task's reference and comparison closure, including dependencies and notices.
- The entry point is `asset.usd`. The four comparison IDs come from the frozen
  camera manifest. Agent and final time budgets are recorded in the release;
  runner repetitions and model limits are configured separately.

Finish asset, image, camera, licence and checksum validation before preparing
the release. Use the input schema expected by the exporter, rather than feeding
its generated `release.json` back as canonical input. Stable task IDs and source
bytes must remain unchanged when adding readable labels or packaging metadata.

## Prepare the export

Build the wheel from the selected source revision and use a Python environment
with Harbor 0.24.0 and the release's locked dependencies. The example below uses
the native 2.0.0 wheel; select the wheel matching the asset-release format.

```sh
python -m build --wheel native
python -m pip install --no-deps ./native/dist/simreadybench_eval-2.0.0-py3-none-any.whl
harbor --version
harbor auth status
python native/tools/prepare_hub_release.py --release ./release-input.json \
  --wheel ./native/dist/simreadybench_eval-2.0.0-py3-none-any.whl \
  --output ../simreadybench-hub-release --asset-freeze-verified \
  > ../simreadybench-hub-release-receipt.json
```

The output must be a fresh directory outside every Git checkout. The helper
validates the release schema, public HF revision, task roster, role hashes,
package inventory and publisher digests. `--asset-freeze-verified` records that
asset validation is complete; it does not perform that validation. The optional
`--public-assets PATH` uses a local mirror with the same manifest and checksum
checks.

Inspect the local receipt, `release.json`, `inventory.json` and dataset namespace.
Only the declared task and dataset files belong in the export. The actor build
verifies the frozen public inputs and installs the named runtime instruction
with its own content identity. Reference USDs, comparison images, results and
source checkouts are excluded from actor inputs.

## Publish

Authenticate with Harbor using an account with publication rights for the
destination. Run publication against the inspected export root:

```sh
harbor publish ../simreadybench-hub-release --public --tag RELEASE_TAG --concurrency 4
```

The dataset name is `hcltech-robotics/simreadybench`; task names use
`hcltech-robotics/nv100_<id>`. Harbor discovers immediate task directories and
synchronizes their digests. Preserve the returned dataset/task content digests
and `dataset.toml`. Tags are mutable, so use content digests for reproducible
consumption. Verify the dataset and tasks are publicly accessible, including
when updating packages that were previously private.

## Verify consumption

Use the returned 64-hex dataset digest in place of `DATASET_DIGEST`:

```sh
harbor download 'hcltech-robotics/simreadybench@sha256:DATASET_DIGEST' \
  --output-dir ../simreadybench-published --export
harbor run --dataset 'hcltech-robotics/simreadybench@sha256:DATASET_DIGEST' \
  --agent nop --n-tasks 1 --n-concurrent 1 --max-retries 0 \
  --job-name nv100-package-check --jobs-dir ../simreadybench-package-check
```

This no-output test checks package acquisition, startup, artifact handling and
known invalid-submission scoring. Inspect collected rewards, exceptions and the
verifier report, rather than only the command exit code. Use the
[verification guide](../../docs/native-harbor-verification.md) for evaluator,
isolation and reference-qualification checks.
