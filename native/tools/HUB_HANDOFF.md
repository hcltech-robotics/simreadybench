# Publish the clean NV100 package from STC

Run this on STC with the existing approved Harbor login. Credentials stay on STC;
do not copy a key, credentials file, home directory, or working checkout into this
handoff, a task, HF, or a container. This handoff prepares local files and makes
one public HF metadata read. It never logs in or publishes automatically.

The runtime is the accepted 1 GiB evaluator from commit
`364d77e46f258724875a925345cc54069cfa08e6`; wheel SHA-256 is
`4ffb2ada910496506f85300493e0ff3475f78c31a8133672472a88d853e6e3e9`.
Its renderer and frozen images are unchanged. The source NVIDIA collection is
`nvidia/simready-assets@f4860aa0c75607adb50a78bf96e77510bffc040f`.

## Inputs from the asset publication owner

Finish all 100 assets / 500 RGBs, masks, cameras, descriptions, licenses and
checksums first. Preserve historical data under `results/` in the existing HF
repository. Regenerate all public instruction bytes with the v6 runtime's
`public_instruction(dimensions)`, so they advertise 1 GiB; update role-file hashes.
Publish/freeze the complete canonical HF dataset and obtain its full commit SHA.
Do not use a tag, branch, placeholder revision or draft role list.

Supply canonical **input** `release-input.json` for `simready_eval.export`, with
exactly these top-level fields: `schema`, `repo_id`, `revision`, `source_commit`,
`source_collection`, `tasks`. Do not pass the exporter's draft output release.json.

- `schema`: `simready-harbor-release-v1`; `repo_id`:
  `hcltech-robotics/simreadybench`; `revision`: immutable 40-hex HF commit.
- `source_commit`: the exact runtime source commit above; `source_collection`:
  the exact NVIDIA repository/revision above.
- `tasks`: exactly 100 objects, each with `id`, `actor`, `verifier`, `reference`,
  `entrypoint`, `views`, `budgets`. Keep frozen `nv100_<hash>` IDs unchanged.
- `actor`/`verifier`: `simready-role-assets-v1` objects with `repo_id`, `revision`,
  `task_id`, `role`, `files`. Each file has actual HF `path`, local `destination`,
  `size_bytes`, `sha256`. Actor gets exactly `observation.png` and `instruction.md`.
  Verifier gets only that task's `reference/` and `comparison/` closure, including
  eligibility, cameras, four RGB/mask pairs and required notices/dependencies.
- `reference`: actual nested source path under `reference/`; `entrypoint`:
  `asset.usd`; `views`: the four frozen comparison IDs. Budgets remain
  `agent_seconds: 231000`, `final_seconds: 600`.

The helper validates this schema. Its freeze flag is the release owner's
acknowledgment of completed source/frame/license/checksum work, not a substitute
for that work. It checks the declared HF revision exists and is public/ungated;
it does not redownload and re-audit the full dataset.

For the reused bed frames, preserve the original execution receipt plus the
verified final-code equivalence proof. The raw bed bridge was `2c90b234...`;
the qualified current bridge is `62a265f1...` after that proof. Bind eligibility
to the qualified current bridge without falsifying the original receipt.
Toolbox executed the current bridge directly. Inventory the proof alongside
the reference. Do not substitute reduced candidate-probe frames.

## Prepare and inspect locally

Verify `SHA256SUMS` before using the extracted handoff. Use the existing dedicated
Harbor 0.24.0 Python environment; install only the bundled neutral wheel there:

```sh
python -m pip install --no-deps ./simreadybench_eval-1.0.0-py3-none-any.whl
harbor --version
harbor auth status
harbor auth org list --search hcltech-robotics --columns name,role
python prepare_hub_release.py --release /path/to/release-input.json \
  --wheel ./simreadybench_eval-1.0.0-py3-none-any.whl \
  --output /tmp/nv100-clean-hub-release --asset-freeze-verified \
  > /tmp/nv100-clean-hub-release-receipt.json
```

The output must be a fresh directory outside every Git checkout. The helper
requires a matching public HF commit, creates exactly 100 task directories plus
the seven declared dataset files, and validates task schema 1.4 / dataset 1.0,
official publisher digests, root file digests and the complete allowlist.
It finalizes the draft README/status truthfully: owner freeze verified and ready
for publication; published consumption remains unverified and no task-specific
oracle is claimed. No asset binaries, results or working repository are bundled.
Inspect the receipt, `release.json`, `inventory.json`, and dataset namespace.

## Publish only after the HF freeze

The following is the publication action; the helper does not execute it:

```sh
harbor publish /tmp/nv100-clean-hub-release --public --tag nv100-v1 --concurrency 4
```

Names in task.toml and dataset.toml select `hcltech-robotics`, not the personal
namespace. Harbor auto-discovers immediate task child directories and synchronizes
digests: pass only the audited clean root, never its parent or a working checkout.
Keep the final publisher output and dataset.toml. Versions/tags, including the
automatically added latest tag, are mutable; retain and use returned content
digests. The public base OCI image is pinned by digest and all Python wheels are
hash-locked. Keep any concrete built-image digest records during consumption.
For existing packages, `--public` does not change prior private visibility; verify
the intended dataset/tasks are public using the official access/visibility tools
and an unauthenticated consumer.

## Consume the published digest on STC

Replace the digest below with the actual publisher result's 64 hex digits
(omit its existing `sha256:` prefix to avoid duplicating it). No `--launch`, model
credentials, paid inference, timeout multiplier, extra HF agent host or security
relaxation is needed:

```sh
harbor download 'hcltech-robotics/nv100@sha256:<DATASET_DIGEST>' \
  --output-dir /tmp/nv100-published-download --export
harbor run --dataset 'hcltech-robotics/nv100@sha256:<DATASET_DIGEST>' \
  --agent nop --n-tasks 1 --n-concurrent 1 --max-retries 0 \
  --job-name nv100-published-consumption --jobs-dir /tmp/nv100-published-jobs
```

This no-output probe tests published package/role acquisition, native startup,
declared artifact handling and frozen-reference validation. Its missing candidate
must produce the declared known-zero invalid-submission result; it is not model
performance or task-specific Oracle evidence. Inspect the actual Harbor result,
exception_info, collected rewards and verifier report: CLI exit zero alone did
not detect the earlier collector failure. Preserve failures unchanged. The full
native Oracle lifecycle already passed on STC with the accepted v2 fixture;
do not repeat it merely to refresh labels. Subsequent real evaluation remains on
STC with the same isolation and runner-selected repetition policy.
