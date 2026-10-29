# Codex + text2cad/build123d

This is an agent package, not a benchmark task variant. It installs the complete
text-to-cad 0.8.2 skill, cadgen 0.8.2 and build123d 0.13.0, explicitly injects the
selected skill into Codex's instruction, and supplies a small CAD-to-USD bridge.
Original task inputs, instructions on disk, feedback tools and verifier stay intact.

`cad_agent/pins.json` records the upstream skill commit, retrieval timestamp,
package versions and file hashes. The original MIT license is included. The
runtime lock is the tested pilot's 67-package hash lock plus usd-core 26.8. Ten
checksum-pinned Debian libraries support headless OCP on Linux amd64/glibc >=2.38.
There is no Blender dependency. Installation requires Python 3.12 with venv,
`dpkg-deb`, network access to the pinned package sources, and writable installation
storage. The hosted wrapper also installs checksum-pinned Node 24.19.0 rootlessly.

## Native Harbor 0.24.0

Install this package with its native extra in the Harbor manager environment:

```sh
uv pip install './agents/codex-text2cad[native]'
```

Merge `examples/native-agent.json` into an existing job configuration, preserving
the original task reference and execution limits. Do not add `name: codex`; that
built-in selector overrides the import path in Harbor's factory. Change
`model_name` independently of `kwargs.reasoning_effort`.

The Codex subclass pins CLI 0.164.0. Its bakeable `install()` hook calls Harbor's
installer, uploads the agent bundle to `/opt/codex-text2cad`, installs the locked
runtime, and checks real cadgen STEP/STL/GLB generation. Its `run()` hook always
checks the baked bundle, registers the 13 skill files for the current user, and
checks the runtime before calling Codex. This per-trial step does not depend on
`install()` being called again. The MCP config is reconstructed in the constructor.
It preserves task-provided MCP servers and does not replace the benchmark verifier.

`render_instruction()` puts the complete selected SKILL.md plus the USD workflow
in the actual prompt; references remain available in the native skill directory.
The original task instruction is included unchanged. Variant identity is written
to both Harbor agent information and the converted ATIF trajectory; parent
trajectory conversion, model identity and usage/cost extraction are preserved.

## Hosted Harbor Hub

`harbor-agent.json` and `uv.lock` implement Harbor's supported GitHub source/ACP
route. The stdio wrapper reuses the same preparation, skill and USD exporter. It
injects the skill into `session/prompt`, preserving original text/image blocks,
task MCP servers and cancellation. Version 0.1.1 also normalizes legacy model IDs
to `model[effort]` and cumulative usage with explicit original-counter metadata.
It pins codex-acp 1.12.0 and sets
`CODEX_PATH` to the separately pinned CLI 0.164.0 rather than the adapter's bundled
0.154.0. ACP is a different transport from native `codex exec`; record that when
comparing results.

`examples/hosted-agent.json` pins the published, locally tested implementation at
`15716cc553a94d3e5c75a91c71cff2b9ec41c309`. Merge that agent entry into the
existing job, select the existing appropriate hosted secret, and
preserve the canonical task reference and limits. Do not use local `import_path`
in the hosted source form. No credential is included in this package.

Model selection comes from `model_name` / `HARBOR_ACP_REQUESTED_MODEL`; reasoning
comes independently from `env.CAD_AGENT_REASONING_EFFORT` (default `medium`). Direct
mode uses the normal provider endpoint. The wrapper honors `HOSTED_INFERENCE_URL`
only when provided and reads the injected `HOSTED_INFERENCE_TOKEN`. It neither
creates credentials nor reads browser sessions. The hosted configuration is a
template, not an executed launch or a verified remote dry run.

That example still identifies the previous published 0.1.0 implementation. The
0.1.1 accounting/retention/thumbnail revision is local pending publication; replace
the source ref with its approved immutable commit before evaluating that revision.

Official source contract:
https://docs.harborframework.com/hosted-harbor/custom-agents

## CAD → USD and evidence

Run reconstruction scripts using `/opt/codex-text2cad/bin/cad-python` (the ACP
prompt supplies its own cache prefix). This launcher supplies native library and
CAD state paths before the Python process starts. Use `Part` local CAD solids in
mm, explicit translations in mm, unit `rotation_wxyz` quaternions and RGB colors.
`Hinge` anchors use world coordinates in mm; the bridge computes local position
and rotation frames for both bodies. USD uses meters, Z up, explicit rigid bodies,
masses, convex-hull collision meshes, material bindings and revolute-joint limits.
It supports static solids and two-body hinges; it does not infer arbitrary physics
from STEP. Dynamics simulation and benchmark quality are not established by local
structural tests.

`export_asset(..., source=__file__, evidence_dir=...)` writes STEP intermediates
and hashes linking reconstruction source, CAD parts and the reopened USD. Source
and evidence must stay outside the submission directory. The actual task output
contract remains authoritative; the supplied workflow targets the existing NV100
`/output/submission/asset.usd` contract.

Agent logs distinguish readiness, instruction injection and a verified artifact
chain. The smoke receipt never counts as reconstruction use. Artifact hashes are
not tamper-proof attestation: inspect native execution/tool events alongside the
generation receipt before reporting skill use. A changed final USD invalidates the
chain. Benchmark F/V/K scores remain separate and unchanged.

The exporter now renders a 256×256 preview from the authored meshes, transforms
and material colors using a CPU-only orthographic renderer. It saves the actual
SR.002 dependency at `.thumbs/256x256/asset.usd.png` and links it through the
default prim's `assetInfo.previews.thumbnail` asset path. It uses no observation
or ground-truth image. The real NVIDIA 2026.8.0 SR.002 rule is tested to fail when
the PNG is removed and pass after generation on both static and hinged exports.

## Usage, low reasoning receipts and safe retention (0.1.1)

Before forwarding a prompt, the wrapper requires an actual matching ACP session
receipt for the configured `model[effort]`; environment settings alone do not
qualify. It converts Harbor's undecorated legacy model request to that explicit
ID. Native `turn_context` records must also match the selected model and effort
before native cumulative usage is marked verified.

The final native `total_token_usage` checkpoint is used once per root session;
checkpoints and repeated prompt receipts are never summed. Corrected ACP
`inputTokens` includes cached input; `cachedReadTokens` is its cached subset, and
`outputTokens` includes reasoning output as reported by Codex. Original ACP usage
remains under `_meta.cad_agent_accounting.upstream_usage`. Missing/ambiguous/reset
or mismatched records remain unverified, with upstream numbers preserved. Other
native sessions are retained but not summed, avoiding fork/resume double counting.

The agent periodically copies only native `rollout-*.jsonl` from this task's
`.codex-text2cad/sessions`, plus receipt-linked reconstruction `.py`, STEP files
and generation.json. It never scans `.codex/sessions` or copies auth/config homes.
Symlinks and references outside authoring are excluded. Known injected secrets,
credential JSON fields and viewer tokens are redacted; retention.json records
original/retained hashes. Files over 128 MiB are explicitly skipped. A hard kill
can lose changes since the last two-second snapshot; retention status must be
checked. No sandbox can make model-generated content a tamper-proof attestation.

Harbor 0.24.0 still omits cache totals from its primary ATIF converter. After
downloading a completed trial, explicitly normalize it using the saved receipt:

```sh
python -m cad_agent.accounting path/to/trajectory.json path/to/cad/accounting.json
```

This leaves `trajectory.upstream.json`, records provenance in ATIF `extra`, fills
cumulative input/output/cache totals and is idempotent. It does not alter USD,
scores, benchmark inputs or dollar cost. The hosted detail page is outside this
package; its cache field can remain incomplete until Harbor's converter changes.
No derived pricing is inserted. See BUDGET.md for verified rates and the remaining
strict-cap limitations; accounting correction is not a spending guard.

## Local verification without inference

```sh
python -m cad_agent.prepare install --prefix /tmp/cad-agent
/tmp/cad-agent/bin/cad-python examples/generate.py /tmp/cad-agent-validation
uv pip install '.[native,test]'
PYTHONPATH=/tmp/cad-agent/runtime/lib/python3.12/site-packages pytest -q
```

`examples/generate.py` executes a real cadgen model and exports static and hinged
assemblies through the same helper used by reconstruction scripts. Tests verify
units, bounds, transforms, triangle indices, materials, mass/collision schemas,
joint-frame coincidence, limits, USD reopening, bundle loading, factory identity,
baked-image lifecycle and ACP request preservation. No model calls are required.
