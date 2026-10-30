# Plain Pi for NVIDIA Build

This package provides native Harbor and hosted ACP entrypoints for Pi; it does
not introduce a new model/agent loop. It adds an explicit model profile and
formal feedback instructions.
It installs no CAD tools, CAD skills, MCP server or model credentials. Original
NV100 tasks, inputs, evaluators, scoring and five-call enforcement stay unchanged.

## Verified versions and differences

- Tested manager: Harbor **0.24.0**, using its installed Pi adapter. Its option is
  `thinking` (alias `reasoning_effort`), not upstream main's renamed primary field.
- Pi: **@earendil-works/pi-coding-agent 1.4.0**, Node 24.19.0. The npm lock records
  the hosted runtime installation; `pi_nvidia/pins.json` records the release
  integrity and inspected source/document hashes. The inherited Harbor installer
  pins the top-level Pi version, but does not use this npm lock for global install.
- Hosted bridge: registry-listed community **pi-acp 0.0.34**, source commit
  `04d0a15d6bb42bb65cce5c9caedcdcb2cdcb6905`, ACP registry revision
  `af322d3bf734436be837bc078f5ae9364449aeba`. The existing bridge is reused
  unchanged. It handles Pi RPC, images, streaming, tool events, cancellation and
  `agent_settled`; our wrapper supplies the guarded configuration and final usage.
- Inspected Harbor main at `7d9a0b688898a11cd03cf14a21d51a9ab45d3b94` and Pi main
  at `908aa7b74baa7502321293797313f64c6fec22ee`. Main is not substituted for the
  tested installed manager. Pi's **bundled 1.4.0 docs** and actual model parser were
  checked as well as upstream docs. The key reference is `$NVIDIA_API_KEY`.
- Both inspected Harbor adapters build a minimal custom model with ID and optional
  reasoning/map, omitting image input, context/output limits and compatibility
  details. This subclass overrides that builder and reuses the parent installer,
  run method, session collection, image handling and ATIF conversion.

## Configure a selected model

The NVIDIA model is intentionally **unselected**. The example is an incomplete
template and fails validation until capabilities are filled truthfully. Do not
interpret its `thinking: off` placeholder as a verified setting for any model.

Install this Python package into the native Harbor manager, then merge
`examples/native-agent.template.json` into the intended existing job. Omit a
built-in `name: pi`; it would override the custom `import_path`. Each agent entry
has its own `model_name`, `kwargs.thinking` and `kwargs.model_profile`. The model
name is `nvidia/` followed by the exact NVIDIA `vendor/model` ID. The profile must
declare input modalities, tool-call support, context and output capacity,
reasoning support, all supported thinking mappings, relevant compatibility flags
and the evidence used to establish them. Unknown model capabilities are not defaults.

`NVIDIA_BASE_URL` must be `https://integrate.api.nvidia.com/v1`, and the API is
`openai-completions`. An operator supplies `NVIDIA_API_KEY` through the existing
authorized environment/secret route; the generated Pi config contains only the
environment-variable reference, never a copied key. No credential setup is done
by this package. It does not use a local `.env` file as hosted credentials.

For reasoning models, `thinkingLevelMap` must contain all seven Pi levels. Set
unsupported levels to null and supported levels to the exact provider value.
The adapter rejects an unsupported requested level instead of allowing Pi to
clamp it. `compat.thinkingFormat`, sampling parameters and per-thinking sampling
overrides carry model-specific behavior. Pi validates remaining compatibility
fields when loading the generated config. The profile declares capabilities;
it cannot prove the remote endpoint honors them without a later authorized probe.

Image tasks default to `require_images: true`; text-only profiles fail rather than
pretend to accept the observation. `require_images: false` is available for a
separately declared text-only condition. Parent Pi image/tool-result handling is
retained. Initial task text and `/task/observation.png` remain intact.

Pricing is **unknown** in this initial adapter. It omits model prices and clears
Pi's generated cost estimates from Harbor/ATIF metrics, retaining upstream cost
fields in diagnostic metric extras. An absent price in Pi internally becomes zero;
that is not evidence of free inference. Complete session receipts supply token
totals including cached reads/writes. Missing, malformed or all-zero default
usage leaves totals unavailable; valid partial step receipts remain reviewable.
Resumed copied prefixes are excluded using the parent adapter's accounting logic.

## Five validator calls, not five inference calls

`feedback.txt` explicitly asks Pi to validate the first complete candidate, inspect
findings, repair actionable mandatory failures and revalidate changed candidates.
It reads feedback health, preserves same-ID retries and reserves the last call
for the final candidate. A passing first candidate may finish after one call.
It stops on success, exhausted allowance, a real runner deadline or investigated
unrepairable faults, reporting unresolved findings and final-validation status.

This is an **instruction policy**, not an added completion gate or inference-call
controller. The original benchmark feedback ledger enforces five admitted
validator calls. An admitted timeout still consumes a call; a busy request does
not; retrying the same ID/content retrieves the existing result. No turn limit,
task timeout, retry count or independent verifier budget is added or changed.
Visual self-checks are not hidden visual/kinematic fidelity feedback.

## Hosted Hub package (0.2.0)

`harbor-agent.json` implements the documented Python-uv/ACP source contract and
runs `python -m pi_nvidia.acp`. Use `examples/hosted-agent.template.json` for a
GitHub source entry with `name: acp`; do not use native `import_path` on Hub.
The implementation is local and **unpublished** until a separately authorized
push. The example's immutable source ref becomes accessible only after publication.

Fill the actual model ID, `env.PI_NVIDIA_PROFILE` JSON and
`env.PI_NVIDIA_THINKING` in each entry. Those controls are independent per agent.
The incomplete example intentionally cannot execute before a model is selected.
Merge this fragment into the intended job without changing its tasks, attempts,
retry counts or timeouts. It uses `credential_mode: direct` and selects the
existing `NVIDIA_API_KEY` hosted secret by name. No key value belongs in the
profile, source or example. Direct mode accepts the selected injected
`HOSTED_INFERENCE_TOKEN` or named environment secret. An unexpected gateway URL
is rejected because no NVIDIA gateway route has been verified.

On startup the wrapper installs checksum-pinned Node and runs `npm ci
--ignore-scripts` using the packaged npm lock. It verifies Pi and pi-acp versions.
Installation can be cached; every start rewrites the per-trial model profile,
thinking settings and launcher. The launcher disables skills, extensions and
prompt templates; no CAD packages or tool servers are installed. Pi's config,
home/session mapping and native sessions are isolated under this trial's agent
logs. No auth/config files from another agent home are inspected or retained.
Harbor's agent-project virtualenv is removed from the child's authoring `PATH`,
restoring the task's Python and its genuine `simready_eval.feedback` client.
The pinned node/Pi executables remain explicit; other task path entries are kept.

The proxy checks Pi's actual model/thinking state before allowing the prompt,
maps the internal `harbor-endpoint` identity to the configured NVIDIA identity,
and rejects model/thinking changes that disagree with the entry. It prepends the
same feedback policy while retaining the original text and image blocks. MCP
requests and session load/fork are explicitly rejected, not silently ignored.
The original NV100 task uses the feedback CLI and needs neither capability.

After Pi settles, the wrapper reads only its matching native session JSONL,
deduplicates exact repeated event IDs and sums all per-call usage including
compaction/summary receipts. Context occupancy from ACP `usage_update` is **not**
treated as cumulative input. Final ACP `usage` carries inclusive input, output,
cached reads/writes and reasoning where recorded. Missing/conflicting usage or
model/thinking receipts leave usage unavailable. Additional native sessions require
reconciliation. Native provider errors become ACP errors rather than the upstream
bridge's normal `end_turn`. Dollar cost is omitted and recorded as unknown;
unverified upstream cost is retained in metadata rather than silently accepted.

`pi-accounting.json`, native sessions and Harbor ACP events retain the evidence.
Harbor 0.24.0's primary ACP ATIF converter omits its cached-token field; the
wrapper's receipt, ACP metadata and AgentContext still retain the cached count.
This package does not modify Harbor's hosted result UI or backfill old jobs.

## Offline verification

The tests invoke no paid model. Install this project with its test extra and set
these paths to the prepared local packages:

```sh
PYTHONPATH=$PWD PI_TEST_PREFIX=/workspace/pi-nvidia-runtime \
NV100_EVAL_WHEEL=/path/to/simreadybench_eval-2.2.0-py3-none-any.whl \
python -m pytest -q tests
```

The feedback tests check the immutable published 2.2.0 wheel hash and exercise its
original ledger with synthetic evaluator results: failure then repair, early
completion, quota exhaustion, changed-content conflicts, transport retries, busy
admission and an admitted simulated timeout. They do not claim NVIDIA validation
or prove that a model follows the policy. Pi parser checks use an in-memory empty
credential store, disable model-network refresh and forbid fetch. No key is read.
Hosted tests use the real registry pi-acp executable and Harbor 0.24.0's typed
ACP runner, with only the Pi RPC backend mocked. They check text/images, tool
events, model identity, early agent_end followed by continuation, final token
ingestion, unknown cost, provider failure and cancellation. The real manifest
entrypoint is initialized without credentials or a model call. A fresh locked
Python environment matches the inspected Harbor Pi source hash.

Before deployment, select the exact NVIDIA model and verify image/tool input,
model/thinking parameter behavior, limits, streamed usage and the resolved hosted
adapter/runtime versions. No run, secret configuration, push or deployment has
been performed. Publication and the model-specific profile are still required;
the hosted protocol/package integration itself is implemented and tested.

Sources: [Harbor Pi](https://github.com/harbor-framework/harbor/blob/7d9a0b688898a11cd03cf14a21d51a9ab45d3b94/src/harbor/agents/installed/pi.py),
[Pi model documentation](https://github.com/earendil-works/pi/blob/908aa7b74baa7502321293797313f64c6fec22ee/packages/coding-agent/docs/models.md),
[pi-acp](https://github.com/svkozak/pi-acp/tree/04d0a15d6bb42bb65cce5c9caedcdcb2cdcb6905),
[Hub custom agents](https://docs.harborframework.com/hosted-harbor/custom-agents).
