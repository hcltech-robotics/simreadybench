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

## Hosted Hub package (0.2.1)

`harbor-agent.json` implements the documented Python-uv/ACP source contract and
runs `python -m pi_nvidia.acp`. Use `examples/hosted-agent.template.json` for a
GitHub source entry with `name: acp`; do not use native `import_path` on Hub.
The implementation is local and **unpublished** until a separately authorized
push. The hosted example pins implementation commit
`07491e32305578d79c1ef0d7ea1bcdbc4fe3759d`; that ref becomes accessible only after publication.

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
been performed. Publication and live model capability verification are still required;
the hosted protocol/package integration itself is implemented and tested.

## Selected Nemotron smoke trial

`examples/nemotron35vl-low-one-trial.json` selects the unchanged original NV100
task `nv100_04ed00317ccc5a90`, one attempt, concurrency one and zero trial retries.
The plain Pi agent has no CAD packages. The original five-validator-call ledger
and evaluator remain unchanged. Its 600-second authoring cap is a smoke-test
condition, not the original full benchmark time budget; setup remains 360 seconds.
The task's verifier timeout remains 600 seconds and both build timeouts 1200 seconds.

The wire model is `nvidia/nemotron-3.5-super-vl-120b-a12b`; the Harbor model name
adds its own `nvidia/` provider prefix. The packaged profile selects low reasoning,
an 8192-token per-request output cap and a conservative 262000-token context cap
below NVIDIA Build's advertised 262K. Unsupported Pi thinking levels fail rather
than silently clamp. Model limits and actual image/tool/usage behavior still need
the authorized live check. NVIDIA's official schema documents low/none/high,
images, tools and streamed usage; Baseten's endpoint capabilities are not substituted.

The example selects the existing Hub `NGC_API_KEY` secret and sets
`PI_NVIDIA_ENV_NAME=NGC_API_KEY`. This is a reference to an injected environment
variable, not a secret value. If Hub metadata instead shows `NVIDIA_API_KEY` as
the real environment-variable name, select that existing secret and set the knob
accordingly. No credential creation or environment-file copying is required.

Verified native usage for this exact model gets a separate
`reference_inference_estimate` in `pi-accounting.json` and ACP metadata. Baseten's
2026-10-30 reference rates are $0.20 uncached input, $0.04 cached input and $0.50
output per million tokens. Cached input is subtracted from inclusive input;
missing cache details receive no discount. Reasoning is already within output
and is never added again. The estimate is not emitted as ACP billed cost:
actual NVIDIA cost stays unknown and Harbor compute is separate. The model is
advertised on NVIDIA's free prototype endpoint; no paid provider fallback is used.

At current Harbor rates the full 10-minute actor/feedback plus 10-minute verifier
execution scenario is $0.4010424. Charging two full cold 20-minute build windows
and six minutes of setup at their corresponding resources gives $1.33905024
before teardown/storage/other overhead. These are conservative scenarios, not
a platform dollar cap. The approved total actual-spend ceiling for this validation
work is $200; no trial has been submitted. Inspect existing job state and resolved
timeouts, secret namespace and source pin in the signed-in Hub browser before the
one submission; retain its receipt and check uncertain outcomes before retrying.

Reference sources: [NVIDIA API schema](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer),
[NVIDIA Build](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b),
[Baseten model pricing](https://www.baseten.co/library/nemotron-35-super-vl/),
[cached-input accounting](https://docs.baseten.co/inference/model-apis/pricing-and-limits),
[reasoning accounting](https://docs.baseten.co/inference/model-apis/reasoning),
[Harbor compute pricing](https://hub.harborframework.com/pricing).

Sources: [Harbor Pi](https://github.com/harbor-framework/harbor/blob/7d9a0b688898a11cd03cf14a21d51a9ab45d3b94/src/harbor/agents/installed/pi.py),
[Pi model documentation](https://github.com/earendil-works/pi/blob/908aa7b74baa7502321293797313f64c6fec22ee/packages/coding-agent/docs/models.md),
[pi-acp](https://github.com/svkozak/pi-acp/tree/04d0a15d6bb42bb65cce5c9caedcdcb2cdcb6905),
[Hub custom agents](https://docs.harborframework.com/hosted-harbor/custom-agents).


## Authentication diagnostics (0.2.1)

`PI_NVIDIA_ENV_NAME` is an optional nonsecret variable-name selector. A nonempty
explicit selection is preserved. Without one, the reference defaults to
`NVIDIA_API_KEY`, with fallback to the existing `NGC_API_KEY` or hosted token.
Both alias directions work. The legacy `PI_NVIDIA_KEY_ENV` remains supported,
but new configurations should not use it: Harbor 0.24.0's sensitive-environment
serializer classifies its KEY-containing name as sensitive and rewrites its
literal value. The new name survives the actual hosted serializer unchanged.
The exact backend stage that removed the legacy selector is not visible here.

The selected variable, hosted token and alternate NVIDIA/NGC variable may coexist
only when their nonempty values agree. Conflicting credentials fail before any
configuration files or inference request; the error contains no values. The
selected nonempty variable takes precedence when equivalent aliases exist.
Aliases are normalized only in the child environment; no key is written into
`models.json`, launch arguments or logs. `pi-auth-status.json` records only the
selected variable name, source name, presence boolean, endpoint and model.
No credential value, suffix or hash is retained.

Sanitized artifacts from job `1c0efefa-ad31-43ae-bb9f-2ae04b6b8935` establish that
its selector was absent, its effective environment supplied `NGC_API_KEY` and no
`NVIDIA_API_KEY`, and the old adapter wrote `apiKey: "$NVIDIA_API_KEY"`. The same
environment replayed against the exact original runtime commit fails at ACP
session/new with zero inference HTTP requests. This identifies a credential
reference mismatch, not a rejected NVIDIA API key or model-quality result. Pi
1.4.0 supports `$NGC_API_KEY` interpolation; no OAuth login is needed.

The regression runs the actual Pi CLI and pi-acp bridge with dummy keys against
a loopback SSE endpoint, after actual Harbor config serialization. It covers the
exact missing-selector/NGC-only case, both explicit alias directions, an empty
selector, hosted token and missing credentials in normal and offline startup.
It verifies the Bearer header, exact model, reasoning low, 8192-token cap and
final usage; non-loopback fetch is forbidden. Missing credentials cannot issue an
inference request. Conflict tests verify that no dummy values appear in errors.
No paid API request is made. Hosted recovery still needs an authorized publication
and one separately approved retry; no retry is performed by this package.
