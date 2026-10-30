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
not; retrying the same ID/content retrieves the existing result. No benchmark
turn limit, task timeout, feedback-ledger rule or independent verifier budget is
changed. Hosted Pi error/provider retries are disabled separately.
Visual self-checks are not hidden visual/kinematic fidelity feedback.

## Hosted Hub package (0.2.4)

`harbor-agent.json` implements the documented Python-uv/ACP source contract and
runs `python -m pi_nvidia.acp`. Use `examples/hosted-agent.template.json` for a
GitHub source entry with `name: acp`; do not use native `import_path` on Hub.
This revision is local and **unpublished** until a separately authorized push.
The hosted examples use immutable implementation pins; use the verified revision
reported in the release handoff, which becomes accessible after publication.

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
thinking settings and launcher. The launcher disables automatic skill, extension and prompt-template discovery,
then explicitly loads one bundled runtime extension for image receipts and
budget guidance. No CAD packages, skills or tool servers are installed. Pi's config,
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
and evaluator remain unchanged. The old 600-second smoke authoring cap is removed
in 0.2.4; the hosted agent enforces 64 model turns. Setup remains 360 seconds.
The canonical task still imposes its inherited 231000-second outer timeout; see
the platform constraint below. This is not an unlimited hosted deadline.
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


## Output-length handling and first candidate (0.2.2)

NVIDIA max_tokens is a cap on each inference response, including reasoning. It
is separate from Harbor's whole authoring timeout. In the observed smoke trial, the final 8192-token response contained
reasoning but no text/tool call and stopped with length. Pi 1.4.0 then settled;
pi-acp 0.0.34 converted the settled run to ACP end_turn. The run had no entrypoint
and no admitted validation calls. This is one failed workflow, not evidence that
the model is generally unsuitable. The source code does not support blaming a
wall-clock timeout for that response-length stop.

The wrapper now retains only last-assistant stop/content flags in the accounting
receipt, maps length to ACP max_tokens, and performs at most one same-session ACP
prompt continuation when a verified final length response has neither text nor
a tool call. The continuation asks for concrete candidate construction and real
validation, preserving existing work and the original task context. It does not
change model/effort/caps or start a new trial. A second truncation returns
max_tokens; partial text/tool calls are not replayed automatically. Cancellation,
provider errors and unverifiable receipts do not trigger recovery. The original
Harbor timeout surrounds the entire agent run, so this continuation cannot reset
it. The generated continuation is also recorded in pi-length-recovery.jsonl. A
receipt captured while continuation is pending is labeled continuation_requested,
not final usage. Both responses remain in native/ACP evidence; cumulative usage and the
Baseten reference estimate include every recorded inference call exactly once.
On an ACP error, final usage remains in the native/accounting logs; a successful
ACP response includes it for Harbor ingestion.

The policy prioritizes an initial complete asset.usd, reopening and basic geometry
checks before optional exploration, then the genuine feedback client. --help is
explicitly setup only. The five admitted-validator-call ledger is unchanged and
a passing first candidate may finish early. This is an instruction policy, not
a claim that a mock server proves a model follows it or a hard validation gate.

`examples/nemotron35vl-none-one-trial.proposed.json` is an unlaunched proposal:
reasoning off (NVIDIA reasoning_effort none), max_tokens 16384, the enforced
64-model-turn budget, 360-second setup cap, one original task/attempt and zero trial
retries. None explicitly requests disabled reasoning; it does not prove absence of
internal computation. 16384 is NVIDIA's documented default and gives code/tool output more
room. This changes two generation settings for a workflow smoke test, not a
controlled reasoning comparison. Larger response capacity does not guarantee an
asset, a validation call, quality or lower cost. No increase to the approved
aggregate budget is implied, and no publication or live trial is performed here.


## Frozen image delivery (since 0.2.3)

Every hosted initial session prompt attaches the existing public
`/task/observation.png` as structured image content before model inference. Set
`PI_NVIDIA_OBSERVATION_PATH` for a different task-provided path; the optional
`PI_NVIDIA_OBSERVATION_SHA256` pins the expected task bytes. The one-asset examples
include the original frozen observation hash. Missing/nonregular PNGs and hash
mismatches fail before inference. Original instruction blocks are preserved and
an identical already-supplied image is not duplicated. Continuations reuse the
session's original image. No observation is rendered, cropped, substituted or
copied into the task package by this agent.

`pi-input-image.json` records source and outgoing ACP payload hashes, not image
bytes. The bundled Pi `before_provider_request` hook also records PNG hashes
from the actual provider payload in `pi-image-wire.jsonl`; if the first expected
image is missing, it aborts the request and the wrapper reports failure. These
are pre-send payload receipts, not a claim about remote server interpretation.
Native session records retain the image and actual tool results as before.

## Enforced 64-turn policy (0.2.4)

The hosted runtime enforces **64 model turns for the entire task**. A turn is
one Pi assistant generation plus its complete tool-result batch, regardless of
how many tools it calls. A failed generation attempt still consumes its turn.
Context notices and health reads do not create extra turns. Internal compaction
requests are not authoring turns; their recorded usage still contributes to
accounting. Model identity, reasoning and response-token caps remain independent.

Before each generation, the transient `context` notice explicitly gives the
current turn, limit 64, remaining including this turn, and remaining afterward.
For example: `turn 1 of 64. 64 turns remain including this turn; 63 remain after
this turn.` Aim to produce the first complete candidate by turn 8. Turns 61–64
are reserved in guidance for final validation/reporting. These milestones are
advisory; the maximum of 64 is enforced. Passing early is allowed. The model must
schedule any last validation in turn 64's tool batch: no 65th summary turn exists.
This reserves an opportunity, not a guarantee that a model validates or succeeds.
The genuine five-admitted-validator-call quota and counting rules are unchanged.

Pi 1.4.0's `turn_end` boundary runs after tool results have been persisted. Its
`BoundaryResult.continue=false` does not stop natural tool continuation. The
extension therefore uses the supported `ctx.abort()` after batch 64, with a
second guard against an attempted next request. All tools in the last batch
finish first. ACP returns `max_turn_requests` and retains saved output and usage;
user cancellation and genuine provider errors keep their distinct outcomes.
A single empty-length continuation cannot reset or exceed the 64-turn budget.
No automatic provider/error/Harbor trial retries or background warming are enabled.

`pi-turn-budget.json` records started/completed/remaining and the definition;
`pi-budget-notices.jsonl` retains exactly the notices sent. The ACP wrapper
requires the extension's startup receipt before admitting a task prompt. The
existing image hash receipts and accounting are preserved. Pi may append a
synthetic empty zero-usage assistant error while unwinding a post-batch abort.
The compatibility layer excludes only the exact abort artifact after 64 recorded
assistant generations, an independently exhausted budget and zero usage; it
labels the exclusion and preserves the original native record. Other errors
and missing usage remain failures/unknown, never invented zero-cost success.

Remove `PI_NVIDIA_TIME_LIMIT_SEC` and `PI_NVIDIA_FINAL_RESERVE_SEC` from old configs;
0.2.4 rejects those obsolete options. Remove `agents[].max_timeout_sec:600` as in
the updated examples. No agent authoring wall clock or time reminders remain.
Do not confuse setup/build/verifier deadlines, per-tool/transport failures or
infrastructure termination with the 64-model-turn budget. The separate native
import-path adapter does not install this hosted extension; use the hosted ACP
manifest for the enforced 64-turn configuration.

### Exact Harbor timeout constraint

Pinned Harbor **0.24.0** supports no outer authoring timeout when the task's
`agent.timeout_sec` is `None`. Its resolution is
`agent.override_timeout_sec or task.agent.timeout_sec`. Therefore an agent-level
null, omitted value or zero **cannot unset a non-null task timeout**. Removing
the 600-second cap from the unchanged pinned NV100 task inherits **231000 seconds
(64 hours 10 minutes)**, not infinity. The examples do not insert that large value
or substitute a huge override: it already belongs to the canonical task. A true
no-authoring-timeout launch with this unchanged task requires a supported Harbor
manager/API capability that distinguishes "unset override" from "explicitly no
timeout"; none is present in the verified installed API. Additional hosted sandbox
lifetime limits are not exposed by this local inspection and must not be claimed
absent. No current job is changed by these example files.

Offline tests cover exact requests 1–64, notices 64–1, both tools completing in
batch 64, no HTTP request 65, saved output, exhausted length recovery, early
completion, explicit off/low/high request mappings, image hashes, real feedback
client validation on turn 64 against the unchanged five-call ledger, and strict
abort-artifact accounting. The feedback test uses a synthetic evaluator: it proves
the client/ledger/budget integration, not live model compliance or asset quality.
No paid model call, secret access, publication or run change is part of these tests.
