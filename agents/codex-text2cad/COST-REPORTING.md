# v0.1.2 cost-reporting verification

No running job, credential or billing setting was changed. No inference, launch,
retry, cancellation or restart was performed while implementing this revision.

The wrapper now reports **estimated cumulative API token cost in USD** through
ACP `session/update.params.update.cost`, before the final prompt response. Harbor
0.24.0 reads that field into AgentContext.cost_usd and ATIF total_cost_usd. The
estimate is labeled `status: estimated`, `billed: false`, with a dated rate-table
hash, official source URLs and declared/observed pricing assumptions. It is not
an OpenAI invoice or a spending cap. Sandbox settlement remains separate.

The live-count test fixture contains 769,755 inclusive input tokens, 713,850
cached tokens and 4,715 output tokens, including 1,177 reasoning tokens. Eleven
short requests, rather than the aggregate context size, yield a Standard/global
estimate of **$0.1151725 for Terra** or **$0.230345 for Sol**. These are test-fixture
estimates, not a claim about the live job's final bill or its per-request details.

Tests cover:

- Inclusive input minus cached reads/writes before uncached pricing; reasoning
  output is a subset of output, never an additional billed output count.
- Per-request long-context thresholds, Fast/priority, regional processing and
  12-hour cache writes; complete reconciliation with native cumulative totals.
- Repeated checkpoints, multiple ACP prompts and duplicate response IDs without
  double counting; conflicting or incomplete evidence leaves cost unavailable.
- Unknown model, policy, tier or malformed retention metadata without invented
  pricing; existing upstream cost reports remain unchanged.
- Actual wrapper stdio ordering: the USD notification precedes the prompt result.
  Later context-only updates preserve the last valid cumulative USD estimate.
- Typed ACP UsageUpdate parsing, actual HarborAcpClient.session_update handling,
  AcpAgent.populate_context_post_run and its ATIF converter. Cache/input/output
  reconcile, and estimate provenance survives the recorded ACP events into ATIF.
- Existing native-agent identity/usage handling, safe retention, low-reasoning
  receipts and genuine NVIDIA SR.002 thumbnail checks for static/hinged assets.

The focused review fixed preservation of upstream costs even when local pricing
is unavailable, final-response checkpoint freshness, conflicting response IDs,
optional-counter reconciliation and oversized/private turn-context copying.
Pricing context retains only model/effort and relevant billing classifications.

The reproducible local test environment uses Harbor 0.24.0 and Python ACP SDK
0.12.1. The hosted harness controls its own SDK installation; this package does
not replace the hosted harness. Final tests and wheel/lock records are retained
under `/workspace/codex-text2cad-validation/cost-*` in the implementation workspace.

For a restart, set the new immutable source ref and retain the selected model,
explicit low effort and existing secret selection. The updated hosted example
adds `CAD_AGENT_PRICING_POLICY=openai-standard-global-30m`. Absent billing metadata
uses that declared Standard/global/30-minute basis; observed classifications
override it. Unreported cache-write counts use the pinned Codex protocol's zero
default. Assumptions are explicit in archived cost-estimate.json and ACP metadata.

The source example is updated only after the implementation commit exists. The
existing live run cannot be changed by selecting this new revision elsewhere.
