# Staged evaluation spending bounds

Approved limits are strictly **Harbor fees < $200** and **OpenAI < $1,000**,
including preflight and the later 600 attempts. No payment settings, credentials,
hosted jobs, task timeouts or evaluator limits were changed in this revision.

## Supported controls and remaining blocker

Harbor 0.24.0's inspected hosted/job/trial schemas expose attempt/retry counts,
concurrency, agent runtime/setup timeouts and environment resources; they expose
no dollar-cap field. Stored credits, auto-recharge being disabled and monitoring
completed costs are not hard spending limits. The pinned Codex config exposes a
weighted rollout-token budget with reminders; this inspection did not establish
a dollar reservation bound including in-flight requests. Do not present it as one.

The agent normalizes observed cumulative tokens after responses. That repairs
accounting, not request admission. A cancellation after observing a token total
can overshoot by the active response and other in-flight work. Native usage does
not include a provider dollar bill, so costs remain unavailable unless calculated
separately with an explicit verified pricing basis. Child-session usage is retained
but not summed into root-session totals, avoiding fork/resume double counting.

Consequently the present self-service configuration cannot **guarantee** either
strict dollar limit for 600 attempts. Do not launch the sweep on a claim that this
agent implements a hard cap. Its current artifact/accounting readiness tests are
not spending-limit tests.

## Verified pricing basis

OpenAI official model/pricing documentation checked 2026-10-29:

| Standard, short context, USD / million tokens | Input | Cached input | Output |
|---|---:|---:|---:|
| gpt-6.1-sol | 2.00 | 0.10 | 10.00 |
| gpt-6.1-terra | 1.00 | 0.05 | 5.00 |

Sources: https://developers.openai.com/api/docs/models/gpt-6.1-sol and
https://developers.openai.com/api/docs/models/gpt-6.1-terra . Longer context,
cache writes, service tier and regional processing can change prices. Any dollar
derivation must state those selections and account for reasoning output. Do not
use this short-context table as a worst-case admission price. No dollar estimate
is inserted into ACP or ATIF metrics by this package.

The parent verified Harbor pricing at https://hub.harborframework.com/pricing :
$0.1561032 per physical core-hour plus $0.0264132 per GiB-hour, billed per second.
Derived rates for the inspected **actual 2.2.0 oily-waste task** are:

| Service | Declared resources | USD / hour |
|---|---|---:|
| Actor | 4 CPU, 8 GiB | 0.8357184 |
| Feedback | 2 CPU, 8 GiB | 0.5235120 |
| Separate verifier | 4 CPU, 16 GiB | 1.0470240 |

The feedback row assumes it is separately billed; verify actual service billing
and whether any co-resident resources are already included before settlement.

Evidence is the extracted canonical task.toml, docker-compose.yaml and separate
verifier configuration under `/workspace/terra-cad-trial/tasks/nv100_04ed00317ccc5a90`.
The actor timeout is 231,000 seconds; actor and verifier build timeouts are each
1,200 seconds; verifier execution timeout is 600 seconds. These are not small
cost bounds. Feedback may remain alive through other phases. Build/setup,
service teardown, cleanup and any separately billed resources must be included.
These files prove one canonical task's resources, **not all 100 release tasks**.
The other 99 resolved task configurations and actual billed service lifetimes
still need verification before asserting an aggregate bound.

Even 600 maximum-duration verifier executions alone cost $104.7024; if verifier
build time also bills at its declared resources, 600 builds plus executions can
reach $314.1072 before actor/feedback/setup. Assuming ten minutes of actor plus
feedback and ten minutes of verifier per attempt gives $240.62544 before builds.
These are conservative scenarios, not measured prices or guaranteed lifetimes.

## Minimal enforceable allocation design

1. Include every already-spent preflight dollar in separate Harbor and OpenAI
   ledgers. Reserve margin for active work, teardown, delayed metering and taxes
   or other applicable charges before admitting new work. Credits do not reset
   the user's spending limit.
2. Resolve exactly 100 canonical tasks, two explicit low agents and three attempts
   each: 600 attempts. Set `retry.max_retries=0`; keep preflight outside that count.
3. Before dispatch, atomically reserve the **worst-case** resource-lifetime and
   inference-request bound for every concurrent attempt. Release a reservation
   only after its final costs/lifetime are settled. Reserve in-flight work even
   when its current usage is absent. Concurrency controls exposure, not total cost.
4. Require for every admitted attempt: `spent + all_reserved + new_bound + margin`
   is strictly below each approved limit. A failed or cancelled attempt still
   consumes its observed cost. Never count a missing cost as zero.
5. A nominal allocation of $0.30 Harbor and $1.50 OpenAI per sweep attempt would
   reserve $180/$900 and leave $20/$100 for preflight/margins. This is an example
   allocation, **not an implemented cap or an approval to spend it**. Subtract
   actual preflight spending and margin first; reduce allocations if necessary.
6. Enforce lifetime bounds including build/setup/feedback/verifier/cleanup and
   reserve inference before each provider request using counted input plus a
   verified maximum output at the applicable worst-case price. Post-response
   accounting alone cannot enforce this. The inspected hosted route does not
   provide the complete required admission hooks, and no new proxy was created.

Reducing the inherited 231,000-second task budget to meet a fee ceiling changes
the evaluation condition and must be recorded as a time-capped condition. It is
not directly equivalent to the old baseline. A measured pilot can inform a plan;
it cannot establish an upper bound across all 100 objects by itself.
