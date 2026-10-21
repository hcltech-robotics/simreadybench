# Execution and recovery

Every Daytona invocation enters the same coordinator, including the default single worker. The worker count controls simultaneous capacity. Each canonical condition retains its model, reference, seed, authoring limits and scoring policy throughout recovery.

The coordinator holds one durable lease per active condition in `attempt-state.sqlite3`. Each state change and its transition record commit together. Measurement, infrastructure, publication and cleanup have separate states. A completed measurement remains immutable while delivery or cleanup is pending, and a measured zero is a completed model outcome.

Read the durable state without taking ownership:

```sh
simready-bench status --output-dir ../simreadybench-results
simready-bench status --output-dir ../simreadybench-results --json
```

Status reports measured conditions, pending publication, pending cleanup and held attempts independently. `complete` becomes true when every planned condition has an accepted measurement, requested publication has completed, every owned executor has confirmed deletion and no infrastructure hold remains. A full measured count alone does not mark the run complete.

## Phase ownership

The coordinator owns executor allocation, condition leases, collection and retirement. The model gateway owns request admission and settlement. The controller seals the authored package before evaluation. The publisher delivers the collected immutable readout using the current remote parent commit.

A phase retry retains the original identity and evidence. Collection reconnects the same executor. Publication sends the same measured result, retains its delivery signature across restarts and permits at most three attempts per delivery. Deletion reconciles the same sandbox identity within its saved cleanup policy. These operations do not create a new authoring attempt or call the model again. An authorised authoring retry consumes its original finite request and spend allowance.

Collection and observation failures have a saved retry counter and elapsed deadline. Deterministic integrity rejection holds the attempt with its retained evidence. Retirement remains outstanding until exact-owner deletion is confirmed. A controller heartbeat records liveness, while completed phases and measurements record progress.

Held cleanup permits three attempts within 600 seconds. Its retry interval is saved with the attempt and uses `--retry-delay-seconds`, which defaults to five seconds. The attempt remains held with pending cleanup when either bound is exhausted.

A changed provider catalogue route or price creates an infrastructure hold. The hold and its trusted admission evidence survive report export and collection, so the coordinator retires the owned executor without replacement authoring. A temporary catalogue outage receives a bounded admission wait within the original deadline.

Resume a run by repeating its command with `--resume` and the original output directory. Resume validates the frozen run settings and staged inputs before reconciling existing workers. Completed measurements are collected and published before any eligible new authoring is considered.

A measured condition whose execution delivery receipt is missing enters a hold. Its accepted measurement and publication remain intact, and the coordinator creates no replacement executor. Cleanup remains pending when the exact resource acknowledgement cannot be recovered.

## Scientific and runtime identities

The scientific contract binds the effective prompts, skill contents, tools, inference limits, reference inputs, evaluator, scorer and capture settings. The execution receipt binds that contract to the executor wheel and container digests. A repair retains its parent receipt and must preserve the protected scientific contract. Changes to instructions, tools, limits or scoring require a declared protocol revision.

New Pi runs use the `bounded-v1` tool contract. Tool execution has a wall-time limit and bounded output, and a timeout stops descendant processes before returning its result to the actor. The authoring protocol identity includes this contract. `--tool-protocol legacy` selects the earlier contract explicitly, while resume retains the saved contract.

## Local acceptance

Install the execution, publication and test dependencies in Python 3.12:

```sh
python -m pip install '.[test,daytona,publish]' build
python -m pytest -q
```

Default discovery includes gateway, controlled-stop, execution-health, recovery, publication and installed-wheel acceptance tests. It excludes the licensed validator, container and paid tiers. These tiers require explicit local selection and opt-in. Licensed validation never runs in hosted CI.

Run installed-wheel acceptance alone:

```sh
python -m pytest -q -m acceptance tests/test_runtime_acceptance.py
```

Acceptance builds and installs the actual executor wheel into a fresh directory, then invokes its CLI from separate processes. The product coordinator, authority ledger, controller, inference gateway, package sealing, scorer, collection and publisher remain in the execution path. Daytona and Hugging Face transports use local durable fixtures, and inference uses loopback HTTP.

The process exits immediately at six durable boundaries: inference admission, candidate sealing, evaluation, collection, remote commit and deletion acknowledgement. A fresh process resumes the same run. Additional cases interrupt collection, publication and deletion once. Each case requires one executor, one admitted generation, one accepted measurement per condition, the unchanged published zero score, bounded transport attempts and confirmed cleanup. Permanent collection failure and corrupted receipts must hold the measurement, publish nothing and retire the exact owner within the saved retry bound. Each passing case writes `acceptance-receipt.json` with its tested wheel hash, runtime image identities, transport counts and outcome.

On WSL, use a Linux filesystem for temporary acceptance roots so the private authority directories retain their required ownership and permissions. Copy the resulting acceptance receipts to durable storage after the run.

## Licensed and runtime gates

Run the pinned NVIDIA checks locally:

```sh
SIMREADY_NATIVE_VALIDATOR_TESTS=1 python -m pytest -q -m licensed --run-licensed
```

Container and paid cases use `-m container --run-container` and `-m paid --run-paid` respectively. A paid case also requires the run's explicit allowance and credentials.

Before scaling a benchmark runtime, run its supported container and validator acceptance against the exact image digests, followed by a small cloud canary containing a successful asset, an invalid or empty asset and an injected infrastructure failure. Retain the canary receipts, itemwise publication and deletion proofs before increasing the tranche size.
