# Local and Daytona deployment

Both deployment targets execute Harbor tasks, SimReady Robotics-Prop validation, USD fidelity scoring and orientation-aligned captures on CPU. Python 3.12 is required.

## Daytona

Follow the [quickstart installation](quickstart.md#install) with the Daytona extra.

Set `DAYTONA_API_KEY`, then import the [published runtime](https://github.com/hcltech-robotics/simreadybench/pkgs/container/simreadybench-runtime) into your organisation:

```sh
simready-bench setup daytona --output ../simreadybench-runtime.json
```

Setup imports the immutable public container image recorded in the installed release and saves its actor, verifier and CPU image identities together with the organisation snapshot name. Its default name includes the runtime version. A matching active snapshot is reused when setup is repeated. Use `--snapshot NAME` to choose a different name if an existing snapshot belongs to another runtime.

The import uses two CPUs, 8 GiB of memory and 10 GiB of disk by default. Each actor has a 4 GiB memory limit. Eight concurrent executors use 64 GiB of sandbox memory in total. `--cpu`, `--memory` and `--disk` select resources when creating a new snapshot, while `--timeout` selects its import timeout in seconds. Setup rejects an existing snapshot whose resources are below the requested values, so use a new `--snapshot NAME` when increasing memory. Use `--release-manifest PATH` to select another published release manifest with its pinned public image and matching internal image identities.

An operator policy can set `runtime.container_resources` with explicit `cpus` and `memory_mb` values for the actor. Daytona passes these bounds to the executor without changing the controller, validation or rendering limits.

Hugging Face routes share a private inference governor across workers and coordinators using the same authority directory. Each route admits at most four simultaneous inference requests, independently of the worker count. A rate-limit response applies a shared cooldown using `Retry-After` or `RateLimit` headers, with exponential backoff and jitter when neither provides a delay. Routes are identified by provider endpoint, model and credential identity, so other providers retain independent capacity. The governor retains leases and cooldowns across coordinator restarts and is deleted after its final owner releases it and all dependent executors have been deleted.

The governor uses an expiring signed preview restricted to its service port. Before workers launch, the pinned governor container checks the authenticated HTTPS endpoint using the worker network configuration. A failed check reports `shared_route_governor_worker_network_unreachable` and deletes an unshared governor. The Daytona organisation's outbound network policy must permit access to its preview domain.

Failed inference does not consume the authoring-round allowance. Transport attempts remain recorded under a separate finite limit and monetary accounting retains any uncertain spend. A successful response truncated at the per-request token limit consumes one authoring round, including when continuation is enabled.

The snapshot contains Docker, actor and verifier images, the validator environment and Storm/Mesa with its Python interpreter at `/opt/storm/bin/python`. Set the model credential, such as `OPENAI_API_KEY` or `HF_TOKEN`, then pass the saved runtime configuration to the benchmark:

```sh
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --deploy daytona --repetitions 4 \
  --runtime-images ../simreadybench-runtime.json \
  --authoring-turns 64 --workers 8 --supervise --output-dir ../simreadybench-results
```

The benchmark installs its package in each executor, executes the reconstruction tasks and collects their reports before deleting the executor.

### Live worker logs

From the repository checkout on the coordinator, attach a live log view to current and subsequent workers:

```sh
python scripts/daytona_live_logs.py /path/to/run \
  --state /path/to/live-logs.json --watch
```

The command uses `DAYTONA_API_KEY`. It follows the run's worker ownership records and attaches a separate read-only Daytona session to each executor. Multiple run directories can be supplied for a model sequence.

Open a running sandbox in Daytona, select Terminal, then Open in new tab and run:

```sh
simready-logs
```

The view shows Pi tool calls, tool results and visible assistant messages, with model request counts and a waiting message every 15 seconds. Output is also available through the `simready-live-logs` Daytona process session and `/work/simready-live.log`. Closing the terminal leaves the benchmark running. The observer exits after all supplied runs finish and their workers are closed.

Controller progress and evaluation events are available with `tail -n 30 -F /work/controller.stdout.log /work/controller.stderr.log`. The standalone reader at `src/simready_bench/worker_logs.py` also supports `--once` for a snapshot inside the controller.

## Local

Local execution uses Linux amd64 with Docker. On Windows, use an Ubuntu WSL shell. Install the validator and configure the actor, verifier and CPU rendering images:

```sh
python -m pip install '.[validator]'
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --deploy local --repetitions 4 \
  --runtime-images /tmp/srb-snapshot/local-runtime-images.json \
  --authoring-turns 64 --output-dir ../simreadybench-local-results
```

`--runtime-images` selects the actor, verifier and CPU image identities in the runtime configuration. `--policy operator.json` can reuse a configured runtime and provider connection.

## Build CPU images

From the installed repository checkout, build the package and runtime images:

```sh
python -m pip install build
python -m build --wheel
python containers/build.py actor \
  --tag simreadybench/actor:1.0.0 --output /tmp/srb-actor
python containers/build.py verifier \
  --tag simreadybench/verifier:1.0.0 --output /tmp/srb-verifier
python containers/build.py cpu \
  --wheel dist/simready_bench-1.0.0-py3-none-any.whl \
  --tag simreadybench/cpu:1.0.0 --output /tmp/srb-cpu
python containers/daytona/prepare_storm.py --output /tmp/srb-storm
docker build --build-arg CPU_BASE=simreadybench/cpu:1.0.0 \
  --tag simreadybench/cpu:storm /tmp/srb-storm
```

Package the images and create the runtime configuration:

```sh
ACTOR_IMAGE=$(docker image inspect --format '{{.Id}}' simreadybench/actor:1.0.0)
VERIFIER_IMAGE=$(docker image inspect --format '{{.Id}}' simreadybench/verifier:1.0.0)
CPU_IMAGE=$(docker image inspect --format '{{.Id}}' simreadybench/cpu:storm)
python containers/daytona/prepare_snapshot.py \
  --actor-image "$ACTOR_IMAGE" --verifier-image "$VERIFIER_IMAGE" \
  --cpu-image "$CPU_IMAGE" --output /tmp/srb-snapshot
```

The generated `local-runtime-images.json` selects the build host's images for local execution. `runtime-images.json` records the corresponding immutable identities used by the snapshot's containerd image store. The snapshot context contains the packaged images for Daytona.

## Create a Daytona snapshot

With `DAYTONA_API_KEY` set, create an organisation snapshot from the prepared context:

```sh
python containers/daytona/create_snapshot.py \
  --context /tmp/srb-snapshot --name simready-bench-cpu \
  --receipt /tmp/srb-snapshot.json
```

The `simready-bench-cpu` snapshot contains the configured images. For a source-built snapshot, pass both `--cpu-snapshot simready-bench-cpu` and `--runtime-images /tmp/srb-snapshot/runtime-images.json` to the benchmark. The image mapping must match the images packaged in that snapshot.

## Publish a runtime release

Push a `runtime-vVERSION` tag to build and publish the complete runtime to `ghcr.io/hcltech-robotics/simreadybench-runtime:VERSION`. The `Publish runtime` workflow builds the actor, verifier, CPU and Storm images, embeds their exact image identities in the outer snapshot image and authenticates to GHCR with the workflow's package token. Before publication, the pinned consumer engine must load and execute all three identities without network access.

The workflow retains `runtime-release.json` and `build-receipt.json` as its `runtime-release` artifact. The manifest binds the public image digest to the internal actor, verifier and CPU identities. Set the GHCR package visibility to public after its first publication, then copy the verified manifest to `src/simready_bench/resources/runtimes/daytona.json` for the next package build. Published runtime tags are retained rather than overwritten.

## Recovery

The [execution contract and local acceptance gates](execution.md) define durable ownership, phase recovery and the evidence required before scaling a runtime.

Transient source-download failures receive up to five attempts per file with exponential backoff and jitter. Each attempt starts with a clean partial file and the completed download must pass the original digest checks. Authentication, certificate and integrity failures retain their cause.

The coordinator retries individual infrastructure failures as soon as a slot becomes available and their backoff has elapsed. Two further attempts per condition are allowed by default. Use `--max-retries` to change this limit, `--retry-delay-seconds` for the initial delay and `--maximum-retry-delay-seconds` for its ceiling. Completed reconstructions retain their scores and artefacts, including measured zero scores. Each retry retains the model, reference, seed, evaluator and authoring limit and receives a separate immutable execution record. An exhausted condition is reported while the remaining conditions continue.

New executor batches contain one reconstruction, so a failed repetition can be collected and retried without waiting for the other repetitions of its asset. Five infrastructure failures without completed-result progress open an execution circuit. The coordinator stops further dispatch, stops the workers it owns, collects their outputs and deletes their sandboxes. The circuit also opens after 65 minutes without a completed result when infrastructure failures are present. Its saved state is `execution-health.json` and survives coordinator restarts. Model traffic does not reset the completion timer.

An explicitly approved transport repair can replace the executor package through `simready_bench.executor_repair.approve_executor_repair`. The approval records both wheel hashes and checks that every evaluator, model configuration and scoring file remains byte-identical. Existing workers must be collected and deleted first. Completed scores and admitted-attempt counters remain in the original run.

New runs reserve an aggregate request allowance for the initial attempt and its permitted retries. A supplied operator grant or explicit spending limit is retained. Workers stop after ten minutes without coordinator contact. A live worker with no provider admissions or reconstruction progress for 65 minutes is stopped, collected and deleted before replacement. Set `runtime.executor_stall_seconds` in the operator policy to change this interval, between 1800 and 86400 seconds. Confirmed missing workers consume their full reserved allowance when their usage cannot be recovered.

Add `--supervise` on Linux or WSL to restart a coordinator automatically after a transient error, process interruption or ten minutes without main-loop progress. `--coordinator-restarts` sets its restart limit, which defaults to five. Clean completion and terminal incomplete results stop supervision.

To continue after a host interruption, repeat the benchmark command with `--resume` and the same output directory. The coordinator reconnects existing executors before considering new work and reuses the original executor wheel. New execution batches are stored under `executions/`. Publication failures remain pending while workers continue, and subsequent uploads retain every completed reconstruction.
