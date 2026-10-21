# Local and Daytona deployment

Both deployment targets execute Harbor tasks, SimReady Robotics-Prop validation, USD fidelity scoring and orientation-aligned captures on CPU. Python 3.12 is required.

## Daytona

Follow the [quickstart installation](quickstart.md#install) with the Daytona extra.

Set `DAYTONA_API_KEY`, then import the [published runtime](https://github.com/hcltech-robotics/simreadybench/pkgs/container/simreadybench-runtime) into your organisation:

```sh
simready-bench setup daytona --output ../simreadybench-runtime.json
```

Setup imports the immutable public container image recorded in the installed release and saves its actor, verifier and CPU image identities together with the organisation snapshot name. Its default name includes the runtime version. A matching active snapshot is reused when setup is repeated. Use `--snapshot NAME` to choose a different name if an existing snapshot belongs to another runtime.

The import uses two CPUs, 12 GiB of memory and 10 GiB of disk by default. Each actor has a 4 GiB memory limit. Four concurrent executors use 48 GiB of sandbox memory in total. `--cpu`, `--memory` and `--disk` select resources when creating a new snapshot, while `--timeout` selects its import timeout in seconds. Setup rejects an existing snapshot whose resources are below the requested values, so use a new `--snapshot NAME` when increasing memory. Use `--release-manifest PATH` to select another published release manifest with its pinned public image and matching internal image identities.

An operator policy can set `runtime.container_resources` with explicit `cpus` and `memory_mb` values for the actor. Daytona passes these bounds to the executor without changing the controller, validation or rendering limits.

The snapshot contains Docker, actor and verifier images, the validator environment and Storm/Mesa with its Python interpreter at `/opt/storm/bin/python`. Set the model credential, such as `OPENAI_API_KEY` or `HF_TOKEN`, then pass the saved runtime configuration to the benchmark:

```sh
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --deploy daytona --repetitions 4 \
  --runtime-images ../simreadybench-runtime.json \
  --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-results
```

The benchmark installs its package in each executor, executes the reconstruction tasks and collects their reports before deleting the executor.

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

Transient source-download failures receive up to five attempts per file with exponential backoff and jitter. Each attempt starts with a clean partial file and the completed download must pass the original digest checks. Authentication, certificate and integrity failures retain their cause.

The coordinator replaces failed workers for unadmitted work as soon as a slot becomes available. Up to two replacement dispatches are allowed by default. Use `--max-retries` to change this limit, `--retry-delay-seconds` for the initial delay and `--maximum-retry-delay-seconds` for its ceiling. Completed reconstructions retain their scores and artefacts, while admitted authoring attempts keep their original model, reference, seed, evaluator and request/spend allowance.

Add `--supervise` on Linux or WSL to restart a coordinator automatically after a transient error or process interruption. `--coordinator-restarts` sets its restart limit, which defaults to five. Clean completion and terminal incomplete results stop supervision.

To continue after a host interruption, repeat the benchmark command with `--resume` and the same output directory. The coordinator reconnects existing executors before considering new work and reuses the original executor wheel. New execution batches are stored under `executions/`. Publication failures remain pending while workers continue, and subsequent uploads retain every completed reconstruction.
