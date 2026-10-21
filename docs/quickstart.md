# Evaluate a model on the NV25 public reference set

The NV25 public reference set contains 25 NVIDIA SimReady assets. The standard NV25 protocol runs four reconstructions per asset with up to 64 authoring turns each, giving 100 attempts. The benchmark executes each reconstruction in a Harbor environment and collects its scores, generated USD and orientation-aligned captures.

## Install

Use Python 3.12 and Git in a Linux or WSL shell:

```sh
git clone https://github.com/hcltech-robotics/simreadybench.git
cd simreadybench
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[daytona,publish]'
```

Set the model and Daytona credentials:

```sh
export OPENAI_API_KEY="your-openai-api-key"
export DAYTONA_API_KEY="your-daytona-api-key"
```

Import the published runtime into your Daytona organisation:

```sh
simready-bench setup daytona --output ../simreadybench-runtime.json
```

Setup creates or reuses the versioned snapshot and saves its matching image configuration. See [deployment](deployment.md) for runtime imports and source builds.

## Run the benchmark

```sh
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --deploy daytona --repetitions 4 \
  --runtime-images ../simreadybench-runtime.json \
  --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-results
```

Pi receives the reference image, brief and dimensions for each reconstruction, then authors the USD with validator feedback. The default accounting allowance covers the permitted requests at full native provider capacity. `--max-spend-usd AMOUNT` adds a separate monetary ceiling. Completed reconstructions are exported during the run, and the Daytona executor is removed after collection.

`--workers` selects a positive number of Daytona workers within your organisation's CPU, memory and disk allocation. Each worker uses a separate sandbox and Harbor environment, and completed reconstructions enter the same report.

Open `../simreadybench-results/report.html` to compare scores and captures. `../simreadybench-results/scores.csv` contains the score table and `../simreadybench-results/results.json` contains the structured readout. The report links each attempt's generated asset, validator report and Harbor trace. See [reporting](reporting.md) for the output layout.

`--supervise` restarts a coordinator after a transient failure or process interruption. To continue after closing the shell or rebooting, repeat the same command with `--resume`. It keeps completed reconstructions, reconnects existing workers and assigns only unadmitted work within the saved allowance. The worker count can change when resuming.

## Publish reconstruction reports

Set `HF_TOKEN` or sign in with `hf auth login`, then add your dataset and run name to the benchmark command:

```sh
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --deploy daytona --repetitions 4 \
  --runtime-images ../simreadybench-runtime.json \
  --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-results \
  --hf-repo YOUR_ACCOUNT/simreadybench-results --hf-run sol-6.1-medium
```

Each completed reconstruction is published with its scores, USD package, orientation-aligned captures and Harbor trace. The Hugging Face dataset viewer reads the published score table. Reconstruction logs, traces and score measurements are also available through Daytona's OpenTelemetry collector.

## Run locally

Install the validator extra on a Linux amd64 Docker host with the [CPU runtime images](deployment.md#local):

```sh
python -m pip install '.[validator]'
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --deploy local --repetitions 4 \
  --runtime-images /tmp/srb-snapshot/local-runtime-images.json \
  --authoring-turns 64 --output-dir ../simreadybench-local-results
```

## Change the experiment

Replace `--model` to compare another model, `--harness pi` with `--harness opencode` to use OpenCode or add `--skills PATH` to supply authoring skills. Keep the same references and repetitions when comparing conditions, and use a separate output directory for each run.

A USD path or JSON asset list can replace `nvidia-25`. The [configuration guide](configuration.md) defines the public reference set selections, custom asset lists, model connections and skills, while [evaluation](evaluation.md) defines the scores.
