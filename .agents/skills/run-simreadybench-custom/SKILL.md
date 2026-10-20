---
name: run-simreadybench-custom
description: Configure and run SimReadyBench experiments with custom USD packages, asset lists, model endpoints, harnesses or authoring skills. Use for experiments whose assets or protocol differ from the standard NV25 run.
---

# Run a custom SimReadyBench experiment

Run the public `simready-bench benchmark` command from a Linux or Windows WSL environment. Work from the selected SimReadyBench checkout and resolve documentation paths against its repository root, including when this skill is installed in a user's skills directory. Read `docs/agent-skills.md` for source installation and `docs/configuration.md` for reference and provider schemas.

## Define the experiment

Use the user's chosen model, reference set, harness, repetitions, authoring turns and deployment. Select Daytona unless the user chooses local execution. Fresh authoring allows 64 turns by default. Record changed factors in the run name and use a fresh output directory for each experiment.

Select a reference USD path, JSON list of USD paths or full experiment manifest. Bare USD paths select their authored dependency closure, including composed layers and material resources. Explicit manifest entries select only their declared files and observations. Use `--assets-root` to locate relative paths and shared dependencies in a JSON list.

The model must accept images and tool calls. Built-in connections use `--provider openai` or `--provider huggingface`. Keep the HF provider suffix in the model identifier and supply `--model-metadata` when required. For another OpenAI-compatible endpoint, use `--policy` with its model specification, context capacity, pricing and credential variable as described in configuration. A Daytona endpoint must be reachable from the executor.

Set `DAYTONA_API_KEY` and the selected provider's credential in the environment. Keep credential values out of commands, manifests and reports.

Import the published runtime with `simready-bench setup daytona --output ../simreadybench-runtime.json`, then pass that file to the benchmark. Reuse a matching runtime configuration when one already exists. `docs/deployment.md` defines source-built snapshots and their image mappings.

## Run and recover

For a custom asset list:

```sh
simready-bench benchmark ./references.json \
  --assets-root ./references --provider openai --model gpt-6.1-sol \
  --harness pi --deploy daytona --supervise --repetitions 2 --authoring-turns 64 \
  --runtime-images ../simreadybench-runtime.json \
  --workers 4 --output-dir ../runs/custom-sol
```

For a custom endpoint already configured in an operator policy:

```sh
simready-bench benchmark ./references/hinged-case/asset.usd \
  --policy ./custom-model-policy.json --model custom-vlm \
  --harness pi --deploy daytona --supervise --repetitions 1 --authoring-turns 64 \
  --runtime-images ../simreadybench-runtime.json \
  --workers 1 --output-dir ../runs/custom-vlm
```

Use `--authoring-turns` for a requested custom turn limit. Select `--harness opencode` when requested. Add actor authoring material with `--skills PATH [PATH ...]` and record its contents as an experimental factor. This operator skill directs the benchmark and is separate from the skills supplied to its actor.

Use `--workers 1..16` for Daytona concurrency. `--cpu-snapshot` and `--runtime-images` select the compatible deployment runtime. For local execution, select `--deploy local` and follow `docs/deployment.md` for its Linux/Docker images and validator installation.

Respect an explicitly requested run budget through `--max-spend-usd`. Otherwise, a fresh run derives its accounting allowance from native request capacity and the selected turn count. This allowance is a reservation bound rather than a predicted charge. Existing operator policies retain their original limits. Add `--hf-repo OWNER/DATASET --hf-run RUN_ID` when the user requests publication.

Use `--supervise` to restart a failed host coordinator, with up to five restarts by default. `--coordinator-restarts` selects that limit. The built-in recovery coordinator retains completed conditions and reconnects saved workers. After a host interruption, repeat the same command and output directory with `--resume`. Preserve the saved scientific inputs, model identity, turn limit and allowance when resuming.

## Read the result

Check `results.json`, `scores.csv` and `report.html`. Report the requested and completed condition counts, model/provider, reference identity, harness, authoring skills, repetitions and turn limit. Retain candidate packages, aligned captures and traces with their hashes. Keep infrastructure failures distinct from measured low scores and report formal, visual and kinematic fidelity with their eligible denominators and coverage.

Changing the model does not itself change an official asset set. Use the NV25 skill when the user requests the standard NV25 reference/protocol identity, including a custom model on that set.
