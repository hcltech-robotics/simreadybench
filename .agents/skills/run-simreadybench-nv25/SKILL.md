---
name: run-simreadybench-nv25
description: Run the standard SimReadyBench NV25 public-reference protocol on Daytona for a selected multimodal model, preserving its packaged 25 assets, four repetitions and 64 authoring turns.
---

# Run the standard NV25 benchmark

Use the installed public CLI from Linux or Windows WSL. Work from the selected SimReadyBench checkout and resolve documentation paths against its repository root, including when this skill is installed in a user's skills directory. Read `docs/agent-skills.md` for source installation. The packaged `NV25` and `nvidia-25` selections identify the same official NV25 public reference set. NV25 is separate from NV100 and the withheld internal reference set.

## Preserve the benchmark identity

Use the packaged selection with all 25 assets, four repetitions per asset and 64 authoring turns. This gives 100 conditions for each model/harness/authoring-skill combination. Retain the packaged source revision, reference closure, scoring profile, seed and provider-native output limits.

Choose the model and provider requested by the user. A custom model endpoint is still an NV25 model condition when the reference set and protocol remain unchanged. Use Pi with no actor authoring skills by default. Record any requested harness or actor-skill condition explicitly and keep its output separate.

Use a fresh output directory for a new model condition. An altered asset list, repetition count or authoring turn limit is a custom protocol and should use the custom experiment skill.

## Run on Daytona

Install the Daytona dependencies and set `DAYTONA_API_KEY` and the selected model credential in the environment. Import the published runtime with `simready-bench setup daytona --output ../simreadybench-runtime.json`, then pass that configuration to the benchmark. Reuse a matching runtime configuration when one already exists. `docs/deployment.md` defines source-built snapshots and their image mappings.

```sh
simready-bench benchmark NV25 \
  --provider openai --model gpt-6.1-sol --reasoning medium \
  --harness pi --deploy daytona --supervise --repetitions 4 --authoring-turns 64 \
  --runtime-images ../simreadybench-runtime.json \
  --workers 4 --output-dir ../runs/nv25-sol-pi-none
```

For an OpenAI-compatible custom model already configured in an operator policy:

```sh
simready-bench benchmark NV25 \
  --policy ./custom-model-policy.json --model custom-vlm \
  --harness pi --deploy daytona --supervise --repetitions 4 --authoring-turns 64 \
  --runtime-images ../simreadybench-runtime.json \
  --workers 4 --output-dir ../runs/nv25-custom-vlm-pi-none
```

For Hugging Face, use `--provider huggingface`, the complete `MODEL:PROVIDER` identifier and any required `--model-metadata PATH`. Verify image/tool support, route identity, context capacity and pricing before admission. `docs/configuration.md` defines these connections.

Concurrency is operational: `--workers 1..16` changes the number of simultaneous Daytona workers without changing the 100-condition matrix. Use `--cpu-snapshot` and `--runtime-images` for a compatible runtime. Pass `--max-spend-usd` when the user requests a budget. Otherwise, the fresh run derives its accounting allowance from native request capacity and all 64 turns, while an existing policy keeps its limits. The allowance is a reservation bound rather than a predicted charge. Add `--hf-repo OWNER/DATASET --hf-run RUN_ID` when publication is requested, using an identity that records the model, harness, authoring skills and protocol.

This skill is an operator instruction. Pass only separately requested actor authoring material through `--skills`; the operator skill is not an authoring input.

## Recover and verify

Use `--supervise` for host coordinator recovery, with up to five restarts by default. `--coordinator-restarts` selects that limit. The built-in coordinator reconnects workers and retains completed conditions. For a host interruption, repeat the same command with `--resume` and the original output directory. Preserve the saved model, references, seed, turn limit and original allowance.

Check the saved experiment and operator settings for the 64-turn protocol and inspect `results.json`, `scores.csv` and `report.html`. A complete standard model condition has all 100 completed rows and retained generated USD packages, aligned captures and traces. Record infrastructure outcomes separately and preserve known scores when recovering work.

Report formal, visual and kinematic fidelity separately, retaining earned credit, eligible denominators, measurement coverage and unavailable results. Give the run location and publication link when one was produced.
