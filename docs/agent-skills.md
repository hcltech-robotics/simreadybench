# Run SimReadyBench with an agent

The repository supplies two operator skills:

| Skill | Use |
| --- | --- |
| [run-simreadybench-custom](../.agents/skills/run-simreadybench-custom/SKILL.md) | Custom USD packages, asset lists, endpoints or experimental protocols |
| [run-simreadybench-nv25](../.agents/skills/run-simreadybench-nv25/SKILL.md) | The standard NV25 public-reference protocol: 25 assets, four repetitions and 64 authoring turns |

A selected model can be custom while its reference set remains official NV25. A changed asset set, repetition count or turn limit defines a custom protocol. Model, harness and actor authoring skills remain explicit experimental factors.

## Install from the repository

Follow the [quickstart installation](quickstart.md#install) from Linux or an Ubuntu WSL shell on Windows. The repository keeps its discoverable skill folders under `.agents/skills/`. To install them in a user's Codex skills directory from the repository root:

```sh
SKILL_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILL_HOME"
cp -R .agents/skills/run-simreadybench-custom "$SKILL_HOME/"
cp -R .agents/skills/run-simreadybench-nv25 "$SKILL_HOME/"
```

Keep an installed copy in sync when updating the repository. The skills use the installed public `simready-bench` command. The [configuration guide](configuration.md) defines provider and asset schemas, and [deployment](deployment.md) describes the compatible Daytona snapshot and local Docker images.

## Invoke a skill

In Codex, examples include:

```text
Use $run-simreadybench-nv25 to evaluate my selected model on NV25 with four Daytona workers and 64 authoring turns. Save the results in a new output directory.
```

```text
Use $run-simreadybench-custom to evaluate my OpenAI-compatible model on references.json with two repetitions, Pi and 64 authoring turns on Daytona.
```

The agent configures the run through the public CLI, observes the saved output and resumes the same identity after an interruption. The examples use `--supervise` to restart a failed host coordinator, with up to five restarts by default. `--coordinator-restarts` selects that limit. Publishing to Hugging Face is an explicit run option using `--hf-repo` and `--hf-run`.

These operator skills are separate from authoring material supplied with `--skills`. Operator skills run the benchmark. Authoring skills are experimental inputs given to the model reconstructing the asset.
