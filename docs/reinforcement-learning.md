# Reinforcement learning with your assets

SimReadyBench supplies Harbor reconstruction episodes for an externally trained policy. The environment presents reference observations, runs the authoring tools and evaluates the submitted OpenUSD asset. Your training system serves the model, collects the resulting trajectories and updates its parameters between episodes.

## Prepare the references and policy connection

Keep each reference asset with its textures, materials and composed USD layers. A single USD entry point selects one training task, while a [JSON asset list](configuration.md#references) selects a reference set. For example:

```text
references/
  hinged_case/
    asset.usd
    textures/
  drawer/
    asset.usd
```

Serve the policy through an OpenAI-compatible endpoint that accepts reference images and tool calls. For an unmetered local endpoint serving the model identifier `reconstruction-policy`, save this specification as `policy-model.json`:

```json
{
  "schema": "simready-operator-model-v1",
  "model": "reconstruction-policy",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_style": "openai-completions",
  "context_window": 32768,
  "maximum_output_tokens": 8192,
  "pricing": {
    "input_per_million_usd": 0,
    "output_per_million_usd": 0
  },
  "pricing_source": "http://127.0.0.1:8000/model-pricing",
  "checked": "2026-10-20"
}
```

Use the capacity and token charges of your served model. The example declares no token charges for its local endpoint. Set `POLICY_API_KEY` to the endpoint's credential. Use the [operator configuration script](configuration.md#operator-policy), replacing its final `init_daytona_policy` call with this local policy setup:

```python
from simready_bench.policy import init_policy

policy = init_policy(
    output=args.output.absolute(),
    workspace=args.workspace.absolute(),
    assets=args.assets.absolute(),
    providers={"selected": provider},
    maximum_spend_usd=args.max_spend_usd,
    native_output=True,
    execution_enabled=True,
)
policy["limits"].update(
    maximum_matrix_runs=args.conditions,
    maximum_condition_spend_usd=args.max_spend_usd / args.conditions,
    inference_requests=args.authoring_turns,
    agent_seconds=authoring_seconds(args.authoring_turns),
)
args.output.write_text(json.dumps(policy, indent=2) + "\n")
```

Create an allowance for 100 episodes:

```sh
python configure_provider.py --model-spec ./policy-model.json --api-key-env POLICY_API_KEY --assets ./references --workspace /dev/shm/srb-training --conditions 100 --max-spend-usd 1 --output ./training.json
```

The same operator policy supplies the request and spend allowance across the experiment. The local runtime uses the actor, verifier and CPU images selected by `runtime-images.json`, as described in [deployment](deployment.md#local).

## Collect one episode

Use one reconstruction and a separate output directory for each rollout:

```sh
simready-bench benchmark ./references/hinged_case/asset.usd --policy ./training.json --model reconstruction-policy --harness pi --deploy local --runtime-images ./runtime-images.json --repetitions 1 --authoring-turns 64 --max-spend-usd 0.01 --output-dir ../rollouts/episode-000
```

Harbor runs the selected harness against your policy endpoint. The benchmark manages the actor environment, private verifier service, final submission and evaluation. Each episode allows up to 64 model requests and five validator calls. This example partitions the experiment's $1 allowance into $0.01 per episode.

The actor receives the reference image, brief, dimensions and selected skills. It authors `/output/asset.usd` with its dependencies and can use these tools:

| Tool | Observation or action |
| --- | --- |
| `get_task_thumbnail` | Read the reference image |
| `validate_candidate` | Evaluate the current candidate and return formal fidelity feedback and the NVIDIA report descriptor |
| `get_validation_report` | Read a selected portion of the NVIDIA findings |
| `submit_asset` | End authoring and submit the generated package |

## Read the reward and evaluation vector

The terminal Harbor reward is formal fidelity:

```json
{"simready_reward": 100.0}
```

It is written to the trial's `/logs/verifier/reward.json`. An incomplete evaluation retains a null score and its failure details in the report. The exported `results.json` provides the three composites for each reconstruction:

```python
import json
from pathlib import Path

readout = Path("../rollouts/episode-000/results.json")
episode = json.loads(readout.read_text())["results"][0]
scores = episode["composite_scores"]
evaluation_vector = {
    name: scores[name]["score_percent"]
    for name in ("formal", "visual", "kinematic")
}
```

Each composite also retains its status, earned credit, eligible denominator and measurement coverage. The readout links the submitted asset, captures, NVIDIA findings and Harbor conversation trace. Keep unknown values null and exclude criteria marked not applicable when constructing a training objective. The [evaluation guide](evaluation.md) defines the measurements and score bounds.

## Update the policy between rollouts

Run the benchmark once per rollout, consume its readout in your trainer and update the model served at the policy endpoint. After collecting `episode-000` above, an orchestration script can call your own `train_step.py`, then collect the remaining episodes:

```sh
python train_step.py ../rollouts/episode-000/results.json
for episode in $(seq -f '%03g' 1 99); do
  simready-bench benchmark ./references/hinged_case/asset.usd --policy ./training.json --model reconstruction-policy --harness pi --deploy local --runtime-images ./runtime-images.json --repetitions 1 --authoring-turns 64 --max-spend-usd 0.01 --output-dir "../rollouts/episode-$episode"
  python train_step.py "../rollouts/episode-$episode/results.json"
done
```

`train_step.py` is the training system's update step. It selects the objective, reads the trajectory and evaluation vector and publishes the next policy checkpoint to the endpoint. Record that checkpoint with the rollout so its generated asset, scores and conversation remain associated with the policy that produced them.

To collect several assets in one rollout batch, replace the USD path with the asset list. Set the operator policy's condition allowance for the total number of asset/reconstruction conditions, and give each batch a distinct output directory. Add `--skills PATH` to supply a selected authoring skill set or use `--deploy daytona` with a policy endpoint reachable over HTTPS from the executor.
