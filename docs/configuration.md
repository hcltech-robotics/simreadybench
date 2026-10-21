# Configure references, models and skills

The `benchmark` command selects the public reference set or custom assets, model, Harbor harness, optional skills and deployment. The NV25 protocol uses four reconstructions per asset and up to 64 authoring turns, giving 100 attempts across its 25 assets. `--authoring-turns` selects between one and 128 turns for a custom protocol and becomes part of the saved condition identity.

The Daytona examples use the runtime configuration created by `simready-bench setup daytona --output ../simreadybench-runtime.json`. See [deployment](deployment.md) to import the published runtime or build a source snapshot.

## References

NV25 and NV100 are distinct public reference sets. The internal benchmark uses a separate withheld reference set. The packaged [`nvidia-25` manifest](../src/simready_bench/resources/benchmarks/nvidia-25.json) contains the 25 assets in NV25.

Use `nvidia-25` to select the NV25 public reference set, a USD/USDA/USDC path to select one asset or a JSON list to select several assets:

```json
[
  "hinged_case/hinged_case.usd",
  "drawer/drawer.usd",
  "pallet/pallet.usda"
]
```

A USD path selects the layers, textures and material resources referenced by that asset. Relative paths are preserved, including tiled textures. `--assets-root` provides the containing directory when assets share resources across package directories:

```sh
simready-bench benchmark ./references.json --assets-root ./references \
  --model gpt-6.1-sol --harness pi --deploy daytona --repetitions 4 \
  --runtime-images ../simreadybench-runtime.json \
  --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-custom-results
```

For an exact file selection, use asset objects with explicit identifiers and dependencies. Paths are relative to `--assets-root`, or to the JSON file's directory when that option is omitted:

```json
{
  "assets": [
    {
      "id": "hinged-case",
      "usd": "hinged_case/asset.usdc",
      "dependencies": [
        "hinged_case/layers/lid.usda",
        "hinged_case/materials/plastic.mdl",
        "hinged_case/textures/plastic.png"
      ],
      "source_view": "hinged_case/reference.png"
    },
    {
      "id": "drawer",
      "usd": "drawer/asset.usda",
      "dependencies": [],
      "source_view": "drawer/reference.png"
    }
  ]
}
```

The explicit selection includes the USD, listed dependencies and reference observations. Automatic selection follows literal MDL resources and MaterialX filename prefixes. Use an explicit manifest for custom MDL imports, computed resource paths, MaterialX substitutions or prefixed include scopes. Every selected path must be a regular file within its asset root. Credentials, repository metadata, virtual environments and executable files are rejected. A bare path uses its filename stem as the asset identifier, with stable relative-path identifiers when several assets share that filename.

An asset manifest can also supply nominal dimensions. The NV25 manifest defines these fields and each asset's dependency paths. The agent receives the reference image, brief and dimensions for each reconstruction.

## Model connections

The model endpoint must accept image inputs and tool calls. `--model` selects its model identifier, while `--provider` selects the built-in OpenAI or Hugging Face connection. An operator policy supplies other OpenAI-compatible endpoints.

| Provider | Transport | Credential | Model example |
| --- | --- | --- | --- |
| OpenAI | Responses API | `OPENAI_API_KEY` | `gpt-6.1-sol` |
| Hugging Face | Routed Chat Completions | `HF_TOKEN` | `google/gemma-4-31B-it:novita` |
| Anthropic through LiteLLM | OpenAI-compatible Chat Completions | `ANTHROPIC_API_KEY` in LiteLLM and `LITELLM_MASTER_KEY` in SimReadyBench | `claude-sonnet-5-5` |

### OpenAI

Set `OPENAI_API_KEY` and select the model:

```sh
simready-bench benchmark nvidia-25 --provider openai --model gpt-6.1-sol --harness pi --deploy daytona --runtime-images ../simreadybench-runtime.json --repetitions 4 --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-openai
```

`--reasoning low|medium|high|xhigh|max` selects reasoning effort for a compatible model. GPT-6.1-Sol uses `medium` by default. `--api-key-env NAME` selects another credential variable.

`--max-output-tokens N` caps each model response, including reasoning tokens counted within the provider's completion allowance. Omitting it retains provider-native output limits. The response cap is separate from `--authoring-turns`, which counts model calls. Both the cap and reasoning effort are saved in the experiment and cannot change when resuming it.

For a bounded NV25 run:

```bash
simready-bench benchmark nvidia-25 --provider openai --model gpt-6.1-sol --reasoning medium --max-output-tokens 8192 --harness pi --deploy daytona --runtime-images ../simreadybench-runtime.json --repetitions 4 --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-medium-8192
```

### Hugging Face

Set `HF_TOKEN` and retain the routed provider suffix in the model identifier:

```sh
simready-bench benchmark nvidia-25 \
  --provider huggingface --model google/gemma-4-31B-it:novita \
  --harness pi --deploy daytona --repetitions 4 \
  --runtime-images ../simreadybench-runtime.json \
  --authoring-turns 64 --workers 4 --supervise --output-dir ../simreadybench-hf-results
```

Use a route from the [priced Hugging Face catalogue](../src/simready_bench/engine/hf_pricing.py). Its live provider identity, capacity and rates are verified before admission. `--model-metadata model.json` supplies image, context and reasoning settings. Other OpenAI-compatible endpoints use an [operator policy](#operator-policy) with their capacity and pricing specification.

### Anthropic through LiteLLM

LiteLLM translates OpenAI-compatible Chat Completions into [Anthropic image and tool requests](https://docs.litellm.ai/docs/providers/anthropic). Run the gateway in a separate Python environment:

```sh
python3.12 -m venv .venv-litellm
.venv-litellm/bin/python -m pip install 'litellm[proxy]==1.104.0'
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LITELLM_MASTER_KEY="$(python -c 'import secrets; print("sk-" + secrets.token_hex(32))')"
```

Save `anthropic.yaml`:

```yaml
model_list:
  - model_name: claude-sonnet-5-5
    litellm_params:
      model: anthropic/claude-sonnet-5-5
      api_key: os.environ/ANTHROPIC_API_KEY
      max_tokens: 128000
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Start the gateway and keep it running while evaluating:

```sh
.venv-litellm/bin/litellm --config anthropic.yaml --port 4000
```

Use the same `LITELLM_MASTER_KEY` in the shell running SimReadyBench. Save the connection's model specification as `anthropic-model.json`:

```json
{
  "schema": "simready-operator-model-v1",
  "model": "claude-sonnet-5-5",
  "base_url": "https://your-gateway.example.org/v1",
  "api_style": "openai-completions",
  "context_window": 1000000,
  "maximum_output_tokens": 128000,
  "pricing": {
    "input_per_million_usd": 4,
    "output_per_million_usd": 10
  },
  "pricing_source": "https://platform.claude.com/docs/en/models/sonnet-5-5/overview",
  "checked": "2026-10-20"
}
```

These capacities follow the [Claude Sonnet 5.5 specification](https://platform.claude.com/docs/en/models/sonnet-5-5/overview). The input reservation uses the one-hour cache-write rate from [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing). The gateway's `max_tokens` uses the model's full output capacity.

Serve the gateway at an HTTPS address reachable from the Daytona executor and set that address in the model specification. Create an [operator policy](#operator-policy) for one reconstruction, then run it on Daytona:

```sh
python configure_provider.py --model-spec ./anthropic-model.json --api-key-env LITELLM_MASTER_KEY --assets ./references --workspace ../simreadybench-claude-operator --conditions 1 --max-spend-usd 500 --output ./anthropic.json
simready-bench benchmark ./references/hinged_case/asset.usd --policy ./anthropic.json --model claude-sonnet-5-5 --harness pi --deploy daytona --runtime-images ../simreadybench-runtime.json --repetitions 1 --authoring-turns 64 --workers 1 --supervise --max-spend-usd 500 --output-dir ../simreadybench-claude
```

The model identifier, endpoint and credential variable must match the gateway configuration.

### Operator policy

The policy records a custom endpoint's model capacity, pricing and run allowance. The Daytona helper creates the host policy from the source installation without requiring a local validator. Save this script as `configure_provider.py`:

```python
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4

from simready_bench.policy import init_daytona_policy
from simready_bench.engine.native_output import authoring_seconds

parser = argparse.ArgumentParser()
parser.add_argument("--model-spec", type=Path, required=True)
parser.add_argument("--api-key-env", required=True)
parser.add_argument("--assets", type=Path, required=True)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--conditions", type=int, required=True)
parser.add_argument("--authoring-turns", type=int, default=64)
parser.add_argument("--max-spend-usd", type=float, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

spec = json.loads(args.model_spec.read_text())
endpoint = urlsplit(spec["base_url"])
provider = {
    "kind": "openai",
    "base_url": spec["base_url"],
    "api_key_env": args.api_key_env,
    "model_spec": spec,
    "project_budget": {
        "project": "simready-bench",
        "authorization_id": "experiment-" + uuid4().hex,
        "approved": True,
        "upstream_origin": endpoint.scheme + "://" + endpoint.netloc,
        "model": spec["model"],
        "expires_at": time.time() + max(86400, args.conditions * (authoring_seconds(args.authoring_turns) + 2400)),
        "maximum_requests": args.authoring_turns * args.conditions,
        "maximum_spend_usd": args.max_spend_usd,
        "pricing_status": "known",
        "external_spend_cap_verified": False,
    },
}
init_daytona_policy(
    output=args.output.absolute(),
    workspace=args.workspace.absolute(),
    assets=args.assets.absolute(),
    providers={"selected": provider},
    maximum_spend_usd=args.max_spend_usd,
    conditions=args.conditions,
    authoring_turns=args.authoring_turns,
)
```

Use a fresh workspace and policy path, an existing asset directory and the total number of asset/reconstruction conditions in the authorised experiment. The helper retains the provider grant and creates private skill, input-view, manifest and authority directories. The policy grants 64 model requests per condition by default. Custom endpoint reservations use the declared input and output capacities, so set the budget for those bounds. Reuse the same policy when continuing the experiment. Local execution uses `simready_bench.policy.init_policy` with the installed validator and [local runtime images](deployment.md).

## Harnesses and skills

Pi and OpenCode are included in the actor runtime. Select them with `--harness pi` or `--harness opencode`. Harbor manages the agent's authoring session and its access to the model and tools.

Supply skill files or directories with `--skills`:

```sh
simready-bench benchmark nvidia-25 \
  --model gpt-6.1-sol --harness pi --skills ./skills/usd-authoring \
  --deploy daytona --repetitions 4 --authoring-turns 64 --workers 4 --supervise \
  --runtime-images ../simreadybench-runtime.json \
  --output-dir ../simreadybench-skills-results
```

The agent receives only the selected skills. Each run records its model, harness, skill contents and authoring settings so that experimental conditions can be compared.

## Runtime and budget

`--deploy local` uses the local Linux/Docker executor and `--deploy daytona` creates a Daytona CPU executor. Built-in providers receive an accounting allowance derived from the turn count, native input capacity and selected output cap, using native output capacity when no cap is set. This is recorded in `allowance.json`. `--max-spend-usd` supplies a separate monetary ceiling, and `--policy operator.json` reuses a configured runtime, provider connection and persistent allowance.

`--runtime-images PATH` selects a JSON configuration containing the actor, verifier and CPU image identities and, for an imported runtime, its Daytona snapshot name. See [deployment](deployment.md) for published runtime imports and source builds.

The NV25 protocol runs four reconstructions of each asset, with up to 64 model calls and five validator calls for each reconstruction. Pi and the inference gateway receive the same turn limit. Before each model call, Pi supplies the current round and the number of rounds remaining after it. The final round includes a reminder to save and submit the candidate. Individual model requests retain their 1800-second deadline within the complete authoring interval.
