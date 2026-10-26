# Reports and results

The selected output directory contains `report.html` for comparing scores and orientation-aligned captures, `scores.csv` for tabular analysis and `results.json` for the complete structured readout. Each per-asset report links its generated USDs, validator reports, captures and Harbor traces.

During a run, `reconstructions/report.html` and `reconstructions/scores.csv` update after each completed reconstruction. Each `reconstructions/condition-*/` directory contains that attempt's report and artefacts.

Authoring and evaluation have separate completion counts. `authored_conditions` counts retained submissions, while `evaluated_conditions` counts submissions whose required formal, fidelity and capture stages have finished. `benchmark_complete` requires every condition to finish those stages. Explicit N/A measurements and captures require recorded eligibility evidence. Missing evaluation outputs keep the benchmark partial and prevent leaderboard admission.

Evaluation-only recovery reuses the retained candidate and reference inputs. It preserves authoring traces, formal results, usage and condition identities, and records the evaluation source and previous publication revision in a recovery receipt.

| Result | What to inspect |
|---|---|
| Formal fidelity | Mandatory SimReady score, strict profile pass and passed, failed and unmeasured requirements out of 46 |
| Visual fidelity | Shape and appearance material correspondence, combined by their geometric mean |
| Kinematic fidelity | Joint recovery and agreement of allowed motion, including missing and additional joints |
| SimReady Robotics-Prop optional | Separate score out of 100 and passed, failed and unmeasured requirements out of 51 |
| Shape | Surface correspondence at the reference scale |
| Rigid parts | Per-part correspondence and body structure |
| Joints | Joint types, connectivity, frames and limits |
| Materials | Supported visual and physical material correspondence |
| Captures | Orientation-aligned reference and reconstruction from matched cameras at the same physical scale |

The three composites appear first in the report and CSV as `formal_fidelity_percent`, `visual_fidelity_percent` and `kinematic_fidelity_percent`. Their structured definitions and factors are retained in `composite_scores`. Each score includes its status, earned credit, measured and eligible counts and coverage. Unknown measurements remain null and inapplicable criteria remain N/A.

When both visual factors are measured, visual fidelity is `sqrt(shape_percent * appearance_percent)`. A zero in either factor makes the combined score zero. Appearance compares supported USD material properties at corresponding surface samples, so similar rendered colours alone do not establish a match. Inspect `composite_scores.visual.factors` and the material feature counts to identify the factor and properties responsible.

The underlying factors include surface precision and recall, per-body correspondence, joint motion errors and per-property material matches. The validator report records the findings behind both profile scores. The generated USD is the submitted asset and the Harbor trajectory records its authoring session.

The main table, CSV and published JSON also contain per-reconstruction model requests, validation calls, request errors, elapsed time, summed model request time, input/output token counts and inference/compute costs in USD. Token usage maps provider `prompt_tokens`/`completion_tokens` to `input_tokens`/`output_tokens`. Cached and reasoning token details are included in their provider totals and are not added again. Total tokens require both input and output counts, and total cost requires both inference and compute costs. Missing measurements remain null in JSON, blank in CSV and `Not measured` in the table.

Usage reconciliation compares retained gateway records with trajectory steps and their final totals, deduplicating canonical request IDs. Pricing-derived inference cost uses the verified model/provider route and recorded tariff. Flat input/output tariffs price each full token total once, while tariffs with separate cache rates retain their cache breakdown. Productive authoring calls and total provider requests have separate counts because failed requests do not consume the authoring allowance. Partial stream usage can corroborate a trajectory but does not establish a complete request cost. Reconciliation receipts retain the priced-request subtotal and coverage when a complete run cost is unavailable.

Elapsed time is the duration of one reconstruction. Model request time sums the recorded durations of its admitted requests, including failed requests. Token totals require terminal usage for every admitted request, while missing or partial usage remains unmeasured. The leaderboard displays means per reconstruction, with measured-attempt counts, and reports a mean only when that metric is measured for every reconstruction. These means do not represent the wall time of a parallel run.

## Compare conditions

Use one output directory per model/harness/skill condition. For the NV25 public reference set, compare all 100 attempts across the 25 assets and four reconstructions per asset. Keep the public reference set, repetitions and authoring settings fixed when comparing models. To measure a skill's effect, run the same model and harness with and without that skill.

## Published results

The leaderboard and reconstruction reports are hosted on [Hugging Face](https://huggingface.co/datasets/hcltech-robotics/simreadybench), together with result tables, generated USDs and captures. The GitHub repository contains the benchmark code, configuration and project page.

Pass `--hf-repo OWNER/DATASET --hf-run RUN` to publish each completed reconstruction. Install the `publish` extra and set `HF_TOKEN` or use `hf auth login`. The dataset viewer reads published Parquet tables, and the report includes captures, generated USDs, validator findings and Harbor trajectories. A run enters the NV25 leaderboard after all 25 assets in that public reference set have four scored reconstructions.

The viewer opens with `references`, the 100 NV100 reference objects and their public observation images. Select `results` for historical NV25 scores and paired images. Reference rows have `type="reference"` and an `id`; result rows have `type="result"` and a `reference_id`. The `references` config has `nv100` and `nv25` splits with the same image schema. Join each result’s `reference_id` to the reference `id` across those splits to retrieve its object. Additional subsets group results by model, harness, skills, asset, reference set, authoring protocol and reasoning level.

## Benchmark registration and leaderboard

Each publication maintains the dataset's `eval.yaml` benchmark definition, discovery metadata and visible leaderboard. NV25 leaderboard admission requires the pinned reference set and all 100 reconstructions, with four attempts for every asset under the declared model, harness, skills and authoring protocol. Formal, visual and kinematic fidelity remain separate scores.

Completing the set automatically adds its entry to `leaderboard.json` and the dataset card in the same commit as the readout. Repeating publication or resuming the same run updates that entry while retaining other completed runs. Partial runs remain available through the viewer and enter the leaderboard when their set is complete. The benchmark definition uses Harbor and declares the NV25 64-turn, four-reconstruction formal, visual and kinematic tasks.

