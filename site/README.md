# Project page

The [project page](https://hcltech-robotics.github.io/simreadybench/) is deployed from `site/` by `.github/workflows/pages.yml` on changes to `main`.

Preview locally:

```sh
python3 -m http.server 8765 --directory site
```

`assets/leaderboard.json` supplies the sortable leaderboard and secondary charts. It records three separate fidelity scores and per-reconstruction resource use. Formal fidelity is the default. Cost and trial duration include retries; hosted compute is not allocated to configurations. Pareto scores use matched object/repetition slots across configurations. Frontiers minimise cost or time and maximise the selected fidelity score.

Rebuild chart data and static figures from the published leaderboard:

```sh
python scripts/build_nv100_results.py --leaderboard leaderboard.json --output charts
```

The explorer uses `assets/explorer/manifest.json` and `delta-manifest.json`. Each candidate links to its exact trial, render and USD package. Derived GLBs and distance clouds are hosted alongside that run on Hugging Face. The gallery selects the first repetition of one configuration consistently across objects.

The page uses the [Academic project page template](https://github.com/eliahuhorwitz/Academic-project-page-template/tree/d38af1ccae1ce82c3404d2820c4c646afd409f81), licensed under CC BY-SA 4.0.

Formal pass rates use the explicit mandatory-pass flag in the per-trial evaluator output. pass@3 estimates the probability of at least one pass in three attempts; pass^3 estimates the probability all three pass. Retry executions contribute resource use but do not add pass-rate samples. The table expands token breakdowns and per-metric cost per point. Cached input remains part of input, and total tokens equal input plus output.
