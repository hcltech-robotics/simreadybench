# Leaderboard statistics

Each leaderboard entry contains the complete NV25 reference set with four
reconstructions per asset. Formal, visual and kinematic fidelity retain their
published scoring definitions and are reported separately on a 0-100 scale.
The leaderboard mean is the arithmetic mean of eligible reconstruction scores.
Reference-inapplicable results are excluded. A missing eligible score leaves
the mean unknown.

## Confidence intervals

The displayed 95% confidence interval is a percentile asset-cluster bootstrap
with 10,000 resamples and seed 20261020. Each resample draws the eligible asset
clusters with replacement and retains every reconstruction within a selected
asset. This preserves dependence between the four reconstructions of the same
reference. The mean in each draw uses the original eligible-attempt weighting.
The interval endpoints are the 2.5th and 97.5th percentiles, calculated with
linear interpolation. At least two eligible asset clusters are required.

This approach follows the cluster bootstrap for repeated measurements described
by [Deen and de Rooij](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/).
The interval describes variation across reference assets and retains the
benchmark's fixed scoring and reference eligibility rules.

## Resource efficiency

Efficiency uses the same eligible reconstructions as each fidelity score:

- Points per million tokens equals mean fidelity divided by mean total tokens
  per eligible reconstruction, multiplied by 1,000,000.
- Points per elapsed minute equals mean fidelity divided by mean elapsed seconds
  per eligible reconstruction, multiplied by 60.

Total tokens include input and output tokens. Elapsed time includes model
requests, tools and validation within the reconstruction, rather than only
inference latency or the wall time of a parallel benchmark run. Resource
measurements must be present for every eligible reconstruction and the mean
denominator must be positive. Missing measurements are retained as unknown.
The JSON records the measured and eligible attempt counts, eligible asset count
and the resource denominator for each ratio.

Efficiency intervals resample the score and its resource use together with the
same asset-cluster draw. They are intervals of the ratio, rather than ratios
of independently calculated interval endpoints. A zero resource denominator
in a bootstrap draw leaves its interval unknown.

## Mandatory pass rates

The attempt pass rate reports the fraction of reconstructions that pass all 46
mandatory requirements. The four-reconstruction asset pass rate reports the
fraction of assets with at least one such reconstruction among their four
attempts. Neither rate replaces the three fidelity scores. Unknown mandatory
outcomes retain their measurement coverage and are not counted as failures.

The machine-readable leaderboard retains the resource means and coverage for
cost, tokens, elapsed time, model requests and validation calls. The score,
token-efficiency and elapsed-time-efficiency figures are regenerated whenever
a complete benchmark entry is published.
