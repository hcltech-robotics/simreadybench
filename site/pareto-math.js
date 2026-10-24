/** Minimise x and maximise y; equality alone never constitutes dominance. */
export function paretoFrontier(points) {
  const eligible = points.filter((point) => Number.isFinite(point.x) && point.x >= 0
    && Number.isFinite(point.y));
  return eligible.filter((point) => !eligible.some((other) => other.x <= point.x
    && other.y >= point.y && (other.x < point.x || other.y > point.y)))
    .sort((a, b) => a.x - b.x || b.y - a.y);
}

/** Each axis uses its available mean; coverage remains attached to that mean. */
export function availableScore(run, metric, scoreName) {
  if (!metric || !Number.isFinite(metric.value) || metric.value < 0) return null;
  const score = run.scores?.[scoreName];
  if (!score || !Number.isFinite(score.value)) return null;
  if (scoreName !== "cumulative" && (score.value < 0 || score.value > 100)) return null;
  return { ...score, x_value: metric.value };
}
