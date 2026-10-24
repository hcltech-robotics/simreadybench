/** Minimise x and maximise y; equality alone never constitutes dominance. */
export function paretoFrontier(points) {
  const eligible = points.filter((point) => Number.isFinite(point.x) && point.x >= 0
    && Number.isFinite(point.y) && point.y >= 0 && point.y <= 100
    && point.complete !== false);
  return eligible.filter((point) => !eligible.some((other) => other.x <= point.x
    && other.y >= point.y && (other.x < point.x || other.y > point.y)))
    .sort((a, b) => a.x - b.x || b.y - a.y);
}

/** The plotted score must describe exactly the observations behind its x value. */
export function pairedScore(run, metric, scoreName) {
  if (!metric || !Number.isFinite(metric.value) || metric.value < 0) return null;
  const score = metric.scores?.[scoreName];
  if (!score || !Number.isFinite(score.value) || score.value < 0 || score.value > 100) return null;
  if (!Number.isFinite(score.x_value) || score.x_value < 0) return null;
  return score;
}
