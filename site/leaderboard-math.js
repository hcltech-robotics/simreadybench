/** Numeric access and stable ordering shared by the table and its tests. */
export const sortFields = {
  formal: ['Formal fidelity', 'desc'], visual: ['Visual fidelity', 'desc'],
  optional: ['Optional criteria', 'desc'],
  kinematic: ['Kinematic fidelity', 'desc'], pass_at_1: ['pass@1', 'desc'],
  pass_at_3: ['pass@3', 'desc'], pass_all_3: ['pass^3', 'desc'],
  tokens: ['Total tokens / asset', 'asc'], input: ['Input tokens / asset', 'asc'],
  output: ['Output tokens / asset', 'asc'], cached_input: ['Cached input / asset', 'asc'],
  cost: ['Inference cost / asset', 'asc'], cost_point: ['Cost / formal point', 'asc'],
  time: ['Trial time / asset', 'asc'], label: ['Configuration', 'asc'],
};

export function valueFor(run, key) {
  if (['formal', 'visual', 'kinematic'].includes(key)) return run.scores[key]?.value ?? null;
  if (key === 'optional') return run.optional_criteria?.value ?? null;
  if (key.startsWith('pass_')) return run.pass_rates?.[key]?.value ?? null;
  if (key === 'tokens') return run.tokens_per_asset?.total?.value ?? null;
  if (['input', 'output', 'cached_input'].includes(key)) return run.tokens_per_asset?.[key]?.value ?? null;
  if (key === 'cost') return run.inference_cost_per_asset?.value ?? null;
  if (key === 'time') return run.trial_seconds_per_asset?.value ?? null;
  if (key === 'cost_point') {
    const metric = run.cost_per_point?.formal;
    return metric?.status === 'infinite' ? Infinity : metric?.value ?? null;
  }
  return key === 'label' ? run.label : null;
}

export function orderRuns(runs, key = 'formal', direction = 'desc', query = '') {
  const match = query.trim().toLocaleLowerCase();
  const present = value => value !== null && value !== undefined && !Number.isNaN(value);
  return runs.filter(run => !match || [run.label, run.model, run.harness, run.requested_model]
    .filter(Boolean).join(' ').toLocaleLowerCase().includes(match)).slice().sort((a, b) => {
    const left = valueFor(a, key), right = valueFor(b, key);
    if (present(left) !== present(right)) return present(left) ? -1 : 1;
    if (present(left) && left !== right) {
      const cmp = typeof left === 'string' ? left.localeCompare(right) : left < right ? -1 : 1;
      return direction === 'asc' ? cmp : -cmp;
    }
    const ac = valueFor(a, 'cost'), bc = valueFor(b, 'cost');
    if (Number.isFinite(ac) && Number.isFinite(bc) && ac !== bc) return ac - bc;
    return a.label.localeCompare(b.label);
  });
}

export function scoreText(metric) {
  if (metric?.status === 'not_applicable') return 'N/A';
  if (!Number.isFinite(metric?.value) && Number.isFinite(metric?.lower_bound_percent) && Number.isFinite(metric?.upper_bound_percent)) {
    const fmt = value => value.toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});
    return `${fmt(metric.lower_bound_percent)}–${fmt(metric.upper_bound_percent)}`;
  }
  return Number.isFinite(metric?.value) && metric.value >= 0 && metric.value <= 100
    ? metric.value.toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2}) : '—';
}
