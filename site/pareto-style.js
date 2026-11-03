/** Model identity stays consistent across fidelity panels and resource axes. */
export function markerStyle(run) {
  const model = `${run.requested_model || ''} ${run.model || ''}`.toLowerCase();
  const colour = /nemotron|nvidia/.test(model) ? '#76b900'
    : /gpt-|chatgpt/.test(model) ? '#0072b2'
    : /deepseek/.test(model) ? '#7851a9' : /claude/.test(model) ? '#d97757' : '#596579';
  const initial = /nemotron/.test(model) ? 'N' : /deepseek/.test(model) ? 'D'
    : /claude/.test(model) ? 'C' : /claude/.test(model) ? 'C' : /terra/.test(model) ? 'T' : /sol/.test(model) ? 'S' : (run.model || '?')[0];
  const harness = run.harness_id || run.harness || '';
  return {colour, initial, custom: !['codex', 'Codex', 'terminus-2', 'Terminus 2', 'claude-code', 'Claude Code', 'claude-code', 'Claude Code'].includes(harness)};
}
