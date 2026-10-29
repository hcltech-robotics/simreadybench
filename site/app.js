import {orderRuns, sortFields, valueFor, scoreText} from './leaderboard-math.js?v=leaderboard-20261029';

const region = document.getElementById('leaderboard-table-region');
const body = document.getElementById('leaderboard-body');
const status = document.getElementById('leaderboard-status');
const archive = document.getElementById('leaderboard-archive');
const search = document.getElementById('leaderboard-search');
const selector = document.getElementById('leaderboard-sort');
const directionButton = document.getElementById('leaderboard-direction');
let rows = [], key = 'formal', direction = 'desc';
const expanded = new Set();
const numeric = (value, digits=2) => Number.isFinite(value)
  ? value.toLocaleString('en-GB', {maximumFractionDigits:digits, minimumFractionDigits:digits}) : '—';
const compact = value => Number.isFinite(value)
  ? value.toLocaleString('en-GB',{notation:'compact',maximumSignificantDigits:4}) : '—';
const token = metric => Number.isFinite(metric?.value) ? (metric.partial?'≥ ':'') + compact(metric.value) : '—';
const money = (metric, digits=4) => !metric || !Number.isFinite(metric.value) ? '—'
  : (metric.partial || metric.status === 'partial_lower_bound' ? '≥ ' : '') + '$' + numeric(metric.value,digits);
function element(tag, text, className) {
  const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;
}
function detail(row, run) {
  const cell=element('td');cell.colSpan=13;
  const grid=element('div',undefined,'leaderboard-detail-grid');
  const tokens=element('div');tokens.append(element('h4','Tokens per asset'));
  for(const [name,label] of [['total','Total'],['input','Input'],['output','Output'],['cached_input','Cached input (part of input)']]) {
    const metric=run.tokens_per_asset[name];tokens.append(element('p',`${label}: ${Number.isFinite(metric.value)&&metric.partial?'≥ ':''}${numeric(metric.value,2)}`));
  }
  const efficiency=element('div');efficiency.append(element('h4','Cost per fidelity point'));
  for(const name of ['formal','visual','kinematic']) {
    const metric=run.cost_per_point[name];
    efficiency.append(element('p',`${name[0].toUpperCase()+name.slice(1)}: ${metric.status==='infinite'?'∞':money(metric,6)}`));
  }
  efficiency.append(element('p','Inference $/asset ÷ mean fidelity on the 0–100 scale.','leaderboard-detail-note'));
  const pass=element('div');pass.append(element('h4','Formal passes'));
  for(const [name,label] of [['pass_at_1','pass@1'],['pass_at_3','pass@3'],['pass_all_3','pass³']]) {
    const metric=run.pass_rates?.[name];pass.append(element('p',metric ? `${label}: ${metric.successes}/${metric.denominator}` : `${label}: —`));
  }
  const link=element('a','Assets and attempt results');link.href=run.source_url;pass.append(link);
  grid.append(tokens,efficiency,pass);cell.append(grid);row.append(cell);
}
function render() {
  const ordered=orderRuns(rows,key,direction,search.value);body.replaceChildren();
  let previous=null, rank=0;
  for(const [index,run] of ordered.entries()) {
    const row=element('tr');row.dataset.configuration=run.id;
    const primary=valueFor(run,key);if(primary!==previous || index===0)rank=index+1;previous=primary;
    row.append(element('td',key==='label'?'':primary===null||primary===undefined?'—':String(rank),'leaderboard-rank'));
    const name=element('th',run.label);name.scope='row';row.append(name);
    for(const metric of ['formal','visual','kinematic']) {
      const score=run.scores[metric];const cell=element('td',scoreText(score),'metric-value');
      cell.title=`${score.measured_attempts}/${score.eligible_attempts} measured scores`;
      if(metric==='formal')cell.classList.add('leaderboard-primary');row.append(cell);
    }
    for(const field of ['pass_at_1','pass_at_3','pass_all_3']) {
      const metric=run.pass_rates?.[field];const cell=element('td',numeric(metric?.value),'metric-value');
      if(metric)cell.title=`${metric.successes}/${metric.denominator}`;row.append(cell);
    }
    const tokens=element('td',undefined,'leaderboard-tokens');const t=run.tokens_per_asset;
    tokens.append(element('strong',token(t.total)));
    tokens.append(element('span',`In ${token(t.input)} · Out ${token(t.output)}`,'leaderboard-submetric'));
    tokens.append(element('span',`Cached input ${token(t.cached_input)}`,'leaderboard-submetric'));
    tokens.title='Total = input + output. Cached input is already included in input.';row.append(tokens);
    const cost=element('td',money(run.inference_cost_per_asset),'metric-value');
    if(run.inference_cost_per_asset.partial)cost.title='Lower bound from available execution costs.';row.append(cost);
    const ratio=run.cost_per_point.formal;const point=element('td',ratio.status==='infinite'?'∞':money(ratio,6),'metric-value');
    point.title='Inference cost per asset divided by mean formal fidelity, on the 0–100 scale.';row.append(point);
    row.append(element('td',numeric(Number.isFinite(run.trial_seconds_per_asset.value)?run.trial_seconds_per_asset.value/60:null),'metric-value'));
    const control=element('td');const button=element('button',expanded.has(run.id)?'−':'+','leaderboard-expand');
    button.type='button';button.setAttribute('aria-label',`Details for ${run.label}`);button.setAttribute('aria-expanded',String(expanded.has(run.id)));
    button.addEventListener('click',()=>{if(expanded.has(run.id))expanded.delete(run.id);else expanded.add(run.id);render();});control.append(button);row.append(control);body.append(row);
    if(expanded.has(run.id)) {const extra=element('tr',undefined,'leaderboard-detail');extra.dataset.detailsFor=run.id;detail(extra,run);body.append(extra);}
  }
  document.querySelectorAll('[data-leaderboard-sort]').forEach(button=>{
    const active=button.dataset.leaderboardSort===key;button.closest('th').setAttribute('aria-sort',active?(direction==='asc'?'ascending':'descending'):'none');
    button.classList.toggle('is-active',active);
  });
  selector.value=key;directionButton.textContent=direction==='asc'?'↑ Ascending':'↓ Descending';
  directionButton.setAttribute('aria-label',`Sort ${direction==='asc'?'descending':'ascending'}`);
  status.hidden=ordered.length>0;status.textContent=ordered.length?'':'No matching configurations.';
}
for(const [field,[label]] of Object.entries(sortFields))selector.append(new Option(label,field));
selector.addEventListener('change',()=>{key=selector.value;direction=sortFields[key][1];render();});
directionButton.addEventListener('click',()=>{direction=direction==='asc'?'desc':'asc';render();});
search.addEventListener('input',render);
document.querySelectorAll('[data-leaderboard-sort]').forEach(button=>button.addEventListener('click',()=>{
  const selected=button.dataset.leaderboardSort;direction=selected===key?(direction==='asc'?'desc':'asc'):sortFields[selected][1];key=selected;render();
}));
region.dataset.loadState='loading';
fetch('assets/leaderboard.json',{credentials:'omit',cache:'no-store'}).then(response=>{if(!response.ok)throw new Error('Leaderboard unavailable');return response.json();}).then(data=>{
  if(!Array.isArray(data.runs)||!data.runs.length)throw new Error('Leaderboard unavailable');rows=data.runs;render();
  document.getElementById('leaderboard-caption').textContent=`NV100 · ${data.reference_count} objects · formal fidelity (0–100)`;
  region.hidden=false;archive.hidden=true;region.dataset.loadState='ready';
}).catch(()=>{region.dataset.loadState='error';status.hidden=false;status.textContent='View the leaderboard on Hugging Face.';archive.hidden=false;});
