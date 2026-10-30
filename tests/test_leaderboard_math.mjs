import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
const source=await readFile(new URL('../site/leaderboard-math.js',import.meta.url),'utf8');
const {orderRuns,valueFor,scoreText,sortFields}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const run=(label,formal,cost,tokens=0)=>({label,model:label,scores:{formal:{value:formal}},inference_cost_per_asset:{value:cost},tokens_per_asset:{total:{value:tokens}},cost_per_point:{formal:{value:formal>0?cost/formal:null,status:formal===0&&cost>0?'infinite':'known'}}});
test('formal is primary; ties use cost and missing scores always sort last',()=>{
 const rows=[run('missing',null,0),run('costly',100,3),run('cheap',100,1),run('zero',0,0)];
 assert.deepEqual(orderRuns(rows).map(r=>r.label),['cheap','costly','zero','missing']);
 assert.deepEqual(orderRuns(rows,'formal','asc').map(r=>r.label),['zero','cheap','costly','missing']);
 assert.equal(rows[0].label,'missing');
});
test('resources sort numerically, zero remains measured, infinity is explicit',()=>{
 const rows=[run('big',100,20,1000),run('small',100,1,100),run('zero',0,0,0),run('no-score',0,1,5)];
 assert.deepEqual(orderRuns(rows,'tokens','asc').map(r=>r.label),['zero','no-score','small','big']);
 assert.equal(valueFor(rows[3],'cost_point'),Infinity);
 assert.equal(valueFor(rows[2],'tokens'),0);
});
test('filtering configurations and sorting individual token fields are independent',()=>{
 const a=run('Sol high',100,1),b=run('Terra high',100,2);a.tokens_per_asset.input={value:4};b.tokens_per_asset.input={value:2};
 assert.deepEqual(orderRuns([a,b],'input','asc').map(r=>r.label),['Terra high','Sol high']);
 assert.deepEqual(orderRuns([a,b],'formal','desc','sol').map(r=>r.label),['Sol high']);
});

test('optional criteria remain separate, bounded unknowns stay unknown, and pass^3 uses a caret',()=>{
 const a=run('baseline',100,1),b=run('CAD',90,2);
 a.optional_criteria={value:null,lower_bound_percent:73.21,upper_bound_percent:73.24};
 b.optional_criteria={value:70.89};
 assert.equal(valueFor(a,'optional'),null);
 assert.equal(scoreText(a.optional_criteria),'73.21–73.24');
 assert.equal(scoreText(b.optional_criteria),'70.89');
 assert.deepEqual(orderRuns([a,b],'optional','desc').map(r=>r.label),['CAD','baseline']);
 assert.equal(sortFields.pass_all_3[0],'pass^3');
});
