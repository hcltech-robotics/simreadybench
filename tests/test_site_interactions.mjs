import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const load = async name => import(`data:text/javascript;base64,${Buffer.from(await readFile(new URL('../site/'+name,import.meta.url),'utf8')).toString('base64')}`);
const {compactPointLabel,placePointLabels}=await load('pareto-labels.js');
const {resetExplorerPreview}=await load('explorer-state.js');

test('point labels use the model ID and separate display effort without changing evidence',()=>{
 const run={requested_model:'openai/gpt-6.1-sol',reasoning:null,reasoning_display:'low'};
 assert.equal(compactPointLabel(run),'gpt-6.1-sol/low');assert.equal(run.reasoning,null);
 assert.equal(compactPointLabel({requested_model:'openai/gpt-6.1-terra',reasoning:'high'}),'gpt-6.1-terra/high');
});

for(const width of [220,900]) test(`tied labels fit without overlap in a ${width}px plot`,()=>{
 const items=Array.from({length:5},(_,i)=>({id:String(i),x:width*(.2+i*.12),y:35,width:100,height:12}));
 const original=JSON.stringify(items);const positions=placePointLabels(items,{left:0,right:width,top:10,bottom:280});
 assert.equal(JSON.stringify(items),original);assert.equal(positions.length,5);
 for(const a of positions){assert.ok(a.left>=0 && a.left+a.width<=width && a.top>=10 && a.top+a.height<=280);
  for(const b of positions)if(a!==b)assert.ok(a.left+a.width<=b.left || b.left+b.width<=a.left || a.top+a.height<=b.top || b.top+b.height<=a.top);
 }
 assert.deepEqual(positions,placePointLabels(items,{left:0,right:width,top:10,bottom:280}));
});

test('snapshot reset removes saved canvases and false readiness before lazy initialisation',()=>{
 const nodes=Object.fromEntries(['canvas','object','variant','mode-swipe','mode-delta','reset','swipe','threshold','stage','divider','delta-controls','readout','status','preview-message'].map(id=>[id,{dataset:{},disabled:false,hidden:false,textContent:'ready',children:[{},{}],replaceChildren(){this.children=[];},setAttribute(key,value){this[key]=value;}}]));
 const root={dataset:{state:'ready'},querySelector(selector){return nodes[selector.replace('#explorer-','')];}};
 resetExplorerPreview(root,{interactive:false});
 assert.equal(root.dataset.state,'idle');assert.equal(nodes.canvas.children.length,0);assert.equal(nodes.object.disabled,true);assert.equal(nodes['mode-swipe'].disabled,true);
 assert.equal(nodes.readout.hidden,true);assert.equal(nodes.divider.hidden,true);assert.equal(nodes['preview-message'].hidden,false);assert.ok(!nodes.status.textContent.includes('ready'));
 assert.ok(nodes.status.textContent.includes('JavaScript'));
});
