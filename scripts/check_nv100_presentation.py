#!/usr/bin/env python3
"""Check current benchmark charts and asset examples in a browser."""
from pathlib import Path
from urllib.parse import unquote
import argparse,json,math,shutil
from playwright.sync_api import sync_playwright


def check(url, output, payload=None):
    output.mkdir(parents=True,exist_ok=True);errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        if payload:
            prefix='https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/main/'
            def route_local(route):
                path=payload/unquote(route.request.url.split(prefix,1)[1].split('?',1)[0])
                if path.is_file():route.fulfill(path=str(path),headers={'Access-Control-Allow-Origin':'*'})
                else:route.continue_()
            page.route(prefix+'**',route_local)
            # Existing published renders are mirrored from their hash-verified
            # local files for an offline preview. Public URLs are checked with
            # the environment's TLS-verifying HTTP client after publication.
            source_plan = Path('.cache/nv100-migration/full-approved-20261028/artifact-plan/plan.json')
            if source_plan.exists():
                files = {i['path']: i for i in json.loads(source_plan.read_text())['files']}
                original = 'https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/e586f232e0b1939f84c59956e88eed38c167bd2f/'
                def route_render(route):
                    name = unquote(route.request.url.split(original,1)[1].split('?',1)[0])
                    item = files.get(name)
                    if item and name.endswith('.png'):
                        import hashlib
                        raw = Path(item['local_path']).read_bytes()
                        assert hashlib.sha256(raw).hexdigest() == item['sha256']
                        route.fulfill(body=raw, content_type='image/png',headers={'Access-Control-Allow-Origin':'*'})
                    else:route.continue_()
                page.route(original+'**',route_render)
        page.goto(url,wait_until='domcontentloaded');page.wait_for_function("document.querySelector('#pareto-root').dataset.loadState === 'ready'")
        page.wait_for_function("document.querySelector('#leaderboard-table-region').dataset.loadState === 'ready'")
        data=page.evaluate("async () => (await fetch('assets/pareto.json')).json()")
        assert page.locator('#leaderboard-body tr').count()==len(data['runs'])
        assert page.locator('[data-pareto-score="cumulative"]').count()==0
        assert page.locator('[data-pareto-score="formal"]').get_attribute('aria-pressed')=='true'
        combinations=[]
        for axis in data['runs'][0]['metrics']:
            page.select_option('#pareto-x',axis)
            for score in ['formal','visual','kinematic','all']:
                page.locator(f'[data-pareto-score="{score}"]').click()
                score_ids=['formal','visual','kinematic'] if score=='all' else [score]
                actual=page.locator('.pareto-point').evaluate_all("nodes => nodes.map(n => ({id:n.dataset.runId,score:n.dataset.score,frontier:n.dataset.frontier==='true'}))")
                expected=[]
                for metric in score_ids:
                    points=[{'id':r['id'],'x':r['metrics'][axis]['value'],'y':r['metrics'][axis]['paired_scores'][metric]['value']} for r in data['runs']]
                    points=[r for r in points if r['x'] is not None and r['y'] is not None]
                    for r in points:
                        undominated=not any(q['x']<=r['x'] and q['y']>=r['y'] and (q['x']<r['x'] or q['y']>r['y']) for q in points)
                        expected.append({'id':r['id'],'score':metric,'frontier':undominated})
                assert sorted(actual,key=lambda r:(r['id'],r['score']))==sorted(expected,key=lambda r:(r['id'],r['score']))
                combinations.append([axis,score,len(actual)])
        page.select_option('#pareto-x','cost_usd_per_asset');page.locator('[data-pareto-score="all"]').click();page.locator('#pareto').scroll_into_view_if_needed();page.screenshot(path=str(output/'charts-desktop.png'),full_page=False)
        page.locator('#explorer').scroll_into_view_if_needed();page.wait_for_function("document.querySelector('#explorer').dataset.state === 'ready'",timeout=60000)
        manifest=page.evaluate("async () => (await fetch('assets/explorer/manifest.json')).json()")
        delta=page.evaluate("async () => (await fetch('assets/explorer/delta-manifest.json')).json()")
        variants=0
        for obj in manifest['objects']:
            page.select_option('#explorer-object',obj['id'])
            for v in obj['variants']:
                assert v['run_id'] in data['job_ids'] and v['repetition']==0
                assert v['run_id'] in v['src'] and v['trial_id'] in v['download_url']
                assert delta['pairs'][obj['id']+'/'+v['id']]['source_candidate_sha256']==v['sha256']
                page.select_option('#explorer-variant',v['id']);variants+=1
        page.select_option('#explorer-object',manifest['objects'][0]['id']);page.select_option('#explorer-variant','sol-medium')
        page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer?.getState().candidateReady === true",timeout=90000)
        page.wait_for_function("document.querySelector('#explorer-candidate-render').complete && document.querySelector('#explorer-candidate-render').naturalWidth > 0",timeout=60000)
        assert page.locator('#explorer-mode-delta').is_enabled();page.locator('#explorer-mode-delta').click()
        page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer.getState().deltaReady === true",timeout=90000)
        page.screenshot(path=str(output/'gallery-desktop.png'),full_page=False)
        page.set_viewport_size({'width':390,'height':844});page.locator('#pareto').scroll_into_view_if_needed();page.locator('[data-pareto-score="formal"]').click();page.screenshot(path=str(output/'charts-mobile.png'))
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 2')
        assert not errors,errors
        result={'status':'passed','chart_combinations':combinations,'configurations':len(data['runs']),'objects':len(manifest['objects']),'current_variants_checked':variants,'current_delta_pairs':len(delta['pairs']),'desktop_and_mobile':True,'javascript_errors':errors}
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');browser.close();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--output',type=Path,required=True);p.add_argument('--payload',type=Path);a=p.parse_args();print(json.dumps(check(a.url,a.output,a.payload),indent=2))
