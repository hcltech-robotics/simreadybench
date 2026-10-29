#!/usr/bin/env python3
"""Check real current-run meshes, viewport visibility and compact chart annotations."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import argparse
import hashlib
import json

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-snapshot', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('.cache/site-interactions'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path('site/assets/explorer/manifest.json').read_text())
    source_hashes = {item['src']: item['sha256'] for obj in manifest['objects']
                     for item in [obj['reference'], *obj['variants']] if item.get('src')}
    cache = {}; fetched = []; errors = []
    def relay(route):
        url = route.request.url
        if urlsplit(url).hostname not in {'huggingface.co', 'fonts.googleapis.com', 'fonts.gstatic.com'}:
            route.continue_(); return
        if url not in cache:
            headers = {'Accept-Encoding': 'identity', 'User-Agent': 'Mozilla/5.0'}
            if route.request.headers.get('origin'): headers['Origin'] = route.request.headers['origin']
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                raw = response.read()
                selected = {k:v for k,v in response.headers.items() if k.lower() in
                            {'content-type', 'access-control-allow-origin', 'cache-control'}}
                cache[url] = (response.status, raw, selected)
            if url in source_hashes:
                assert raw[:4] == b'glTF' and hashlib.sha256(raw).hexdigest() == source_hashes[url]
                fetched.append({'url':url, 'sha256':source_hashes[url], 'bytes':len(raw)})
        status, raw, headers = cache[url]
        route.fulfill(status=status, body=raw, headers=headers)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory='site'))
    Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/'
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium' if Path('/usr/bin/chromium').exists() else None,
            args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        if args.update_snapshot:
            page = browser.new_page(viewport={'width':1440,'height':1100}); page.route('**/*',relay)
            page.goto(url+'?snapshot=1', wait_until='domcontentloaded')
            page.wait_for_function("document.querySelector('#leaderboard-table-region').dataset.loadState==='ready' && document.querySelector('#pareto-root').dataset.loadState==='ready' && document.querySelector('#explorer').dataset.state==='idle'")
            assert page.locator('#explorer-canvas canvas').count() == 0
            Path('site/index.html').write_text(page.content()); page.close()
        for name, width, height in [('desktop',1440,1100), ('mobile',390,844)]:
            page = browser.new_page(viewport={'width':width,'height':height}); page.route('**/*',relay)
            page.on('pageerror',lambda error: errors.append(str(error)))
            page.goto(url,wait_until='domcontentloaded')
            page.wait_for_function("document.querySelector('#explorer').dataset.state==='idle'")
            assert page.locator('#explorer-canvas canvas').count() == 0
            assert page.locator('#explorer-object').is_disabled() and page.locator('#explorer-reset').is_disabled()
            assert 'ready' not in page.locator('#explorer-status').inner_text()
            # Reproduce old snapshots: constructor must replace, rather than append to, both canvases.
            page.locator('#explorer-canvas').evaluate('(el)=>el.append(document.createElement("canvas"),document.createElement("canvas"))')
            page.locator('#explorer').scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer?.getState().referenceReady && document.querySelector('#explorer-stage').comparisonViewer?.getState().candidateReady",timeout=90000)
            assert page.locator('#explorer-canvas canvas').count() == 1
            frame = page.evaluate('''() => {
              const viewer=document.querySelector('#explorer-stage').comparisonViewer;
              viewer.render();const gl=viewer.renderer.getContext();gl.finish();
              const w=gl.drawingBufferWidth,h=gl.drawingBufferHeight,pixels=new Uint8Array(w*h*4);
              gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
              let left=0,right=0;const background=[...pixels.slice(0,3)];
              for(let y=0;y<h;y++)for(let x=0;x<w;x++){
                const i=(y*w+x)*4;
                if(background.some((v,k)=>Math.abs(pixels[i+k]-v)>24)){if(x<w/2)left++;else right++;}
              }
              const a=viewer.canvas.getBoundingClientRect(),b=viewer.container.getBoundingClientRect();
              return {left,right,width:w,height:h,canvasTop:a.top,containerTop:b.top,canvasHeight:a.height,containerHeight:b.height,errors:viewer.getState().errors,glError:gl.getError()};
            }''')
            assert frame['left'] > 100 and frame['right'] > 100 and frame['glError'] == 0, frame
            assert abs(frame['canvasTop']-frame['containerTop']) < 1 and abs(frame['canvasHeight']-frame['containerHeight']) < 1, frame
            before = page.evaluate("document.querySelector('#explorer-stage').comparisonViewer.getState().cameraPosition")
            page.locator('#explorer-canvas canvas').focus();page.keyboard.press('ArrowRight')
            assert before != page.evaluate("document.querySelector('#explorer-stage').comparisonViewer.getState().cameraPosition")
            page.locator('#explorer-reset').click();page.locator('#explorer-stage').screenshot(path=str(args.output/(name+'-3d.png')))
            page.locator('#explorer-mode-delta').click()
            page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer.getState().deltaReady",timeout=90000)
            assert page.evaluate("document.querySelector('#explorer-stage').comparisonViewer.getState().pointCount") == 24000
            page.locator('#explorer-stage').screenshot(path=str(args.output/(name+'-delta.png')))
            page.locator('#pareto').scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#pareto-root').dataset.loadState==='ready'")
            for metric in ['formal','visual','kinematic','all']:
                page.locator(f'[data-pareto-score="{metric}"]').click()
                labels = page.locator('.pareto-point-label').evaluate_all('''nodes=>nodes.map(node=>{
                  const b=node.getBoundingClientRect();return {text:node.textContent,left:b.left,right:b.right,top:b.top,bottom:b.bottom};})''')
                assert len(labels) == (15 if metric=='all' else 5)
                for i,a in enumerate(labels):
                    assert a['text'].startswith('gpt-6.1-') and '/' in a['text'] and 'Reasoning' not in a['text']
                    for b in labels[i+1:]:
                        assert a['right'] <= b['left']+.5 or b['right'] <= a['left']+.5 or a['bottom'] <= b['top']+.5 or b['bottom'] <= a['top']+.5, (name,metric,a,b)
            page.locator('[data-pareto-score="formal"]').click();page.locator('#pareto').screenshot(path=str(args.output/(name+'-pareto.png')))
            assert 'ChatGPT 6.1 Sol' in page.locator('#leaderboard-body').inner_text()
            results.append({'viewport':name,'single_visible_canvas':True,'real_mesh_pixels':frame,'camera_controls':True,'delta_points':24000,'compact_nonoverlapping_labels':True});page.close()
        browser.close()
    server.shutdown();server.server_close()
    assert not errors,errors
    receipt={'status':'passed','backend':'Chromium WebGL2 with SwiftShader software rendering','results':results,'actual_current_glbs_sha_verified':fetched,'errors':errors}
    (args.output/'verified.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
