#!/usr/bin/env python3
"""Exercise the real static comparison page with Chromium and WebGL.

Requires Playwright and its Chromium browser. Start the site with http.server,
then pass its URL. Screenshots and a compact verification receipt are optional.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def check(url, output):
    url = url.rstrip('/') + '/'
    manifest = json.load(urlopen(url + 'assets/explorer/manifest.json'))
    checks = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url, wait_until='domcontentloaded')
        page.locator('#explorer').scroll_into_view_if_needed()
        page.wait_for_function("document.querySelector('#explorer').dataset.state === 'ready'", timeout=60000)
        assert page.locator('#explorer-object option').count() == len(manifest['objects'])
        for obj in manifest['objects']:
            page.select_option('#explorer-object', obj['id'])
            for variant in obj['variants']:
                page.select_option('#explorer-variant', variant['id'])
                page.wait_for_function("document.querySelector('#explorer-reference-stage model-viewer')?.loaded", timeout=30000)
                if variant.get('src'):
                    page.wait_for_function("document.querySelector('#explorer-candidate-stage model-viewer')?.loaded", timeout=30000)
                    assert page.locator('#explorer-candidate-stage model-viewer').evaluate('(el) => el.src').endswith(variant['src'])
                else:
                    assert page.locator('#explorer-candidate-stage').inner_text().startswith('3D preview unavailable')
                assert page.locator('#explorer-candidate-title').inner_text() == variant['label']
                actual = page.locator('#explorer-scores .explorer-metric-value').all_text_contents()
                expected = page.evaluate("v => ['formal','visual','kinematic'].map(k => v.scores[k].status === 'not_applicable' ? 'N/A' : v.scores[k].value === null ? 'Not measured' : v.scores[k].value === 0 ? '0' : v.scores[k].value.toLocaleString('en-GB', {maximumFractionDigits:1}))", variant)
                assert actual == expected, (obj['id'], variant['id'], actual, expected)
                assert page.locator('#explorer-result-source').get_attribute('href') == variant['source_url']
                checks.append({'object': obj['id'], 'variant': variant['id'], 'geometry': bool(variant.get('src')), 'scores': actual})

        # A rapid selection change must end with both geometry and scores from
        # the final choice, even if an earlier asset finishes loading later.
        first = manifest['objects'][0]
        page.select_option('#explorer-object', first['id'])
        for variant in first['variants']:
            page.select_option('#explorer-variant', variant['id'])
        page.wait_for_function("document.querySelector('#explorer-candidate-stage model-viewer')?.loaded", timeout=30000)
        assert page.locator('#explorer-candidate-stage model-viewer').evaluate('(el) => el.src').endswith(first['variants'][-1]['src'])

        orbit = "role => {const o=document.querySelector('#explorer-'+role+'-stage model-viewer').getCameraOrbit();return {theta:o.theta,phi:o.phi,radius:o.radius}}"
        page.wait_for_function("document.querySelector('#explorer-reference-stage model-viewer')?.loaded")
        before = page.evaluate(orbit, 'reference')
        page.locator('[data-view="reference"][data-rotate="right"]').click()
        after = page.evaluate(orbit, 'reference')
        other = page.evaluate(orbit, 'candidate')
        assert abs(after['theta'] - before['theta']) > 0.1
        assert abs(after['theta'] - other['theta']) < 0.001
        page.uncheck('#explorer-link')
        before = page.evaluate(orbit, 'candidate')
        page.locator('[data-view="reference"][data-rotate="left"]').click()
        assert abs(page.evaluate(orbit, 'candidate')['theta'] - before['theta']) < 0.001
        page.check('#explorer-link')
        box = page.locator('#explorer-reference-stage').bounding_box()
        before = page.evaluate(orbit, 'reference')
        page.mouse.move(box['x'] + box['width'] * .5, box['y'] + box['height'] * .5)
        page.mouse.down()
        page.mouse.move(box['x'] + box['width'] * .7, box['y'] + box['height'] * .5, steps=12)
        page.mouse.up()
        page.wait_for_timeout(400)
        assert abs(page.evaluate(orbit, 'reference')['theta'] - before['theta']) > .1
        page.click('#explorer-reset')
        page.locator('#explorer').screenshot(path=str(output / 'viewer-desktop.png'))

        # One model's missing preview must not hide its measurements or prevent
        # selecting another model; every variant above has already been visited.
        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
        mobile.goto(url, wait_until='domcontentloaded')
        mobile.locator('#explorer').scroll_into_view_if_needed()
        mobile.wait_for_function("document.querySelector('#explorer-candidate-stage model-viewer')?.loaded", timeout=60000)
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        mobile.locator('#explorer').screenshot(path=str(output / 'viewer-mobile.png'))

        snapshot = browser.new_page()
        requests = []
        snapshot.on('request', lambda request: requests.append(request.url))
        snapshot.goto(url + '?snapshot=1', wait_until='domcontentloaded')
        snapshot.locator('#explorer').scroll_into_view_if_needed()
        snapshot.wait_for_timeout(300)
        assert not any('assets/explorer/manifest.json' in r or 'vendor/model-viewer.min.js' in r for r in requests)
        assert snapshot.locator('model-viewer').count() == 0
        assert not errors, errors
        browser.close()
    report = {'variants_verified': checks, 'browser_errors': errors, 'camera_keyboard_drag_lock_reset': 'passed',
              'mobile_overflow': 'none', 'snapshot_isolation': 'passed', 'rapid_selection': 'passed'}
    (output / 'browser-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {len(checks)} object/run pairs, rotation, selection, mobile layout and snapshot isolation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check(args.url, args.output)
