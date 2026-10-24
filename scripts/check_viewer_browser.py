#!/usr/bin/env python3
"""Exercise the real static comparison page with Chromium and WebGL.

Requires Playwright and its Chromium browser. Start the site with http.server,
then pass its URL. Screenshots and a compact verification receipt are optional.
"""
import argparse
import json
import math
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


STATE = "document.querySelector('#explorer-stage')?.comparisonViewer?.getState()"


def viewer_state(page):
    return page.evaluate(f"() => {STATE}")


def rendered(page):
    """Allow the requested frame to reach the shared canvas before inspection."""
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def ready_pair(page, obj, variant):
    page.wait_for_function(
        f"""expected => {{
            const state = {STATE};
            return state && !state.loading && state.referenceReady
                && state.referenceSrc?.endsWith(expected.reference)
                && (expected.candidate
                    ? state.candidateReady && state.candidateSrc?.endsWith(expected.candidate)
                    : !state.candidateReady && !state.candidateSrc);
        }}""",
        arg={'reference': obj['reference']['src'], 'candidate': variant.get('src')},
        timeout=60000,
    )
    rendered(page)


def assert_metrics(page, obj, variant):
    assert page.locator('#explorer-candidate-title').inner_text() == variant['label']
    actual = page.locator('#explorer-scores .explorer-metric-value').all_text_contents()
    expected = page.evaluate("v => ['formal','visual','kinematic'].map(k => v.scores[k].status === 'not_applicable' ? 'N/A' : v.scores[k].value === null ? 'Not measured' : v.scores[k].value === 0 ? '0' : v.scores[k].value.toLocaleString('en-GB', {maximumFractionDigits:1}))", variant)
    assert actual == expected, (obj['id'], variant['id'], actual, expected)
    actual_usage = page.locator('#explorer-usage .explorer-metric-value').all_text_contents()
    expected_usage = page.evaluate("""u => {
        const valid = v => typeof v === 'number' && v >= 0;
        const n = (v, digits=0) => (v === 0 ? 0 : v).toLocaleString('en-GB', {maximumFractionDigits:digits});
        const cost = valid(u.total_cost_usd) ? u.total_cost_usd : u.inference_cost_usd;
        const costLabel = !valid(cost) ? 'Not measured' : cost > 0 && cost < .0001 ? '< $0.0001'
            : '$' + (cost === 0 ? 0 : cost).toLocaleString('en-GB', {minimumFractionDigits:2, maximumFractionDigits:4});
        const time = u.elapsed_seconds;
        const timeLabel = !valid(time) ? 'Not measured' : time < 60 ? n(time,1)+' s'
            : time < 3600 ? n(time/60,1)+' min' : n(time/3600,2)+' h';
        return [costLabel, valid(u.total_tokens) ? n(u.total_tokens) : 'Not measured', timeLabel];
    }""", variant['usage'])
    assert actual_usage == expected_usage, (obj['id'], variant['id'], actual_usage, expected_usage)
    assert page.locator('#explorer-result-source').get_attribute('href') == variant['source_url']
    return {'object': obj['id'], 'variant': variant['id'], 'geometry': bool(variant.get('src')),
            'scores': actual, 'cost_tokens_time': actual_usage}


def set_range(page, selector, value):
    """Set up a range value; actual pointer and keyboard behaviour is checked below."""
    page.locator(selector).evaluate("(el, value) => {el.value = value; el.dispatchEvent(new Event('input', {bubbles:true}));}", str(value))
    rendered(page)


def assert_swipe(page, touch=False):
    slider = page.locator('#explorer-swipe')
    assert slider.get_attribute('type') == 'range'
    assert slider.get_attribute('aria-label') == 'Comparison split'
    slider.focus()
    slider.press('Home')
    page.wait_for_function(f"{STATE}?.split === 0")
    rendered(page)
    left_image = page.locator('#explorer-stage canvas').screenshot()
    slider.press('End')
    page.wait_for_function(f"{STATE}?.split === 1")
    rendered(page)
    right_image = page.locator('#explorer-stage canvas').screenshot()
    assert left_image != right_image, 'Swipe endpoints must render different geometry'
    slider.press('ArrowLeft')
    assert 0 < viewer_state(page)['split'] < 1

    set_range(page, '#explorer-swipe', 50)
    slider.scroll_into_view_if_needed()
    box = slider.bounding_box()
    start = {'x': box['x'] + box['width'] * .5, 'y': box['y'] + box['height'] * .5}
    finish = {'x': box['x'] + box['width'] * .7, 'y': start['y']}
    before = viewer_state(page)
    if touch:
        session = page.context.new_cdp_session(page)
        session.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [start]})
        for step in range(1, 9):
            session.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [
                {'x': start['x'] + (finish['x'] - start['x']) * step / 8, 'y': start['y']}],})
        session.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        session.detach()
    else:
        page.mouse.move(**start)
        page.mouse.down()
        page.mouse.move(**finish, steps=10)
        page.mouse.up()
    page.wait_for_function(f"Math.abs({STATE}?.split - 0.5) > 0.1")
    after = viewer_state(page)
    assert math.dist(before['cameraPosition'], after['cameraPosition']) < .00001, 'Dragging the divider must not orbit the camera'
    set_range(page, '#explorer-swipe', 50)


def check(url, output):
    url = url.rstrip('/') + '/'
    manifest = json.load(urlopen(url + 'assets/explorer/manifest.json'))
    checks = []
    errors = []
    delta_checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url, wait_until='domcontentloaded')
        page.locator('#explorer').scroll_into_view_if_needed()
        page.wait_for_function("document.querySelector('#explorer').dataset.state === 'ready'", timeout=60000)
        assert page.locator('#explorer-object option').count() == len(manifest['objects'])
        assert page.locator('[data-rotate], model-viewer, #explorer-link').count() == 0
        assert page.locator('#explorer-stage canvas').count() == 1

        for obj in manifest['objects']:
            page.select_option('#explorer-object', obj['id'])
            for variant in obj['variants']:
                page.select_option('#explorer-variant', variant['id'])
                ready_pair(page, obj, variant)
                checks.append(assert_metrics(page, obj, variant))
                if variant.get('src'):
                    page.click('#explorer-mode-delta')
                    page.wait_for_function(f"{STATE}?.mode === 'delta' && {STATE}?.deltaReady", timeout=60000)
                    state = viewer_state(page)
                    assert state['pointCount'] > 0, (obj['id'], variant['id'], state)
                    assert 0 <= state['visiblePointCount'] <= state['pointCount']
                    assert page.locator('#explorer-mode-delta').get_attribute('aria-pressed') == 'true'
                    assert page.locator('#explorer-delta-controls').is_visible()
                    delta_checks.append({'object': obj['id'], 'variant': variant['id'], 'point_count': state['pointCount']})
                    page.click('#explorer-mode-swipe')
                    assert viewer_state(page)['mode'] == 'swipe'
                else:
                    assert page.locator('#explorer-mode-delta').is_disabled()
                    assert viewer_state(page)['referenceReady']
                    assert not viewer_state(page)['deltaReady']

        # Interleave object and model changes without waiting for assets. Both
        # loaded sources and measured values must belong to the final choice.
        first = manifest['objects'][0]
        final_variant = first['variants'][-1]
        page.select_option('#explorer-object', manifest['objects'][1]['id'])
        page.select_option('#explorer-object', first['id'])
        for variant in first['variants']:
            page.select_option('#explorer-variant', variant['id'])
        ready_pair(page, first, final_variant)
        assert_metrics(page, first, final_variant)

        assert_swipe(page)
        canvas = page.locator('#explorer-stage canvas')
        page.click('#explorer-reset')
        rendered(page)
        initial = viewer_state(page)
        canvas.focus()
        canvas.press('ArrowRight')
        page.wait_for_function(f"before => Math.hypot(...{STATE}.cameraPosition.map((v,i) => v-before[i])) > .001", arg=initial['cameraPosition'])
        page.wait_for_timeout(200)
        canvas.scroll_into_view_if_needed()
        box = canvas.bounding_box()
        before_drag = viewer_state(page)
        page.mouse.move(box['x'] + box['width'] * .35, box['y'] + box['height'] * .25)
        page.mouse.down()
        page.mouse.move(box['x'] + box['width'] * .55, box['y'] + box['height'] * .25, steps=12)
        page.mouse.up()
        page.wait_for_function(f"before => Math.hypot(...{STATE}.cameraPosition.map((v,i) => v-before[i])) > .001", arg=before_drag['cameraPosition'])
        before_zoom = viewer_state(page)
        page.mouse.wheel(0, -240)
        page.wait_for_function(f"before => {{const s={STATE};return Math.hypot(...s.cameraPosition.map((v,i)=>v-s.target[i])) < before;}}", arg=math.dist(before_zoom['cameraPosition'], before_zoom['target']))
        canvas.focus()
        canvas.press('Home')
        page.wait_for_function(f"before => Math.hypot(...{STATE}.cameraPosition.map((v,i) => v-before[i])) < .00001", arg=initial['cameraPosition'])
        canvas.press('ArrowLeft')
        page.wait_for_function(f"before => Math.hypot(...{STATE}.cameraPosition.map((v,i) => v-before[i])) > .001", arg=initial['cameraPosition'])
        page.click('#explorer-reset')
        page.wait_for_function(f"before => Math.hypot(...{STATE}.cameraPosition.map((v,i) => v-before[i])) < .00001", arg=initial['cameraPosition'])
        page.locator('#explorer').screenshot(path=str(output / 'viewer-desktop.png'))

        # The threshold is a real geometric distance filter, not just a legend.
        page.click('#explorer-mode-delta')
        page.wait_for_function(f"{STATE}?.deltaReady", timeout=60000)
        threshold = page.locator('#explorer-threshold')
        threshold.focus()
        threshold.press('Home')
        rendered(page)
        all_points = viewer_state(page)
        threshold.press('End')
        rendered(page)
        filtered_points = viewer_state(page)
        assert filtered_points['pointCount'] == all_points['pointCount']
        assert 0 <= filtered_points['visiblePointCount'] < all_points['visiblePointCount']
        assert filtered_points['thresholdMetres'] > all_points['thresholdMetres']
        assert page.locator('#explorer-threshold-value').inner_text().strip()
        page.locator('#explorer').screenshot(path=str(output / 'viewer-delta.png'))
        page.click('#explorer-mode-swipe')

        mobile = browser.new_page(viewport={'width': 390, 'height': 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
        mobile.on('pageerror', lambda error: errors.append(str(error)))
        mobile.goto(url, wait_until='domcontentloaded')
        mobile.locator('#explorer').scroll_into_view_if_needed()
        ready_pair(mobile, first, first['variants'][0])
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        assert_swipe(mobile, touch=True)
        mobile.click('#explorer-mode-delta')
        mobile.wait_for_function(f"{STATE}?.deltaReady", timeout=60000)
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        mobile.click('#explorer-mode-swipe')
        mobile.locator('#explorer').screenshot(path=str(output / 'viewer-mobile.png'))

        # A failed candidate download keeps reference and measurements usable;
        # choosing another candidate recovers without reloading the page.
        failed = browser.new_page()
        failed.on('pageerror', lambda error: errors.append(str(error)))
        failed.route('**/' + first['variants'][0]['src'], lambda route: route.abort())
        failed.goto(url, wait_until='domcontentloaded')
        failed.locator('#explorer').scroll_into_view_if_needed()
        failed.wait_for_function(f"{STATE}?.referenceReady && !{STATE}?.loading && !!{STATE}?.error", timeout=60000)
        assert not viewer_state(failed)['candidateReady']
        assert_metrics(failed, first, first['variants'][0])
        failed.select_option('#explorer-variant', first['variants'][1]['id'])
        ready_pair(failed, first, first['variants'][1])
        assert not viewer_state(failed)['error']
        failed.close()

        missing_reference = browser.new_page()
        missing_reference.on('pageerror', lambda error: errors.append(str(error)))
        missing_reference.route('**/' + first['reference']['src'], lambda route: route.abort())
        missing_reference.goto(url, wait_until='domcontentloaded')
        missing_reference.locator('#explorer').scroll_into_view_if_needed()
        missing_reference.wait_for_function(f"{STATE}?.candidateReady && !{STATE}?.loading && !!{STATE}?.errors.reference", timeout=60000)
        assert missing_reference.locator('#explorer-mode-delta').is_disabled()
        assert_metrics(missing_reference, first, first['variants'][0])
        missing_reference.close()

        failed_delta = browser.new_page()
        failed_delta.on('pageerror', lambda error: errors.append(str(error)))
        failed_delta.route('**/assets/explorer/delta/**', lambda route: route.abort())
        failed_delta.goto(url, wait_until='domcontentloaded')
        failed_delta.locator('#explorer').scroll_into_view_if_needed()
        ready_pair(failed_delta, first, first['variants'][0])
        failed_delta.click('#explorer-mode-delta')
        failed_delta.wait_for_function(f"!!{STATE}?.errors.delta && !{STATE}?.deltaLoading", timeout=60000)
        assert failed_delta.locator('#explorer-point-count').inner_text() == 'Points unavailable'
        assert failed_delta.locator('#explorer-preview-message').inner_text() == 'Mesh delta unavailable'
        failed_delta.unroute('**/assets/explorer/delta/**')
        failed_delta.click('#explorer-mode-swipe')
        failed_delta.click('#explorer-mode-delta')
        failed_delta.wait_for_function(f"{STATE}?.deltaReady", timeout=60000)
        assert not viewer_state(failed_delta)['errors'].get('delta')
        failed_delta.close()

        # A device context can be lost while a user changes the selected model.
        # Restoring it must load that latest pair without a second selection.
        canvas.evaluate("""el => {
            const extension = el.getContext('webgl2').getExtension('WEBGL_lose_context');
            if (!extension) throw new Error('Context-loss test extension unavailable');
            el.restoreTestContext = () => extension.restoreContext();
            extension.loseContext();
        }""")
        page.wait_for_function(f"!!{STATE}?.errors.webgl")
        restored_variant = first['variants'][1]
        page.select_option('#explorer-variant', restored_variant['id'])
        assert not viewer_state(page)['referenceReady']
        canvas.evaluate('el => el.restoreTestContext()')
        ready_pair(page, first, restored_variant)
        assert not viewer_state(page)['error']
        assert_metrics(page, first, restored_variant)

        snapshot = browser.new_page()
        requests = []
        snapshot.on('request', lambda request: requests.append(request.url))
        snapshot.goto(url + '?snapshot=1', wait_until='domcontentloaded')
        snapshot.locator('#explorer').scroll_into_view_if_needed()
        snapshot.wait_for_timeout(300)
        assert not any('/assets/explorer/' in r or 'comparison-viewer' in r or '/vendor/three' in r or 'model-viewer' in r for r in requests), requests
        assert snapshot.locator('#explorer-stage canvas, model-viewer').count() == 0
        assert not errors, errors
        browser.close()
    report = {'variants_verified': checks, 'delta_pairs_verified': delta_checks,
              'browser_errors': errors, 'shared_camera_keyboard_drag_zoom_reset': 'passed',
              'swipe_pointer_keyboard_touch': 'passed', 'delta_threshold_filter': 'passed',
              'mobile_overflow': 'none', 'snapshot_isolation': 'passed',
              'rapid_selection': 'passed', 'candidate_download_failure_recovery': 'passed',
              'missing_reference_disables_delta': 'passed', 'delta_download_failure_recovery': 'passed',
              'webgl_restore_preserves_selection': 'passed'}
    (output / 'browser-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {len(checks)} object/run pairs, {len(delta_checks)} delta clouds, swipe, camera, mobile, download recovery and snapshot isolation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check(args.url, args.output)
