#!/usr/bin/env python3
"""Verify the published Pareto chart with a real browser, including fallbacks."""
import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


def ready(page, url):
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#pareto-root').dataset.loadState === 'ready'", timeout=90000)
    page.wait_for_function("['ready', 'error'].includes(document.querySelector('#leaderboard-table-region').dataset.loadState)", timeout=90000)
    page.locator("#pareto").scroll_into_view_if_needed()


def check(url, output):
    url = url.rstrip("/") + "/"
    output.mkdir(parents=True, exist_ok=True)
    errors, combinations = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        ready(page, url)
        assert page.locator('#pareto-root [data-pareto-score="all"]').get_attribute("aria-pressed") == "true"
        rows = page.locator("#pareto-table-body tr").count()
        assert rows > 0
        axes = page.locator("#pareto-x option").evaluate_all("options => options.map(option => option.value)")
        assert len(axes) >= 3
        for axis in axes:
            page.select_option("#pareto-x", axis)
            for score in ["all", "formal", "visual", "kinematic"]:
                button = page.locator(f'#pareto-root [data-pareto-score="{score}"]')
                button.click()
                assert button.get_attribute("aria-pressed") == "true"
                points = page.locator("#pareto-plot .pareto-point")
                if score != "all":
                    assert all(item == score for item in points.evaluate_all("nodes => nodes.map(n => n.dataset.score)"))
                assert not page.locator('#pareto-plot [transform*="NaN"], #pareto-plot [d*="NaN"]').count()
                if points.count():
                    points.first.focus()
                    assert page.locator("#pareto-tooltip").is_visible()
                    tooltip = page.locator("#pareto-tooltip").inner_text()
                    assert "per asset" in tooltip and "reconstructions" in tooltip
                    assert page.locator("#pareto-tooltip a").get_attribute("href").startswith("https://")
                    assert not re.search(r"Orichalcum(?:@|\s+)[0-9a-f]{9,64}\b", tooltip, re.I)
                    page.keyboard.press("Escape")
                    assert page.locator("#pareto-tooltip").is_hidden()
                combinations.append({"axis": axis, "score": score, "points": points.count()})
            page.locator('#pareto-root [data-pareto-score="all"]').click()
            if page.locator("#pareto-subsets-label").is_visible():
                before = page.locator("#pareto-plot .pareto-point").count()
                page.locator("#pareto-subsets").check()
                assert page.locator("#pareto-plot .pareto-point").count() >= before
                page.locator("#pareto-subsets").uncheck()
        page.locator("#pareto-separate").check()
        separated = page.locator("#pareto-plot svg").bounding_box()["height"]
        page.locator("#pareto-separate").uncheck()
        assert separated > page.locator("#pareto-plot svg").bounding_box()["height"]
        page.select_option("#pareto-x", "elapsed_seconds_per_asset" if "elapsed_seconds_per_asset" in axes else axes[0])
        assert page.locator("#pareto-plot .pareto-frontier").count() > 0
        assert page.locator('#pareto-plot .pareto-point[data-frontier="false"][data-complete="true"]').count() > 0
        page.locator("#pareto").screenshot(path=str(output / "desktop.png"))

        # The new section must not disrupt the neighbouring shared 3D viewer.
        page.locator("#explorer").scroll_into_view_if_needed()
        page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer?.getState().candidateReady", timeout=60000)
        assert page.locator("#explorer-stage canvas").count() == 1
        page.locator("#explorer-swipe").focus()
        page.keyboard.press("Home")
        page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer.getState().split === 0")
        page.locator("#explorer-mode-delta").click()
        page.wait_for_function("document.querySelector('#explorer-stage').comparisonViewer.getState().deltaReady", timeout=60000)
        page.locator("#explorer-mode-swipe").click()

        # Exercise the exact rendered-HTML lifecycle used by GitHub Pages.
        ready(page, url + "?snapshot=1")
        assert page.locator("#explorer-stage canvas").count() == 0
        captured = page.content()
        svg_count = page.locator("#pareto-plot svg").count()
        snapshot_context = browser.new_context()
        snapshot_context.route(re.compile(r"/\?captured=1$"), lambda route: route.fulfill(content_type="text/html", body=captured))
        snapshot = snapshot_context.new_page()
        ready(snapshot, url + "?captured=1")
        assert snapshot.locator("#pareto-table-body tr").count() == rows
        assert snapshot.locator("#pareto-plot svg").count() == svg_count == 1
        snapshot_context.close()

        # A failed refresh preserves the rendered chart and its numeric table.
        fallback_context = browser.new_context()
        fallback_context.route(re.compile(r"/\?captured=1$"), lambda route: route.fulfill(content_type="text/html", body=captured))
        fallback_context.route("**/assets/pareto.json", lambda route: route.abort())
        fallback = fallback_context.new_page()
        fallback.goto(url + "?captured=1", wait_until="domcontentloaded")
        fallback.wait_for_function("document.querySelector('#pareto-root').dataset.loadState === 'error'")
        assert fallback.locator("#pareto-table-body tr").count() == rows
        assert fallback.locator("#pareto-plot svg").count() == 1
        fallback_context.close()

        mobile_context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, device_scale_factor=1)
        mobile = mobile_context.new_page()
        ready(mobile, url)
        mobile.select_option("#pareto-x", "elapsed_seconds_per_asset" if "elapsed_seconds_per_asset" in axes else axes[0])
        point = mobile.locator("#pareto-plot .pareto-point").first
        point.scroll_into_view_if_needed()
        target_id = point.get_attribute("data-point-id")
        target_label = point.get_attribute("aria-label").split(". Formal fidelity:")[0]
        box = point.bounding_box()
        mobile.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        assert mobile.locator("#pareto-tooltip").is_visible()
        # Near-identical measurements must stay at their true coordinates while
        # remaining separately selectable on touch screens.
        nearby = mobile.locator("#pareto-nearby")
        assert nearby.is_visible()
        assert nearby.locator("option").count() >= 2
        nearby.select_option(target_id)
        assert mobile.locator("#pareto-tooltip strong").inner_text() == target_label
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        mobile.locator("#pareto-tooltip button").click()
        mobile.locator("#pareto").screenshot(path=str(output / "mobile.png"))
        mobile_context.close()
        assert not errors, errors
        browser.close()
    receipt = {"url": url, "configurations": rows, "combinations": combinations,
               "snapshot_and_fallback": "passed", "mobile_touch": "passed", "javascript_errors": errors}
    (output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.url, args.output)
