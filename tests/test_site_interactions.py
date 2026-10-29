"""Published HTML must not pretend that a serialized WebGL canvas is live."""
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = {}; self.canvases = 0; self.status = []; self.in_status = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get('id'): self.ids[attributes['id']] = attributes
        if tag == 'canvas': self.canvases += 1
        if attributes.get('id') == 'explorer-status': self.in_status = True

    def handle_endtag(self, tag):
        if tag == 'p': self.in_status = False

    def handle_data(self, data):
        if self.in_status: self.status.append(data)


def test_static_explorer_has_no_saved_canvas_or_false_ready_state():
    parsed = Elements(); parsed.feed((ROOT/'site/index.html').read_text())
    assert parsed.canvases == 0
    assert parsed.ids['explorer']['data-state'] == 'idle'
    assert 'ready' not in ''.join(parsed.status)
    for control in ['object','variant','mode-swipe','mode-delta','reset','swipe','threshold']:
        assert 'disabled' in parsed.ids['explorer-'+control]


def test_compact_label_layout_and_preview_reset_in_node():
    node = shutil.which('node')
    if not node: pytest.skip('Node.js is not installed')
    subprocess.run([node,'--test',str(ROOT/'tests/test_site_interactions.mjs')],check=True,capture_output=True,text=True)
