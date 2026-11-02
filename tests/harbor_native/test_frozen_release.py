"""Guard the published release's executable source provenance."""
from pathlib import Path
import subprocess
import sys


def test_evaluator_source_matches_accepted_release_inventory():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / 'native/tools/verify_frozen_wheel.py')],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"package_files_verified": 31' in result.stdout
