#!/usr/bin/env python3
"""Verify source and optional wheels against the accepted 2.2.0 package inventory."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-wheel', type=Path)
    parser.add_argument('--built-wheel', type=Path)
    args = parser.parse_args()
    native = Path(__file__).resolve().parents[1]
    receipt = json.loads((native / 'frozen-2.2.0.json').read_text())
    expected = receipt['package_files_sha256']
    source = {p.relative_to(native / 'src').as_posix(): digest(p.read_bytes())
              for p in (native / 'src/simready_eval').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if source != expected:
        raise SystemExit('Source differs from the frozen 2.2.0 inventory')
    identities = []
    for label, path in [('frozen', args.frozen_wheel), ('built', args.built_wheel)]:
        if path is None:
            continue
        if label == 'frozen' and digest(path.read_bytes()) != receipt['wheel_sha256']:
            raise SystemExit('Frozen wheel archive digest mismatch')
        with ZipFile(path) as wheel:
            actual = {n: digest(wheel.read(n)) for n in wheel.namelist()
                      if n.startswith('simready_eval/') and not n.endswith('/')}
            if actual != expected:
                raise SystemExit(label + ' wheel package differs from frozen source')
            metadata = BytesParser().parsebytes(wheel.read('simreadybench_eval-2.2.0.dist-info/METADATA'))
            if metadata['Version'] != '2.2.0' or metadata['Name'] != 'simreadybench-eval':
                raise SystemExit(label + ' wheel distribution identity mismatch')
            identities.append((metadata['Requires-Python'], sorted(metadata.get_all('Requires-Dist', [])),
                               sorted(metadata.get_all('Provides-Extra', [])),
                               wheel.read('simreadybench_eval-2.2.0.dist-info/entry_points.txt')))
    if len(identities) == 2 and identities[0] != identities[1]:
        raise SystemExit('Built/frozen wheel dependencies or entry points differ')
    print(json.dumps({'evaluator_version': '2.2.0', 'package_files_verified': len(expected),
                      'frozen_wheel_verified': args.frozen_wheel is not None,
                      'built_wheel_verified': args.built_wheel is not None,
                      'frozen_wheel_sha256': receipt['wheel_sha256']}, sort_keys=True))


if __name__ == '__main__':
    main()
