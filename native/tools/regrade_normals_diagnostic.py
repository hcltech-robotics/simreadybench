"""Reproject one frozen complete report; never dispatch SDK, renders or models.

Usage: PYTHONPATH=native/src python native/tools/regrade_normals_diagnostic.py \
  --report ORIGINAL/verifier/report.json --candidate ORIGINAL/artifacts/candidate \
  --output NEW_VERSION
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from simready_eval import contract, formal, validator


def regrade(report, candidate_root):
    corrected = deepcopy(report)
    nvidia = corrected['formal_report']['nvidia_validator_report']
    execution = nvidia['rule_execution']
    inventory = nvidia['inventory']
    if nvidia.get('engine_dispatch_completed') is not True or execution.get('completed') is not True:
        raise ValueError('Complete frozen SDK execution evidence required')
    before = formal.score_validation(nvidia)
    if before != report['formal_report']['simready']:
        raise ValueError('Original formal grade must reproduce from saved evidence')
    if before['unknown_count'] != 1:
        raise ValueError('Exactly one unknown mandatory requirement required')
    # Check every immutable package member against the verifier frozen snapshot.
    expected = report['candidate']['files']
    root = Path(candidate_root).resolve()
    actual_files = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    if actual_files != sorted(row['path'] for row in expected):
        raise ValueError('Frozen candidate file roster changed')
    for row in expected:
        path = root / row['path']
        if path.resolve().parent != root and root not in path.resolve().parents:
            raise ValueError('Candidate path escapes package')
        if path.stat().st_size != row['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Immutable candidate hash mismatch: ' + row['path'])
    entrypoint = next(row for row in expected if row['path'] == report['candidate']['entrypoint'])
    if nvidia.get('asset_sha256') != entrypoint['sha256']:
        raise ValueError('Saved SDK report does not identify the frozen entrypoint')
    issues = [i for key in ('issues', 'soft_issues', 'warnings', 'infos') for i in nvidia[key]]
    if any(not isinstance(i, dict) for i in issues):
        raise ValueError('Malformed raw issues')
    kwargs = dict(versions=inventory['versions'], sources=execution['sdk_source_sha256'])
    matches = [i for i in issues if validator.is_candidate_normals_diagnostic(i, **kwargs)]
    if not matches:
        raise ValueError('No exact pinned candidate diagnostic')
    projected = validator.requirement_outcomes(inventory['mandatory_requirements'], execution['buckets'], issues, **kwargs)
    original_rows = execution['requirements']
    if len(projected) != len(original_rows):
        raise ValueError('Incomplete recorded requirements')
    for old, new in zip(original_rows, projected):
        expected_row = deepcopy(old)
        if old['code'] == 'com.nvidia.usd.VG.029':
            if old['disposition'] != 'infrastructure_error' or new['disposition'] != 'failed':
                raise ValueError('Normals requirement has unresolved execution failures')
            expected_row['disposition'] = 'failed'
        if expected_row != new:
            raise ValueError('Unrelated or malformed recorded requirement: ' + old['code'])
    execution['requirements'] = projected
    execution['diagnostic_mapping'] = validator.NORMALS_DIAGNOSTIC_ADAPTER_ID
    def matched_error(error):
        return (isinstance(error, dict) and error.get('code') == 'validator_rule_execution_error'
                and error.get('issue') in matches)
    errors = nvidia['infrastructure_errors']
    if not isinstance(errors, list) or any(not isinstance(e, dict) for e in errors):
        raise ValueError('Malformed infrastructure errors')
    if sum(matched_error(e) for e in errors) != len(matches):
        raise ValueError('Diagnostic infrastructure entries do not match raw findings')
    nvidia['infrastructure_errors'] = [e for e in errors if not matched_error(e)]
    nvidia['passed'] = False
    nvidia['outcome'] = 'infrastructure_error' if nvidia['infrastructure_errors'] else 'failure'
    score = formal.score_validation(nvidia)
    if score['unknown_count'] or score['passed_count'] != before['passed_count'] or score['failed_count'] != before['failed_count'] + 1:
        raise ValueError('Unexpected corrected formal score')
    f = corrected['formal_report']
    f.update(outcome='incomplete', score_percent=score['score_percent'], reward=score['score_percent'],
             simready=score, optional_score=score['optional_score'])
    f['infrastructure_errors'] = [e for e in f['infrastructure_errors'] if not matched_error(e)]
    f['coverage'].update(measured=46, execution_fraction=1.0)
    corrected['scores']['formal'] = contract.component(score['score_percent'], coverage=1,
        lower=score['score_lower_bound'], upper=score['score_upper_bound'], evidence=score)
    corrected['outcome'] = 'complete' if all(corrected['scores'][d]['score_percent'] is not None for d in contract.DIMENSIONS) else 'partial'
    assert nvidia['issues'] == report['formal_report']['nvidia_validator_report']['issues']
    assert all(corrected['scores'][d] == report['scores'][d] for d in corrected['scores'] if d != 'formal')
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = args.report.read_bytes()
    report = regrade(json.loads(original), args.candidate)
    report['grading_correction'] = {
        'id': validator.NORMALS_DIAGNOSTIC_ADAPTER_ID,
        'original_report_sha256': hashlib.sha256(original).hexdigest(),
        'method': 'deterministic_reprojection_of_frozen_SDK_report_no_execution',
        'regrade_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'validator_source_sha256': hashlib.sha256(Path(validator.__file__).read_bytes()).hexdigest(),
    }
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    contract.write_outputs(report, args.output)
    assert args.report.read_bytes() == original
    print(json.dumps({'output': str(args.output), 'formal': report['scores']['formal']['score_percent'],
                      'reward': contract.reward_record(report)}, indent=2))


if __name__ == '__main__':
    main()
