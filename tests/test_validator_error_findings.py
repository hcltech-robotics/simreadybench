"""Requirement-scoped SDK errors are asset failures, not execution failures."""

from simready_bench.engine.validator import requirement_outcomes


def outcomes(issues):
    return requirement_outcomes(
        [{'code': 'VG.029', 'rule': 'sdk.Normals'}],
        [{'declared_requirement_codes': ['VG.029'], 'enabled_rules': ['sdk.Normals'],
          'method_calls': [{'rule': 'sdk.Normals', 'method': 'CheckPrim', 'count': 1}]}],
        issues,
    )[0]


def test_requirement_scoped_error_is_a_failed_requirement():
    result = outcomes([{'rule_id': 'VG.029', 'rule': 'sdk.Normals', 'severity': 'error'}])
    assert result['disposition'] == 'failed'
    assert result['rule_invoked']


def test_unscoped_execution_error_keeps_requirement_unknown():
    result = outcomes([{'rule_id': None, 'rule': 'sdk.Normals', 'severity': 'error'}])
    assert result['disposition'] == 'infrastructure_error'


def test_execution_error_is_not_hidden_by_an_asset_finding():
    result = outcomes([
        {'rule_id': 'VG.029', 'rule': 'sdk.Normals', 'severity': 'error'},
        {'rule_id': None, 'rule': 'sdk.Normals', 'severity': 'error'},
    ])
    assert result['disposition'] == 'infrastructure_error'
