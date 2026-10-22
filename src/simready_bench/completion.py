"""Separate finished authoring from complete benchmark measurements."""

from simready_bench.composites import scores


def evaluation_completion(row, *, fidelity_enabled=True, captures_enabled=False, views=()):
    missing = []
    if row.get('status') != 'completed':
        missing.append('authoring')
    fidelity = row.get('fidelity') or {}
    if fidelity_enabled:
        if fidelity.get('status') != 'measured':
            missing.append('fidelity_worker')
        else:
            for name, metric in scores(row).items():
                if name != 'schema' and metric.get('status') not in ('measured', 'not_applicable'):
                    missing.append(name)
    capture_status = 'disabled'
    if captures_enabled:
        if fidelity.get('status') == 'measured' and (
            fidelity.get('candidate_invalid') is True or fidelity.get('alignment') is None
        ):
            capture_status = 'not_applicable_no_alignable_candidate'
        else:
            illustration = row.get('illustrations') or {}
            images = row.get('images') or illustration.get('images') or []
            actual = {item.get('view') for item in images}
            expected = {role + ' / ' + view for role in ('reference', 'candidate') for view in views}
            capture_status = 'complete' if illustration.get('status') == 'rendered' and expected <= actual else 'incomplete'
            if capture_status == 'incomplete':
                missing.append('captures')
    return {
        'schema': 'simready-evaluation-completion/v1',
        'complete': not missing,
        'authoring_complete': row.get('status') == 'completed',
        'missing': missing,
        'captures': capture_status,
        'recovery_scope': 'evaluation_only' if missing and 'authoring' not in missing else None,
    }


def completion_summary(rows, planned):
    authored = sum(row.get('status') == 'completed' for row in rows)
    known = any('evaluation_completion' in row or 'fidelity' in row for row in rows)
    evaluated = 0
    for row in rows:
        state = row.get('evaluation_completion')
        if state is None and 'fidelity' in row:
            illustrations = row.get('illustrations') or {}
            state = evaluation_completion(
                row, fidelity_enabled=row['fidelity'].get('status') != 'disabled',
                captures_enabled=illustrations.get('renderer', 'none') != 'none',
                views=list((illustrations.get('rig') or {}).get('views', ['isometric', 'front', 'side'])),
            )
        evaluated += (state or {}).get('complete') is True
    return {
        'authored_conditions': authored,
        'evaluated_conditions': evaluated if known else None,
        'benchmark_complete': authored == planned and (evaluated == planned if known else True),
    }
