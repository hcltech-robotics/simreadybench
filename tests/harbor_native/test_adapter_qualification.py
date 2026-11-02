"""Historical references remain sealed while the qualified runtime advances."""
from copy import deepcopy
import pytest
from simready_eval import qualification as q,render
from tests.harbor_native.test_render_qualification import gpu_receipt


def test_qualified_bridge_pin_matches_actual_source(monkeypatch):
    real_hash=render._hash
    monkeypatch.setattr(q.shutil,'which',lambda _: '/qualified/blender')
    monkeypatch.setattr(render,'_hash',lambda path:q.CPU_BINARY if str(path)=='/qualified/blender'else real_hash(path))
    assert q.evaluator_identity()['bridge_code_sha256']==q.CPU_BRIDGE


def historical(monkeypatch):
    identity={'recipe_sha256':q.CPU_RECIPE,'bridge_code_sha256':q.CPU_BRIDGE,'renderer_binary_sha256':q.CPU_BINARY}
    monkeypatch.setattr(q,'evaluator_identity',lambda:dict(identity))
    raw=gpu_receipt();current=q.compatibility(raw)
    old={**current,'qualification':q.HISTORICAL_QUALIFICATION,'qualification_evidence':dict(q.HISTORICAL_EVIDENCE),
         'candidate_evaluator':{**identity,'bridge_code_sha256':q.HISTORICAL_CPU_BRIDGE}}
    return raw,current,old


def test_exact_historical_seal_admits_current_qualified_adapter_without_mutation(monkeypatch):
    raw,current,old=historical(monkeypatch);before=deepcopy(old)
    assert q.validate_compatibility(old,raw)==current
    assert old==before


@pytest.mark.parametrize('fault',['binary','bridge','recipe','evidence','qualification'])
def test_altered_historical_seal_is_rejected(monkeypatch,fault):
    raw,current,old=historical(monkeypatch)
    if fault in('binary','bridge','recipe'):
        key={'binary':'renderer_binary_sha256','bridge':'bridge_code_sha256','recipe':'recipe_sha256'}[fault]
        old['candidate_evaluator'][key]='0'*64
    elif fault=='evidence':old['qualification_evidence']['paired_calibration_sha256']='0'*64
    else:old['qualification']='unknown'
    with pytest.raises(ValueError,match='compatibility mismatch'):q.validate_compatibility(old,raw)
