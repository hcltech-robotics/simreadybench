"""Diagnostic cache staging preserves the real package admission boundary."""
import importlib.util
import os
from pathlib import Path

import pytest


def helper():
    path=Path(__file__).parents[2]/'native/tools/validate_gpu_capture.py'
    spec=importlib.util.spec_from_file_location('validate_gpu_capture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_staging_breaks_cache_hardlinks_without_relaxing_regular_boundary(tmp_path):
    tool=helper()
    from simready_eval.cli import _regular
    source=tmp_path/'cache';source.mkdir()
    (source/'LICENSE.txt').write_text('Original license bytes')
    os.link(source/'LICENSE.txt',source/'LICENSE-copy.txt')
    with pytest.raises(ValueError): _regular(source,'LICENSE.txt')
    target=tmp_path/'stage'
    tool.stage_regular(source,target,maximum_bytes=1024)
    assert _regular(target,'LICENSE.txt').read_bytes()==(source/'LICENSE.txt').read_bytes()
    assert (target/'LICENSE.txt').stat().st_nlink==1
    assert (target/'LICENSE-copy.txt').stat().st_nlink==1
    assert (source/'LICENSE.txt').stat().st_nlink==2


@pytest.mark.parametrize('fault',['symlink','executable','byte_bound','entry_bound'])
def test_cache_staging_rejects_unsafe_or_unbounded_input(tmp_path,fault):
    tool=helper();source=tmp_path/'cache';source.mkdir()
    file=source/'data';file.write_bytes(b'123')
    if fault=='symlink': (source/'alias').symlink_to(file)
    if fault=='executable': file.chmod(0o755)
    with pytest.raises(ValueError):
        tool.stage_regular(source,tmp_path/'stage',maximum_bytes=2 if fault=='byte_bound' else 1024,
                           maximum_entries=0 if fault=='entry_bound' else 100)


def test_result_inspection_rejects_cli_success_without_measured_render():
    tool=helper()
    from simready_eval import contract
    report={'outcome':'complete','errors':[],'candidate_invalid':False,
            'scores':{name:contract.component(100,coverage=1) for name in contract.DIMENSIONS},
            'render':{'status':'unknown'}}
    with pytest.raises(ValueError):tool.check_result(report,contract.reward_record(report),{})


@pytest.mark.parametrize('fault',['inside_task','inside_capture','inside_candidate','symlink_parent'])
def test_rejected_output_does_not_write_into_protected_inputs(tmp_path,fault):
    tool=helper()
    task=tmp_path/'task';capture=tmp_path/'capture';candidate=tmp_path/'candidate'
    for directory in (task,capture,candidate):directory.mkdir()
    if fault=='symlink_parent':
        alias=tmp_path/'alias';alias.symlink_to(task,target_is_directory=True);output=alias/'leak'
    else:output={'inside_task':task,'inside_capture':capture,'inside_candidate':candidate}[fault]/'leak'
    before={str(p) for p in tmp_path.rglob('*')}
    with pytest.raises(SystemExit):
        tool.main(['--task',str(task),'--receipt',str(capture/'diagnostic-receipt.json'),
          '--capture-root',str(capture),'--submission',str(candidate),'--output',str(output)])
    assert {str(p) for p in tmp_path.rglob('*')}==before
