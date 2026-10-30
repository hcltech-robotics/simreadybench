"""Real OpenUSD, no inference/feedback service: transactional candidate authoring."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

HELPER = Path(__file__).parents[1] / 'pi_nvidia/candidate.py'
TASK_PYTHON = Path(os.environ.get('PI_TASK_PYTHON_DIR', '/nonexistent')) / 'python'
pytestmark = pytest.mark.skipif(not TASK_PYTHON.exists(), reason='requires genuine task Python with usd-core 26.8')


def invoke(tmp, command=None, *, mode='build'):
    runtime = tmp / 'logs/pi/config/runtime.json'
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({'submission_entrypoint':str(tmp / 'output/submission/asset.usd')}))
    args = [str(TASK_PYTHON), str(HELPER), '--runtime', str(runtime), mode]
    if command is not None:
        args += ['--command', command]
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return result, json.loads(result.stdout)


def author(tmp, body):
    script=tmp/'author.py'
    script.write_text('import os\nfrom pathlib import Path\nfrom pxr import Usd,UsdGeom\np=Path(os.environ["PI_CANDIDATE_DIR"])/"asset.usd"\n'+body+'\n')
    return shlex.join([str(TASK_PYTHON),str(script)])


def good(tmp, name='Cube'):
    return author(tmp, f's=Usd.Stage.CreateNew(str(p)); UsdGeom.Cube.Define(s,"/{name}"); s.GetRootLayer().Save()')


def digest(tmp):
    return hashlib.sha256((tmp/'output/submission/asset.usd').read_bytes()).hexdigest()


def test_create_new_then_exception_does_not_destroy_previous_candidate(tmp_path):
    command=good(tmp_path)+' && '+shlex.join([str(TASK_PYTHON),'-c','import os,pathlib; (pathlib.Path(os.environ["PI_CANDIDATE_DIR"])/"texture.bin").write_bytes(b"original")'])
    assert invoke(tmp_path, command)[0].returncode == 0
    before=digest(tmp_path)
    result, receipt=invoke(tmp_path, author(tmp_path, 's=Usd.Stage.CreateNew(str(p)); (p.parent/"texture.bin").unlink(); raise RuntimeError("synthetic bad USD API")'))
    assert result.returncode != 0 and receipt['promoted'] is False
    assert digest(tmp_path)==before
    assert receipt['reason']=='build_failed'
    assert (tmp_path/'output/submission/texture.bin').read_bytes()==b'original'


def test_invalid_usd_does_not_replace_previous_candidate(tmp_path):
    invoke(tmp_path, good(tmp_path)); before=digest(tmp_path)
    result, receipt=invoke(tmp_path, author(tmp_path, 'p.write_text("this is not USD")'))
    assert result.returncode != 0 and receipt['reason']=='invalid_openusd'
    assert digest(tmp_path)==before


def test_valid_empty_stage_is_explicitly_allowed(tmp_path):
    invoke(tmp_path, good(tmp_path)); before=digest(tmp_path)
    result, receipt=invoke(tmp_path, author(tmp_path, 's=Usd.Stage.CreateNew(str(p)); s.GetRootLayer().Save()'))
    assert result.returncode == 0 and receipt['promoted'] is True
    assert receipt['usd_valid'] is True and digest(tmp_path)!=before
    assert len(list((tmp_path/'logs/pi/candidates/verified').iterdir()))==2


def test_missing_reference_is_not_an_added_dependency_or_quality_gate(tmp_path):
    command=author(tmp_path, 's=Usd.Stage.CreateNew(str(p)); s.DefinePrim("/Asset").GetReferences().AddReference("missing.usd"); s.GetRootLayer().Save()')
    result,receipt=invoke(tmp_path,command)
    assert result.returncode==0 and receipt['usd_valid'] is True


def test_success_replaces_whole_package_and_guard_restores_unpromoted_changes(tmp_path):
    command=good(tmp_path,'First')
    command+=' && '+shlex.join([str(TASK_PYTHON),'-c','import os,pathlib; (pathlib.Path(os.environ["PI_CANDIDATE_DIR"])/"texture.bin").write_bytes(b"first")'])
    assert invoke(tmp_path,command)[0].returncode==0
    command=good(tmp_path,'Second')
    command+=' && '+shlex.join([str(TASK_PYTHON),'-c','import os,pathlib; (pathlib.Path(os.environ["PI_CANDIDATE_DIR"])/"texture.bin").write_bytes(b"second")'])
    assert invoke(tmp_path,command)[0].returncode==0
    before=digest(tmp_path)
    final=tmp_path/'output/submission'
    (final/'asset.usd').write_text('#usda 1.0\n')
    (final/'texture.bin').unlink()
    result,receipt=invoke(tmp_path,mode='guard')
    assert result.returncode==0 and receipt['action']=='restored_last_valid'
    assert digest(tmp_path)==before and (final/'texture.bin').read_bytes()==b'second'


@pytest.mark.parametrize("failure", ["exchange", "pointer"])
def test_failed_atomic_exchange_preserves_final_and_store_pointer(tmp_path, failure):
    invoke(tmp_path,good(tmp_path));before=digest(tmp_path)
    script=tmp_path/'failed-promotion.py'
    runtime=tmp_path/'logs/pi/config/runtime.json'
    script.write_text(f'''import importlib.util,sys
s=importlib.util.spec_from_file_location("candidate",{str(HELPER)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def fail(*args): raise OSError("synthetic exchange failure")
if {failure!r}=="exchange": m.exchange=fail
else:
 original=m.atomic_json
 def fail_pointer(path,value):
  if path.name=="current.json": raise OSError("synthetic pointer failure")
  original(path,value)
 m.atomic_json=fail_pointer
sys.argv=["candidate","--runtime",{str(runtime)!r},"build","--command",{good(tmp_path,'Replacement')!r}]
m.main()
''')
    result=subprocess.run([str(TASK_PYTHON),str(script)],capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and digest(tmp_path)==before
    assert json.loads(result.stdout)['promoted'] is False
    assert invoke(tmp_path,mode='guard')[1]['action']=='unchanged'


def test_build_that_ignores_staging_restores_prior_whole_package(tmp_path):
    invoke(tmp_path,good(tmp_path));before=digest(tmp_path)
    final=tmp_path/'output/submission/asset.usd'
    command=shlex.join([str(TASK_PYTHON),'-c',f'from pxr import Usd; Usd.Stage.CreateNew({str(final)!r})'])
    result,receipt=invoke(tmp_path,command)
    assert result.returncode!=0 and receipt['reason']=='build_wrote_final_instead_of_staging'
    assert digest(tmp_path)==before
