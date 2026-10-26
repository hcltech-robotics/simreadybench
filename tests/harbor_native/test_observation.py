"""Frozen per-task observation sampling, exact comparison reuse and framing."""
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from simready_eval.prepare import freeze_cameras
from simready_eval.render import validate_cameras


def rig(bounds=((-1,-2,-.5),(1,2,.5)), resolution=(512,512)):
    low,high=map(list,bounds)
    triangles=np.array([[low,high,[low[0],high[1],low[2]]]],float)
    result=freeze_cameras({'triangles':triangles},resolution=resolution[1])
    result['resolution']=list(resolution)
    return result


def test_observation_is_deterministic_idempotent_and_changes_only_its_two_fields():
    from simready_eval.observation import freeze_observation,comparison_identity
    original=rig();before=copy.deepcopy(original);bounds=[[-1,-2,-.5],[1,2,.5]]
    new=freeze_observation(original,bounds,'asset-001')
    assert original==before
    assert new==freeze_observation(original,bounds,'asset-001')
    assert new==freeze_observation(new,{'reference_bounds_m_Z_up':bounds},'asset-001')
    assert {key for key in new if new[key]!=original.get(key)}=={'observation','observation_policy'}
    assert json.dumps(new['views'])==json.dumps(original['views'])
    assert comparison_identity(new)==comparison_identity(original)
    assert new['observation_policy']['comparison_sha256']==comparison_identity(original)
    validate_cameras(new)


def test_task_sampling_uses_explicit_sha256_bits_and_is_independent_of_order():
    from simready_eval.observation import freeze_observation,PROTOCOL_SALT
    original=rig();bounds=[[-1,-2,-.5],[1,2,.5]];ids=[f'asset-{i:03}'for i in range(100)]
    forward={task:freeze_observation(original,bounds,task)['observation_policy']for task in ids}
    reverse={task:freeze_observation(original,bounds,task)['observation_policy']for task in reversed(ids)}
    assert forward==reverse
    assert len({(p['azimuth_degrees'],p['elevation_degrees'])for p in forward.values()})==100
    quadrants=[0]*4
    for task,p in forward.items():
        seed=hashlib.sha256(PROTOCOL_SALT.encode()+b'\0'+task.encode()+b'\0'+(0).to_bytes(4,'big')).digest()
        assert p['seed_sha256']==seed.hex()and p['sample_index']==0
        assert p['azimuth_degrees']==360*((int.from_bytes(seed[:8],'big')>>11)/2**53)
        assert p['elevation_degrees']==15+55*((int.from_bytes(seed[8:16],'big')>>11)/2**53)
        assert 0<=p['azimuth_degrees']<360 and 15<=p['elevation_degrees']<=70
        quadrants[int(p['azimuth_degrees']//90)]+=1
    assert min(quadrants)>10


@pytest.mark.parametrize('bounds,resolution',[
    ([[-1,-2,-3],[1,2,3]],[512,512]),
    ([[-5,-.1,-.2],[5,.1,.2]],[1024,256]),
    ([[-.2,-8,0],[.2,8,0]],[256,1024]),
    ([[1e9,-2e8,3e7],[1e9+.01,-2e8+.02,3e7+.03]],[512,512]),
])
def test_all_reference_corners_have_margin_positive_depth_and_rigid_camera(bounds,resolution):
    from simready_eval.observation import freeze_observation
    original=rig(bounds,resolution)
    new=freeze_observation(original,bounds,'framing-case')
    view=new['observation'];matrix=np.array(view['camera_to_world']);rotation=matrix[:3,:3]
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-14)
    assert np.linalg.det(rotation)==pytest.approx(1)
    corners=np.array(list(itertools.product(*zip(*bounds))))
    local=(corners-matrix[:3,3])@rotation
    half_height=view['orthographic_scale']/2;half_width=half_height*resolution[0]/resolution[1]
    assert np.max(np.abs(local[:,0]))<=half_width/1.15+1e-7
    assert np.max(np.abs(local[:,1]))<=half_height/1.15+1e-7
    depth=-local[:,2]
    assert np.min(depth)>view['clip_start']>0
    assert np.max(depth)<view['clip_end']
    center=np.array(bounds[0])+(np.array(bounds[1])-bounds[0])/2
    target=matrix[:3,3]-matrix[:3,2]*new['observation_policy']['camera_distance_m']
    np.testing.assert_allclose(target,center,atol=1e-7,rtol=0)
    validate_cameras(new)


def test_collision_with_comparison_is_deterministically_resampled():
    from simready_eval.observation import freeze_observation
    bounds=[[-1,-2,-.5],[1,2,.5]];original=rig()
    first=freeze_observation(original,bounds,'collision')
    original['views'][0]={**first['observation'],'id':original['views'][0]['id']}
    second=freeze_observation(original,bounds,'collision')
    assert second['observation_policy']['sample_index']>0
    assert second==freeze_observation(second,bounds,'collision')
    assert second['views']==original['views']
    validate_cameras(second)


@pytest.mark.parametrize('bounds',[[[0,0,0],[0,0,0]],[[1,0,0],[0,1,1]],
                                  [[0,0,0],[float('nan'),1,1]],[[0,0],[1,1]]])
def test_invalid_reference_bounds_are_rejected(bounds):
    from simready_eval.observation import freeze_observation
    with pytest.raises(ValueError):freeze_observation(rig(),bounds,'invalid')


def test_portable_cli_has_no_site_package_dependency_and_is_hash_seed_independent(tmp_path):
    import simready_eval.observation as module
    original=rig();cameras=tmp_path/'old.json';bounds=tmp_path/'bounds.json'
    cameras.write_text(json.dumps(original));bounds.write_text(json.dumps({'reference_bounds_m_Z_up':[[-1,-2,-.5],[1,2,.5]]}))
    results=[]
    for seed in ['7','9001']:
        output=tmp_path/(seed+'.json');env={**os.environ,'PYTHONHASHSEED':seed};env.pop('PYTHONPATH',None)
        process=subprocess.run([sys.executable,'-S',module.__file__,'--cameras',str(cameras),'--bounds',str(bounds),
                                '--task-id','portable-case','--candidate-index','4','--reselection-reason','Reviewed framing.','--output',str(output)],env=env,capture_output=True,text=True,timeout=10)
        assert process.returncode==0,process.stdout+process.stderr
        assert json.loads(process.stdout)['status']=='frozen'
        assert json.loads(output.read_text())['observation_policy']['candidate_index']==4
        results.append(output.read_bytes())
    assert results[0]==results[1]
    assert json.loads(cameras.read_text())==original


def test_default_prepare_camera_factory_routes_task_id_without_changing_legacy_factory():
    from simready_eval.observation import freeze_observation
    observation={'triangles':np.array([[[-1,-2,-.5],[1,2,.5],[-1,2,-.5]]])}
    legacy=freeze_cameras(observation,resolution=64)
    seeded=freeze_cameras(observation,resolution=64,task_id='new-task')
    assert seeded==freeze_observation(legacy,[[-1,-2,-.5],[1,2,.5]],'new-task')
    assert 'observation_policy'not in legacy


def test_candidate_index_reselection_is_frozen_and_preserves_comparison_views():
    from simready_eval.observation import freeze_observation,comparison_identity
    original=rig();bounds=[[-1,-2,-.5],[1,2,.5]]
    default=freeze_observation(original,bounds,'reviewed-task')
    assert default==freeze_observation(original,bounds,'reviewed-task',candidate_index=0)
    selected=freeze_observation(original,bounds,'reviewed-task',candidate_index=3,reselection_reason='Reviewed framing.')
    assert selected['observation_policy']['candidate_index']==3
    assert selected['observation_policy']['seed_sha256']!=default['observation_policy']['seed_sha256']
    assert selected['observation']!=default['observation']
    assert comparison_identity(selected)==comparison_identity(original)
    matrix=np.array(selected['observation']['camera_to_world']);corners=np.array(list(itertools.product(*zip(*bounds))))
    local=(corners-matrix[:3,3])@matrix[:3,:3];half=selected['observation']['orthographic_scale']/2
    assert np.max(np.abs(local[:,:2]))<=half/1.15+1e-12
    assert np.min(-local[:,2])>selected['observation']['clip_start']>0
    assert np.max(-local[:,2])<selected['observation']['clip_end']
    assert selected==freeze_observation(selected,bounds,'reviewed-task',candidate_index=3,reselection_reason='Reviewed framing.')
    validate_cameras(selected)


@pytest.mark.parametrize('index',[-1,2**32,True,1.5,'1',None])
def test_candidate_index_requires_uint32(index):
    from simready_eval.observation import freeze_observation
    with pytest.raises(ValueError,match='candidate_index'):
        freeze_observation(rig(),[[-1,-2,-.5],[1,2,.5]],'task',candidate_index=index)


def test_prepare_without_explicit_cameras_passes_seeded_observation_to_renderer(tmp_path,monkeypatch):
    import simready_eval.prepare as preparation
    from simready_eval.observation import comparison_identity
    package=tmp_path/'package';package.mkdir();source=package/'cube.usda'
    source.write_text('#usda 1.0\n(defaultPrim="Cube"\nmetersPerUnit=1\nupAxis="Z")\ndef Cube "Cube"\n{\n double size=2\n}\n')
    before=source.read_bytes();captured=[]
    def stop_after_camera(source,cameras,*args,**kwargs):
        captured.append(cameras)
        raise RuntimeError('Camera routing inspected without rendering')
    monkeypatch.setattr(preparation.render,'render_asset',stop_after_camera)
    result=preparation.prepare(source,package,tmp_path/'prepared','routing-task')
    assert result['reason']=='preparation_unavailable'
    assert len(captured)==1
    assert captured[0]['observation_policy']['task_id']=='routing-task'
    assert captured[0]['observation_policy']['candidate_index']==0
    assert captured[0]['observation_policy']['reference_bounds_m_Z_up']==[[-1.,-1.,-1.],[1.,1.,1.]]
    legacy=rig([[-1,-1,-1],[1,1,1]])
    assert comparison_identity(captured[0])==comparison_identity(legacy)
    assert source.read_bytes()==before


def test_nonzero_candidate_requires_recorded_review_reason():
    from simready_eval.observation import freeze_observation
    with pytest.raises(ValueError,match='reselection_reason'):
        freeze_observation(rig(),[[-1,-2,-.5],[1,2,.5]],'task',candidate_index=1)


def test_review_reason_is_verbatim_and_does_not_change_seed_or_pose():
    from simready_eval.observation import freeze_observation
    bounds=[[-1,-2,-.5],[1,2,.5]];original=rig();reason='  Full object visible after review.  '
    first=freeze_observation(original,bounds,'review',candidate_index=2,reselection_reason=reason)
    other=freeze_observation(original,bounds,'review',candidate_index=2,reselection_reason='Alternative review note.')
    assert first['observation_policy']['reselection_reason']==reason
    assert first['observation']==other['observation']
    assert first['observation_policy']['seed_sha256']==other['observation_policy']['seed_sha256']
    assert freeze_observation(original,bounds,'review')['observation_policy']['reselection_reason']is None


@pytest.mark.parametrize('reason',['','   ','line\nnext','\x1b[0m','x'*1025,True])
def test_reselection_reason_rejects_blank_control_or_oversized_values(reason):
    from simready_eval.observation import freeze_observation
    with pytest.raises(ValueError,match='reselection_reason'):
        freeze_observation(rig(),[[-1,-2,-.5],[1,2,.5]],'task',candidate_index=1,reselection_reason=reason)
