"""Scoring-version, geometry controls and unchanged motion regression checks."""
from copy import deepcopy
import numpy as np
import pytest
from simready_eval import correspondence as cor, kinematics as kin
from simready_eval.cloud_geometry import VERSION, SAMPLES
from simready_eval.dependencies import runtime_lock


def box():
    points=np.array([[-.5,-.4,0],[.5,-.4,0],[.5,.4,0],[-.5,.4,0],[-.5,-.4,2],[.5,-.4,2],[.5,.4,2],[-.5,.4,2]])
    return points[np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])]


def test_self_match_and_reordered_geometry_have_documented_sampling_floor():
    triangles=box();cfg=cor.settings({'samples':128})
    same,trace=cor.surface(triangles,triangles,cfg,.01)
    reordered,_=cor.surface(triangles,triangles[::-1],cfg,.01)
    assert same['schema']==VERSION and same['score_percent']==100
    assert same['samples_each_direction']==SAMPLES==65536
    assert 97 < reordered['score_percent'] <= 100
    assert all(len(value)==128 for value in trace)
    assert same['protocol']['alignment'].startswith('unchanged ')


def test_candidate_scale_is_not_normalized_away_and_flat_impostor_is_penalized():
    triangles=box();flat=triangles.copy();flat[:,:,1]=0
    cfg=cor.settings({'samples':128})
    scaled,_=cor.surface(triangles,triangles*1.4,cfg,.01)
    plane,_=cor.surface(triangles,flat,cfg,.01)
    assert scaled['score_percent']<25
    assert plane['score_percent']<25
    assert scaled['reference_robust_diagonal_m']==plane['reference_robust_diagonal_m']


def test_geometry_sample_protocol_does_not_change_the_frozen_material_roster():
    triangles=box()
    a,tracea=cor.surface(triangles,triangles,cor.settings({'samples':128}),.01)
    b,traceb=cor.surface(triangles,triangles,cor.settings({'samples':256}),.01)
    assert a==b
    assert len(tracea[0])==128 and len(traceb[0])==256
    assert cor.VERSION=='neutral-usd-correspondence-v1'
    assert cor.SCORING_VERSION=='neutral-usd-correspondence-v2'


def test_motion_landmarks_are_computed_once_without_changing_the_17_state_score(monkeypatch):
    triangles=box();joint={'type':'PhysicsRevoluteJoint','bodies':['base','leaf'],'anchors':[[0,0,0],[0,0,0]],'axes':[[0,0,1],[0,0,1]],'frames':[np.eye(3).tolist(),np.eye(3).tolist()],'enabled':True,'limits':{'state':'bounded','units':'radians','bounds':{'physics:lowerLimit':{'state':'finite','value':0.},'physics:upperLimit':{'state':'finite','value':1.}}}}
    reference={'triangles':np.concatenate((triangles,triangles+[2,0,0])),'triangle_bodies':['base']*12+['leaf']*12,'bodies':{'base':{'enabled':True,'kinematic':True},'leaf':{'enabled':True,'kinematic':False}},'joints':[deepcopy(joint)for _ in range(3)]}
    candidate=deepcopy(reference);calls=[];original=kin._landmarks
    def track(observation,body):calls.append(body);return original(observation,body)
    monkeypatch.setattr(kin,'_landmarks',track)
    result=kin.compare_kinematics(reference,candidate,{'base':'base','leaf':'leaf'},cor.settings(),4.)
    assert result['score_percent']==100 and result['settings']['sweep_states']==17
    assert sorted(calls)==['base','leaf']
    assert all(j['motion_rms_m']==0 for j in result['joints'])
    assert reference.keys()==candidate.keys()
    np.testing.assert_array_equal(reference['triangles'],candidate['triangles'])


def test_scipy_is_pinned_only_for_the_geometry_verifier_role():
    assert 'scipy=='not in runtime_lock('actor')
    assert 'scipy=='not in runtime_lock('feedback')
    assert 'scipy==1.18.0 --hash=sha256:1f55797419e16e7f30cf88ffb3113ce0467f00cfe3f70d5c281730b21769bfc2'in runtime_lock('verifier')


def test_new_release_binds_new_evaluator_without_renaming_frozen_asset_paths():
    from simready_eval.export import release_version
    assert release_version({'schema':'simready-harbor-release-v2-evaluator-2.2.0'})==('v2','2.2.0')
    assert release_version({'schema':'simready-harbor-release-v2'})==('v2','2.0.0')
