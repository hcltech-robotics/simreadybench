import hashlib
import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade
from simready_eval.usd import observe
from simready_eval.correspondence import _appearance_component


def material_fixture(path, opacity, *, as_observation=False):
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, '/Cube')
    mat = UsdShade.Material.Define(stage, '/Material')
    shader = UsdShade.Shader.Define(stage, '/Material/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    if opacity != 'missing':
        inp = shader.CreateInput('opacity', Sdf.ValueTypeNames.Float)
        if opacity == 'connected':
            unknown = UsdShade.Shader.Define(stage, '/Material/Unknown')
            unknown.CreateIdAttr('UnsupportedOpacity')
            unknown.CreateOutput('result',Sdf.ValueTypeNames.Float)
            inp.ConnectToSource(unknown.ConnectableAPI(),'result')
        elif opacity != 'unvalued':
            inp.Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    observation=observe(path)
    value=observation['materials'][0]
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    return observation if as_observation else value


@pytest.mark.parametrize('opacity, expected',[('missing',1),('unvalued',1),(0.,0),(.35,.35),(1.,1)])
def test_opacity_fallback_and_authored_values(tmp_path,opacity,expected):
    value=material_fixture(tmp_path/'material.usda',opacity)
    assert value['visual:opacity']==pytest.approx(expected)


def test_connected_unsupported_opacity_is_unmeasured(tmp_path):
    value=material_fixture(tmp_path/'material.usda','connected')
    assert 'visual:opacity' not in value


def test_opacity_alone_cannot_qualify_full_appearance():
    result=_appearance_component([{'sample':0,'feature':'visual:assignment','passed':True}, {'sample':0,'feature':'visual:opacity','passed':True}])
    assert result['status']=='unknown'
    assert result['score_percent'] is None


def test_property_diagnostics_survive_unknown_full_appearance():
    rows=[{'sample':0,'feature':'visual:assignment','passed':True}, {'sample':0,'feature':'visual:opacity','passed':True}]
    result=_appearance_component(rows)
    assert result['property_diagnostics']['visual:opacity']=={'measured':1,'earned':1}
    assert result['measured']==0
    rows += [{'sample':0,'feature':key,'passed':True} for key in ('visual:diffuseColor','visual:roughness','visual:metallic')]
    assert _appearance_component(rows)['score_percent']==100


def test_unsupported_expected_property_cannot_qualify_candidate_appearance():
    rows=[{'sample':0,'feature':'visual:assignment','passed':True}]
    rows += [{'sample':0,'feature':key,'passed':True,'supported':key!='visual:roughness'} for key in ('visual:diffuseColor','visual:roughness','visual:metallic','visual:opacity')]
    result=_appearance_component(rows)
    assert result['status']=='unknown'
    assert result['property_diagnostics']['visual:roughness']['measured']==0



def test_opaque_default_matches_explicit_opaque_diagnostics(tmp_path):
    from simready_eval.correspondence import compare, settings
    omitted=material_fixture(tmp_path/'omitted.usda','missing',as_observation=True)
    explicit=material_fixture(tmp_path/'explicit.usda',1.,as_observation=True)
    cfg=settings({'samples':128})
    control=compare(explicit,explicit,cfg)
    result=compare(explicit,omitted,cfg)
    assert result['components']['shape']==control['components']['shape']
    assert result['components']['materials']==control['components']['materials']
