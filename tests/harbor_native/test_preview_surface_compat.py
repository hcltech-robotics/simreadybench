"""Real USD graphs exercise standard shading semantics without rewriting assets."""
import hashlib
import numpy as np
import pytest
from pxr import Sdf, UsdGeom, UsdShade
from simready_eval import render
from tests.harbor_native.test_render import textured_quad, write_cube


def extract(source, output):
    output.mkdir()
    return render._extract_scene(source, output, np.eye(4))


@pytest.mark.parametrize('workflow', [None, 0])
def test_inactive_specular_color_preserves_identical_scene(tmp_path, workflow):
    source=tmp_path/'asset.usda';stage=write_cube(source)
    baseline,support=extract(source,tmp_path/'baseline')
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set((.15,.2,.3))
    if workflow is not None:shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(workflow)
    stage.GetRootLayer().Save();before=source.read_bytes()
    scene,support=extract(source,tmp_path/'inactive')
    assert support['status']=='supported',support
    assert scene==baseline
    assert source.read_bytes()==before


def test_active_specular_workflow_stays_explicitly_unknown(tmp_path):
    source=tmp_path/'asset.usda';stage=write_cube(source)
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set((.15,.2,.3))
    stage.GetRootLayer().Save()
    scene,support=extract(source,tmp_path/'active')
    assert support['status']=='unknown'


@pytest.mark.parametrize('fallback', [None,(.2,.4)])
def test_missing_primvar_uses_reader_fallback_before_transform(tmp_path, fallback):
    source=tmp_path/'asset.usda';stage=textured_quad(source,mapping=True)
    stage.GetPrimAtPath('/Quad').RemoveProperty('primvars:st')
    if fallback is not None:UsdShade.Shader(stage.GetPrimAtPath('/Mat/UV')).CreateInput('fallback',Sdf.ValueTypeNames.Float2).Set(fallback)
    stage.GetRootLayer().Save();before=source.read_bytes()
    scene,support=extract(source,tmp_path/'fallback')
    assert support['status']=='supported',support
    expected=[.25,.5] if fallback is None else [-.95,.9]
    for row in scene['triangle_materials']:assert np.allclose(row['uv']['diffuseColor'],[expected]*3)
    assert source.read_bytes()==before


def test_inherited_constant_primvar_is_used_instead_of_fallback(tmp_path):
    source=tmp_path/'asset.usda';stage=textured_quad(source)
    mesh=stage.GetPrimAtPath('/Quad');mesh.RemoveProperty('primvars:st')
    # A constant primvar on an ancestor is legal; a geometry-local reader must find it.
    root=UsdGeom.Xform.Define(stage,'/Root')
    UsdGeom.PrimvarsAPI(root).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.constant).Set([(.3,.6)])
    stage.GetRootLayer().Save()
    # Test the same real reader on the ancestor prim; inherited binding is covered
    # with a child geometry copied into the new namespace.
    Sdf.CopySpec(stage.GetRootLayer(),'/Quad',stage.GetRootLayer(),'/Root/Quad')
    stage.RemovePrim('/Quad');stage.GetRootLayer().Save()
    scene,support=extract(source,tmp_path/'inherited')
    assert support['status']=='supported',support
    for row in scene['triangle_materials']:assert np.allclose(row['uv']['diffuseColor'],[[.3,.6]]*3)


def test_opacity_threshold_is_preserved_as_a_shader_field(tmp_path):
    source=tmp_path/'asset.usda';stage=write_cube(source,opacity=.1)
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(.1)
    stage.GetRootLayer().Save();before=source.read_bytes()
    scene,support=extract(source,tmp_path/'cutout')
    assert support['status']=='supported',support
    assert all(m['opacityThreshold']==pytest.approx(.1) for m in scene['materials'])
    assert source.read_bytes()==before


def test_two_readers_of_missing_primvar_keep_distinct_fallbacks(tmp_path):
    source=tmp_path/'asset.usda';stage=textured_quad(source)
    stage.GetPrimAtPath('/Quad').RemoveProperty('primvars:st')
    uv=UsdShade.Shader(stage.GetPrimAtPath('/Mat/UV'))
    uv.CreateInput('fallback',Sdf.ValueTypeNames.Float2).Set((.2,.4))
    second=UsdShade.Shader.Define(stage,'/Mat/SecondUV');second.CreateIdAttr('UsdPrimvarReader_float2')
    second.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    second.CreateInput('fallback',Sdf.ValueTypeNames.Float2).Set((.7,.8))
    second.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex=UsdShade.Shader.Define(stage,'/Mat/SecondTex');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tmp_path/'colour.png')))
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(second.ConnectableAPI(),'result')
    tex.CreateOutput('r',Sdf.ValueTypeNames.Float)
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),'r')
    stage.GetRootLayer().Save()
    scene,support=extract(source,tmp_path/'distinct')
    assert support['status']=='supported',support
    for row in scene['triangle_materials']:
        assert np.allclose(row['uv']['diffuseColor'],[[.2,.4]]*3)
        assert np.allclose(row['uv']['roughness'],[[.7,.8]]*3)
