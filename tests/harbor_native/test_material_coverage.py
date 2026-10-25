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


def test_16bit_texture_diagnostics_do_not_report_truncated_values(tmp_path):
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    texture = tmp_path / 'precise.png'
    texture.write_bytes(b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 16, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(b'\x00' + struct.pack('>HHH', 32769, 1001, 65534)))
        + chunk(b'IEND', b''))
    path = tmp_path / 'asset.usda'
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, '/Mesh')
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
        'faceVarying').Set([(0, 0), (1, 0), (0, 1)])
    material = UsdShade.Material.Define(stage, '/Material')
    shader = UsdShade.Shader.Define(stage, '/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    tex = UsdShade.Shader.Define(stage, '/Material/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('precise.png'))
    reader = UsdShade.Shader.Define(stage, '/Material/UV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    observation = observe(path)
    value = observation['materials'][0]
    assert 'visual:diffuseColor' not in value
    assert value['visual:opacity'] == 1
