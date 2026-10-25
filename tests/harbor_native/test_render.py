"""Neutral camera, coverage and real CPU Blender regressions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from simready_eval.render import compare_views, render_asset, validate_cameras


def cameras(size=64):
    # Blender camera basis, looking toward origin from +Z; world is Z-up.
    return {'schema':'simready-cameras-v1','resolution':[size,size],
            'observed_camera_id':'front','views':[
        {'id':name,'projection':'orthographic','camera_to_world':matrix,
         'orthographic_scale':4.0,'clip_start':.01,'clip_end':100}
        for name,matrix in [
            ('front',[[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]]),
            ('back',[[-1,0,0,0],[0,1,0,0],[0,0,-1,-6],[0,0,0,1]]),
            ('right',[[0,0,1,6],[1,0,0,0],[0,1,0,0],[0,0,0,1]]),
            ('left',[[0,0,-1,-6],[-1,0,0,0],[0,1,0,0],[0,0,0,1]])]]}


def image_rows(tmp_path,colour=(120,40,20),coverage=True):
    tmp_path.mkdir(exist_ok=True)
    rows=[]
    for view in cameras()['views']:
        rgb=np.zeros((64,64,3),np.uint8)
        mask=np.zeros((64,64),np.uint8)
        rgb[24:40,24:40]=colour
        if coverage: mask[24:40,24:40]=255
        file=tmp_path/(view['id']+'.png'); mf=tmp_path/(view['id']+'-mask.png')
        Image.fromarray(rgb).save(file); Image.fromarray(mask).save(mf)
        rows.append({'camera_id':view['id'],'file':str(file),'mask_file':str(mf)})
    return rows


def test_foreground_ssim_and_fixed_four_view_bounds(tmp_path):
    ref=image_rows(tmp_path/'ref'); cand=image_rows(tmp_path/'cand')
    result=compare_views(ref,cand,cameras())
    assert result['score_percent']==pytest.approx(100)
    assert result['observed_score_percent']==pytest.approx(100)
    assert result['novel_score_percent']==pytest.approx(100)
    partial=compare_views(ref,cand[:3],cameras())
    assert partial['status']=='unknown' and partial['score_percent'] is None
    assert partial['measured']==3 and partial['eligible']==4
    assert partial['lower_bound_percent']==pytest.approx(75)
    assert partial['upper_bound_percent']==pytest.approx(100)
    # One very bad view retains the same weight as each perfect view.
    bad=image_rows(tmp_path/'bad',colour=(0,0,0))
    mixed=cand[:3]+[bad[3]]
    mixed_result=compare_views(ref,mixed,cameras())
    scores=[view['score_percent'] for view in mixed_result['views']]
    assert mixed_result['score_percent']==pytest.approx(sum(scores)/4)
    assert 70<mixed_result['score_percent']<80
    wrong=image_rows(tmp_path/'wrong',colour=(0,0,0),coverage=False)
    assert compare_views(ref,wrong,cameras())['score_percent']<10


def test_empty_reference_mask_and_invalid_cameras_are_unknown(tmp_path):
    ref=image_rows(tmp_path/'ref',coverage=False)
    assert compare_views(ref,ref,cameras())['status']=='unknown'
    malformed=cameras(); malformed['views'][0]['camera_to_world'][0][0]=2
    with pytest.raises(ValueError,match='rigid'): validate_cameras(malformed)


def write_cube(path,opacity='omitted',colour=(.6,.12,.05),shader='UsdPreviewSurface',scale=1):
    from pxr import Usd,UsdGeom,UsdShade,Sdf
    stage=Usd.Stage.CreateNew(str(path)); UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    cube=UsdGeom.Cube.Define(stage,'/Cube'); cube.CreateSizeAttr(scale*2)
    mat=UsdShade.Material.Define(stage,'/Mat'); node=UsdShade.Shader.Define(stage,'/Mat/Surface')
    node.CreateIdAttr(shader)
    node.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(colour)
    node.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.5)
    node.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0)
    if opacity=='unvalued': node.CreateInput('opacity',Sdf.ValueTypeNames.Float)
    elif opacity!='omitted': node.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(node.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    return stage


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_real_blender_opacity_defaults_fixed_camera_and_repeat(tmp_path):
    source=tmp_path/'cube.usda'; write_cube(source)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    first=render_asset(source,cameras(),tmp_path/'first',blender='/usr/bin/blender')
    assert first['status']=='measured',first
    mask_pixels=np.asarray(Image.open(first['images'][0]['mask_file']))
    assert not np.any(mask_pixels[:8,:8]), 'Empty background must have exactly zero object coverage'
    repeat=render_asset(source,cameras(),tmp_path/'repeat',blender='/usr/bin/blender')
    assert [r['sha256'] for r in first['images']]==[r['sha256'] for r in repeat['images']]
    assert first['scene_sha256']==repeat['scene_sha256']
    explicit=tmp_path/'opaque.usda'; write_cube(explicit,1)
    opaque=render_asset(explicit,cameras(),tmp_path/'opaque',blender='/usr/bin/blender')
    assert compare_views(first,opaque,cameras())['score_percent']==pytest.approx(100)
    transparent=tmp_path/'transparent.usda'; write_cube(transparent,0)
    zero=render_asset(transparent,cameras(),tmp_path/'zero',blender='/usr/bin/blender')
    assert zero['images'][0]['mask_sha256']==first['images'][0]['mask_sha256']
    assert zero['images'][0]['sha256']!=first['images'][0]['sha256']
    small=tmp_path/'small.usda'; write_cube(small,1,scale=.5)
    smaller=render_asset(small,cameras(),tmp_path/'smaller',blender='/usr/bin/blender')
    assert smaller['images'][0]['foreground_pixels']<first['images'][0]['foreground_pixels']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before


def test_unsupported_shader_and_connected_opacity_are_typed_unknown(tmp_path):
    from pxr import UsdShade,Sdf
    source=tmp_path/'mdl.usda'; write_cube(source,shader='OmniPBR')
    receipt=render_asset(source,cameras(),tmp_path/'mdl')
    assert receipt['status']=='unknown' and receipt['reason']=='unsupported_scene'
    source=tmp_path/'connected.usda'; stage=write_cube(source)
    node=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    upstream=UsdShade.Shader.Define(stage,'/Mat/Unsupported'); upstream.CreateIdAttr('Unsupported')
    upstream.CreateOutput('result',Sdf.ValueTypeNames.Float)
    node.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(upstream.ConnectableAPI(),'result')
    stage.GetRootLayer().Save()
    receipt=render_asset(source,cameras(),tmp_path/'connected')
    assert receipt['status']=='unknown' and receipt['reason']=='unsupported_scene'
    assert any('opacity' in str(item) for item in receipt['support']['unsupported'])


def test_receipt_camera_recipe_and_content_identity(tmp_path):
    ref=image_rows(tmp_path/'ref')
    assert compare_views({'status':'measured','images':ref,'camera_sha256':'bad'},ref,cameras())['measured']==0
    tampered=[dict(row) for row in ref]
    tampered[0]['sha256']='bad'
    result=compare_views(ref,tampered,cameras())
    assert result['status']=='unknown' and result['measured']==3
    assert 'digest mismatch' in result['views'][0]['reason']


def run_cli(*arguments):
    import subprocess,sys,os
    import simready_eval.render as module
    environment=dict(os.environ)
    environment['PYTHONPATH']=str(Path(module.__file__).resolve().parents[1])
    return subprocess.run([sys.executable,'-m','simready_eval.render',*map(str,arguments)],
                          capture_output=True,text=True,env=environment,timeout=30)


def test_render_cli_schema_error_is_json_unknown_and_nonzero(tmp_path):
    manifest=tmp_path/'cameras.json'; manifest.write_text(json.dumps({'schema':'wrong'}))
    process=run_cli('--source',tmp_path/'asset.usda','--cameras',manifest,'--output',tmp_path/'render')
    assert process.returncode!=0
    receipt=json.loads(process.stdout)
    assert receipt['status']=='unknown' and receipt['reason']=='invalid_render_input'


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_render_cli_real_render_and_unsupported_source(tmp_path):
    manifest=tmp_path/'cameras.json'; manifest.write_text(json.dumps(cameras()))
    source=tmp_path/'cube.usda'; write_cube(source)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    process=run_cli('--source',source,'--cameras',manifest,'--output',tmp_path/'render','--timeout','20')
    assert process.returncode==0,process.stderr+process.stdout
    receipt=json.loads(process.stdout)
    assert receipt['status']=='measured' and len(receipt['images'])==4
    assert len(list((tmp_path/'render').glob('*-mask.png')))==4
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    unsupported=tmp_path/'mdl.usda';write_cube(unsupported,shader='OmniPBR')
    process=run_cli('--source',unsupported,'--cameras',manifest,'--output',tmp_path/'unsupported')
    assert process.returncode!=0
    assert json.loads(process.stdout)['reason']=='unsupported_scene'


def textured_quad(path,*,mapping=False,normal=False,active_coat=0):
    from pxr import Usd,UsdGeom,UsdShade,Sdf
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    mesh=UsdGeom.Mesh.Define(stage,'/Quad');mesh.CreatePointsAttr([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)])
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying).Set([(0,0),(1,0),(1,1),(0,1)])
    mat=UsdShade.Material.Define(stage,'/Mat');shader=UsdShade.Shader.Define(stage,'/Mat/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('clearcoat',Sdf.ValueTypeNames.Float).Set(active_coat)
    shader.CreateInput('clearcoatRoughness',Sdf.ValueTypeNames.Float).Set(.03)
    shader.CreateInput('specular',Sdf.ValueTypeNames.Float).Set(.5)
    reader=UsdShade.Shader.Define(stage,'/Mat/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    uv=reader
    if mapping:
        uv=UsdShade.Shader.Define(stage,'/Mat/Mapping');uv.CreateIdAttr('UsdTransform2d')
        uv.CreateInput('in',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        uv.CreateInput('scale',Sdf.ValueTypeNames.Float2).Set((2,3))
        uv.CreateInput('rotation',Sdf.ValueTypeNames.Float).Set(90)
        uv.CreateInput('translation',Sdf.ValueTypeNames.Float2).Set((.25,.5))
        uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    file=path.parent/'normal.png' if normal else path.parent/'colour.png'
    Image.fromarray(np.full((2,2,3),(218,128,218) if normal else (210,90,20),np.uint8)).save(file)
    texture=UsdShade.Shader.Define(stage,'/Mat/Tex');texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
    texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw' if normal else 'sRGB')
    if normal:
        texture.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set((2,2,2,2))
        texture.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set((-1,-1,-1,-1))
    texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.CreateInput('normal' if normal else 'diffuseColor',Sdf.ValueTypeNames.Normal3f if normal else Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat);stage.GetRootLayer().Save()
    return stage


def test_transform2d_uv_scale_rotate_translate_and_inactive_exporter_inputs(tmp_path):
    from simready_eval.render import _extract_scene
    source=tmp_path/'mapped.usda';textured_quad(source,mapping=True)
    output=tmp_path/'scene';output.mkdir()
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    points=np.asarray(scene['triangles'])
    for triangle,row in zip(points,scene['triangle_materials']):
        original=(triangle[:,:2]+1)/2
        expected=np.column_stack([-3*original[:,1]+.25,2*original[:,0]+.5])
        assert np.allclose(row['uv']['diffuseColor'],expected)
    assert any('specular' in item['reason'] for item in support['diagnostics'])


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_raw_tangent_normal_texture_scale_bias_affects_real_render(tmp_path):
    from simready_eval.render import _extract_scene
    source=tmp_path/'normal.usda';stage=textured_quad(source,normal=True)
    output=tmp_path/'scene';output.mkdir()
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    normal=scene['materials'][0]['normal']
    assert normal['sRGB'] is False and normal['scale']==[2,2,2] and normal['bias']==[-1,-1,-1]
    mapped=render_asset(source,cameras(),tmp_path/'mapped',blender='/usr/bin/blender')
    assert mapped['status']=='measured',mapped
    from pxr import UsdShade
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'));shader.GetInput('normal').DisconnectSource()
    shader.GetInput('normal').Set((0,0,1));stage.GetRootLayer().Save()
    flat=render_asset(source,cameras(),tmp_path/'flat',blender='/usr/bin/blender')
    assert flat['status']=='measured',flat
    assert mapped['images'][0]['sha256']!=flat['images'][0]['sha256']
    assert mapped['images'][0]['mask_sha256']==flat['images'][0]['mask_sha256']


def test_active_clearcoat_stays_unknown(tmp_path):
    source=tmp_path/'coated.usda';textured_quad(source,active_coat=.5)
    receipt=render_asset(source,cameras(),tmp_path/'render')
    assert receipt['status']=='unknown' and receipt['reason']=='unsupported_scene'


def five_cameras(size=64):
    rig=cameras(size)
    rig['observation']={'id':'observation','projection':'orthographic',
      'camera_to_world':[[-1,0,0,0],[0,0,1,6],[0,1,0,0],[0,0,0,1]],
      'orthographic_scale':4.,'clip_start':.01,'clip_end':100.}
    rig['observed_camera_id']='observation'
    return rig


def test_separate_observation_rig_validates_five_unique_cameras(tmp_path):
    rig=five_cameras();validate_cameras(rig)
    ref=image_rows(tmp_path/'ref')
    compared=compare_views(ref,ref,rig)
    assert compared['measured']==4 and compared['score_percent']==pytest.approx(100)
    assert compared['observed_score_percent'] is None
    assert compared['novel_score_percent']==pytest.approx(100)
    assert all(view['role']=='novel' for view in compared['views'])
    duplicate=five_cameras();duplicate['observation']['camera_to_world']=duplicate['views'][0]['camera_to_world']
    with pytest.raises(ValueError,match='distinct'):validate_cameras(duplicate)


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_real_separate_observation_render_cli_role(tmp_path):
    rig=five_cameras();source=tmp_path/'cube.usda';write_cube(source)
    comparison=render_asset(source,rig,tmp_path/'comparison',blender='/usr/bin/blender',role='comparison')
    assert comparison['status']=='measured' and len(comparison['images'])==4
    manifest=tmp_path/'cameras.json';manifest.write_text(json.dumps(rig))
    process=run_cli('--source',source,'--cameras',manifest,'--output',tmp_path/'observation','--role','observation')
    assert process.returncode==0,process.stdout+process.stderr
    observation=json.loads(process.stdout)
    assert observation['status']=='measured' and observation['role']=='observation'
    assert len(observation['images'])==1 and observation['images'][0]['camera_id']=='observation'
    assert Path(observation['images'][0]['file']).name=='observation.png'
    assert comparison['camera_sha256']==observation['camera_sha256']
    assert {row['camera_id'] for row in comparison['images']}.isdisjoint({row['camera_id'] for row in observation['images']})


def test_fileless_texture_uses_authored_fallback_scale_bias_and_channel(tmp_path):
    from pxr import UsdShade,Sdf
    from simready_eval.render import _extract_scene
    source=tmp_path/'fallback.usda';stage=textured_quad(source)
    node=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'));node.GetPrim().RemoveProperty('inputs:file')
    node.CreateInput('fallback',Sdf.ValueTypeNames.Float4).Set((.1,.2,.3,.4))
    node.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set((2,3,4,5))
    node.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set((.01,.02,.03,.04))
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    node.CreateOutput('g',Sdf.ValueTypeNames.Float)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).ConnectToSource(node.ConnectableAPI(),'g')
    stage.GetRootLayer().Save();output=tmp_path/'scene';output.mkdir()
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    colour=scene['materials'][0]['diffuseColor']
    assert colour['fallback_only'] is True and colour['fallback']==pytest.approx([.1,.2,.3,.4])
    assert colour['scale']==[2,3,4]
    assert scene['materials'][0]['metallic']['channels']==[1]
    assert any('fallback' in row['reason'].lower() for row in support['diagnostics'])


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_udim_native_tiles_preserve_tile_coordinates_and_channel_wiring(tmp_path):
    from pxr import UsdShade,Sdf,UsdGeom
    source=tmp_path/'tiles.usda';stage=textured_quad(source)
    node=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'))
    node.GetInput('file').Set(Sdf.AssetPath(str(tmp_path/'tiles.<UDIM>.png')))
    node.GetInput('sourceColorSpace').Set('raw')
    Image.fromarray(np.full((4,4,3),(200,5,5),np.uint8)).save(tmp_path/'tiles.1001.png')
    Image.fromarray(np.full((4,4,3),(5,200,5),np.uint8)).save(tmp_path/'tiles.1002.png')
    pv=UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/Quad')).GetPrimvar('st')
    pv.Set([(0,0),(2,0),(2,1),(0,1)])
    # Indexed face-varying data must flatten without swapping authored channels.
    pv.SetIndices([0,1,2,3]);stage.GetRootLayer().Save()
    result=render_asset(source,cameras(),tmp_path/'render',blender='/usr/bin/blender')
    assert result['status']=='measured',result
    pixels=np.asarray(Image.open(result['images'][0]['file']))
    assert pixels[32,24,0]>pixels[32,24,1]+60
    assert pixels[32,40,1]>pixels[32,40,0]+60
    scene=json.loads((tmp_path/'render/scene.json').read_text())
    assert sorted(scene['materials'][0]['diffuseColor']['tiles'])==['1001','1002']
    assert any(np.max(np.asarray(row['uv']['diffuseColor'])[:,0])==2 for row in scene['triangle_materials'])


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_emission_and_explicit_unbound_display_policy(tmp_path):
    from pxr import UsdShade,Sdf,UsdGeom
    source=tmp_path/'emission.usda';stage=textured_quad(source)
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set((0,2,0));stage.GetRootLayer().Save()
    emissive=render_asset(source,cameras(),tmp_path/'emissive',blender='/usr/bin/blender')
    assert emissive['status']=='measured',emissive
    assert np.asarray(Image.open(emissive['images'][0]['file']))[32,32,1]==255
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Quad'))
    UsdShade.MaterialBindingAPI(mesh).UnbindAllBindings()
    mesh.CreateDisplayColorAttr([(0,0,.8)]);mesh.CreateDisplayOpacityAttr([.35]);stage.GetRootLayer().Save()
    unbound=render_asset(source,cameras(),tmp_path/'unbound',blender='/usr/bin/blender')
    assert unbound['status']=='measured',unbound
    assert unbound['support']['source_material_status']['unbound_triangles']==2
    scene=json.loads((tmp_path/'unbound/scene.json').read_text())
    assert scene['materials'][0]['source_material_status']=='unbound'
    assert scene['materials'][0]['material_diagnostic_status']=='not_applicable'
    assert scene['triangle_materials'][0]['displayColor']==[[0,0,pytest.approx(.8)]]*3
    assert scene['triangle_materials'][0]['displayOpacity']==pytest.approx([.35]*3)
    mesh.GetPrim().RemoveProperty('primvars:displayColor');mesh.GetPrim().RemoveProperty('primvars:displayOpacity');stage.GetRootLayer().Save()
    default=render_asset(source,cameras(),tmp_path/'default',blender='/usr/bin/blender')
    assert default['status']=='measured'
    assert default['images'][0]['mask_sha256']==unbound['images'][0]['mask_sha256']


def test_nondefault_literal_ior_and_resource_errors_are_typed(tmp_path,monkeypatch):
    from pxr import UsdShade,Sdf
    from simready_eval.render import _extract_scene
    import simready_eval.usd as usd
    source=tmp_path/'ior.usda';stage=textured_quad(source)
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.3);stage.GetRootLayer().Save()
    output=tmp_path/'scene';output.mkdir();scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    assert scene['materials'][0]['ior']==pytest.approx(1.3)
    def unavailable(*args,**kwargs):raise MemoryError('surface resource bound')
    monkeypatch.setattr(usd,'observe',unavailable)
    receipt=render_asset(source,cameras(),tmp_path/'failed')
    assert receipt['status']=='unknown' and receipt['reason']=='render_resource_bound'


def test_constant_texture_fallback_needs_no_st_or_primvar(tmp_path):
    from pxr import UsdShade,UsdGeom,Sdf
    source=tmp_path/'constant.usda';stage=textured_quad(source)
    texture=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'))
    texture.GetPrim().RemoveProperty('inputs:file');texture.GetPrim().RemoveProperty('inputs:st')
    texture.CreateInput('fallback',Sdf.ValueTypeNames.Float4).Set((.2,.4,.6,1))
    UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/Quad')).RemovePrimvar('st');stage.GetRootLayer().Save()
    output=tmp_path/'scene';output.mkdir()
    from simready_eval.render import _extract_scene
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    assert scene['materials'][0]['diffuseColor']['fallback_only']
    assert scene['materials'][0]['diffuseColor']['fallback']==pytest.approx([.2,.4,.6,1])


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_nested_entrypoint_can_read_sibling_texture_with_explicit_package_root(tmp_path):
    from pxr import UsdShade,Sdf
    models=tmp_path/'models';models.mkdir();source=models/'mesh.usda';stage=textured_quad(source)
    Image.fromarray(np.full((2,2,3),(20,180,40),np.uint8)).save(tmp_path/'colour.png')
    texture=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'));texture.GetInput('file').Set(Sdf.AssetPath('../colour.png'))
    stage.GetRootLayer().Save()
    denied=render_asset(source,cameras(),tmp_path/'denied',blender='/usr/bin/blender')
    assert denied['status']=='unknown'
    receipt=render_asset(source,cameras(),tmp_path/'allowed',blender='/usr/bin/blender',package_root=tmp_path)
    assert receipt['status']=='measured',receipt
    with pytest.raises(ValueError,match='package root'):
        render_asset(source,cameras(),tmp_path/'outside',package_root=tmp_path/'elsewhere')


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_prepare_nested_bound_sibling_texture_uses_qualified_root(tmp_path):
    from test_prepare import fixture
    from simready_eval.prepare import prepare,freeze_cameras
    from simready_eval.usd import observe
    from pxr import Usd,UsdShade,UsdGeom,Sdf
    root=tmp_path/'package';original=fixture(root);stage=Usd.Stage.Open(str(original))
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Material/Shader'))
    reader=UsdShade.Shader.Define(stage,'/Material/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    texture=UsdShade.Shader.Define(stage,'/Material/Tex');texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('../colour.png'))
    texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(),'rgb')
    for name in ['/Base/Cube','/Leaf/Cube']:
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(name)).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.constant).Set([(.5,.5)])
    stage.GetRootLayer().Save();(root/'models').mkdir();source=root/'models/mesh.usda';original.rename(source)
    Image.fromarray(np.full((2,2,3),(30,150,70),np.uint8)).save(root/'colour.png')
    result=prepare(source,root,tmp_path/'prepared','nested',cameras=freeze_cameras(observe(source),resolution=32),timeout=60)
    assert result['status']=='prepared',result
    assert (tmp_path/'prepared/reference/models/mesh.usda').is_file()
    assert (tmp_path/'prepared/reference/colour.png').is_file()
    assert len(list((tmp_path/'prepared/comparison').glob('*.png')))==8


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_connected_emission_and_missing_udim_tile_use_authored_fallback(tmp_path):
    from pxr import UsdShade,UsdGeom,Sdf
    source=tmp_path/'emissive-tiles.usda';stage=textured_quad(source)
    node=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'))
    node.GetInput('file').Set(Sdf.AssetPath(str(tmp_path/'emission.<UDIM>.png')))
    node.GetInput('sourceColorSpace').Set('raw')
    node.CreateInput('fallback',Sdf.ValueTypeNames.Float4).Set((0,2,0,1))
    Image.fromarray(np.full((4,4,3),(200,5,5),np.uint8)).save(tmp_path/'emission.1001.png')
    UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/Quad')).GetPrimvar('st').Set([(0,0),(2,0),(2,1),(0,1)])
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(node.ConnectableAPI(),'rgb')
    stage.GetRootLayer().Save()
    result=render_asset(source,cameras(),tmp_path/'render',blender='/usr/bin/blender')
    assert result['status']=='measured',result
    pixels=np.asarray(Image.open(result['images'][0]['file']))
    assert int(pixels[32,24,0])>int(pixels[32,24,1])+80
    assert int(pixels[32,40,1])>int(pixels[32,40,0])+80


def test_effective_instance_proxy_shader_active_features_are_checked(tmp_path):
    from pxr import Usd,UsdGeom,UsdShade,Sdf
    prototype=tmp_path/'prototype.usda';stage=Usd.Stage.CreateNew(str(prototype))
    model=UsdGeom.Xform.Define(stage,'/Model');stage.SetDefaultPrim(model.GetPrim())
    cube=UsdGeom.Cube.Define(stage,'/Model/Cube');cube.CreateSizeAttr(2)
    material=UsdShade.Material.Define(stage,'/Model/Mat')
    shader=UsdShade.Shader.Define(stage,'/Model/Mat/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('clearcoat',Sdf.ValueTypeNames.Float).Set(.5)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set((.4,.2,.1))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material);stage.GetRootLayer().Save()
    source=tmp_path/'instance.usda';composed=Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(composed,1);UsdGeom.SetStageUpAxis(composed,UsdGeom.Tokens.z)
    instance=UsdGeom.Xform.Define(composed,'/Instance').GetPrim()
    instance.GetReferences().AddReference('prototype.usda');instance.SetInstanceable(True);composed.GetRootLayer().Save()
    assert not any(p.IsA(UsdShade.Shader)for p in composed.Traverse())
    assert any(p.IsA(UsdShade.Shader)and p.IsInstanceProxy()for p in composed.Traverse(Usd.TraverseInstanceProxies()))
    from simready_eval.render import _extract_scene
    output=tmp_path/'scene';output.mkdir();scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='unknown',support
    assert any('clearcoat' in row['reason'] for row in support['unsupported'])
    # Printer-like active values on the same proxy must be translated exactly.
    shader.GetInput('clearcoat').Set(0)
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set((.24657011,.7064786,.05984024))
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.3);stage.GetRootLayer().Save()
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    assert scene['materials'][0]['emissiveColor']==pytest.approx([.24657011,.7064786,.05984024])
    assert scene['materials'][0]['ior']==pytest.approx(1.3)


@pytest.mark.parametrize('name,reverse,unbound,expected',[
 ('mapped',False,False,'6b599cae5249338af166db3cbaac1b2f42b2dcb5a32bc2cec9001cda1b3a9fca'),
 ('left_handed',True,False,'9f731f6b2ce9819447b3b107f7118b4bab0fc389c98d6d9ff90ba37a10b003d5'),
 ('unbound',False,True,'281966869744a4837a2047df585af71f3cc1b48d75de223924c7e44b18b644aa')])
def test_large_scene_optimizations_preserve_frozen_scene_identity(tmp_path,name,reverse,unbound,expected):
    from pxr import UsdShade,UsdGeom,Sdf
    from simready_eval.render import _extract_scene,_identity
    source=tmp_path/'asset.usda';stage=textured_quad(source,mapping=True,normal=True)
    shader=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Surface'));tex=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'))
    for key,channel in [('diffuseColor','rgb'),('roughness','g'),('metallic','b')]:
        tex.CreateOutput(channel,Sdf.ValueTypeNames.Float3 if channel=='rgb'else Sdf.ValueTypeNames.Float)
        shader.CreateInput(key,Sdf.ValueTypeNames.Color3f if key=='diffuseColor'else Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),channel)
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Quad'))
    if reverse:mesh.CreateOrientationAttr('leftHanded')
    if unbound:
        UsdShade.MaterialBindingAPI(mesh).UnbindAllBindings();mesh.CreateDisplayColorAttr([(.2,.4,.6)]);mesh.CreateDisplayOpacityAttr([.35])
    stage.GetRootLayer().Save();output=tmp_path/'scene';output.mkdir()
    scene,support=_extract_scene(source,output,np.eye(4));assert support['status']=='supported',support
    for material in scene['materials']:
        for value in material.values():
            if isinstance(value,dict)and'file'in value:value['file']=Path(value['file']).name
    assert _identity(scene)==expected


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_repeated_position_face_and_zero_normals_use_declared_auto_policy(tmp_path):
    from pxr import UsdGeom
    from simready_eval.render import _extract_scene
    source=tmp_path/'degenerate-corner.usda';stage=textured_quad(source)
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Quad'))
    mesh.GetPointsAttr().Set([(0,0,0),(1,0,0),(1,1,0),(1,1,0)])
    mesh.CreateNormalsAttr([(0,0,1),(0,0,1),(0,0,0),(0,0,0)]);mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    stage.GetRootLayer().Save();output=tmp_path/'scene';output.mkdir()
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    assert len(scene['triangles'])==1
    assert any('Repeated-position' in row['reason']for row in support['diagnostics'])
    assert any('Zero-length authored normal' in row['reason']for row in support['diagnostics'])
    assert scene['corner_normals'][0][2]==[0,0,0]
    zero=render_asset(source,cameras(),tmp_path/'zero',blender='/usr/bin/blender')
    assert zero['status']=='measured',zero
    mesh.GetNormalsAttr().Set([(0,0,1)]*4);stage.GetRootLayer().Save()
    explicit=render_asset(source,cameras(),tmp_path/'explicit',blender='/usr/bin/blender')
    assert explicit['status']=='measured',explicit
    assert [image['sha256']for image in zero['images']]==[image['sha256']for image in explicit['images']]
    mesh.GetNormalsAttr().Set([(float('nan'),0,1)]*4);stage.GetRootLayer().Save()
    invalid=render_asset(source,cameras(),tmp_path/'invalid',blender='/usr/bin/blender')
    assert invalid['status']=='unknown'and invalid['reason']=='unsupported_scene'


def test_chunked_scene_encoding_preserves_legacy_json_bytes_and_identity(tmp_path):
    import simready_eval.render as rendering
    payload={'unicode':'pixelλ\n'*180000,'values':[0.,-0.,1e-15,1e20],'nested':{'a':[1,2,3]}}
    expected=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert rendering._identity(payload)==hashlib.sha256(expected).hexdigest()
    target=tmp_path/'scene.json';rendering._write_scene_json(target,payload)
    assert target.read_bytes()==json.dumps(payload,allow_nan=False).encode()


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_all_role_matches_separate_five_frozen_renders(tmp_path):
    rig=five_cameras();source=tmp_path/'cube.usda';write_cube(source)
    comparison=render_asset(source,rig,tmp_path/'comparison',blender='/usr/bin/blender',role='comparison')
    observation=render_asset(source,rig,tmp_path/'observation',blender='/usr/bin/blender',role='observation')
    combined=render_asset(source,rig,tmp_path/'all',blender='/usr/bin/blender',role='all')
    assert combined['status']=='measured'and combined['role']=='all'and len(combined['images'])==5
    separate={row['camera_id']:row for row in comparison['images']+observation['images']}
    assert len({row['camera_id']for row in combined['images']})==5
    for row in combined['images']:
        assert row['sha256']==separate[row['camera_id']]['sha256']
        assert row['mask_sha256']==separate[row['camera_id']]['mask_sha256']
    assert combined['scene_sha256']==comparison['scene_sha256']==observation['scene_sha256']
    assert compare_views(combined,combined,rig)['score_percent']==pytest.approx(100)
    manifest=tmp_path/'cameras.json';manifest.write_text(json.dumps(rig))
    result=run_cli('--source',source,'--cameras',manifest,'--output',tmp_path/'cli-all','--role','all')
    assert result.returncode==0,result.stdout+result.stderr
    assert len(json.loads(result.stdout)['images'])==5


def test_worker_decoder_shares_only_bitwise_equal_uvs_within_each_triangle(monkeypatch):
    import importlib.util,sys,types,copy
    import simready_eval.render as rendering
    monkeypatch.setitem(sys.modules,'bpy',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'mathutils',types.SimpleNamespace(Matrix=object,Vector=object))
    spec=importlib.util.spec_from_file_location('neutral_worker_unit',Path(rendering.__file__).with_name('render_worker.py'))
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    positive=[[0.,0.],[1.,0.],[1.,1.]];negative=[[-0.,0.],[1.,0.],[1.,1.]]
    row={'material':0,'uv':{'diffuseColor':positive,'roughness':copy.deepcopy(positive),'normal_frame':negative}}
    encoded=json.dumps({'triangle_materials':[row,copy.deepcopy(row)],'material':{'normal':[0,0,1]}})
    decoded=json.loads(encoded,object_hook=worker._intern_uv_corners)
    assert json.dumps(decoded)==json.dumps(json.loads(encoded))
    first,second=decoded['triangle_materials']
    assert first['uv']['diffuseColor'] is first['uv']['roughness']
    assert first['uv']['diffuseColor'] is not first['uv']['normal_frame']
    assert first['uv']['diffuseColor'] is not second['uv']['diffuseColor']


def write_rgba16_png(path,pixels):
    import struct,zlib
    values=np.asarray(pixels,dtype=np.uint16);height,width,_=values.shape
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\x00'+row.astype('>u2').tobytes()for row in values)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,16,6,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_native_16bit_texture_low_samples_are_not_truncated(tmp_path):
    from pxr import UsdShade,Sdf
    from simready_eval.render import _extract_scene
    import subprocess
    source=tmp_path/'source.usda';stage=textured_quad(source)
    texture=tmp_path/'precision.png';write_rgba16_png(texture,[[[1,257,32769,65535]]])
    node=UsdShade.Shader(stage.GetPrimAtPath('/Mat/Tex'));node.GetInput('file').Set(Sdf.AssetPath(str(texture)))
    node.GetInput('sourceColorSpace').Set('raw');stage.GetRootLayer().Save()
    output=tmp_path/'scene';output.mkdir();scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    exported=Path(scene['materials'][0]['diffuseColor']['file'])
    assert exported.read_bytes()==texture.read_bytes()
    assert exported.read_bytes()[24]==16
    expression="import bpy,json;im=bpy.data.images.load("+repr(str(exported))+");im.colorspace_settings.name='Non-Color';print('NATIVE_PIXEL:'+json.dumps(list(im.pixels[:4])))"
    process=subprocess.run(['/usr/bin/blender','--background','--factory-startup','--threads','1','--python-exit-code','1','--python-expr',expression],capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    pixels=json.loads(next(line.split(':',1)[1]for line in process.stdout.splitlines()if line.startswith('NATIVE_PIXEL:')))
    assert pixels==pytest.approx([1/65535,257/65535,32769/65535,1],abs=4e-8)


def test_texture_streaming_deduplicates_content_without_retaining_rgba(tmp_path,monkeypatch):
    from simready_eval.render import _TextureBridge,RECIPE
    source=tmp_path/'asset.usda';source.write_text('#usda 1.0\n')
    pixels=np.full((4,4,3),(20,40,80),np.uint8)
    first=tmp_path/'a.png';second=tmp_path/'b.png';third=tmp_path/'c.png'
    Image.fromarray(pixels).save(first,compress_level=0);Image.fromarray(pixels).save(second,compress_level=9)
    Image.fromarray(np.full((4,4,3),(80,40,20),np.uint8)).save(third)
    monkeypatch.setitem(RECIPE,'maximum_decoded_texture_bytes',176)
    bridge=_TextureBridge(source,texture_dir=tmp_path/'encoded')
    a=bridge._read_image(first);b=bridge._read_image(second);c=bridge._read_image(third)
    assert a is b and a is not c
    assert not any(isinstance(value,np.ndarray)for value in bridge.images.values())
    assert bridge.texture_statistics()['retained_rgba_bytes']==0
    assert bridge.texture_statistics()['peak_decoded_rgba_bytes']<=176
    monkeypatch.setitem(RECIPE,'maximum_decoded_texture_bytes',175)
    with pytest.raises(MemoryError):_TextureBridge(source,texture_dir=tmp_path/'small')._read_image(first)


def test_streamed_8bit_canonical_export_discards_source_icc_metadata(tmp_path):
    from PIL import ImageCms
    from simready_eval.render import _TextureBridge
    source=tmp_path/'asset.usda';source.write_text('#usda 1.0\n')
    pixels=np.full((2,2,3),(20,40,80),np.uint8)
    first=tmp_path/'icc.png';plain=tmp_path/'plain.png'
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(pixels).save(first,icc_profile=profile)
    Image.fromarray(pixels).save(plain)
    expected=tmp_path/'legacy.png';Image.fromarray(np.asarray(Image.open(plain).convert('RGBA'))).save(expected)
    bridge=_TextureBridge(source,texture_dir=tmp_path/'encoded')
    exported=bridge._read_image(first);same=bridge._read_image(plain)
    assert exported is same
    assert exported.file.read_bytes()==expected.read_bytes()
    assert not Image.open(exported.file).info


def test_texture_peak_bound_returns_resource_unknown_before_blender(tmp_path,monkeypatch):
    import simready_eval.render as rendering
    source=tmp_path/'asset.usda';textured_quad(source)
    monkeypatch.setitem(rendering.RECIPE,'maximum_decoded_texture_bytes',1)
    result=rendering.render_asset(source,cameras(),tmp_path/'bounded',blender='must-not-launch')
    assert result['status']=='unknown'and result['reason']=='render_resource_bound'
    assert result['error_type']=='MemoryError'and result['error_phase']=='extract'
    assert result['images']==[]
