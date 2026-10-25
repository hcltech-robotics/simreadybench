"""Image agreement cannot replace full-surface three-dimensional agreement."""
from pathlib import Path
import math
import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Sdf
from simready_eval.correspondence import compare, settings
from simready_eval.render import render_asset, compare_views, _extract_scene
from simready_eval.usd import observe
from simready_eval.contract import visual_score
from test_render import cameras, write_cube


def bind_surface(stage,prim,path,colour=(.6,.12,.05),texture=None):
    mat=UsdShade.Material.Define(stage,path)
    shader=UsdShade.Shader.Define(stage,path+'/Surface'); shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(colour)
    if texture:
        reader=UsdShade.Shader.Define(stage,path+'/UV'); reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        node=UsdShade.Shader.Define(stage,path+'/Texture'); node.CreateIdAttr('UsdUVTexture')
        node.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
        node.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
        node.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        node.CreateOutput('rgb',Sdf.ValueTypeNames.Float3); node.CreateOutput('a',Sdf.ValueTypeNames.Float)
        shader.GetInput('diffuseColor').ConnectToSource(node.ConnectableAPI(),'rgb')
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(node.ConnectableAPI(),'a')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def sphere(path,colour=(.6,.12,.05)):
    stage=Usd.Stage.CreateNew(str(path)); UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    mesh=UsdGeom.Sphere.Define(stage,'/Sphere'); mesh.CreateRadiusAttr(1)
    bind_surface(stage,mesh.GetPrim(),'/Mat',colour)
    stage.GetRootLayer().Save()


def fourfold(path,rig,images,*,transparent=True,double_sided=True):
    stage=Usd.Stage.CreateNew(str(path)); UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    for i,(view,image) in enumerate(zip(rig['views'],images)):
        rgb=np.asarray(Image.open(image['file']).convert('RGB'))
        mask=np.asarray(Image.open(image['mask_file']).convert('L'))
        texture=path.parent/f'view-{i}-{transparent}.png'
        alpha=np.where(mask>0,255,0).astype(np.uint8) if transparent else np.full(mask.shape,255,np.uint8)
        Image.fromarray(np.dstack([rgb,alpha])).save(texture)
        basis=np.asarray(view['camera_to_world'])[:3,:3]
        # A camera-sized plane carries exactly its corresponding rendered view;
        # four thin crossed surfaces still lack the sphere's full 3D surface.
        local=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]])
        points=local@basis.T
        mesh=UsdGeom.Mesh.Define(stage,f'/Billboard{i}'); mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateDoubleSidedAttr(double_sided); mesh.CreateSubdivisionSchemeAttr('none')
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set([(0,0),(1,0),(1,1),(0,1)])
        bind_surface(stage,mesh.GetPrim(),f'/Mat{i}',texture=texture)
    stage.GetRootLayer().Save()


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_literal_textured_fourfold_billboard_and_wrong_colour_full_shape(tmp_path):
    rig=cameras(); source=tmp_path/'sphere.usda'; sphere(source)
    reference=render_asset(source,rig,tmp_path/'reference',blender='/usr/bin/blender')
    assert reference['status']=='measured',reference
    reference_observation=observe(source)
    assert compare(reference_observation,reference_observation,settings({'samples':128}))['components']['shape']['score_percent']==pytest.approx(100)
    assert compare_views(reference,reference,rig)['score_percent']==pytest.approx(100)
    for transparent,double_sided in [(False,False),(True,False),(True,True)]:
        candidate=tmp_path/f'planes-{transparent}-{double_sided}.usda'
        fourfold(candidate,rig,reference['images'],transparent=transparent,double_sided=double_sided)
        geometry=compare(reference_observation,observe(candidate),settings({'samples':128}))['components']['shape']
        assert geometry['score_percent']<25
        # Even granting an artificially perfect image match cannot make this V
        # exceed sqrt(G*100); material opacity never changes this geometry term.
        assert math.sqrt(geometry['score_percent']*100)<50
        rendered=render_asset(candidate,rig,tmp_path/f'rendered-{transparent}-{double_sided}',blender='/usr/bin/blender')
        assert rendered['status']=='measured',rendered
        appearance=compare_views(reference,rendered,rig)
        assert appearance['measured']==4
        combined=visual_score(geometry,appearance)
        assert combined['score_percent']==pytest.approx(math.sqrt(geometry['score_percent']*appearance['score_percent']))
        assert combined['score_percent']<50
    wrong=tmp_path/'wrong-colour.usda'; sphere(wrong,(.01,.8,.02))
    wrong_observation=observe(wrong)
    assert compare(reference_observation,wrong_observation,settings({'samples':128}))['components']['shape']['score_percent']==pytest.approx(100)
    wrong_render=render_asset(wrong,rig,tmp_path/'wrong-colour',blender='/usr/bin/blender')
    wrong_appearance=compare_views(reference,wrong_render,rig)
    assert wrong_appearance['score_percent']<90
    assert visual_score({'status':'measured','score_percent':100,'coverage_fraction':1},wrong_appearance)['score_percent']<100


@pytest.mark.parametrize('opacity',[0.,.35,1.,'omitted','unvalued'])
def test_v8_renderer_bridge_opacity_value(tmp_path,opacity):
    source=tmp_path/'cube.usda'; write_cube(source,opacity)
    output=tmp_path/'scene'; output.mkdir()
    scene,support=_extract_scene(source,output,np.eye(4))
    assert support['status']=='supported',support
    expected=1. if isinstance(opacity,str) else opacity
    assert all(row['opacity']==pytest.approx(expected) for row in scene['materials'])


def cubes(path,positions,*,opacity=1):
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    for index,position in enumerate(positions):
        cube=UsdGeom.Cube.Define(stage,f'/Cube{index}');cube.CreateSizeAttr(2)
        cube.AddTranslateOp().Set(position);bind_surface(stage,cube.GetPrim(),f'/Mat{index}')
        shader=UsdShade.Shader(stage.GetPrimAtPath(f'/Mat{index}/Surface'))
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
    stage.GetRootLayer().Save()


def test_missing_and_extra_full_surface_geometry_cannot_self_match(tmp_path):
    reference=tmp_path/'reference.usda';cubes(reference,[(0,0,0),(4,0,0)])
    ref=observe(reference)
    for name,positions in [('missing',[(0,0,0)]),('extra',[(0,0,0),(4,0,0),(8,0,0)])]:
        candidate=tmp_path/(name+'.usda');cubes(candidate,positions)
        geometry=compare(ref,observe(candidate),settings({'samples':128}))['components']['shape']
        assert geometry['score_percent']<90
        best_images={'status':'measured','score_percent':100,'coverage_fraction':1}
        assert visual_score(geometry,best_images)['score_percent']<95


@pytest.mark.skipif(not Path('/usr/bin/blender').exists(),reason='public Blender unavailable')
def test_transparent_only_and_invisible_geometry_plus_textured_billboards(tmp_path):
    rig=cameras();source=tmp_path/'source.usda';cubes(source,[(0,0,0)])
    reference=render_asset(source,rig,tmp_path/'reference',blender='/usr/bin/blender')
    ref_observation=observe(source)
    transparent=tmp_path/'transparent.usda';cubes(transparent,[(0,0,0)],opacity=0)
    rendered=render_asset(transparent,rig,tmp_path/'transparent',blender='/usr/bin/blender')
    appearance=compare_views(reference,rendered,rig)
    geometry=compare(ref_observation,observe(transparent),settings({'samples':128}))['components']['shape']
    assert geometry['score_percent']==pytest.approx(100)
    assert appearance['score_percent']<90
    assert visual_score(geometry,appearance)['score_percent']<95
    assert [r['mask_sha256']for r in reference['images']]==[r['mask_sha256']for r in rendered['images']]
    candidate=tmp_path/'billboard-invisible-cube.usda';fourfold(candidate,rig,reference['images'],transparent=True,double_sided=True)
    stage=Usd.Stage.Open(str(candidate));hidden=UsdGeom.Cube.Define(stage,'/InvisibleCube');hidden.CreateSizeAttr(2)
    bind_surface(stage,hidden.GetPrim(),'/HiddenMat')
    UsdShade.Shader(stage.GetPrimAtPath('/HiddenMat/Surface')).CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(0)
    stage.GetRootLayer().Save()
    attacked=render_asset(candidate,rig,tmp_path/'attacked',blender='/usr/bin/blender')
    assert attacked['status']=='measured',attacked
    geometry=compare(ref_observation,observe(candidate),settings({'samples':128}))['components']['shape']
    appearance=compare_views(reference,attacked,rig)
    assert geometry['score_percent']<90
    assert visual_score(geometry,appearance)['score_percent']<90
    assert all(view['foreground_union_pixels']>=np.count_nonzero(np.array(Image.open(row['mask_file'])))for view,row in zip(appearance['views'],reference['images']))
