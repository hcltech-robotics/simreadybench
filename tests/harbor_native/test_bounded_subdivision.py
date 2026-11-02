"""Declared Blender subdivision rendering; raw candidate USD remains immutable."""
from pathlib import Path
import numpy as np
from pxr import Sdf,Usd,UsdGeom,UsdShade
from simready_eval import render


def cube(path):
    stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    mesh=UsdGeom.Mesh.Define(stage,'/Cube')
    mesh.CreatePointsAttr([(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)])
    mesh.CreateFaceVertexCountsAttr([4]*6);mesh.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
    mesh.CreateSubdivisionSchemeAttr('catmullClark')
    material=UsdShade.Material.Define(stage,'/Mat');shader=UsdShade.Shader.Define(stage,'/Mat/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set((.5,.12,.04))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save();return stage,mesh


def extract(path,target):
    target.mkdir();return render._extract_scene(path,target,np.eye(4))


def test_subdivision_preserves_control_faces_and_declares_bounded_protocol(tmp_path):
    source=tmp_path/'asset.usda';stage,mesh=cube(source);before=source.read_bytes()
    scene,support=extract(source,tmp_path/'render')
    assert support['status']=='supported',support
    assert scene['triangles']==[]
    part=scene['subdivision_meshes'][0]
    assert len(part['points'])==8 and len(part['faces'])==6
    assert all(len(f)==4 for f in part['faces'])
    assert part['protocol']['levels']==part['protocol']['render_levels']==3
    assert part['protocol']['boundary_smooth']=='PRESERVE_CORNERS'
    assert part['protocol']['uv_smooth']=='PRESERVE_CORNERS_AND_JUNCTIONS'
    assert source.read_bytes()==before


def test_holes_are_retained_as_topology_input_and_excluded_by_renderer(tmp_path):
    source=tmp_path/'asset.usda';stage,mesh=cube(source);mesh.CreateHoleIndicesAttr([0]);stage.GetRootLayer().Save()
    scene,support=extract(source,tmp_path/'render')
    assert support['status']=='supported',support
    part=scene['subdivision_meshes'][0]
    assert part['hole_indices']==[0] and len(part['faces'])==6
    assert part['estimated_triangles']==2*5*4*16


def test_nonzero_unqualified_creases_remain_unknown_not_silently_ignored(tmp_path):
    source=tmp_path/'asset.usda';stage,mesh=cube(source)
    mesh.CreateCreaseIndicesAttr([0,1]);mesh.CreateCreaseLengthsAttr([2]);mesh.CreateCreaseSharpnessesAttr([2]);stage.GetRootLayer().Save()
    scene,support=extract(source,tmp_path/'render')
    assert support['status']=='unknown'
    assert any('crease' in r['reason'].lower() for r in support['unsupported'])


def test_polygonal_mesh_keeps_existing_triangle_path(tmp_path):
    source=tmp_path/'asset.usda';stage,mesh=cube(source);mesh.CreateSubdivisionSchemeAttr('none');stage.GetRootLayer().Save()
    scene,support=extract(source,tmp_path/'render')
    assert support['status']=='supported'
    assert len(scene['triangles'])==12
    assert 'subdivision_meshes' not in scene
