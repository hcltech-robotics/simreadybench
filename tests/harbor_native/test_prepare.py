"""Standalone preparation: real neutral fixture, immutable closure and exact roles."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from simready_eval import correspondence, cli
from simready_eval.prepare import freeze_cameras, prepare


def fixture(root, shader_id='UsdPreviewSurface'):
    root.mkdir()
    source=root/'source.usda'
    stage=Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,'Z')
    material=UsdShade.Material.Define(stage,'/Material')
    shader=UsdShade.Shader.Define(stage,'/Material/Shader')
    shader.CreateIdAttr(shader_id)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.6,.2,.1))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.5)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    for name,position in [('Base',(0,0,0)),('Leaf',(1.5,0,0))]:
        body=UsdGeom.Xform.Define(stage,'/'+name)
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube=UsdGeom.Cube.Define(stage,'/'+name+'/Cube')
        cube.CreateSizeAttr(1)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    joint=UsdPhysics.RevoluteJoint.Define(stage,'/Hinge')
    joint.CreateBody0Rel().SetTargets(['/Base'])
    joint.CreateBody1Rel().SetTargets(['/Leaf'])
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(0)
    joint.CreateUpperLimitAttr(90)
    stage.GetRootLayer().Save()
    (root/'LICENSE.txt').write_text('Synthetic test fixture license.\n')
    return source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_exact_reference_only_camera_arrays():
    triangles=np.array([[[-1.,-1,-1],[1,1,1],[1,-1,1]]])
    cameras=freeze_cameras({'triangles':triangles},resolution=32)
    assert cameras['observed_camera_id']=='observation'
    assert cameras['observation']['id']=='observation'
    assert [v['id'] for v in cameras['views']]==['isometric','front','side','top']
    a=1/np.sqrt(2); d=np.sqrt(2.64)
    expected=[[[-a,-.8*a/d,1/d,6/d],[a,-.8*a/d,1/d,6/d],[0,2*a/d,.8/d,4.8/d],[0,0,0,1]],
              [[1,0,0,0],[0,0,-1,-6],[0,1,0,0],[0,0,0,1]],
              [[0,0,1,6],[1,0,0,0],[0,1,0,0],[0,0,0,1]],
              [[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]]]
    observation=[[a,-.8*a/d,1/d,6/d],[a,.8*a/d,-1/d,-6/d],[0,2*a/d,.8/d,4.8/d],[0,0,0,1]]
    np.testing.assert_allclose(cameras['observation']['camera_to_world'],observation,atol=1e-15)
    for view,matrix in zip(cameras['views'],expected):
        np.testing.assert_allclose(view['camera_to_world'],matrix,atol=1e-15)
        assert view['orthographic_scale']==3.3
    assert len({json.dumps(view['camera_to_world']) for view in cameras['views']+[cameras['observation']]})==5
    assert cameras==freeze_cameras({'triangles':triangles[::-1]},resolution=32)


def test_real_articulated_prepare_has_five_rgb_four_masks_and_cli_inventory(tmp_path,monkeypatch):
    from simready_eval.usd import observe
    source=fixture(tmp_path/'package')
    before={p.name:digest(p) for p in source.parent.iterdir()}
    cameras=freeze_cameras(observe(source),resolution=32)
    target=tmp_path/'task'
    import simready_eval.render as rendering
    real_render=rendering.render_asset
    calls=[]
    def capture(*args,**kwargs):
        calls.append(kwargs['role'])
        return real_render(*args,**kwargs)
    monkeypatch.setattr(rendering,'render_asset',capture)
    result=prepare(source,source.parent,target,'articulated',cameras=cameras,timeout=60,
                   metadata={'source_id':'synthetic-articulated','license':'test'})
    assert result['status']=='prepared',result
    config=json.loads((target/'evaluator.json').read_text())
    eligibility,frozen,images=cli._fixtures(target,cli._configuration(config))
    assert frozen==cameras and len(images)==4
    assert eligibility['profile']['joint_count']==1
    assert eligibility['profile']['settings']['samples']==2048
    assert eligibility['profile']['settings']['enabled'] is True
    assert correspondence.digest(eligibility['profile'])==eligibility['profile_sha256']
    generated=list((target/'comparison').glob('*.png'))+list((target/'public').glob('*.png'))
    assert sum(Image.open(p).mode=='RGB' for p in generated)==5
    assert sum(Image.open(p).mode=='L' for p in generated)==4
    assert calls==['all']
    observed=json.loads((target/'reference/observation-render-receipt.json').read_text())
    assert observed['role']=='observation' and len(observed['images'])==1
    assert digest(target/'public/observation.png')==observed['images'][0]['sha256']
    assert not (target/'comparison/observation.png').exists()
    assert digest(target/'public/observation.png') not in {row['sha256'] for row in images}
    assert 'Do not use Blender for authoring' in (target/'public/instruction.md').read_text()
    instruction=(target/'public/instruction.md').read_text()
    dimensions_line=next(line for line in instruction.splitlines() if line.startswith('Nominal dimensions in canonical metres XYZ: '))
    assert json.loads(dimensions_line.split(': ',1)[1])==[2.5,1.,1.]
    inventory=json.loads((target/'asset-inventory.json').read_text())
    assert inventory['metadata']['source_id']=='synthetic-articulated'
    assert inventory['metadata']['nominal_dimensions_m']==[2.5,1.,1.]
    assert 'asset-inventory.json' not in {row['path'] for row in inventory['files']}
    for row in inventory['files']:
        assert digest(target/row['path'])==row['sha256']
    assert (target/'reference/LICENSE.txt').read_bytes()==(source.parent/'LICENSE.txt').read_bytes()
    assert before=={p.name:digest(p) for p in source.parent.iterdir()}
    assert digest(target/'asset-inventory.json')==result['asset_inventory_sha256']


def test_unsupported_shader_receipt_and_cli_are_unknown_with_no_partial_payload(tmp_path):
    source=fixture(tmp_path/'package','OmniPBR')
    target=tmp_path/'task'
    result=prepare(source,source.parent,target,'unsupported',timeout=30)
    assert result['status']=='unknown' and result['reason']=='unsupported_scene'
    assert result['support']['unsupported']
    assert not target.exists()
    process=subprocess.run([sys.executable,'-m','simready_eval.prepare','--source',str(source),
        '--package-root',str(source.parent),'--output',str(target),'--task-id','unsupported'],
        capture_output=True,text=True,timeout=30)
    assert process.returncode!=0
    assert json.loads(process.stdout)['status']=='unknown'
    assert not target.exists()


@pytest.mark.parametrize('bad',['symlink','executable','special','orichalcum','oldresults','extra_usd','missing_dependency'])
def test_preflight_rejects_unqualified_closure_before_render(tmp_path,monkeypatch,bad):
    import simready_eval.render as rendering
    source=fixture(tmp_path/'package')
    if bad=='symlink': (source.parent/'link').symlink_to(source)
    elif bad=='executable':
        file=source.parent/'program.bin'; file.write_bytes(b'program'); file.chmod(0o755)
    elif bad=='special':
        import os
        os.mkfifo(source.parent/'pipe')
    elif bad in ('orichalcum','oldresults'):
        (source.parent/(bad+'.txt')).write_text(bad)
    elif bad=='extra_usd':
        unrelated=Usd.Stage.CreateNew(str(source.parent/'unrelated.usda'))
        unrelated.GetRootLayer().Save()
    else:
        stage=Usd.Stage.Open(str(source)); stage.GetRootLayer().subLayerPaths=['missing.usda']; stage.GetRootLayer().Save()
    monkeypatch.setattr(rendering,'render_asset',lambda *a,**kw:pytest.fail('Render before closure qualification'))
    target=tmp_path/'task'
    receipt=prepare(source,source.parent,target,'bad')
    assert receipt['status']=='unknown'
    assert not target.exists()


def test_rejects_nonfresh_output_and_output_inside_package(tmp_path):
    source=fixture(tmp_path/'package')
    output=tmp_path/'task'; output.mkdir(); (output/'keep.txt').write_text('keep')
    assert prepare(source,source.parent,output,'existing')['status']=='unknown'
    assert (output/'keep.txt').read_text()=='keep'
    assert prepare(source,tmp_path,tmp_path/'nested','nested')['status']=='unknown'


def test_recipe_and_renderer_identity_are_frozen_in_eligibility(tmp_path):
    from simready_eval.usd import observe
    source=fixture(tmp_path/'package')
    target=tmp_path/'task'
    result=prepare(source,source.parent,target,'identity',cameras=freeze_cameras(observe(source),resolution=32))
    assert result['status']=='prepared',result
    eligibility=json.loads((target/'reference/eligibility.json').read_text())
    receipt=json.loads((target/'comparison/render-receipt.json').read_text())
    assert eligibility['render_recipe_sha256']==receipt['recipe_sha256']
    assert eligibility['render_camera_sha256']==receipt['camera_sha256']
    assert eligibility['render_bridge_code_sha256']==receipt['bridge_code_sha256']
    assert eligibility['renderer_binary_sha256']==receipt['renderer_binary_sha256']
    assert receipt['renderer_binary_sha256'] and receipt['bridge_code_sha256']
    assert receipt['renderer']
    assert str(tmp_path) not in json.dumps(receipt)
    assert all('/' not in row['file'] and '/' not in row['mask_file'] for row in receipt['images'])


def test_relative_layer_dependency_is_retained_and_license_is_exception(tmp_path,monkeypatch):
    import simready_eval.render as rendering
    source=fixture(tmp_path/'package')
    parts=source.parent/'parts'; parts.mkdir()
    dependency=Usd.Stage.CreateNew(str(parts/'dependency.usda'))
    root=UsdGeom.Xform.Define(dependency,'/Part')
    dependency.SetDefaultPrim(root.GetPrim())
    dependency.GetRootLayer().Save()
    stage=Usd.Stage.Open(str(source))
    stage.DefinePrim('/Linked').GetReferences().AddReference('parts/dependency.usda')
    stage.GetRootLayer().Save()
    seen=[]
    def sentinel(entrypoint,cameras,target,**kwargs):
        seen.append(Path(entrypoint).parent/'parts/dependency.usda')
        assert seen[-1].read_bytes()==(parts/'dependency.usda').read_bytes()
        return {'status':'unknown','reason':'verified_dependency_test','images':[]}
    monkeypatch.setattr(rendering,'render_asset',sentinel)
    target=tmp_path/'task'
    receipt=prepare(source,source.parent,target,'relative')
    assert receipt['reason']=='verified_dependency_test',receipt
    assert seen and not target.exists()



def test_nested_entrypoint_relative_dependencies_are_preserved(tmp_path):
    from simready_eval.usd import observe
    source=fixture(tmp_path/'package')
    root=source.parent
    nested=root/'nested'; nested.mkdir()
    moved=nested/'original.usda'; source.rename(moved)
    dependency=Usd.Stage.CreateNew(str(root/'shared.usda'))
    prim=UsdGeom.Xform.Define(dependency,'/Auxiliary')
    dependency.SetDefaultPrim(prim.GetPrim())
    dependency.GetRootLayer().Save()
    stage=Usd.Stage.Open(str(moved))
    stage.DefinePrim('/Linked').GetReferences().AddReference('../shared.usda')
    stage.GetRootLayer().Save()
    before={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}
    cameras=freeze_cameras(observe(moved),resolution=32)
    output=tmp_path/'nested-task'
    receipt=prepare(moved,root,output,'nested',cameras=cameras)
    assert receipt['status']=='prepared',receipt
    config=json.loads((output/'evaluator.json').read_text())
    assert config['reference']=='reference/nested/original.usda'
    eligibility,frozen,images=cli._fixtures(output,cli._configuration(config))
    checked=cli._worker('reference',{
        'reference':str(output/config['reference']), 'reference_root':str(output/'reference'),
        'profile':eligibility['profile'], 'settings':config['fidelity_settings'],
        'images':images, 'cameras':frozen,
    })
    assert checked['status']=='verified',checked
    inventory=json.loads((output/'asset-inventory.json').read_text())
    assert inventory['source_entrypoint']=='nested/original.usda'
    for name,digest_before in before.items():
        assert digest(output/'reference'/name)==digest_before
        assert digest(root/name)==digest_before
    assert not (output/'reference/asset.usd').exists()


def test_old_four_camera_manifest_cannot_reuse_comparison_as_observation(tmp_path,monkeypatch):
    import simready_eval.render as rendering
    from simready_eval.usd import observe
    source=fixture(tmp_path/'package')
    cameras=freeze_cameras(observe(source),resolution=32)
    cameras.pop('observation',None)
    cameras['observed_camera_id']='isometric'
    monkeypatch.setattr(rendering,'render_asset',lambda *a,**kw:pytest.fail('Independent observation camera missing'))
    receipt=prepare(source,source.parent,tmp_path/'task','separate',cameras=cameras)
    assert receipt['status']=='unknown'
    assert not (tmp_path/'task').exists()
