"""The approved pair keeps real producer identities and rejects substitutions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from simready_eval import render


def gpu_receipt():
    recipe = {**render.RECIPE, 'samples': 256, 'threads': 4,
              'blender_version': '4.5.11 LTS', 'device': 'OPTIX'}
    return {'schema': 'simready-render-receipt-v1', 'status': 'measured',
            'role': 'all', 'variant': 'raw',
            'alignment': [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
            'recipe': recipe, 'recipe_sha256': render._identity(recipe),
            'bridge_code_sha256': '2e515052b919ac7eff777d2119cf9fc41e29d99aae0b922f70712572d580b419',
            'renderer_binary_sha256': 'dc72290ee8651c93c4a946c012c5f2a034946fd320e6c3ab214fa23181427428',
            'camera_sha256': 'c'*64, 'scene_sha256': 'd'*64,
            'renderer': {'version': '4.5.11 LTS', 'engine': 'Cycles', 'device': 'OPTIX',
                         'threads': 4, 'recipe': recipe}, 'images': []}


def test_approved_gpu_producer_preserves_actual_identity_and_common_science():
    from simready_eval.qualification import validate_producer, scientific_recipe
    receipt = gpu_receipt()
    assert receipt['recipe_sha256'] == '514dbf82a3fbf2ba24f8fb1462a54df6132eb895361f7424389b303034ff9d64'
    before = deepcopy(receipt)
    producer = validate_producer(receipt)
    assert producer['bridge_code_sha256'] == receipt['bridge_code_sha256']
    assert producer['recipe_sha256'] != render._identity(render.RECIPE)
    assert scientific_recipe(receipt['recipe']) == scientific_recipe(render.RECIPE)
    assert receipt == before


@pytest.mark.parametrize('field', ['recipe_sha256','bridge_code_sha256','renderer_binary_sha256'])
def test_unapproved_gpu_identity_is_rejected(field):
    from simready_eval.qualification import validate_producer
    receipt = gpu_receipt(); receipt[field] = '0'*64
    with pytest.raises(ValueError, match='producer'):
        validate_producer(receipt)


@pytest.mark.parametrize('change', ['samples','lighting','device','threads','variant'])
def test_producer_cannot_smuggle_different_science_or_alignment(change):
    from simready_eval.qualification import validate_producer
    receipt = gpu_receipt()
    if change == 'variant': receipt['variant'] = 'globally_aligned'
    else:
        if change == 'lighting': receipt['recipe']['world_strength'] = 5
        elif change == 'device': receipt['recipe']['device'] = 'CUDA'
        elif change == 'threads': receipt['recipe']['threads'] = 8
        else: receipt['recipe']['samples'] = 8
        receipt['recipe_sha256'] = render._identity(receipt['recipe'])
    with pytest.raises(ValueError): validate_producer(receipt)


def capture_fixture(tmp_path):
    """Synthetic bytes exercise admission; this is not GPU render evidence."""
    from PIL import Image
    from simready_eval import correspondence
    from simready_eval.prepare import _row, _write
    from simready_eval.qualification import LEGACY
    from tests.harbor_native.test_evaluator_cli import camera_manifest
    root = tmp_path/'old'; capture = tmp_path/'capture'
    for directory in (root/'reference',root/'comparison',root/'public',capture):
        directory.mkdir(parents=True)
    cameras = camera_manifest()
    cameras['resolution'] = [512,512]
    cameras['observed_camera_id'] = 'observation'
    cameras['observation'] = {**cameras['views'][0], 'id':'observation'}
    cameras['observation']['camera_to_world'] = [[1,0,0,0],[0,1,0,1],[0,0,1,6],[0,0,0,1]]
    _write(root/'comparison/cameras.json', cameras)
    (root/'reference/source.usda').write_text('#usda 1.0\n')
    (root/'public/instruction.md').write_text('Original public constraints.\n')
    receipt = gpu_receipt(); receipt['camera_sha256'] = render._identity(cameras)
    for view in cameras['views']+[cameras['observation']]:
        name = view['id']; image = capture/(name+'.png'); mask = capture/(name+'-mask.png')
        Image.new('RGB',(512,512),(50,70,90)).save(image)
        Image.new('L',(512,512),255).save(mask)
        receipt['images'].append({'camera_id':name,'file':str(image),'mask_file':str(mask),
            'sha256':render._hash(image),'mask_sha256':render._hash(mask),'foreground_pixels':512**2})
        if name == 'observation':
            (root/'public/observation.png').write_bytes(image.read_bytes())
            (root/'reference/observation-mask.png').write_bytes(mask.read_bytes())
        else:
            (root/'comparison'/image.name).write_bytes(image.read_bytes())
            (root/'comparison'/mask.name).write_bytes(mask.read_bytes())
    old = {**receipt, 'recipe_sha256':LEGACY['render_recipe_sha256'],
           'bridge_code_sha256':LEGACY['render_bridge_code_sha256'],
           'renderer_binary_sha256':LEGACY['renderer_binary_sha256']}
    _write(root/'comparison/render-receipt.json',old)
    _write(root/'reference/observation-render-receipt.json',old)
    files = [_row(p,root) for role in ('reference','comparison') for p in sorted((root/role).rglob('*')) if p.is_file()]
    eligibility = {'schema':'simready-reference-eligibility-v1', 'files':files,
                   'profile':{},'profile_sha256':correspondence.digest({}),**LEGACY,
                   'render_camera_sha256':render._identity(cameras)}
    _write(root/'reference/eligibility.json',eligibility)
    config = {'schema':'simready-evaluator-config-v1','task_id':'synthetic','entrypoint':'asset.usd',
              'reference':'reference/source.usda','cameras':'comparison/cameras.json',
              'eligibility':'reference/eligibility.json','fidelity_settings':{'enabled':True},'final_seconds':600}
    _write(root/'evaluator.json',config)
    source_files = [_row(root/'reference/source.usda',root/'reference')]
    inventory = {'schema':'simready-asset-inventory-v1','task_id':'synthetic','metadata':{},
                 'source_entrypoint':'source.usda','source_files':source_files,'files':[]}
    inventory['files'] = [_row(p,root) for p in sorted(root.rglob('*')) if p.is_file()]
    _write(root/'asset-inventory.json',inventory)
    _write(capture/'render-receipt.json',receipt)
    return root,capture,receipt,config


def test_import_reuses_pixels_binds_old_source_scene_and_separates_identities(tmp_path,monkeypatch):
    from simready_eval.capture import import_capture
    from simready_eval import cli
    root,capture,raw,config = capture_fixture(tmp_path)
    monkeypatch.setattr(render,'render_asset',lambda *a,**kw:pytest.fail('Capture import must not render'))
    output = tmp_path/'new'
    result = import_capture(root,capture/'render-receipt.json',capture,output)
    assert result['status'] == 'prepared',result
    eligibility,_,_ = cli._fixtures(output,cli._configuration(config))
    assert eligibility['schema'] == 'simready-reference-eligibility-v2'
    producer = eligibility['render_compatibility']['reference_producer']
    evaluator = eligibility['render_compatibility']['candidate_evaluator']
    assert producer['bridge_code_sha256'] == raw['bridge_code_sha256']
    assert producer != evaluator
    assert (output/'comparison/front.png').read_bytes() == (capture/'front.png').read_bytes()
    assert (output/'reference/source.usda').read_bytes() == (root/'reference/source.usda').read_bytes()
    assert json.loads((output/'comparison/raw-producer-receipt.json').read_text()) == raw


@pytest.mark.parametrize('fault',['cpu_bridge','cpu_binary','cpu_recipe','producer_pixels','producer_scene','qualification','comparison_bytes'])
def test_final_fixture_gate_rejects_changed_qualified_identity_or_receipt(tmp_path,fault):
    from simready_eval.capture import import_capture
    from simready_eval import cli
    root,capture,raw,config = capture_fixture(tmp_path)
    output=tmp_path/'new'
    assert import_capture(root,capture/'render-receipt.json',capture,output)['status']=='prepared'
    eligibility_path=output/'reference/eligibility.json'
    eligibility=json.loads(eligibility_path.read_text())
    seal=eligibility['render_compatibility']
    if fault.startswith('cpu_'):
        key={'cpu_bridge':'bridge_code_sha256','cpu_binary':'renderer_binary_sha256','cpu_recipe':'recipe_sha256'}[fault]
        seal['candidate_evaluator'][key]='0'*64
    elif fault == 'qualification': seal['qualification']='arbitrary-renderer'
    elif fault == 'comparison_bytes':
        image_path=output/'comparison/front.png'
        from PIL import Image
        Image.new('RGB',(512,512),(200,10,100)).save(image_path)
        row=next(row for row in eligibility['files'] if row['path']=='comparison/front.png')
        row.update(size_bytes=image_path.stat().st_size,sha256=cli._hash(image_path))
    else:
        receipt_path=output/'comparison/raw-producer-receipt.json'
        changed=json.loads(receipt_path.read_text())
        if fault == 'producer_pixels': changed['images'][0]['sha256']='0'*64
        else: changed['scene_sha256']='0'*64
        receipt_path.write_text(json.dumps(changed))
        # Rehashing an inventory must not bless a contradictory producer receipt.
        row=next(row for row in eligibility['files'] if row['path']=='comparison/raw-producer-receipt.json')
        row.update(size_bytes=receipt_path.stat().st_size,sha256=cli._hash(receipt_path))
    eligibility_path.write_text(json.dumps(eligibility))
    with pytest.raises(ValueError): cli._fixtures(output,cli._configuration(config))


def test_unqualified_cpu_bridge_cannot_mint_new_compatibility(tmp_path,monkeypatch):
    from simready_eval.qualification import compatibility
    fake=tmp_path/'render.py';fake.write_text('# changed renderer')
    (tmp_path/'render_worker.py').write_text('# changed worker')
    monkeypatch.setattr(render,'__file__',str(fake))
    with pytest.raises(ValueError,match='bridge'): compatibility(gpu_receipt())


def role_inventory(root,task_id,version='v1'):
    from simready_eval.prepare import _row
    value={'schema':'simready-task-role-inventory-v1','repo_id':'hcltech-robotics/simreadybench',
           'task_id':task_id,'revision_binding':'Bind outside self-hashed content.','actor':[],'verifier':[]}
    for role,directories in (('actor',('public',)),('verifier',('reference','comparison'))):
        for directory in directories:
            for path in sorted((root/directory).rglob('*')):
                if path.is_file():
                    row=_row(path,root);relative=row.pop('path')
                    value[role].append({**row,'path':f'benchmark/nv100/{version}/tasks/{task_id}/'+relative,
                        'destination':relative.removeprefix('public/') if role=='actor' else relative})
    return value


def test_legitimate_role_inventory_is_verified_then_rebuilt_for_v2(tmp_path):
    from simready_eval.capture import import_capture
    root,capture,raw,_=capture_fixture(tmp_path)
    old=role_inventory(root,'synthetic')
    (root/'role-inventory.json').write_text(json.dumps(old))
    output=tmp_path/'new'
    result=import_capture(root,capture/'render-receipt.json',capture,output)
    assert result['status']=='prepared',result
    updated=json.loads((output/'role-inventory.json').read_text())
    assert all(row['path'].startswith('benchmark/nv100/v2/tasks/synthetic/')
               for role in ('actor','verifier') for row in updated[role])
    for role in ('actor','verifier'):
        for row in updated[role]:
            path=output/('public/'+row['destination'] if role=='actor' else row['destination'])
            assert render._hash(path)==row['sha256']
    assert 'comparison/raw-producer-receipt.json' in {row['destination'] for row in updated['verifier']}
    assert (root/'role-inventory.json').read_text()==json.dumps(old)


@pytest.mark.parametrize('fault',['schema','hash','extra_reference','actor_reference','missing'])
def test_role_inventory_is_not_an_unchecked_extra_file(tmp_path,fault):
    from simready_eval.capture import import_capture
    root,capture,_,_=capture_fixture(tmp_path)
    old=role_inventory(root,'synthetic')
    if fault=='schema':old['schema']='unknown'
    elif fault=='hash':old['actor'][0]['sha256']='0'*64
    elif fault=='extra_reference':old['verifier'].append({**old['verifier'][0],'destination':'reference/untracked'})
    elif fault=='actor_reference':old['actor'][0]['path']=old['actor'][0]['path'].replace('public/','reference/')
    else:old['verifier'].pop()
    (root/'role-inventory.json').write_text(json.dumps(old))
    result=import_capture(root,capture/'render-receipt.json',capture,tmp_path/'new')
    assert result['status']=='unknown'


@pytest.mark.parametrize('fault', ['source','camera','scene','pixels','mask','producer','extra_view','link'])
def test_import_rejects_mismatched_capture_without_partial_release(tmp_path, fault):
    from simready_eval.capture import import_capture
    root,capture,raw,_ = capture_fixture(tmp_path)
    if fault == 'source': (root/'reference/source.usda').write_text('changed')
    elif fault == 'pixels': (capture/'front.png').write_bytes(b'wrong pixels')
    elif fault == 'mask': (capture/'front-mask.png').write_bytes(b'wrong mask')
    elif fault == 'camera': raw['camera_sha256'] = '0'*64
    elif fault == 'scene': raw['scene_sha256'] = '0'*64
    elif fault == 'producer': raw['bridge_code_sha256'] = '0'*64
    elif fault == 'extra_view': raw['images'].append(raw['images'][0])
    else:
        image=capture/'front.png'; image.rename(capture/'other.png'); image.symlink_to(capture/'other.png')
    (capture/'render-receipt.json').write_text(json.dumps(raw))
    output=tmp_path/'new'
    result=import_capture(root,capture/'render-receipt.json',capture,output)
    assert result['status'] == 'unknown'
    assert not output.exists()
