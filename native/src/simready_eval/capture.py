"""Freeze existing qualified GPU captures against an immutable NV100 v1 task.

No rendering, network, source modification or publication occurs here. The old
source/scene/camera binding and every new pixel are verified before promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from . import cli, render
from .prepare import _row, _write, _normalized_render
from .qualification import compatibility, validate_producer


def _role_inventory(root,task_id,version):
    value={'schema':'simready-task-role-inventory-v1','repo_id':'hcltech-robotics/simreadybench',
           'task_id':task_id,
           'revision_binding':'Bind these exact entries to the immutable published HF commit outside self-hashed benchmark content. This inventory is not a runtime acquisition manifest.',
           'actor':[],'verifier':[]}
    for role,directories in (('actor',('public',)),('verifier',('reference','comparison'))):
        for directory in directories:
            for path in sorted((root/directory).rglob('*')):
                if path.is_file():
                    relative=path.relative_to(root).as_posix()
                    row=_row(cli._regular(root,relative),root)
                    value[role].append({
                        'path':f'benchmark/nv100/{version}/tasks/{task_id}/'+relative,
                        'destination':relative.removeprefix('public/') if role=='actor' else relative,
                        'sha256':row['sha256'],'size_bytes':row['size_bytes']})
    return value


def _validate_role_inventory(root,task_id,value,version):
    from .fetch import validate_manifest, REPO_ID, SCHEMA
    if (not isinstance(value,dict) or set(value)!={'schema','repo_id','task_id','revision_binding','actor','verifier'}
            or value['schema']!='simready-task-role-inventory-v1' or value['repo_id']!=REPO_ID
            or value['task_id']!=task_id or not isinstance(value['revision_binding'],str)
            or not 1<=len(value['revision_binding'])<=1024):
        raise ValueError('Invalid typed role inventory')
    expected=_role_inventory(root,task_id,version)
    for role in ('actor','verifier'):
        # This sidecar deliberately has no revision. Validate its exact role
        # shape locally; an immutable commit is bound only after publication.
        validate_manifest({'schema':SCHEMA,'repo_id':REPO_ID,'task_id':task_id,
                           'revision':'0'*40,'role':role,'files':value[role]})
        if sorted(value[role],key=lambda row:row['path'])!=sorted(expected[role],key=lambda row:row['path']):
            raise ValueError('Role inventory file set or hashes mismatch')


def _capture_images(receipt, root, cameras):
    if receipt.get('role') != 'all' or cameras['resolution'] != [512,512]:
        raise ValueError('Five-view 512-pixel raw reference capture required')
    views = cameras['views']+[cameras['observation']]
    rows = receipt.get('images', [])
    indexed = {row['camera_id']: row for row in rows}
    if len(rows) != 5 or set(indexed) != {view['id'] for view in views}:
        raise ValueError('Exactly five distinct frozen capture views required')
    checked = []
    for view in views:
        row = indexed[view['id']]
        files = {}
        for key,suffix,mode,digest in (('file','.png','RGB','sha256'),
                                      ('mask_file','-mask.png','L','mask_sha256')):
            name = view['id']+suffix
            # Raw paths describe the producer's container. Resolve only these
            # expected basenames inside the separately supplied capture root.
            if Path(row[key]).name != name:
                raise ValueError('Unexpected capture artifact name')
            path = cli._regular(root,name)
            if path.stat().st_size > 16*1024**2 or cli._hash(path) != row.get(digest):
                raise ValueError('Capture image digest mismatch: '+name)
            with Image.open(path) as image:
                if image.mode != mode or list(image.size) != cameras['resolution']:
                    raise ValueError('Capture image format mismatch')
                if mode == 'L' and not np.any(np.asarray(image)):
                    raise ValueError('Empty reference foreground')
            files[key] = path
        checked.append((row,files))
    return checked


def import_capture(task, receipt_file, capture_root, output):
    result = {'schema':'simready-capture-import-v1','status':'unknown'}
    try:
        task,capture_root,output = (Path(path).absolute() for path in (task,capture_root,output))
        receipt_file = Path(receipt_file).absolute()
        if output.exists() or output.is_symlink() or output.parent.resolve() != output.parent:
            raise ValueError('Fresh regular output location required')
        if task.resolve() != task or capture_root.resolve() != capture_root:
            raise ValueError('Regular input directories required')
        if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (task,capture_root)):
            raise ValueError('Output must be separate from inputs')
        config = cli._configuration(cli._json(cli._regular(task,'evaluator.json')))
        eligibility,cameras,_ = cli._fixtures(task,config,_legacy_import=True)
        inventory = cli._json(cli._regular(task,'asset-inventory.json'))
        if inventory.get('schema') != 'simready-asset-inventory-v1' or inventory.get('task_id') != config['task_id']:
            raise ValueError('Original task inventory required')
        declared = inventory.get('files', [])
        for row in declared:
            path = cli._regular(task,row['path'])
            if path.stat().st_size != row['size_bytes'] or cli._hash(path) != row['sha256']:
                raise ValueError('Original task inventory digest mismatch')
        actual = {p.relative_to(task).as_posix() for p in task.rglob('*') if p.is_file() or p.is_symlink()}
        # The published roster inventory lists role files; the evaluator and
        # source provenance are separate fixed-name release metadata files.
        metadata_files = {'asset-inventory.json','evaluator.json','provenance.json','role-inventory.json'}
        if actual - metadata_files != {row['path'] for row in declared} - metadata_files:
            raise ValueError('Original task has unrecorded or missing files')
        for name in actual & metadata_files:
            cli._regular(task,name)
        if 'role-inventory.json' in actual:
            _validate_role_inventory(task,config['task_id'],cli._json(task/'role-inventory.json'),'v1')
        raw = cli._json(cli._regular(receipt_file.parent,receipt_file.name))
        validate_producer(raw)
        if raw.get('task',config['task_id']) != config['task_id']:
            raise ValueError('Raw producer task mismatch')
        old = cli._json(cli._regular(task,'comparison/render-receipt.json'))
        scene = old.get('scene_sha256')
        if not isinstance(scene,str) or len(scene) != 64 or raw.get('scene_sha256') != scene:
            raise ValueError('Capture does not match the original full source scene')
        if raw.get('camera_sha256') != render._identity(cameras):
            raise ValueError('Capture does not match frozen cameras')
        checked = _capture_images(raw,capture_root,cameras)
        seal = compatibility(raw)
        original_hashes = {path:cli._hash(cli._regular(task,path)) for path in actual}
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.simready-capture-',dir=output.parent) as temporary:
            staging = Path(temporary)/'task'
            shutil.copytree(task,staging)
            for row,files in checked:
                name = row['camera_id']
                observation = name == cameras['observation']['id']
                rgb = staging/('public/observation.png' if observation else 'comparison/'+name+'.png')
                mask = staging/('reference/observation-mask.png' if observation else 'comparison/'+name+'-mask.png')
                shutil.copyfile(files['file'],rgb); shutil.copyfile(files['mask_file'],mask)
                if cli._hash(rgb) != row['sha256'] or cli._hash(mask) != row['mask_sha256']:
                    raise ValueError('Capture changed while copying')
            # Preserve the actual raw receipt bytes and its actual bridge IDs.
            shutil.copyfile(receipt_file,staging/'comparison/raw-producer-receipt.json')
            if cli._json(staging/'comparison/raw-producer-receipt.json') != raw:
                raise ValueError('Producer receipt changed during import')
            comparison = {**raw,'role':'comparison','render_batch_role':'all',
                          'images':[row for row in raw['images'] if row['camera_id'] != cameras['observation']['id']]}
            observed = {**raw,'role':'observation','render_batch_role':'all',
                        'images':[row for row in raw['images'] if row['camera_id'] == cameras['observation']['id']]}
            _write(staging/'comparison/render-receipt.json',_normalized_render(comparison))
            observed = _normalized_render(observed)
            observed['images'][0].update(file='../public/observation.png',mask_file='observation-mask.png')
            _write(staging/'reference/observation-render-receipt.json',observed)
            source_paths = {row['path'] for row in inventory['source_files']}
            for relative in source_paths:
                frozen = 'reference/'+relative
                if frozen not in original_hashes or cli._hash(staging/frozen) != original_hashes[frozen]:
                    raise ValueError('Original source closure changed')
            _write(staging/'reference/capture-import.json',{
                'schema':'simready-capture-import-provenance-v1',
                'baseline_eligibility_sha256':cli._hash(task/config['eligibility']),
                'baseline_inventory_sha256':cli._hash(task/'asset-inventory.json'),
                'raw_producer_receipt_sha256':cli._hash(receipt_file),
                'source_scene_sha256':scene,'render_compatibility':seal})
            if (staging/'provenance.json').exists():
                provenance = cli._json(staging/'provenance.json')
                provenance['capture'] = {
                    'recipe_sha256':raw['recipe_sha256'],
                    'bridge_code_sha256':raw['bridge_code_sha256'],
                    'renderer_binary_sha256':raw['renderer_binary_sha256'],
                    'camera_sha256':raw['camera_sha256'],
                    'scene_sha256':raw['scene_sha256'],
                    'image_count':5,'mask_count':5,
                    'render_compatibility':seal}
                _write(staging/'provenance.json',provenance)
            eligibility.update(schema='simready-reference-eligibility-v2',
                render_compatibility=seal,render_recipe_sha256=raw['recipe_sha256'],
                render_bridge_code_sha256=raw['bridge_code_sha256'],
                renderer_binary_sha256=raw['renderer_binary_sha256'])
            eligibility['files'] = [_row(path,staging) for role in ('reference','comparison')
                for path in sorted((staging/role).rglob('*')) if path.is_file() and path != staging/config['eligibility']]
            _write(staging/config['eligibility'],eligibility)
            # Rebuild derived roles from the new image/eligibility bytes. Never
            # carry v1 paths or eight-sample observation hashes into the v2 set.
            roles=_role_inventory(staging,config['task_id'],'v2')
            _validate_role_inventory(staging,config['task_id'],roles,'v2')
            _write(staging/'role-inventory.json',roles)
            inventory['files'] = [_row(path,staging) for path in sorted(staging.rglob('*'))
                                  if path.is_file() and path.name != 'asset-inventory.json']
            for row in inventory['files']:
                row['role'] = Path(row['path']).parts[0] if '/' in row['path'] else 'configuration'
            _write(staging/'asset-inventory.json',inventory)
            cli._fixtures(staging,config)
            # Verify source task remained unchanged throughout the transaction.
            for relative,digest in original_hashes.items():
                if cli._hash(cli._regular(task,relative)) != digest:
                    raise ValueError('Original task changed during import')
            if output.exists() or output.is_symlink():
                raise ValueError('Fresh output required')
            staging.rename(output)
        result.update(status='prepared',task_id=config['task_id'],output=str(output),
                      render_compatibility=seal,asset_inventory_sha256=cli._hash(output/'asset-inventory.json'))
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        result.update(reason='capture_import_unavailable',error=str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('task','receipt','capture-root','output'):
        parser.add_argument('--'+argument,required=True,type=Path)
    args = parser.parse_args(argv)
    result = import_capture(args.task,args.receipt,args.capture_root,args.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status'] == 'prepared' else 2


if __name__ == '__main__':
    raise SystemExit(main())
