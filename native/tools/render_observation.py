#!/usr/bin/env python3
"""Render one selected task observation with the already qualified GPU producer.

This offline driver preserves comparison captures and the actual producer code.
Run in the existing GPU render environment with the original task read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

SELECTOR_SHA256='be40496be6cff86727d2f01f62e34fc17290b22205a3808c9761450fe0aa0f6b'
PRODUCER_SHA256='c723f99be642c2d3b2d886df45532d7d6ae15616ebeec72ad3959d0e3e1988c2'
WORKER_SHA256='b63697790a679b558fc2e3d4a55c51fc2514a1439a3af01599e09cf914b39912'
BINARY_SHA256='dc72290ee8651c93c4a946c012c5f2a034946fd320e6c3ab214fa23181427428'
RECIPE_SHA256='514dbf82a3fbf2ba24f8fb1462a54df6132eb895361f7424389b303034ff9d64'
BRIDGE_SHA256='2e515052b919ac7eff777d2119cf9fc41e29d99aae0b922f70712572d580b419'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024**2):value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>16*1024**2:raise ValueError('Bounded regular JSON required')
    return json.loads(path.read_text())


def load_producer(path):
    path=Path(path).absolute()
    if (path.resolve()!=path or digest(path)!=PRODUCER_SHA256
            or digest(path.with_name('render_worker.py'))!=WORKER_SHA256):
        raise ValueError('Unqualified GPU producer source')
    spec=importlib.util.spec_from_file_location('simready_eval.render',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if module._identity(module.RECIPE)!=RECIPE_SHA256:raise ValueError('Unqualified GPU recipe')
    return module


def load_selector(path):
    path=Path(path).absolute()
    if path.resolve()!=path or digest(path)!=SELECTOR_SHA256:raise ValueError('Unqualified observation selector')
    spec=importlib.util.spec_from_file_location('observation_selector',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check_output(output,inputs):
    output=Path(output).absolute()
    if output.resolve()!=output or any(output.is_relative_to(Path(path).resolve()) or Path(path).resolve().is_relative_to(output) for path in inputs):
        raise ValueError('Output must be regular and separate from protected inputs')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Fresh empty output directory required')
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--cameras',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--producer-render',type=Path,default=Path('/diag/render.py'))
    parser.add_argument('--blender',type=Path,default=Path('/blender/blender'))
    parser.add_argument('--selector',type=Path,default=Path(__file__).with_name('observation.py'))
    parser.add_argument('--previous-selection',type=Path)
    args=parser.parse_args(argv)
    result={'status':'unknown','role':'observation'}
    try:
        task=args.task.absolute()
        if task.resolve()!=task:raise ValueError('Regular task directory required')
        output=check_output(args.output,[task,args.cameras.parent])
        meta=read_json(task/'provenance.json');old_cameras=read_json(task/'comparison/cameras.json')
        selected=read_json(args.cameras);policy=selected.get('observation_policy',{})
        selector=load_selector(args.selector)
        expected=selector.freeze_observation(old_cameras,meta['reference_bounds_m_Z_up'],meta['task_id'],
            candidate_index=policy.get('candidate_index',0),reselection_reason=policy.get('reselection_reason'))
        if selected!=expected:raise ValueError('Selected camera differs from canonical task/index policy')
        previous_sha=None
        if policy['candidate_index']>0:
            if args.previous_selection is None:raise ValueError('Pixel-reviewed reselection requires the previous selection file')
            previous=read_json(args.previous_selection);prior_policy=previous.get('observation_policy',{})
            prior_expected=selector.freeze_observation(old_cameras,meta['reference_bounds_m_Z_up'],meta['task_id'],
                candidate_index=prior_policy.get('candidate_index',0),reselection_reason=prior_policy.get('reselection_reason'))
            if previous!=prior_expected or prior_policy['candidate_index']>=policy['candidate_index']:
                raise ValueError('Invalid previous task selection')
            previous_sha=selector._identity(previous)
        if digest(args.blender)!=BINARY_SHA256:raise ValueError('Unqualified GPU Blender binary')
        producer=load_producer(args.producer_render)
        entry=PurePosixPath(meta['source_entrypoint'])
        if entry.is_absolute() or '..' in entry.parts:raise ValueError('Confined source entrypoint required')
        reference=task/'reference';source=reference/entry
        if not source.resolve().is_relative_to(reference.resolve()):raise ValueError('Source leaves its declared closure')
        output.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.cameras,output/'cameras.json')
        started=time.monotonic()
        result=producer.render_asset(source,selected,output,blender=str(args.blender),timeout=180,
                                     role='observation',package_root=reference)
        result.update(whole_render_seconds=time.monotonic()-started,task=meta['task_id'],asset=meta.get('asset_id'),
            original_triangles=meta.get('reference_triangle_count'),observation_policy=policy,
            observation_selector_sha256=SELECTOR_SHA256,observation_driver_sha256=digest(__file__),
            previous_selection_camera_sha256=previous_sha,selected_camera_file_sha256=digest(args.cameras))
        if result.get('status')=='measured' and (len(result.get('images',[]))!=1
                or result.get('bridge_code_sha256')!=BRIDGE_SHA256
                or result.get('renderer_binary_sha256')!=BINARY_SHA256
                or result.get('recipe_sha256')!=RECIPE_SHA256):
            result.update(status='unknown',reason='observation_producer_receipt_mismatch')
        (output/'diagnostic-receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        result.update(status='unknown',reason='observation_driver_unavailable',error=str(exc))
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status']=='measured' else 2


if __name__=='__main__':raise SystemExit(main())
