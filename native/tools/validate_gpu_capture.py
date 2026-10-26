#!/usr/bin/env python3
"""Stage one cached task, import completed GPU frames and inspect a CPU final.

Run in the pinned, offline CPU verifier container. This diagnostic uses a
supplied exact-source candidate; it never calls a model or publishes anything.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import time

from simready_eval import cli, contract, render
from simready_eval.capture import import_capture
from simready_eval.limits import CANDIDATE_BYTES, MAX_ENTRIES
from simready_eval.qualification import QUALIFICATION, evaluator_identity


def stage_regular(source, target, *, maximum_bytes, maximum_entries=MAX_ENTRIES+8):
    """Copy bounded regular cache files independently; retain strict admission.

    A shared read-only download cache may hardlink identical licenses. Each
    staged file gets its own inode before the normal evaluator boundary runs.
    Symlinks, special files and executables are never admitted by this helper.
    """
    source,target=Path(source).absolute(),Path(target).absolute()
    if source.resolve()!=source or not source.is_dir() or target.exists() or target.is_symlink():
        raise ValueError('Regular source and fresh staging directory required')
    if target.parent.resolve()!=target.parent or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Staging must be separate from source')
    files=[];total=0;entries=0
    for directory,dirs,names in os.walk(source,followlinks=False):
        for name in sorted(dirs+names):
            path=Path(directory)/name;mode=path.lstat()
            entries+=1
            if entries>maximum_entries:raise ValueError('Cache entry bound exceeded')
            if stat.S_ISDIR(mode.st_mode):continue
            if not stat.S_ISREG(mode.st_mode) or mode.st_mode&0o111:
                raise ValueError('Cache must contain regular nonexecutable files')
            total+=mode.st_size
            if total>maximum_bytes:raise ValueError('Cache byte bound exceeded')
            files.append(path)
    target.mkdir()
    for source_file in files:
        relative=source_file.relative_to(source)
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_file,destination)
        cli._regular(target,relative.as_posix())
    return {'files':len(files),'size_bytes':total}


def check_result(report,reward,eligibility):
    if (report.get('outcome')!='complete' or report.get('errors')
            or report.get('candidate_invalid') or reward!=contract.reward_record(report)
            or reward.get('evaluation_complete')!=1):
        raise ValueError('Complete, measured report and matching rewards required')
    receipt=report.get('render',{})
    if receipt.get('status')!='measured':raise ValueError('Measured candidate render required')
    expected=evaluator_identity()
    if any(receipt.get(key)!=value for key,value in expected.items()):
        raise ValueError('Candidate evaluator identity mismatch')
    actual=receipt.get('renderer',{})
    if (actual.get('version')!='4.3.2' or actual.get('device')!='CPU'
            or actual.get('threads')!=4 or actual.get('samples')!=256
            or actual.get('recipe')!=render.RECIPE):
        raise ValueError('Actual CPU256/four-thread execution required')
    appearance=receipt.get('appearance',{})
    if (len(receipt.get('images',[]))!=4 or len(appearance.get('views',[]))!=4
            or appearance.get('coverage_fraction')!=1
            or any(view.get('status')!='measured' for view in appearance['views'])):
        raise ValueError('Four measured candidate appearance views required')
    seal=eligibility.get('render_compatibility')
    if (eligibility.get('schema')!='simready-reference-eligibility-v2' or not isinstance(seal,dict)
            or seal.get('qualification')!=QUALIFICATION
            or report.get('reference',{}).get('render_compatibility')!=seal):
        raise ValueError('Separate producer/evaluator provenance required')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('task','receipt','capture-root','submission','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    root=args.output.absolute()
    if root.exists() or root.is_symlink():parser.error('output must be fresh')
    protected=[path.absolute().resolve() for path in (args.task,args.capture_root,args.submission)]
    if root.resolve()!=root or any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        parser.error('output must have regular ancestors and be separate from all inputs')
    root.mkdir(parents=True)
    result={'schema':'simready-qualified-capture-validation-v1','status':'failed'}
    try:
        result['task_stage']=stage_regular(args.task,root/'original-task',maximum_bytes=2*1024**3)
        result['candidate_stage']=stage_regular(args.submission,root/'candidate',maximum_bytes=CANDIDATE_BYTES)
        imported=import_capture(root/'original-task',args.receipt,args.capture_root,root/'fixtures-v2')
        result['capture_import']=imported
        if imported['status']!='prepared':raise ValueError(imported.get('error','Capture import failed'))
        config=cli._configuration(root/'fixtures-v2/evaluator.json')
        if config['final_seconds']!=600:raise ValueError('Existing600-second final budget required')
        started=time.monotonic()
        report=cli.evaluate_submission(root/'candidate',root/'fixtures-v2',config,root/'evaluation')
        result['evaluation_seconds']=time.monotonic()-started
        reward=cli._json(root/'evaluation/reward.json')
        eligibility=cli._json(root/'fixtures-v2/reference/eligibility.json')
        check_result(report,reward,eligibility)
        result.update(status='passed',rewards=reward,
            scores={key:value.get('score_percent') for key,value in report['scores'].items()},
            report_sha256=cli._hash(root/'evaluation/report.json'),
            reward_sha256=cli._hash(root/'evaluation/reward.json'))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
    (root/'validation.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
