"""Harness paths must not introduce an unfulfillable NP.005 naming failure."""
from pathlib import Path
from simready_eval import cli
from simready_eval.feedback import FeedbackService
from tests.harbor_native.test_feedback import archive, digest


def test_feedback_uses_canonical_asset_root_and_preserves_candidate(tmp_path):
    data=archive({'asset.usd':b'#usda 1.0\n','dependency.usda':b'#usda 1.0\n'})
    seen=[]
    def inspect(root):
        root=Path(root);seen.append(root)
        assert root.parent.name=='asset'
        assert root.name=='candidate'
        assert sorted(p.name for p in root.iterdir())==['asset.usd','dependency.usda']
        assert (root/'asset.usd').read_bytes()==b'#usda 1.0\n'
        return {'outcome':'incomplete','score_percent':0}
    service=FeedbackService(tmp_path/'state',evaluator=inspect)
    try:
        status,result=service.validate(data,'canonical',digest(data))
        assert status==200
        assert result['result']['outcome']=='incomplete',result
        assert len(seen)==1
    finally:service.close()


def test_final_freeze_and_formal_worker_share_canonical_asset_root(tmp_path,monkeypatch):
    root=tmp_path/'submission';root.mkdir();(root/'asset.usd').write_bytes(b'#usda 1.0\n')
    seen=[]
    monkeypatch.setattr(cli,'_configuration',lambda c:c)
    monkeypatch.setattr(cli,'_fixtures',lambda *args:({'files':[],'profile':{},'profile_sha256':'p','render_recipe_sha256':'r','render_camera_sha256':'c'},None,None))
    def worker(mode,request,*args,**kwargs):
        if mode=='reference':return {'status':'verified'}
        assert mode=='formal'
        path=Path(request['root']);seen.append(path)
        assert path.parent.name=='asset'
        assert path.name=='candidate'
        assert (path/'asset.usd').read_bytes()==(root/'asset.usd').read_bytes()
        return {'simready':{'applicable_count':46,'passed_count':45,'failed_count':1,'score_percent':95*45/46},'outcome':'incomplete'}
    monkeypatch.setattr(cli,'_run_worker',worker)
    report=cli.evaluate_submission(root,tmp_path,{'task_id':'layout','reference':'reference/asset.usd','entrypoint':'asset.usd','final_seconds':30,'fidelity_settings':{'enabled':False}},tmp_path/'out')
    assert len(seen)==1
    assert report['errors']==[]
