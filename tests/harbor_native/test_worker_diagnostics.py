import json
from pathlib import Path
import subprocess
import sys
import time


def test_preserves_worker_result_and_records_actual_allowance_and_bounded_log(tmp_path):
    from simready_eval.diagnostics import observed_process
    output=tmp_path/'out';worker=tmp_path/'scratch'/'compare';worker.mkdir(parents=True)
    log=worker/'worker.log';calls=[]
    def original(command,path,**kwargs):
        calls.append((command,path,kwargs));Path(path).write_bytes(b'x'*70000)
        return {'status':'unknown','reason':'worker_timeout'}
    wrapped=observed_process(original,output,time.monotonic()+590)
    command=['python','-m','simready_eval.cli','_worker','compare','request','result']
    result=wrapped(command,log,timeout=123.5,address_space_bytes=8589934592)
    assert result=={'status':'unknown','reason':'worker_timeout'}
    assert calls==[(command,log,{'timeout':123.5,'address_space_bytes':8589934592})]
    row=json.loads((output/'worker-diagnostics.jsonl').read_text())
    assert row['phase']=='compare' and row['timeout_seconds']==123.5
    assert 580<row['remaining_final_seconds_at_start']<=590
    assert row['log_truncated'] is True
    assert (output/row['worker_log']).stat().st_size==65536
    assert 'command' not in row and 'environment' not in row


def test_diagnostic_write_failure_does_not_change_evaluation_result(tmp_path):
    from simready_eval.diagnostics import observed_process
    output=tmp_path/'file';output.write_text('occupied')
    result={'status':'complete'}
    wrapped=observed_process(lambda *a,**kw:result,output,time.monotonic()+600)
    assert wrapped(['worker'],tmp_path/'compare'/'worker.log',timeout=300) is result


def test_real_child_output_and_returncode_are_captured(tmp_path):
    from simready_eval.diagnostics import observed_process
    def original(command,log,**kwargs):
        Path(log).parent.mkdir(parents=True,exist_ok=True)
        with Path(log).open('wb') as stream:
            child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=kwargs['timeout'])
        return {'status':'complete' if child.returncode==0 else 'unknown','returncode':child.returncode}
    out=tmp_path/'out'
    wrapped=observed_process(original,out,time.monotonic()+600)
    r=wrapped([sys.executable,'-c','import sys; print("diagnostic stderr",file=sys.stderr); sys.exit(7)'],
              tmp_path/'scratch'/'formal'/'worker.log',timeout=2)
    assert r['returncode']==7
    row=json.loads((out/'worker-diagnostics.jsonl').read_text())
    assert row['worker_status']['returncode']==7
    assert 'diagnostic stderr' in (out/row['worker_log']).read_text()


def test_obvious_credential_text_is_redacted_from_retained_log(tmp_path):
    from simready_eval.diagnostics import observed_process
    log=tmp_path/'compare'/'worker.log';log.parent.mkdir()
    sample='sk-'+'examplecredential'*3
    def original(*args,**kwargs):
        log.write_text(sample+'\nAuthorization: Bearer example-token\n')
        return {'status':'complete'}
    output=tmp_path/'out'
    observed_process(original,output,time.monotonic()+300)(['worker'],log,timeout=300)
    retained=(output/'worker-compare.log').read_text()
    assert sample not in retained and 'example-token' not in retained
    assert retained.count('[REDACTED]')==2


def test_entrypoint_wraps_workers_and_restores_cli_functions(tmp_path,monkeypatch):
    from simready_eval import cli,diagnostics
    scratch=tmp_path/'scratch'/'formal';scratch.mkdir(parents=True)
    output=tmp_path/'out'
    def process(command,log,**kwargs):
        Path(log).write_text('worker evidence\n');return {'status':'complete'}
    def worker(mode,request,scratch,deadline,**kwargs):
        return cli._run_process(['worker'],Path(scratch)/'worker.log',timeout=123)
    def main(args):
        assert cli._run_worker('formal',{},scratch,time.monotonic()+500)=={'status':'complete'}
        return 0
    monkeypatch.setattr(cli,'_run_process',process)
    monkeypatch.setattr(cli,'_run_worker',worker)
    monkeypatch.setattr(cli,'main',main)
    assert diagnostics.main(['evaluate','--output',str(output)])==0
    assert cli._run_worker is worker and cli._run_process is process
    row=json.loads((output/'worker-diagnostics.jsonl').read_text())
    assert row['phase']=='formal' and row['timeout_seconds']==123
