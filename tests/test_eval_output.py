"""Exclusive evaluation writers: actual processes and adversarial filesystem changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from scripts.eval_output import EvaluationOutput, OutputOwnershipError

ROOT = Path(__file__).resolve().parents[1]


def child(code, *args):
    return subprocess.Popen([sys.executable, '-B', '-c',
        'import sys; sys.path.insert(0, ' + repr(str(ROOT)) + '); ' + code, *map(str,args)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_two_real_processes_race_one_output(tmp_path):
    output, gate = tmp_path/'output.json', tmp_path/'start'
    code = '''
from pathlib import Path
import time,json
from scripts.eval_output import EvaluationOutput
out,gate,mark=map(Path,sys.argv[1:])
deadline=time.monotonic()+10
while not gate.exists():
    if time.monotonic()>deadline: raise TimeoutError('barrier')
    time.sleep(.01)
try:
    with EvaluationOutput(out) as writer:
        mark.write_text('initialized')
        writer.write({'complete':False,'owner':mark.name})
        time.sleep(.2)
        writer.write({'complete':True,'owner':mark.name})
    print('owner')
except FileExistsError:
    print('blocked')
'''
    processes=[child(code,output,gate,tmp_path/f'model-{i}') for i in range(2)]
    try:
        gate.write_text('start')
        results=[p.communicate(timeout=15) for p in processes]
        assert sorted(stdout.strip() for stdout,stderr in results)==['blocked','owner'],results
        assert all(p.returncode==0 for p in processes),results
        assert len(list(tmp_path.glob('model-*')))==1
        assert json.loads(output.read_text())['complete'] is True
        assert not output.with_name(output.name+'.writer-lock').exists()
        with pytest.raises(FileExistsError):
            with EvaluationOutput(output):
                pytest.fail('completed evidence must not be reused')
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill();p.communicate()


def test_crashed_process_reservation_is_not_reclaimed(tmp_path):
    output=tmp_path/'output.json'
    p=child("from scripts.eval_output import EvaluationOutput; import os; w=EvaluationOutput(sys.argv[1]); w.__enter__(); os._exit(9)",output)
    p.communicate(timeout=10)
    assert p.returncode==9
    before=output.read_bytes()
    with pytest.raises(FileExistsError):
        with EvaluationOutput(output):
            pytest.fail('crashed ownership must not be stolen')
    assert output.read_bytes()==before
    assert json.loads(before)['complete'] is False
    assert output.with_name(output.name+'.writer-lock').exists()


@pytest.mark.parametrize('replace_inode',[False,True])
def test_foreign_evidence_is_never_overwritten_after_detection(tmp_path, replace_inode):
    output=tmp_path/'output.json'
    with EvaluationOutput(output) as writer:
        writer.write({'complete':False,'records':[1]})
        if replace_inode:
            foreign=tmp_path/'foreign'
            foreign.write_bytes(b'FOREIGN EVIDENCE')
            foreign.replace(output)
        else:
            output.write_bytes(b'FOREIGN EVIDENCE')
        with pytest.raises(OutputOwnershipError):
            writer.write({'complete':True})
        with pytest.raises(OutputOwnershipError):
            writer.write({'complete':False,'abort':'second attempt'})
    assert output.read_bytes()==b'FOREIGN EVIDENCE'


def test_failed_atomic_update_keeps_previous_valid_json_and_cleans_temp(tmp_path,monkeypatch):
    output=tmp_path/'output.json'
    with EvaluationOutput(output) as writer:
        writer.write({'complete':False,'records':[1]})
        before=output.read_bytes()
        def fail(*args):
            raise PermissionError('simulated Windows file busy')
        monkeypatch.setattr(Path,'replace',fail)
        with pytest.raises(PermissionError):
            writer.write({'complete':True,'records':[1,2]})
        assert output.read_bytes()==before
        assert json.loads(before)['records']==[1]
        assert not list(tmp_path.glob('*.tmp'))
    assert not writer.lock_path.exists()


def test_serialization_failure_and_foreign_lock_are_preserved(tmp_path):
    output=tmp_path/'output.json'
    with EvaluationOutput(output) as writer:
        before=output.read_bytes()
        with pytest.raises(ValueError):
            writer.write({'value':float('nan')})
        assert output.read_bytes()==before
        writer.lock_path.write_bytes(b'foreign lock')
        with pytest.raises(OutputOwnershipError):
            writer.write({'complete':True})
    assert writer.lock_path.read_bytes()==b'foreign lock'


@pytest.mark.parametrize('entrypoint',['paired','recall'])
def test_cli_reserves_before_initialization_and_preserves_foreign_change(tmp_path,monkeypatch,entrypoint):
    from scripts import paired_acceptance as paired, evaluate_context_recall as recall
    output=tmp_path/'output.json';gold=tmp_path/'gold.json'
    gold.write_text(json.dumps([{'id':'a','question':'q','gold_chunk_ids':['a']}]))
    def metadata(*args):
        assert output.exists()
        output.write_text('FOREIGN EVIDENCE')
        return {'gold_sha256':'unused'}
    if entrypoint=='paired':
        rt=NS(cfg=NS(**paired.COMMON_FLAGS,ENABLE_RERANKER=False,RERANKER_BACKEND='llm',TOP_K=5,RERANK_TOP_N=3),
              index_snapshot=nullcontext,reset_circuit_breaker=lambda:None)
        def runtime(root):
            assert json.loads(output.read_text())['batch_status']=='initializing'
            return rt
        monkeypatch.setattr(paired,'_runtime',runtime)
        monkeypatch.setattr(paired,'_metadata',metadata)
        monkeypatch.setattr(paired,'evaluate_pair',lambda *a,**kw:pytest.fail('must stop before model'))
        monkeypatch.chdir(paired.DEFAULT_ROOT)
        code=paired.main(['--gold',str(gold),'--output',str(output)])
    else:
        monkeypatch.setattr(recall,'_run_metadata',metadata)
        monkeypatch.setattr(recall,'evaluate_version',lambda *a,**kw:pytest.fail('must stop before model'))
        monkeypatch.setattr(sys,'argv',['recall','--gold',str(gold),'--output',str(output)])
        code=recall.main()
    assert code==2 and output.read_text()=='FOREIGN EVIDENCE'
    assert not output.with_name(output.name+'.writer-lock').exists()


@pytest.mark.parametrize('entrypoint',['paired','recall'])
def test_existing_lock_stops_cli_before_runtime(tmp_path,monkeypatch,entrypoint):
    from scripts import paired_acceptance as paired, evaluate_context_recall as recall
    output=tmp_path/'output.json';gold=tmp_path/'gold.json'
    gold.write_text('[{"id":"a","question":"q","gold_chunk_ids":["a"]}]')
    lock=output.with_name(output.name+'.writer-lock');lock.write_bytes(b'existing lock')
    monkeypatch.setattr(paired,'_runtime',lambda *a:pytest.fail('runtime not allowed'))
    monkeypatch.setattr(recall,'_run_metadata',lambda *a:pytest.fail('metadata not allowed'))
    with pytest.raises(SystemExit) as error:
        if entrypoint=='paired':
            paired.main(['--gold',str(gold),'--output',str(output)])
        else:
            monkeypatch.setattr(sys,'argv',['recall','--gold',str(gold),'--output',str(output)])
            recall.main()
    assert error.value.code==2 and lock.read_bytes()==b'existing lock' and not output.exists()


def test_template_creation_race_does_not_overwrite(tmp_path,monkeypatch):
    from scripts import evaluate_context_recall as recall
    output=tmp_path/'gold.json'
    def load(path):
        output.write_bytes(b'OTHER TEMPLATE')
        return [{'question':'q'}]
    monkeypatch.setattr(recall,'_load_json',load)
    monkeypatch.setattr(sys,'argv',['recall','--init-template','--gold',str(output)])
    with pytest.raises(SystemExit) as error:
        recall.main()
    assert error.value.code==2 and output.read_bytes()==b'OTHER TEMPLATE'


def test_initialization_failure_keeps_aborted_json_and_releases_lock(tmp_path,monkeypatch):
    from scripts import paired_acceptance as paired
    output=tmp_path/'output.json';gold=tmp_path/'gold.json'
    gold.write_text('[{"id":"a","question":"q","gold_chunk_ids":["a"]}]')
    def fail(root):
        raise RuntimeError('private initialization error')
    monkeypatch.setattr(paired,'_runtime',fail)
    monkeypatch.chdir(paired.DEFAULT_ROOT)
    assert paired.main(['--gold',str(gold),'--output',str(output)])==2
    report=json.loads(output.read_text())
    assert report['complete'] is False and report['batch_status']=='aborted'
    assert 'private' not in output.read_text()
    assert not output.with_name(output.name+'.writer-lock').exists()


def test_interrupt_retains_checkpoint_records_and_releases_lock(tmp_path, monkeypatch):
    from scripts import evaluate_context_recall as recall
    output=tmp_path/'output.json';gold=tmp_path/'gold.json'
    gold.write_text('[{"id":"a","question":"q","gold_chunk_ids":["a"]}]')
    def interrupt(version, cases, **kwargs):
        kwargs['progress_fn']([{'id':'completed-evidence','context_recall':1}])
        raise KeyboardInterrupt()
    monkeypatch.setattr(recall,'evaluate_version',interrupt)
    monkeypatch.setattr(sys,'argv',['recall','--gold',str(gold),'--output',str(output),
                                    '--reference-time','2026-11-02T12:00:00+08:00'])
    with pytest.raises(KeyboardInterrupt):
        recall.main()
    report=json.loads(output.read_text())
    assert report['complete'] is False and report['batch_status']=='aborted'
    assert report['results'][0]['records']==[{'id':'completed-evidence','context_recall':1}]
    assert report['results'][0]['summary'] is None
    assert not output.with_name(output.name+'.writer-lock').exists()


@pytest.mark.parametrize('entrypoint',['paired','recall'])
@pytest.mark.parametrize('failure_mode',['final_once','final_and_abort','cleanup'])
def test_final_write_and_cleanup_failures_never_return_success(tmp_path,monkeypatch,entrypoint,failure_mode):
    from scripts import paired_acceptance as paired, evaluate_context_recall as recall
    from tests.test_paired_acceptance import runtime, cases
    output=tmp_path/'output.json';gold=tmp_path/'gold.json'
    gold.write_text(json.dumps(cases()[:1]))
    monkeypatch.setattr(paired,'_source_hashes',lambda root:{})
    actual_write=EvaluationOutput.write
    actual_close=EvaluationOutput.close
    failures=[]
    def write(writer,payload):
        if failure_mode!='cleanup' and (payload.get('complete') is True or
                failure_mode=='final_and_abort' and payload.get('batch_status')=='aborted'):
            failures.append(payload.get('batch_status'))
            raise PermissionError('simulated disk failure')
        return actual_write(writer,payload)
    def close(writer):
        actual_close(writer)
        if failure_mode=='cleanup':
            raise PermissionError('simulated cleanup failure')
    monkeypatch.setattr(EvaluationOutput,'write',write)
    monkeypatch.setattr(EvaluationOutput,'close',close)
    if entrypoint=='paired':
        rt=runtime(tmp_path)
        monkeypatch.setattr(paired,'_runtime',lambda root:rt)
        monkeypatch.setattr(paired,'_metadata',lambda *args:{'source_sha256':{},
            'gold_sha256':paired._hash(gold),'corpus_sha256':paired._hash(rt.corpus_path)})
        monkeypatch.setattr(paired,'_resolved_revisions',lambda:{})
        monkeypatch.chdir(paired.DEFAULT_ROOT)
        code=paired.main(['--gold',str(gold),'--output',str(output)])
    else:
        def evaluate(version,items,**kwargs):
            record=recall.score_context_recall_case(items[0],[])
            record['status']='empty'
            kwargs['progress_fn']([record])
            return {'version':version,'records':[record], 'summary':paired.summarize([record])}
        monkeypatch.setattr(recall,'evaluate_version',evaluate)
        monkeypatch.setattr(sys,'argv',['recall','--gold',str(gold),'--output',str(output),'--versions','V2',
                                       '--reference-time','2026-11-02T12:00:00+08:00'])
        code=recall.main()
    assert code==2
    saved=json.loads(output.read_text())
    if failure_mode=='cleanup':
        # Evaluation and publication succeeded; cleanup failure is not a successful process exit.
        assert saved['complete'] is True and saved['batch_status']=='completed'
    else:
        assert saved['complete'] is False
        assert all(result['summary'] is None for result in saved['results'])
        assert all(len(result['records'])==1 for result in saved['results'])
        if failure_mode=='final_once':
            assert saved['batch_status']=='aborted' and failures==['completed']
        else:
            assert saved['batch_status']=='running' and failures==['completed','aborted']


def test_invalid_paired_candidate_limits_report_value_error_not_parser_name_error(tmp_path,monkeypatch):
    from scripts import paired_acceptance as paired
    from tests.test_paired_acceptance import runtime
    output=tmp_path/'output.json';gold=tmp_path/'gold.json'
    gold.write_text('[{"id":"a","question":"q","gold_chunk_ids":["a"]}]')
    rt=runtime(tmp_path);rt.cfg.TOP_K=0
    monkeypatch.setattr(paired,'_runtime',lambda root:rt)
    monkeypatch.chdir(paired.DEFAULT_ROOT)
    assert paired.main(['--gold',str(gold),'--output',str(output)])==2
    saved=json.loads(output.read_text())
    assert saved['complete'] is False and saved['abort']['error_type']=='ValueError'
