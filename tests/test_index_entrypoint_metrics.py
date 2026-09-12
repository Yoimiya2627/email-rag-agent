"""Index entry points share metrics and only invalidate changed generations."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from core.index_metrics import current_metrics, add_count, measure_stage, set_outcome


class Plan(list):
    closed = False
    def close(self):
        self.closed = True


@pytest.mark.parametrize('entrypoint',['api','background','cli','sync'])
@pytest.mark.parametrize('outcome',['unchanged','published','pending'])
def test_entrypoints_collect_preparation_and_index_and_skip_only_true_noop(monkeypatch,capsys,entrypoint,outcome):
    from api import main as api
    from scripts import index_emails as cli
    from scripts import sync_gmail_readonly as sync
    import core.ingestion as ingestion
    import core.embedder as embedder
    import core.retriever as retriever
    plan=Plan([1,2]);chunks=['c1','c2','c3'];reports=[]
    def prepare(*args):
        assert current_metrics() is not None
        reports.append(current_metrics())
        with measure_stage('validation'):
            add_count('input_emails',len(plan))
        return plan,chunks
    def index(*args,**kwargs):
        assert current_metrics() is reports[0]
        add_count('input_chunks',3)
        add_count('reused_chunks',3 if outcome=='unchanged' else 1)
        add_count('embedded_chunks',0 if outcome=='unchanged' else 2)
        if outcome!='pending': set_outcome(outcome)
        return 3
    monkeypatch.setattr(ingestion,'prepare_email_chunks',prepare)
    monkeypatch.setattr(cli,'prepare_email_chunks',prepare)
    monkeypatch.setattr(api,'index_chunks',index)
    monkeypatch.setattr(cli,'index_chunks',index)
    monkeypatch.setattr(embedder,'index_chunks',index)
    invalidate=Mock();monkeypatch.setattr(retriever,'invalidate_bm25_cache',invalidate)
    monkeypatch.setattr(cli,'get_collection_stats',lambda:{'chunk_count':3})
    monkeypatch.setattr(embedder,'get_collection_stats',lambda:{'chunk_count':3})
    if entrypoint=='api':
        from api.security import Identity
        from models.schemas import IndexRequest
        result=api.index_emails(IndexRequest(),Identity('local')).model_dump()
    elif entrypoint=='background':
        result=api._run_background_job({'kind':'index','owner':'local','id':'job','request':{},'checkpoint':None},None,Mock(),Mock())
    elif entrypoint=='cli':
        monkeypatch.setattr('sys.argv',['index_emails.py','--json-report'])
        cli.main();result=json.loads(capsys.readouterr().out)
    else:
        result=sync.index_email_json('synthetic-input')
    assert plan.closed
    assert invalidate.call_count == int(outcome!='unchanged')
    metrics=result['index_metrics']
    assert metrics['outcome']==outcome
    assert metrics['counts']['input_chunks']==3
    assert metrics['stages_seconds']['validation']>=0
    assert metrics['total_seconds']>=metrics['stages_seconds']['validation']


@pytest.mark.parametrize('entrypoint',['cli','sync'])
def test_force_reembed_is_explicit_and_forwarded(monkeypatch,capsys,entrypoint):
    from scripts import index_emails as cli
    from scripts import sync_gmail_readonly as sync
    import core.embedder as embedder
    plan=Plan([1]);chunks=['c']
    monkeypatch.setattr(cli,'prepare_email_chunks',lambda *a:(plan,chunks))
    encode=Mock(return_value=1)
    monkeypatch.setattr(cli,'index_chunks',encode);monkeypatch.setattr(embedder,'index_chunks',encode)
    monkeypatch.setattr(cli,'get_collection_stats',lambda:{'chunk_count':1})
    monkeypatch.setattr(embedder,'get_collection_stats',lambda:{'chunk_count':1})
    monkeypatch.setattr('core.retriever.invalidate_bm25_cache',Mock())
    if entrypoint=='cli':
        monkeypatch.setattr('sys.argv',['index_emails.py','--force-reembed'])
        cli.main()
    else:
        sync.index_email_json('synthetic',force_reembed=True)
    assert encode.call_args.kwargs['force_reembed'] is True
    assert plan.closed


def test_sync_closes_plan_after_index_failure(monkeypatch):
    from scripts import index_emails as cli
    from scripts.sync_gmail_readonly import index_email_json
    plan=Plan([1])
    monkeypatch.setattr(cli,'prepare_email_chunks',lambda *a:(plan,['c']))
    monkeypatch.setattr('core.embedder.index_chunks',Mock(side_effect=RuntimeError('synthetic index failure')))
    invalidate=Mock();monkeypatch.setattr('core.retriever.invalidate_bm25_cache',invalidate)
    with pytest.raises(RuntimeError): index_email_json('synthetic')
    assert plan.closed
    invalidate.assert_not_called()


def test_cli_failure_json_report_has_no_exception_body(monkeypatch,tmp_path):
    from scripts import index_emails as cli
    report_path=tmp_path/'report.json'
    monkeypatch.setattr('sys.argv',['index_emails.py','--json-report',str(report_path)])
    monkeypatch.setattr(cli,'prepare_email_chunks',Mock(side_effect=ValueError('sensitive synthetic body')))
    with pytest.raises(ValueError): cli.main()
    report=json.loads(report_path.read_text(encoding='utf-8'))
    assert report['index_metrics']['outcome']=='failed'
    assert 'sensitive synthetic body' not in report_path.read_text(encoding='utf-8')


def test_index_ui_whitelists_numeric_progress_and_final_counts():
    root=Path(__file__).resolve().parents[1]
    module=ast.parse((root/'frontend/app.py').read_text(encoding='utf-8'))
    function=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='_render_index_metrics')
    captions=[];bars=[]
    namespace={'st':SimpleNamespace(caption=captions.append,progress=lambda value,**kwargs:bars.append((value,kwargs)))}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual frontend index renderer>','exec'),namespace)
    namespace['_render_index_metrics']({'stage':'index_embed','completed_chunks':5,'total_chunks':10,
        'reused_chunks':3,'embedded_chunks':2,'chunks_per_second':1.5,'remaining_seconds':float('inf'),
        'access_token':'MUST_NOT_RENDER','untrusted':'MUST_NOT_RENDER'},
        {'outcome':'published','counts':{'new_emails':2,'deleted_emails':1},'total_seconds':4.2})
    text=' '.join(captions)
    assert bars[0][0]==0.5 and '复用 3' in text and '新编码 2' in text
    assert '1.5 片段/秒' in text and '索引已发布' in text
    assert 'MUST_NOT_RENDER' not in text and 'inf' not in text
    namespace['_render_index_metrics']({'stage':{},'total_chunks':10**400}, {'outcome':[]})


def test_tasks_benchmark_commands_do_not_install_models():
    text=(Path(__file__).resolve().parents[1]/'tasks.ps1').read_text(encoding='utf-8')
    assert '"benchmark-check" { Invoke-Python scripts/benchmark_index.py --check --input data/emails.json --max-emails 10000 }' in text
    assert '"benchmark-index" { Invoke-Python scripts/benchmark_index.py --run --input data/emails.json --max-emails 10000 }' in text


def test_background_resume_returns_metrics_and_preserves_noop_cache(monkeypatch):
    from api import main as api
    def resume(identifier):
        assert identifier=='generation-synthetic'
        assert current_metrics() is not None
        set_outcome('unchanged');add_count('input_chunks',4)
        return 4
    monkeypatch.setattr('core.embedder.resume_index_generation',resume)
    invalidator=Mock();monkeypatch.setattr('core.retriever.invalidate_bm25_cache',invalidator)
    result=api._run_background_job({'kind':'index','owner':'local','id':'resume-job','request':{},
        'checkpoint':{'generation_id':'generation-synthetic'}},None,Mock(),Mock())
    assert result['index_metrics']['outcome']=='unchanged'
    assert result['chunk_count']==4
    invalidator.assert_not_called()


def test_actual_frontend_job_renders_index_work_without_untrusted_progress(monkeypatch):
    st_testing=pytest.importorskip('streamlit.testing.v1')
    def get(url,**kwargs):
        if url.endswith('/jobs'):
            value={'jobs':[{'id':'index-synthetic','kind':'index','status':'succeeded',
                'progress':{'stage':'index_complete','completed_chunks':10,'total_chunks':10,'reused_chunks':8,
                            'embedded_chunks':2,'chunks_per_second':5,'remaining_seconds':0,'debug':'NEVER_RENDER_BODY'},
                'result':{'index_metrics':{'outcome':'published','counts':{'input_chunks':10,'reused_chunks':8,
                    'embedded_chunks':2,'unchanged_emails':4},'total_seconds':2}}}]}
        elif url.endswith('/health'): value={'status':'ok'}
        elif url.endswith('/index/status'): value={'email_count':5,'chunk_count':10}
        else: value={'sessions':[],'facts':[],'turns':[],'approvals':[]}
        return SimpleNamespace(ok=True,status_code=200,json=lambda:value,raise_for_status=lambda:None)
    monkeypatch.setattr('requests.get',get)
    app=st_testing.AppTest.from_file(str(Path(__file__).resolve().parents[1]/'frontend/app.py'),default_timeout=15).run()
    next(box for box in app.checkbox if box.label=='显示后台任务').check().run()
    assert not app.exception
    captions=' '.join(str(element.value) for element in app.caption)
    assert '复用 8' in captions and '新编码 2' in captions and '索引已发布' in captions
    assert 'NEVER_RENDER_BODY' not in captions


@pytest.mark.parametrize('outcome',['unchanged','published','pending'])
@pytest.mark.parametrize('entrypoint',['api','sync_empty'])
def test_clear_entrypoints_report_metrics_and_only_skip_cache_on_unchanged(monkeypatch,capsys,outcome,entrypoint):
    from api import main as api
    from scripts import sync_gmail_readonly as sync
    def clear():
        assert current_metrics() is not None
        if outcome!='pending': set_outcome(outcome)
        add_count('input_chunks',0)
    monkeypatch.setattr(api,'clear_collection',clear)
    monkeypatch.setattr('core.embedder.clear_collection',clear)
    invalidator=Mock();monkeypatch.setattr('core.retriever.invalidate_bm25_cache',invalidator)
    if entrypoint=='api':
        from api.security import Identity
        result=api.clear_index(Identity('local'))
        assert result['index_metrics']['outcome']==outcome
    else:
        monkeypatch.setattr('sys.argv',['sync_gmail_readonly.py','--index'])
        monkeypatch.setattr(sync,'GmailReadOnlyProvider',Mock())
        monkeypatch.setattr(sync,'sync_gmail_to_json',lambda **kwargs:{'failed':0,'total':0})
        sync.main()
        text=capsys.readouterr().out
        decoder=json.JSONDecoder();first,offset=decoder.raw_decode(text)
        second,_=decoder.raw_decode(text[offset:].lstrip())
        assert second['index']['index_metrics']['outcome']==outcome
    assert invalidator.call_count==int(outcome!='unchanged')


@pytest.mark.parametrize('use_default',[False,True])
@pytest.mark.parametrize('resume',[False,True])
def test_cli_report_cannot_overwrite_resolved_input_before_any_action(monkeypatch,tmp_path,use_default,resume):
    from scripts import index_emails as cli
    corpus=tmp_path/'emails.json';original=b'[{"id":"synthetic-preserve"}]'
    corpus.write_bytes(original)
    monkeypatch.setattr(cli.cfg,'EMAIL_DATA_PATH',str(corpus))
    target=tmp_path/'not-created'/'..'/'emails.json'
    args=['index_emails.py','--json-report',str(target)]
    if not use_default: args.extend(['--data-path',str(corpus)])
    if resume: args.extend(['--resume-generation','synthetic-generation'])
    monkeypatch.setattr('sys.argv',args)
    prepare=Mock();resumer=Mock()
    monkeypatch.setattr(cli,'prepare_email_chunks',prepare)
    monkeypatch.setattr(cli,'resume_index_generation',resumer)
    with pytest.raises(SystemExit) as caught: cli.main()
    assert caught.value.code==2
    assert corpus.read_bytes()==original
    prepare.assert_not_called();resumer.assert_not_called()


def test_report_atomic_replace_failure_preserves_previous_complete_file(tmp_path):
    from scripts import index_emails as cli
    from unittest.mock import patch
    target=tmp_path/'report.json';original=b'{"previous":"complete"}'
    target.write_bytes(original)
    with patch('core.index_manifest.os.replace',side_effect=OSError('synthetic replace interruption')):
        with pytest.raises(OSError): cli._write_report(target,{'index_metrics':{'outcome':'published'}})
    assert target.read_bytes()==original
    assert not list(tmp_path.glob('.index-*.tmp'))


@pytest.mark.parametrize('cancelled',[False,True])
@pytest.mark.parametrize('stdout',[False,True])
def test_resume_failure_still_emits_requested_safe_metrics(monkeypatch,tmp_path,capsys,cancelled,stdout):
    from scripts import index_emails as cli
    from agents.runtime import RunCancelled
    path=tmp_path/'resume-report.json'
    args=['index_emails.py','--resume-generation','synthetic-generation','--json-report']
    if not stdout: args.append(str(path))
    monkeypatch.setattr('sys.argv',args)
    error_type=RunCancelled if cancelled else RuntimeError
    monkeypatch.setattr(cli,'resume_index_generation',Mock(side_effect=error_type('DO_NOT_WRITE_PRIVATE_BODY')))
    with pytest.raises(error_type): cli.main()
    raw=capsys.readouterr().out if stdout else path.read_text(encoding='utf-8')
    report=json.loads(raw)
    assert report['index_metrics']['outcome']==('cancelled' if cancelled else 'failed')
    assert 'indexed_chunks' not in report
    assert 'DO_NOT_WRITE_PRIVATE_BODY' not in raw
