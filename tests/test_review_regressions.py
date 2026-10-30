"""Synthetic regressions for the full-project review; no provider/model calls."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def preserve_reranker_backend(monkeypatch):
    import config.settings as cfg
    monkeypatch.setattr(cfg,'RERANKER_BACKEND',cfg.RERANKER_BACKEND)


def test_graph_grading_failure_is_not_a_successful_no_results_answer(monkeypatch,tmp_path):
    from agents import graph_workflow as graph
    from models.schemas import SearchResult, AgentRequest, IntentType
    from core.session_repository import SessionRepository
    source=SearchResult(chunk_id='one_0',email_id='one',content='Monday meeting',score=1,metadata={})
    monkeypatch.setattr(graph,'_get_client',lambda:object())
    monkeypatch.setattr(graph,'create_completion',Mock(side_effect=ValueError('synthetic grade failure')))
    state=graph.node_grade_contexts({'query':'查询会议','results':[source]})
    monkeypatch.setattr(graph,'generate_answer',Mock(side_effect=AssertionError('Must not generate from failed grading')))
    graph.node_generate(state)
    monkeypatch.setattr(graph,'get_graph',lambda:NS(invoke=lambda state_in:state))
    monkeypatch.setattr('agents.coordinator.classify_intent',lambda *a,**k:IntentType.RETRIEVE)
    result=graph.run_graph(AgentRequest(query='查询会议'))
    assert result.metadata['status']=='error'
    assert result.metadata['error_code']=='context_grading_failed'
    assert '未找到相关邮件' not in result.answer
    repo=SessionRepository(tmp_path/'sessions.sqlite3')
    repo.append_turns('owner','s',[{'turn_id':'t','query':'查询会议','answer':result.answer,
        'metadata':result.metadata,'include_in_context':True}],expected_revision=0)
    assert not repo.get_turn('owner','s','t')['include_in_context']


def test_graph_successful_retry_clears_previous_grading_failure(monkeypatch):
    from agents import graph_workflow as graph
    from models.schemas import SearchResult
    source=SearchResult(chunk_id='one_0',email_id='one',content='Monday meeting',score=1,metadata={})
    monkeypatch.setattr(graph,'_get_client',lambda:object())
    monkeypatch.setattr(graph,'create_completion',lambda *a,**k:NS(choices=[NS(message=NS(content='[0]'),finish_reason='stop')]))
    monkeypatch.setattr(graph,'generate_answer',lambda *a,**k:'Monday')
    state=graph.node_grade_contexts({'query':'meeting','results':[source],'grading_error':'context_grading_failed'})
    graph.node_generate(state)
    assert state['grading_error'] is None and state['answer_metadata']['status']=='success'


def test_table_parser_upgrade_refreshes_stored_mail(monkeypatch,tmp_path):
    from core import imap_mime, html_tables
    from agents import gmail_readonly
    from tests import test_mail_sync as fixture
    raw=b'Subject: Synthetic\r\nFrom: sender@example.test\r\nDate: Sat, 12 Sep 2026 08:00:00 +0000\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<table><tr><td>Monday</td></tr></table>'
    monkeypatch.setattr(fixture,'RAW',raw)
    provider=fixture.Provider({'INBOX':('1',[1])})
    store=fixture.MailSyncStore(tmp_path,'account-a')
    fixture.perform(provider,store)
    versions=(imap_mime.parser_version(),gmail_readonly.parser_version())
    original_read=Path.read_bytes
    table_path=Path(html_tables.__file__).resolve()
    monkeypatch.setattr(Path,'read_bytes',lambda path:original_read(path)+(b'\n# helper upgrade' if path.resolve()==table_path else b''))
    original_data=html_tables.TableBuilder.data
    monkeypatch.setattr(html_tables.TableBuilder,'data',lambda self,text:original_data(self,text.replace('Monday','Tuesday')))
    assert versions[0]!=imap_mime.parser_version() and versions[1]!=gmail_readonly.parser_version()
    refreshed=fixture.perform(provider,store)
    assert refreshed['last_run']['attempted']==1 and len(provider.calls)==2
    row=store.messages()['items'][0]
    assert 'Tuesday' in store.message(row['key'])['email']['body']
    assert row['parser_version']==imap_mime.parser_version()


@pytest.mark.parametrize('completion,reason,text', [('incomplete','length','The meeting is'),('error','stop','')])
def test_ragas_skips_scoring_unfinished_answers_and_retains_outcome(monkeypatch,completion,reason,text):
    from core import generator,pipeline
    from models.schemas import SearchResult
    from scripts import run_ragas_eval as evaluation
    source=SearchResult(chunk_id='one_0',email_id='one',content='Monday meeting',score=1,metadata={})
    monkeypatch.setattr(pipeline,'retrieve',lambda query:[source])
    monkeypatch.setattr(generator,'_get_client',lambda:object())
    monkeypatch.setattr(generator,'create_completion',lambda *a,**k:NS(choices=[NS(message=NS(content=text),finish_reason=reason)]))
    score=Mock(side_effect=AssertionError('Incomplete answers must not reach judge'))
    monkeypatch.setattr(evaluation,'score_response',score)
    monkeypatch.setattr(evaluation.time,'sleep',lambda *a:None)
    report=evaluation.evaluate_version('V1',[{'question':'When?','ground_truth':'Monday'}],1,object())
    saved=json.loads(json.dumps(report,allow_nan=False))
    assert saved['records'][0]['answer_metadata']['completion_status']==completion
    assert saved['records'][0]['scoring_status']=='skipped'
    assert saved['scoring']['answer_completion_counts']=={completion:1}
    assert saved['scoring']['unscored_count']==1
    assert saved['avg']==dict.fromkeys(evaluation.METRICS,0.0)
    score.assert_not_called()


def test_reranker_benchmark_excludes_failures_and_open_breaker(monkeypatch):
    from core import reranker
    from scripts import measure_reranker_latency as bench
    cases=bench.build_cases([{'question':'query','ground_truth':'answer'}],1,2)
    model=NS(predict=Mock(side_effect=ValueError('synthetic model unavailable')))
    monkeypatch.setattr(reranker,'_get_cross_encoder',lambda:model)
    try:
        report=bench.measure_target(bench.RERANKER_TARGETS['V7'],cases,4)
        assert model.predict.call_count==3
        assert (report['attempted'],report['succeeded'],report['failed'],report['skipped'],report['n'])==(4,0,3,1,0)
        assert report['mean_trimmed_ms'] is None and report['raw_ms']==[]
        assert report['attempts'][-1]['reason']=='circuit_open'
        json.dumps(report,allow_nan=False)
    finally:
        reranker.reset_circuit_breaker()


def test_reranker_benchmark_success_and_disabled_baseline_remain_distinct(monkeypatch):
    from core import reranker
    from scripts import measure_reranker_latency as bench
    cases=bench.build_cases([{'question':'query','ground_truth':'answer'}],1,2)
    model=NS(predict=Mock(return_value=[1.,0.]))
    monkeypatch.setattr(reranker,'_get_cross_encoder',lambda:model)
    ranked=bench.measure_target(bench.RERANKER_TARGETS['V7'],cases,2)
    baseline=bench.measure_target(bench.RERANKER_TARGETS['V2'],cases,2)
    assert ranked['n']==ranked['succeeded']==2 and ranked['failed']==ranked['skipped']==0
    assert baseline['n']==baseline['disabled']==2 and baseline['succeeded']==0
    assert model.predict.call_count==2


def test_generated_ids_remain_unique_across_existing_suffix_collisions(tmp_path):
    from scripts.generate_emails import generate_batch
    from core.ingestion import prepare_email_chunks
    row={'id':'same','subject':'Synthetic','sender':'sender@example.test','recipients':['reader@example.test'],
         'date':'2026-09-13','body':'Synthetic meeting schedule and detailed agenda. '*5}
    reply=NS(choices=[NS(message=NS(content=json.dumps([row,row,row])))])
    used={'same','same_b0','same_b0_2'}
    rows=generate_batch(NS(chat=NS(completions=NS(create=lambda **k:reply))),0,3,used)
    ids=[item['id'] for item in rows]
    assert len(set(ids))==3 and not set(ids).intersection({'same','same_b0','same_b0_2'})
    path=tmp_path/'generated.json';path.write_text(json.dumps(rows),encoding='utf-8')
    emails,chunks=prepare_email_chunks(path)
    try:
        assert len(emails)==3 and len(chunks)>0
    finally:
        emails.close()


def test_langchain_retriever_follows_active_generation_and_checks_embedding(monkeypatch,tmp_path):
    chromadb=pytest.importorskip('chromadb')
    documents=pytest.importorskip('langchain_core.documents')
    retrievers=pytest.importorskip('langchain_core.retrievers')
    from chromadb.config import Settings
    from core import embedder,index_manifest
    from langchain_version import rag_chain
    from models.schemas import EmailChunk
    import config.settings as cfg
    monkeypatch.setattr(cfg,'CHROMA_PERSIST_DIR',str(tmp_path/'chroma'))
    monkeypatch.setattr(cfg,'EMBEDDING_MODEL_REVISION','offline-v1')
    monkeypatch.setattr(cfg,'EMBEDDING_DIMENSION',2)
    client=chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR,settings=Settings(anonymized_telemetry=False))
    monkeypatch.setattr(embedder,'_get_client',lambda:client)
    monkeypatch.setattr(embedder,'resolved_model_revision',lambda:'offline-v1')
    monkeypatch.setattr(embedder,'embed_texts',lambda texts:[[1.,0.] for text in texts])
    monkeypatch.setattr(rag_chain,'_HAS_LANGCHAIN',True)
    for name,value in {'Document':documents.Document,'BaseRetriever':retrievers.BaseRetriever,
                       'ConversationalRetrievalChain':NS(from_llm=lambda **k:k['retriever']),
                       'ChatOpenAI':lambda **k:object(),'ConversationBufferWindowMemory':lambda **k:object(),
                       'PromptTemplate':lambda **k:object()}.items():
        monkeypatch.setattr(rag_chain,name,value,raising=False)
    try:
        retriever=rag_chain.build_chain(k=1)
        for label in ('first','second'):
            embedder.index_chunks([EmailChunk(chunk_id=label+'_0',email_id=label,chunk_index=0,
                content=label,metadata={'source_start':0,'source_end':len(label)})],replace=True)
            assert retriever.invoke('query')[0].page_content==label
        monkeypatch.setattr(embedder,'resolved_model_revision',lambda:'different-revision')
        with pytest.raises(index_manifest.IndexCompatibilityError,match='revision'):
            retriever.invoke('query')
    finally:
        client.close()


def test_charts_derive_labels_from_new_data_and_handle_no_success(monkeypatch,tmp_path):
    matplotlib=pytest.importorskip('matplotlib');matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from scripts import generate_charts as charts
    monkeypatch.setattr(charts,'OUT_DIR',tmp_path)
    figures=[]
    monkeypatch.setattr(Figure,'savefig',lambda self,*a,**k:figures.append(self))
    summaries=[{'version':v,'records':[{},{}],'avg':dict.fromkeys(('answer_relevancy','faithfulness','context_precision'),score)}
               for v,score in [('V1',1.0),('V2',0.0),('V4',0.5),('V5',0.5)]]
    charts.make_radar(summaries)
    labels=figures[-1].axes[0].get_legend_handles_labels()[1]
    assert 'V1 (Relevancy/Faithfulness/Precision max)' in labels and 'V2' in labels
    assert figures[-1].axes[0].get_ylim()==(0.,1.)
    assert 'V1=2' in figures[-1].axes[0].get_title() and '30 题' not in figures[-1].axes[0].get_title()
    charts.make_latency([{'version':'V2','mean_trimmed':1.,'p95':1.1,'attempted':3,'succeeded':2},
                         {'version':'V4','mean_trimmed':None,'p95':None,'attempted':3,'succeeded':0}])
    axis=figures[-1].axes[0]
    assert '8.7' not in axis.get_title() and '24.2' not in axis.get_title()
    assert [t.get_text() for t in axis.get_xticklabels()]==['V2\n2/3 成功','V4\n0/3 成功']
    assert '无可用耗时' in [t.get_text() for t in axis.texts]
