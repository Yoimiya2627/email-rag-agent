"""Offline negative contracts for strict input, scoped retrieval and recovery."""
import io
import json
import logging
import tempfile
import threading
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config.settings as cfg
from agents import graph_workflow as graph
from agents.gmail_readonly import GmailReadOnlyProvider, GmailPageTokenError
from core import embedder, ingestion, loader, pipeline, reranker, retriever
from core.filters import FilterSpec, FilterCoverageError
from core.model_outcomes import ModelText
from models.schemas import AgentRequest, SearchResult
from scripts import sync_gmail_readonly as sync
from tests.test_sync_integrity import FakeProvider


def record(identifier='one', **updates):
    return {'id':identifier, 'subject':'s', 'sender':'a@example.test', 'recipients':[],
            'date':'2026-09-10','body':'safe body', **updates}


def hit(identifier='one', **metadata):
    return SearchResult(chunk_id=identifier+'_chunk_0', email_id=identifier,
                        content='visible evidence',score=1., metadata=metadata)


class StrictInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'input.json'

    def test_invalid_corpora_rejected_before_index_initialization(self):
        for records in [[], {}, [None], [record(),{}], [record(),record()],
                        [record(id=' ')], [record(body={'private':'CANARY'})]]:
            self.path.write_text(json.dumps(records),encoding='utf-8')
            with self.subTest(records=records), patch.object(embedder,'_get_client') as client:
                with self.assertRaises(ValueError):
                    ingestion.prepare_email_chunks(self.path)
                client.assert_not_called()

    def test_validation_logs_and_traceback_do_not_contain_input_values(self):
        self.path.write_text(json.dumps([record(body={'private':'SYNTHETIC_BODY_CANARY'})]),encoding='utf-8')
        stream=io.StringIO()
        handler=logging.StreamHandler(stream)
        loader.logger.addHandler(handler)
        self.addCleanup(loader.logger.removeHandler,handler)
        try:
            loader.load_emails(self.path)
        except ValueError:
            rendered=traceback.format_exc()
        else:
            self.fail('invalid corpus accepted')
        self.assertNotIn('SYNTHETIC_BODY_CANARY',rendered+stream.getvalue())
        self.assertIn('record 1',rendered)

    def test_bad_json_has_safe_error_and_valid_input_roundtrips(self):
        self.path.write_text('{"secret":"CANARY"',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'expected UTF-8 JSON'):
            loader.load_emails(self.path)
        self.path.write_text(json.dumps([record()]),encoding='utf-8')
        emails,chunks=ingestion.prepare_email_chunks(self.path)
        self.assertEqual(len(emails),1)
        self.assertTrue(chunks)


def matches_where(meta, where):
    if '$or' in where:
        return any(matches_where(meta,part) for part in where['$or'])
    if '$and' in where:
        return all(matches_where(meta,part) for part in where['$and'])
    return all(meta.get(key)==predicate['$eq'] for key,predicate in where.items())


class ScopeCollection:
    def __init__(self):
        self.rows=[{'id':f'e{i}_chunk_0','content':'invoice raretoken' if i>=20 else 'invoice ordinary',
                    'metadata':{'email_id':f'e{i}','chunk_index':0,
                                'sender':'RARE@example.test' if i>=20 else 'usual@example.test',
                                'labels':'["work", "urgent"]' if i!=21 else '["work"]',
                                'date':'2026-09-10T00:00:00Z'},'distance':i/100}
                   for i in range(22)]
        for row in self.rows:
            row['metadata'].update(source_start=0,source_end=len(row['content']))
        self.get_calls=[]
        self.queried=[]
    def count(self):return len(self.rows)
    def get(self,include,limit=None,offset=0,**kwargs):
        self.get_calls.append(include)
        rows=[r for r in self.rows if not kwargs.get('where') or matches_where(r['metadata'],kwargs['where'])]
        rows=rows[offset:offset+limit] if limit is not None else rows[offset:]
        out={'ids':[row['id'] for row in rows], 'metadatas':[row['metadata'] for row in rows]}
        if 'documents' in include:out['documents']=[row['content'] for row in rows]
        return out
    def query(self,n_results,where=None,**kwargs):
        rows=[row for row in self.rows if where is None or matches_where(row['metadata'],where)]
        self.queried.extend(row['id'] for row in rows)
        rows=sorted(rows,key=lambda r:r['distance'])[:n_results]
        return {'ids':[[r['id'] for r in rows]],'documents':[[r['content'] for r in rows]],
                'metadatas':[[r['metadata'] for r in rows]],'distances':[[r['distance'] for r in rows]]}


class FilterRecallTests(unittest.TestCase):
    def setUp(self):
        self.collection=ScopeCollection()
        self.scope=FilterSpec.from_mapping({'sender':'rare','labels':['WORK','urgent'],
                                           'date_hint':'2026-09-10'},now=datetime(2026,9,10,tzinfo=timezone.utc))
        for obj,name,value in [(embedder,'_get_collection',lambda:self.collection),
                               (embedder,'embed_texts',lambda texts:[[1.,0.] for _ in texts]),
                               (embedder,'get_corpus_revision',lambda:'fixed'),
                               (retriever,'get_corpus_revision',lambda:'fixed')]:
            context=patch.object(obj,name,value)
            context.start();self.addCleanup(context.stop)

    def test_vector_scope_recalls_beyond_global_top20_without_reading_other_bodies(self):
        with patch.object(cfg,'FILTER_METADATA_PAGE_SIZE',3,create=True):
            out=embedder.search_similar('invoice',top_k=3,filters=self.scope)
        self.assertEqual([r['email_id'] for r in out],['e20'])
        self.assertTrue(all(include==['metadatas'] for include in self.collection.get_calls))
        self.assertEqual(self.collection.queried,['e20_chunk_0'])

    def test_budget_exceeded_is_explicit_and_does_not_claim_empty(self):
        with patch.object(cfg,'FILTER_METADATA_SCAN_LIMIT',10,create=True), patch.object(embedder,'embed_texts') as encode:
            with self.assertRaises(FilterCoverageError):
                embedder.search_similar('invoice',filters=self.scope)
            encode.assert_not_called()

    def test_scope_date_and_all_labels_are_shared(self):
        self.assertTrue(self.scope.matches(self.collection.rows[20]['metadata']))
        self.assertFalse(self.scope.matches(self.collection.rows[21]['metadata']))
        self.assertFalse(self.scope.matches({**self.collection.rows[20]['metadata'],'date':'2026-09-11T00:00:00Z'}))
        self.assertFalse(self.scope.matches({**self.collection.rows[20]['metadata'],'labels':'"work"'}))

    def test_bm25_filters_before_topk_and_keeps_low_global_rank(self):
        rows=[{'chunk_id':row['id'],'content':row['content'],'metadata':row['metadata']} for row in self.collection.rows]
        index=SimpleNamespace(get_scores=lambda _:list(range(22,0,-1)))
        with patch.object(retriever,'_get_filtered_bm25_index',return_value=(index,rows,[r['content'] for r in rows])):
            out=retriever.bm25_search('invoice',top_k=1,filters=self.scope)
        self.assertEqual([r.email_id for r in out],['e20'])

    def test_lexical_scope_reads_only_matching_documents_and_bounds_total_text(self):
        chunks=embedder.get_filtered_chunks(self.scope)
        self.assertEqual([row['chunk_id'] for row in chunks],['e20_chunk_0'])
        with patch.object(cfg,'FILTER_LEXICAL_CHAR_LIMIT',1,create=True):
            before=len(self.collection.get_calls)
            with self.assertRaises(FilterCoverageError):embedder.get_filtered_chunks(self.scope)
            self.assertTrue(all(call==['metadatas'] for call in self.collection.get_calls[before:]))

    def test_single_document_scope_does_not_discard_negative_okapi_hit(self):
        retriever.invalidate_bm25_cache();self.addCleanup(retriever.invalidate_bm25_cache)
        out=retriever.bm25_search('invoice',top_k=1,filters=self.scope)
        self.assertEqual([row.email_id for row in out],['e20'])

    def test_real_bm25_and_vector_use_same_scope(self):
        retriever.invalidate_bm25_cache()
        self.addCleanup(retriever.invalidate_bm25_cache)
        with patch.object(cfg,'ENABLE_BM25',True),patch.object(cfg,'ENABLE_RRF',True):
            out=retriever.hybrid_search('raretoken',top_k=1,filters=self.scope)
        self.assertEqual([r.email_id for r in out],['e20'])

    def test_revision_change_fails_without_mixed_scope_results(self):
        with patch.object(embedder,'get_corpus_revision',side_effect=['old','new']):
            with self.assertRaises(FilterCoverageError):
                embedder.search_similar('invoice',filters=self.scope)

    def test_batched_vector_results_keep_global_score_order(self):
        scope=FilterSpec.from_mapping({'sender':'rare'},now=datetime(2026,9,10,tzinfo=timezone.utc))
        with patch.object(cfg,'FILTER_VECTOR_BATCH_SIZE',1,create=True):
            out=embedder.search_similar('invoice',top_k=1,filters=scope)
        self.assertEqual([r['email_id'] for r in out],['e20'])


class ResolvedQueryTests(unittest.TestCase):
    def test_followup_rerank_sees_resolved_entity(self):
        with patch.object(pipeline,'rewrite_query',return_value='Acme invoice amount'), \
             patch.object(pipeline,'extract_filters',return_value={'query':'invoice amount'}), \
             patch.object(pipeline,'hybrid_search',return_value=[hit()]), \
             patch.object(pipeline,'rerank',return_value=[]) as score:
            pipeline.retrieve('how much was it?',history=[{'role':'user','content':'Acme invoice'}])
        self.assertEqual(score.call_args.args[0],'Acme invoice amount')

    def test_grader_sees_resolved_question_and_evidence_beyond_old_300_chars(self):
        create=Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[0]'))]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        source=hit();source.content='x'*400+' AMOUNT 750'
        state={'query':'how much was it?','rewritten_query':'Acme invoice amount','results':[source]}
        with patch.object(graph,'_get_client',return_value=client):
            graph.node_grade_contexts(state)
        prompt=next(message['content'] for message in create.call_args.kwargs['messages'] if message['role']=='user')
        self.assertIn('how much was it?',prompt)
        self.assertIn('Acme invoice amount',prompt)
        self.assertIn('AMOUNT 750',prompt)

    def test_graph_preserves_incomplete_generation_metadata(self):
        state={'query':'q','relevant_results':[hit()]}
        with patch.object(graph,'generate_answer',return_value=ModelText('partial',completion_status='incomplete',finish_reason='length')):
            graph.node_generate(state)
        state['answer']=str(state['answer'])
        with patch.object(graph,'get_graph',return_value=SimpleNamespace(invoke=lambda _:state)), \
             patch('agents.coordinator.classify_intent',return_value='retrieve'):
            response=graph.run_graph(AgentRequest(query='q'))
        self.assertEqual(response.metadata['completion_status'],'incomplete')
        self.assertEqual(response.metadata['finish_reason'],'length')
        self.assertTrue(response.metadata['grounded'])


class BreakerRecoveryTests(unittest.TestCase):
    def setUp(self):
        reranker.reset_circuit_breaker();self.addCleanup(reranker.reset_circuit_breaker)
        self.rows=[hit('one'),hit('two')]
        self.now=[100.]
        for obj,name,value in [(cfg,'ENABLE_RERANKER',True),(cfg,'RERANKER_BACKEND','cross_encoder'),
                               (cfg,'RERANKER_COOLDOWN_SECONDS',10.),(reranker.time,'monotonic',lambda:self.now[0])]:
            p=patch.object(obj,name,value,create=True);p.start();self.addCleanup(p.stop)
    def open_breaker(self):
        with patch.object(reranker,'_rerank_with_cross_encoder',side_effect=RuntimeError('transient')):
            for _ in range(3):reranker.rerank('q',self.rows)
    def test_cooldown_recovers_and_closed_calls_resume(self):
        self.open_breaker()
        with patch.object(reranker,'_rerank_with_cross_encoder',return_value=self.rows) as backend:
            reranker.rerank('q',self.rows);backend.assert_not_called()
            self.now[0]=111.
            reranker.rerank('q',self.rows)
            reranker.rerank('q',self.rows)
            self.assertEqual(backend.call_count,2)
        self.assertEqual(reranker.get_circuit_breaker_state()['state'],'closed')
    def test_only_one_half_open_probe_can_run_concurrently(self):
        self.open_breaker();self.now[0]=111.
        entered=threading.Event();release=threading.Event()
        def probe(*args):
            entered.set();release.wait(2);return self.rows
        with patch.object(reranker,'_rerank_with_cross_encoder',side_effect=probe) as backend:
            thread=threading.Thread(target=reranker.rerank,args=('q',self.rows))
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                reranker.rerank('q',self.rows)
                self.assertEqual(backend.call_count,1)
            finally:
                release.set();thread.join(2)
            self.assertFalse(thread.is_alive())
    def test_failed_probe_reopens_and_cancellation_releases_probe_slot(self):
        self.open_breaker();self.now[0]=111.
        with patch.object(reranker,'_rerank_with_cross_encoder',side_effect=RuntimeError('still unavailable')) as backend:
            reranker.rerank('q',self.rows);reranker.rerank('q',self.rows)
            self.assertEqual(backend.call_count,1)
        self.now[0]=122.
        from agents.runtime import RunDeadlineExceeded
        with patch.object(reranker,'_rerank_with_cross_encoder',side_effect=RunDeadlineExceeded('deadline')):
            with self.assertRaises(RunDeadlineExceeded):reranker.rerank('q',self.rows)
        self.assertEqual(reranker.get_circuit_breaker_state()['state'],'open')
    def test_disabled_reranker_does_not_touch_breaker_or_backend(self):
        with patch.object(cfg,'ENABLE_RERANKER',False),patch.object(reranker,'_claim_attempt') as claim:
            reranker.rerank('q',self.rows);claim.assert_not_called()


class HttpFailure(RuntimeError):
    def __init__(self,status):self.resp=SimpleNamespace(status=status)


class PagedProvider(FakeProvider):
    def __init__(self,ids):
        super().__init__(ids);self.pages=[];self.bad_tokens=set();self.errors={}
    def list_message_page(self,*,query,page_token,page_size):
        self.pages.append(page_token)
        if page_token in self.bad_tokens:raise GmailPageTokenError('stale token')
        start=int(page_token or 0);end=min(start+page_size,len(self.ids))
        return {'message_ids':self.ids[start:end],'next_page_token':str(end) if end<len(self.ids) else None}
    def get_message(self,mid):
        if mid in self.errors:
            self.reads.append(mid);raise self.errors[mid]
        return super().get_message(mid)


class SyncContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.output=self.directory/'emails.json';self.state=self.directory/'state.json'
    def sync(self,p,limit=2,query=''):
        return sync.sync_gmail_to_json(p,self.output,self.state,query,limit)
    def status(self):return json.loads(self.state.read_text(encoding='utf-8'))
    def test_deleted_pending_is_terminal_and_healthy_message_progresses(self):
        p=PagedProvider(['deleted','healthy']);p.errors['deleted']=HttpFailure(404)
        self.assertEqual(self.sync(p)['added'],1)
        self.assertEqual(self.status()['deleted_message_ids'],['deleted'])
        p.reads=[];self.sync(p)
        self.assertNotIn('deleted',p.reads)
        self.assertEqual(self.status()['pending_message_ids'],[])
    def test_auth_failures_are_not_swallowed_or_marked_deleted(self):
        for status in (401,403):
            p=PagedProvider(['blocked','healthy']);p.errors['blocked']=HttpFailure(status)
            with self.subTest(status=status),self.assertRaises(HttpFailure):self.sync(p)
            self.assertEqual(self.status()['deleted_message_ids'],[])
            self.assertNotIn('healthy',p.reads)
    def test_repeated_runs_backfill_past_seen_prefix(self):
        p=PagedProvider(['one','two','three'])
        self.assertEqual(self.sync(p)['total'],2)
        self.assertFalse(self.status()['backfill_complete'])
        self.assertEqual(self.status()['pagination']['next_page_token'],'2')
        self.assertEqual(self.sync(p)['total'],3)
        self.assertTrue(self.status()['backfill_complete'])
        self.assertEqual(p.pages,[None,'2'])
    def test_interruption_after_listing_preserves_pending_and_cursor(self):
        p=PagedProvider(['one','two','three']);p.errors['one']=KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):self.sync(p)
        self.assertEqual(self.status()['pending_message_ids'],['one','two'])
        p.errors={}
        self.assertEqual(self.sync(p)['total'],3)
        self.assertEqual(self.status()['pending_message_ids'],[])
    def test_invalid_token_restarts_once_then_resumes(self):
        p=PagedProvider(['one','two','three']);self.sync(p)
        p.bad_tokens={'2'};self.sync(p)
        self.assertEqual(self.status()['pagination']['cursor_resets'],1)
        p.bad_tokens=set()
        self.assertEqual(self.sync(p)['total'],3)
    def test_persistently_invalid_token_exhausts_budget(self):
        p=PagedProvider(['one','two','three']);self.sync(p)
        p.bad_tokens={'2'};self.sync(p)
        before=len(p.pages)
        with self.assertRaisesRegex(GmailPageTokenError,'budget exhausted'):self.sync(p)
        self.assertEqual(len(p.pages)-before,1)
        self.assertEqual(self.status()['status'],'aborted')
    def test_restored_corpus_restarts_cursor_without_skipping_lost_rows(self):
        p=PagedProvider(['one','two','three']);self.sync(p)
        old=json.loads(self.output.read_text());self.output.write_text(json.dumps(old[1:]))
        self.sync(p)
        self.assertIsNone(p.pages[-1])
        self.assertEqual({r['id'] for r in json.loads(self.output.read_text())},{'gmail_one','gmail_two'})
    def test_query_change_resets_cursor_and_repeated_tokens_fail_closed(self):
        p=PagedProvider(['one','two','three']);self.sync(p,query='first');self.sync(p,query='second')
        self.assertIsNone(p.pages[-1])
        p.list_message_page=lambda **kw:{'message_ids':[],'next_page_token':kw['page_token']}
        with self.assertRaisesRegex(ValueError,'did not advance'):self.sync(p,query='second')
    def test_empty_pages_have_a_per_run_budget(self):
        p=PagedProvider([])
        p.list_message_page=lambda **kw:{'message_ids':[],'next_page_token':str(int(kw['page_token'] or 0)+1)}
        with patch.object(cfg,'GMAIL_SYNC_MAX_PAGES_PER_RUN',2,create=True):self.sync(p)
        self.assertEqual(self.status()['pagination']['next_page_token'],'2')
        self.assertFalse(self.status()['backfill_complete'])


class GmailPageContractTests(unittest.TestCase):
    def test_legacy_reader_subclass_is_not_bypassed_by_new_page_method(self):
        class LegacyReader(GmailReadOnlyProvider):
            def list_message_ids(self,**kwargs):return []
        with tempfile.TemporaryDirectory() as directory:
            reader=LegacyReader()
            with patch.object(reader,'_get_service',side_effect=AssertionError('legacy transport bypassed')):
                result=sync.sync_gmail_to_json(reader,Path(directory)/'emails.json',
                    Path(directory)/'state.json','',2)
            self.assertEqual(result['fetched'],0)
            self.assertFalse(json.loads((Path(directory)/'state.json').read_text())['backfill_complete'])

    def test_continuation_400_is_recoverable_but_auth_is_not(self):
        class Service:
            def users(self):return self
            def messages(self):return self
            def list(self,**kwargs):return SimpleNamespace(execute=lambda:(_ for _ in ()).throw(HttpFailure(status)))
        p=GmailReadOnlyProvider(service=Service())
        for status in (400,401,403,404):
            expected=GmailPageTokenError if status==400 else HttpFailure
            with self.subTest(status=status),self.assertRaises(expected):
                p.list_message_page(page_token='cursor')
        status=400
        with self.assertRaises(HttpFailure):p.list_message_page()


if __name__=='__main__':unittest.main()
