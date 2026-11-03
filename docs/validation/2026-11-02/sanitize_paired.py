"""Allowlist-only public paired evidence export; offline, no repository imports.

Chunk identifiers use full SHA-256 consistently across gold, pool, returned and
hit arrays. This is pseudonymization, not a confidentiality guarantee against
guessing low-entropy IDs. All free-text mail/query/filter content is omitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def pick(value, names):
    return {k: value[k] for k in names if k in value}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunk_hash(value):
    return 'sha256:' + hashlib.sha256(value.encode('utf-8')).hexdigest()


def score(row):
    out = pick(row, ('id', 'status', 'error_type', 'cause_type', 'error_stage', 'hit_count', 'gold_count', 'context_recall', 'has_any_hit', 'has_perfect_recall'))
    for name in ('gold_chunk_ids', 'retrieved_chunk_ids', 'hit_chunk_ids'):
        out[name] = [chunk_hash(v) for v in row[name]]
    return out


def sanitize(path):
    raw = json.loads(path.read_text(encoding='utf-8'))
    out = pick(raw, ('schema_version', 'mode', 'complete', 'full_suite', 'planned_n', 'batch_status', 'top_n', 'fetch_k', 'completed_n', 'finished_at'))
    out['publication'] = {'format': 'allowlisted-paired-evidence-v1', 'raw_report_sha256': sha(path), 'sanitizer_sha256': sha(__file__), 'chunk_id_transform': 'sha256(utf8(original_id)); full 256 bits; same mapping in all arrays', 'privacy_limit': 'Unkeyed hashes of predictable identifiers are pseudonyms and may be guessable. No mail/query/filter text or local paths are included.'}
    out['run'] = pick(raw['run'], ('started_at','reference_time','git_head','source_sha256','acceptance_script_sha256','gold_sha256','corpus_sha256','corpus_revision','provider_endpoint_origin','packages','resolved_model_revisions'))
    out['run']['manifest'] = pick(raw['run']['manifest'], ('generation','collection','corpus_sha256','source_sha256','chunk_count','email_count','config_fingerprint','resolved_model_revision'))
    out['run']['settings'] = pick(raw['run']['settings'], ('DEEPSEEK_MODEL','MODEL_REVISION','DEEPSEEK_THINKING_MODE','AGENT_PLANNER_MODEL','EMBEDDING_MODEL','EMBEDDING_MODEL_REVISION','EMBEDDING_DEVICE','EMBEDDING_CPU_THREADS','CROSS_ENCODER_MODEL','CROSS_ENCODER_DEVICE','CROSS_ENCODER_MAX_LENGTH','TOP_K','RERANK_TOP_N','RERANK_INPUT_CHAR_LIMIT','RETRIEVAL_TIMEZONE','RETRIEVAL_TIMEZONE_OFFSET_HOURS','MODEL_RUN_TOKEN_LIMIT','MODEL_RUN_COST_LIMIT','AGENT_RUN_TIMEOUT'))
    out['paired_cases'] = []
    for c in raw['paired_cases']:
        row = pick(c, ('id','complete','duration_seconds'))
        row['versions'] = {v: score(c['versions'][v]) for v in ('V2','V7')}
        row['candidate_ids'] = [chunk_hash(v) for v in c.get('candidate_ids', [])]
        row['v7_reranker'] = pick(c.get('v7_reranker', {}), ('status','reason','backend'))
        d = c['diagnostics']
        row['diagnostics'] = pick(d, ('status','stage','top_n','fetch_k','reference_time','hybrid_candidate_count','post_filter_candidate_count','rerank_candidate_count','returned_count'))
        row['diagnostics']['extraction'] = pick(d.get('extraction', {}), ('status','attempt_count','error_type'))
        row['diagnostics']['reranker'] = pick(d.get('reranker', {}), ('status','reason','backend'))
        for field in ('hybrid_candidate_ids','rerank_candidate_ids'):
            if field in d:
                row['diagnostics'][field] = [chunk_hash(v) for v in d[field]]
        u = c['model_usage']
        row['model_usage'] = pick(u, ('call_count','actual_total_tokens','actual_cost','unknown_usage_calls','unknown_cost_calls','unpriced_calls','budget_accounted_tokens','budget_accounted_cost','reserved_tokens','reserved_cost','calls_truncated'))
        row['model_usage']['calls'] = [pick(call, ('stage','duration_ms','status','input_tokens','output_tokens','total_tokens','actual_cost','estimated_input_tokens','reserved_output_tokens','estimated_max_cost','token_estimation_method','usage_source')) for call in u['calls']]
        out['paired_cases'].append(row)
    out['results'] = []
    for result in raw['results']:
        out['results'].append({'version':result['version'], 'flags':pick(result['flags'], ('ENABLE_BM25','ENABLE_RRF','ENABLE_QUERY_REWRITE','ENABLE_RERANKER','RERANKER_BACKEND')), 'summary':pick(result['summary'], ('n','mean_context_recall','chunk_hit_rate','perfect_recall_rate','ok_count','empty_count','error_count')), 'records':[score(r) for r in result['records']]})
    verify(raw, out)
    return out


def verify(raw, public):
    assert public['run']['git_head'] == raw['run']['git_head']
    assert len(public['paired_cases']) == len(raw['paired_cases']) == 30
    for src, dst in zip(raw['paired_cases'], public['paired_cases']):
        assert dst['id'] == src['id']
        assert dst['candidate_ids'] == list(map(chunk_hash, src.get('candidate_ids', [])))
        for version in ('V2','V7'):
            v = dst['versions'][version]
            assert v == score(src['versions'][version])
            expected = [x for x in v['gold_chunk_ids'] if x in set(v['retrieved_chunk_ids'])]
            assert v['hit_chunk_ids'] == expected
            assert v['context_recall'] == round(len(expected)/len(v['gold_chunk_ids']), 4)
            assert set(v['retrieved_chunk_ids']) <= set(dst['candidate_ids'])
    forbidden = {'question','subject','sender','body','body_text','body_html','date','email_id','source_email_ids','gold_scope_checks','gold_path','root','python_executable','parsed_filters','applied_filters','filters','search_query','scoring_query','rewritten_query','scope'}
    def walk(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden), 'Forbidden key in publication'
            for v in value.values(): walk(v)
        elif isinstance(value, list):
            for v in value: walk(v)
    walk(public)
    text = json.dumps(public, ensure_ascii=False)
    for pattern in (r'\bsk-[A-Za-z0-9_-]{20,}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', r'Bearer\s+[A-Za-z0-9._~-]{20,}', r'\bemail_\d+_chunk_\d+\b', r'[A-Za-z]:\\', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'):
        assert not re.search(pattern, text), 'Sensitive content pattern in publication'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise SystemExit('Refusing to overwrite publication evidence')
    result = sanitize(a.report)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'complete':result['complete'], 'cases':len(result['paired_cases']), 'raw_report_sha256':result['publication']['raw_report_sha256'], 'public_report_sha256':sha(a.output), 'privacy_validation':'passed allowlist and common sensitive-pattern checks'}))
