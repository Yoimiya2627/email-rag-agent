"""Offline independent report auditor. Stdlib only; no tested-project imports.

Reads explicitly supplied report/gold/corpus and source files only. Never opens
an index, environment file, private mailbox, or network connection. Output omits
questions, filters, subjects, addresses, bodies, and local user paths.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(report_path, baseline_path, root, corpus_path, expected_head=None, gold_path=None):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    gold_path = gold_path or root / 'data/gold_chunks.json'
    gold = json.loads(gold_path.read_text(encoding='utf-8'))
    failures, checks = [], []
    def check(value, label):
        (checks if value else failures).append(label)
    check(report.get('complete') is True, 'report.complete')
    check(report.get('full_suite') is True, 'report.full_suite')
    check(report.get('batch_status') == 'completed', 'report.batch_status')
    check(report.get('mode') == 'shared_candidates', 'report.mode')
    check(report.get('planned_n') == report.get('completed_n') == len(gold) == 30, 'report.counts_30')
    check(report.get('top_n') == 3 and report.get('fetch_k') == 20, 'report.top3_fetch20')
    records = report.get('paired_cases', [])
    goldmap = {g['id']: g for g in gold}
    check(len(goldmap) == len(gold) == 30, 'gold.unique_30')
    check(len(records) == 30 and len({c['id'] for c in records}) == 30, 'cases.unique_30')
    check([c['id'] for c in records] == [g['id'] for g in gold], 'cases.gold_order')
    meta, oldmeta = report['run'], baseline['run']
    check(meta.get('reference_time') == '2026-11-02T12:00:00+08:00', 'reference.fixed')
    check(meta.get('git_status') == [], 'source.clean_at_start')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    check(meta.get('git_head') == (expected_head or head), 'source.expected_commit')
    check(meta.get('gold_sha256') == sha(gold_path) == oldmeta['gold_sha256'], 'gold.hash_unchanged')
    check(meta.get('corpus_sha256') == sha(corpus_path) == oldmeta['corpus_sha256'], 'corpus.hash_unchanged')
    check(meta.get('manifest') == oldmeta['manifest'], 'manifest.unchanged')
    check(meta.get('corpus_revision') == meta['manifest']['generation'] == oldmeta['corpus_revision'], 'manifest.generation')
    check(meta.get('packages') == oldmeta['packages'], 'packages.unchanged')
    check(meta.get('settings') == oldmeta['settings'], 'settings.unchanged')
    check(meta.get('provider_endpoint_origin') == oldmeta['provider_endpoint_origin'], 'provider.origin_unchanged')
    check(meta.get('resolved_model_revisions') == oldmeta['resolved_model_revisions'], 'models.revisions_unchanged')
    source_dirs = ('agents', 'api', 'core', 'config', 'eval', 'frontend', 'langchain_version', 'models', 'scripts')
    paths = {p for directory in source_dirs for p in (root / directory).rglob('*.py')}
    paths.update(root.glob('*.py'))
    paths.update(root.glob('requirements*.txt'))
    paths.update(root / p for p in ('pyproject.toml', 'uv.lock', 'poetry.lock', 'Pipfile', 'Pipfile.lock') if (root / p).is_file())
    source_hashes = {p.relative_to(root).as_posix(): sha(p) for p in paths}
    check(meta.get('source_sha256') == source_hashes, 'source.all_file_hashes_current')
    committed_paths = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', meta['git_head']], cwd=root, text=True).splitlines())
    check(set(source_hashes) <= committed_paths, 'source.all_hashed_sources_committed')
    diff = subprocess.run(['git', 'diff', '--quiet', meta['git_head'], '--', *sorted(source_hashes)], cwd=root)
    check(diff.returncode == 0, 'source.committed_content_matches_with_git_eol_rules')
    check(meta.get('acceptance_script_sha256') == source_hashes['scripts/paired_acceptance.py'], 'source.acceptance_script_hash')
    check(datetime.fromisoformat(report['finished_at']) >= datetime.fromisoformat(meta['started_at']), 'report.timestamps')
    totals = {v: [] for v in ('V2', 'V7')}
    paired = []
    rerank_outcomes = Counter()
    token_total = call_total = unknown_cost_total = unknown_usage_total = 0
    for c in records:
        cid = c['id']; g = goldmap[cid]; d = c['diagnostics']; usage = c['model_usage']
        check(c.get('complete') is True, cid + '.complete')
        check(c.get('question') == g['question'], cid + '.question_matches_gold')
        candidates = c.get('candidate_ids', [])
        check(len(candidates) == len(set(candidates)) <= 20, cid + '.unique_candidates_within20')
        if 'candidate_ids' in c:
            check(candidates == d.get('rerank_candidate_ids'), cid + '.shared_pool')
            check(d.get('rerank_candidate_count') == len(candidates) == d.get('returned_count'), cid + '.pool_counts')
            check(d.get('scoring_query') == g['question'], cid + '.original_scoring_query')
            check(d.get('reference_time') == meta['reference_time'], cid + '.fixed_reference')
            retrieval_reranker = d.get('reranker', {})
            check(retrieval_reranker.get('status') == 'disabled' or (not candidates and retrieval_reranker.get('status') == 'skipped' and retrieval_reranker.get('reason') == 'empty_candidates'), cid + '.retrieval_reranker_disabled_or_empty')
        for v in totals:
            row = c['versions'][v]; got = row['retrieved_chunk_ids']; labels = g['gold_chunk_ids']
            check(row.get('id') == cid and row.get('question') == g['question'] and row.get('gold_chunk_ids') == labels, cid + '.' + v + '.identity_gold')
            check(len(got) == len(set(got)) <= 3, cid + '.' + v + '.unique_top3')
            check(set(got) <= set(candidates), cid + '.' + v + '.from_shared_pool')
            if v == 'V2' and row['status'] != 'error':
                check(got == candidates[:3], cid + '.V2_pool_prefix')
            check(row['status'] in ('ok', 'empty', 'error'), cid + '.' + v + '.status')
            check((bool(got) if row['status'] == 'ok' else not got), cid + '.' + v + '.status_rows')
            hits = [x for x in labels if x in set(got)]
            expected = dict(hit_count=len(hits), gold_count=len(labels), hit_chunk_ids=hits,
                            context_recall=round(len(hits)/len(labels), 4),
                            has_any_hit=bool(hits), has_perfect_recall=len(hits)==len(labels))
            check(all(row[k] == val for k,val in expected.items()), cid + '.' + v + '.independent_score')
            if row['status'] == 'error':
                check(expected['context_recall'] == 0 and not got, cid + '.' + v + '.failure_zero')
            totals[v].append(dict(expected, status=row['status']))
        rd = c.get('v7_reranker', {})
        key = str(rd.get('status')) + ':' + str(rd.get('reason'))
        rerank_outcomes[key] += 1
        if c['versions']['V7']['status'] != 'error':
            legit = (rd.get('status') == 'succeeded' and rd.get('backend') == 'cross_encoder') if len(candidates) > 1 else (rd.get('status') == 'skipped' and rd.get('reason') == ('empty_candidates' if not candidates else 'insufficient_candidates'))
            check(legit, cid + '.V7_cross_encoder_or_trivial_skip')
        calls = usage['calls']
        check(usage['calls_truncated'] is False and usage['call_count'] == len(calls), cid + '.ledger_complete')
        check(usage['reserved_tokens'] == 0 and usage['reserved_cost'] == 0, cid + '.ledger_no_pending')
        actual = sum(x['total_tokens'] or 0 for x in calls)
        accounted = sum(x['total_tokens'] if x['total_tokens'] is not None else x['estimated_input_tokens'] + x['reserved_output_tokens'] for x in calls)
        check(usage['actual_total_tokens'] == actual and usage['budget_accounted_tokens'] == accounted, cid + '.ledger_token_totals')
        check(usage['unknown_usage_calls'] == sum(x['total_tokens'] is None for x in calls), cid + '.ledger_unknown_usage')
        check(usage['unknown_cost_calls'] == sum(x['actual_cost'] is None for x in calls), cid + '.ledger_unknown_cost')
        check(usage['unpriced_calls'] == sum(x['estimated_max_cost'] is None for x in calls), cid + '.ledger_unpriced')
        check(abs(usage['actual_cost'] - sum(x['actual_cost'] or 0 for x in calls)) < 1e-10, cid + '.ledger_actual_cost')
        check(abs(usage['budget_accounted_cost'] - sum(x['actual_cost'] if x['actual_cost'] is not None else x['estimated_max_cost'] or 0 for x in calls)) < 1e-10, cid + '.ledger_budget_cost')
        for i,x in enumerate(calls):
            check(x['stage'] == 'filter', cid + '.ledger_filter_only.' + str(i))
            check(x['usage_source'] == ('provider' if x['total_tokens'] is not None else 'unavailable'), cid + '.ledger_usage_source.' + str(i))
            if x['total_tokens'] is not None and x['input_tokens'] is not None and x['output_tokens'] is not None:
                check(x['total_tokens'] == x['input_tokens'] + x['output_tokens'], cid + '.ledger_token_components.' + str(i))
        extraction = d.get('extraction', {})
        if extraction.get('attempt_count') is not None:
            check(extraction['attempt_count'] == len(calls), cid + '.extraction_calls')
        token_total += actual; call_total += len(calls)
        unknown_cost_total += usage['unknown_cost_calls']; unknown_usage_total += usage['unknown_usage_calls']
        old = next(x for x in baseline['paired_cases'] if x['id'] == cid)
        paired.append({'id': cid, 'candidate_count': len(candidates), 'candidate_pool_changed_from_history': candidates != old.get('candidate_ids', []),
                       'V2': c['versions']['V2']['context_recall'], 'V7': c['versions']['V7']['context_recall'],
                       'old_V2': old['versions']['V2']['context_recall'], 'old_V7': old['versions']['V7']['context_recall']})
    aggregates = {}
    check([r['version'] for r in report['results']] == ['V2', 'V7'], 'results.versions')
    for v, rows in totals.items():
        n = len(rows)
        summary = {'n': n, 'mean_context_recall': round(sum(x['context_recall'] for x in rows)/n, 4),
                   'chunk_hit_rate': round(sum(x['has_any_hit'] for x in rows)/n, 4),
                   'perfect_recall_rate': round(sum(x['has_perfect_recall'] for x in rows)/n, 4),
                   **{s+'_count': sum(x['status'] == s for x in rows) for s in ('ok','empty','error')}}
        aggregates[v] = summary
        result = next(r for r in report['results'] if r['version'] == v)
        check(result['summary'] == summary, v + '.aggregate_recomputed')
        check(result['records'] == [c['versions'][v] for c in records], v + '.duplicated_records_match')
        check(result['flags'] == dict(ENABLE_BM25=True, ENABLE_RRF=True, ENABLE_QUERY_REWRITE=False, ENABLE_RERANKER=v=='V7', RERANKER_BACKEND='cross_encoder' if v=='V7' else 'llm'), v + '.flags')
    raw = report_path.read_text(encoding='utf-8')
    sensitive = Counter()
    def walk(x):
        if isinstance(x, dict):
            for k,v in x.items():
                if k.lower() in ('question','subject','sender','body','body_text','body_html','email_id','root','python_executable','gold_path') and v:
                    sensitive[k] += 1
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(report)
    secret_patterns = {'api_key_shaped': r'\bsk-[A-Za-z0-9_-]{20,}', 'private_key_header': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', 'bearer_token': r'Bearer\s+[A-Za-z0-9._~-]{20,}'}
    secret_counts = {k: len(re.findall(pattern, raw)) for k,pattern in secret_patterns.items()}
    check(not any(secret_counts.values()), 'release.no_common_secret_patterns')
    changes = sorted(k for k in set(meta['source_sha256']) | set(oldmeta['source_sha256']) if meta['source_sha256'].get(k) != oldmeta['source_sha256'].get(k))
    return {'schema_version': 1, 'verdict': 'fail' if failures else 'pass', 'checks_passed': len(checks), 'failures': failures,
            'report_sha256': sha(report_path), 'baseline_sha256': sha(baseline_path), 'auditor_sha256': sha(__file__), 'git_head': meta['git_head'],
            'aggregates': aggregates, 'ledger': {'call_count': call_total, 'actual_total_tokens': token_total, 'unknown_usage_calls': unknown_usage_total, 'unknown_cost_calls': unknown_cost_total, 'cost_interpretation': 'unknown when unpriced; zero is not evidence of free usage'},
            'rerank_outcomes': dict(rerank_outcomes), 'source_files_changed_from_history': changes,
            'comparability': {key: meta.get(key) == oldmeta.get(key) for key in ('reference_time','gold_sha256','corpus_sha256','corpus_revision','manifest','packages','settings','resolved_model_revisions')},
            'paired_cases': paired, 'release': {'raw_report_verdict': 'limited' if sensitive else 'pass', 'sensitive_field_occurrences': dict(sensitive), 'common_secret_pattern_counts': secret_counts, 'note': 'Pattern scan is not proof of no secrets. Raw questions, filters, subjects, identifiers and local paths require publication authorization or redaction.'},
            'limitations': ['Offline evidence audit; no replay or remote attestation of model execution.', 'Cross-encoder execution established by inspected predict path, diagnostic status, model revision, and source hashes; report contains no raw reranker scores.', 'Historical manifest compared without opening any index.', 'Package/settings values compared as report metadata, not re-imported from runtime.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--gold', type=Path)
    p.add_argument('--expected-head')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite audit output')
    out = audit(a.report, a.baseline, a.root, a.corpus, a.expected_head, a.gold)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('verdict','checks_passed','failures','aggregates','ledger','rerank_outcomes','comparability','release')}, ensure_ascii=True))
    raise SystemExit(1 if out['failures'] else 0)
