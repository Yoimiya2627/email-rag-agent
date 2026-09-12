"""Pure, deterministic evaluation metrics and reproducibility metadata."""
import hashlib
import json
import platform
import sys
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINT_FORMAT = 'text-lf-v1'
EVALUATION_CONTRACT_VERSION = 4

SAFE_CONFIG_FIELDS = (
    'DEEPSEEK_MODEL','AGENT_PLANNER_MODEL','EMBEDDING_MODEL','EMBEDDING_DEVICE',
    'EMBEDDING_MODEL_REVISION','CROSS_ENCODER_MODEL','CROSS_ENCODER_DEVICE','CROSS_ENCODER_MAX_LENGTH',
    'CROSS_ENCODER_MODEL_REVISION','CHUNK_SIZE','CHUNK_OVERLAP','MIN_CHUNK_SIZE','TOP_K',
    'VECTOR_WEIGHT','BM25_WEIGHT','RERANK_TOP_N','RERANKER_BACKEND','RERANK_INPUT_CHAR_LIMIT',
    'GENERATION_CONTEXT_CHAR_LIMIT','ENABLE_BM25','ENABLE_RRF','ENABLE_RERANKER','ENABLE_QUERY_REWRITE',
    'LLM_TIMEOUT','AGENT_MAX_STEPS','AGENT_MAX_TOOL_CALLS','AGENT_RUN_TIMEOUT','AGENT_CONTEXT_CHAR_LIMIT',
    'AGENT_MAX_REPEAT','AGENT_TOOL_OUTPUT_LIMIT','AGENT_MAX_TOKENS','AGENT_TOOL_BACKEND',
    'RETRIEVAL_TIMEZONE_OFFSET_HOURS','MAIL_PROVIDER','CHROMA_COLLECTION',
    'EVAL_JUDGE_TIMEOUT_SECONDS','EVAL_JUDGE_CONTEXT_CHAR_LIMIT',
    'MODEL_CONTEXT_TOKENS','MODEL_OUTPUT_RESERVE_TOKENS','MODEL_RUN_TOKEN_LIMIT','MODEL_RUN_COST_LIMIT',
    'MODEL_INPUT_COST_PER_MILLION','MODEL_OUTPUT_COST_PER_MILLION','MODEL_TOKEN_PRICES','MODEL_STREAM_INCLUDE_USAGE',
    'EMBEDDING_DIMENSION','BM25_MAX_CHUNKS','BM25_CHAR_LIMIT','STATS_METADATA_SCAN_LIMIT',
    'FILTER_METADATA_SCAN_LIMIT','FILTER_METADATA_PAGE_SIZE','FILTER_VECTOR_BATCH_SIZE',
    'FILTER_LEXICAL_MAX_CHUNKS','FILTER_LEXICAL_CHAR_LIMIT','RETRIEVAL_TIMEZONE',
    'RERANKER_COOLDOWN_SECONDS','SELF_RAG_GRADE_CHAR_LIMIT','SESSION_CONTEXT_TURNS','SESSION_CACHE_BYTES',
    'MAX_ACTIVE_REQUESTS','MAX_BACKGROUND_JOBS','MAIL_APPROVAL_TIMEOUT_SECONDS',
    'MAX_EMAIL_RECORD_BYTES','MAX_EMAIL_JSON_DEPTH','MAX_INDEX_INPUT_BYTES','MAX_INDEX_INPUT_EMAILS',
    'MAX_EMAIL_CHUNKS','MAX_INDEX_INPUT_CHUNKS',
    'EVIDENCE_VERIFY_CHAR_LIMIT',
)


def effective_config(settings):
    values = {name:getattr(settings,name) for name in SAFE_CONFIG_FIELDS if hasattr(settings,name)}
    # Model selectors may be local absolute paths rather than public hub IDs.
    for name in ('EMBEDDING_MODEL','CROSS_ENCODER_MODEL'):
        value = values.get(name)
        if isinstance(value,str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
                                      or value.startswith(('./','../','.\\','..\\','~'))):
            values[name] = {'kind':'local_path','sha256':hashlib.sha256(value.encode()).hexdigest()}
    base_url = str(getattr(settings,'DEEPSEEK_BASE_URL',''))
    values['model_endpoint_sha256'] = hashlib.sha256(base_url.encode('utf-8')).hexdigest()
    return values


def artifact_fingerprint(path):
    if path is None:
        return {'status':'unavailable','reason':'not_supplied'}
    path = Path(path)
    if not path.is_file():
        return {'status':'unavailable','reason':'not_found','name':path.name}
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return {'status':'available','name':path.name,'sha256':digest.hexdigest(),'bytes':path.stat().st_size}


def environment_provenance():
    packages = sorted({(d.metadata['Name'], d.version) for d in metadata.distributions() if d.metadata.get('Name')})
    return {'python':platform.python_version(),'implementation':sys.implementation.name,
            'platform':platform.platform(),'packages':dict(packages)}


def value_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                      separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def _assertions(case):
    assertions = case.get('tool_assertions', [])
    if not isinstance(assertions,list) or len(assertions)>100:
        raise ValueError('tool_assertions must be a list of at most 100 assertions')
    for index, assertion in enumerate(assertions):
        if (not isinstance(assertion,dict) or not isinstance(assertion.get('tool'),str)
                or type(assertion.get('occurrence',1)) is not int or assertion.get('occurrence',1)<1
                or assertion.get('source') not in {'arguments','result'}
                or not isinstance(assertion.get('path'),str) or not assertion['path']
                or 'equals' not in assertion):
            raise ValueError(f'invalid tool assertion {index}')
    return assertions


def make_tool_observer(case):
    """Compare canonical scalar/JSON fields in memory; record hashes, not bodies."""
    assertions = _assertions(case)
    counts = {}
    spec_hash = value_digest(assertions)
    def observe(name, arguments, result, tool_call_id):
        counts[name] = counts.get(name,0) + 1
        observations = []
        for index, assertion in enumerate(assertions):
            if assertion['tool'] != name or assertion.get('occurrence',1) != counts[name]:
                continue
            value = arguments if assertion['source']=='arguments' else result
            present = True
            for key in assertion['path'].split('.'):
                try:
                    value = value[int(key)] if isinstance(value,list) else value[key]
                except (KeyError,IndexError,TypeError,ValueError):
                    present = False
                    break
            observations.append({'assertion_index':index,'present':present,
                                 'value_sha256':value_digest(value) if present else None})
        return {'spec_sha256':spec_hash,'observations':observations}
    return observe


def check_tool_assertions(case, steps):
    assertions = _assertions(case)
    if not assertions:
        return {'passed':True,'required':0,'matched':0,'failed_indices':[]}
    spec_hash, observed = value_digest(assertions), {}
    counts = {}
    for step in steps:
        name = step.get('tool')
        counts[name] = counts.get(name,0) + 1
        checks = step.get('evaluation_checks',{})
        if not isinstance(checks,dict) or checks.get('spec_sha256') != spec_hash:
            continue
        for row in checks.get('observations',[]):
            if isinstance(row,dict) and type(row.get('assertion_index')) is int:
                index = row['assertion_index']
                if (0 <= index < len(assertions) and assertions[index]['tool']==name
                        and assertions[index].get('occurrence',1)==counts[name]):
                    observed.setdefault(index,[]).append(row)
    failed = []
    for index, assertion in enumerate(assertions):
        rows = observed.get(index,[])
        if (len(rows)!=1 or rows[0].get('present') is not True
                or rows[0].get('value_sha256') != value_digest(assertion['equals'])):
            failed.append(index)
    return {'passed':not failed,'required':len(assertions),'matched':len(assertions)-len(failed),
            'failed_indices':failed}


def _canonical_text_bytes(path):
    """Checkout line endings do not change Python/JSON text semantics."""
    return path.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')


def aggregate(records):
    n = len(records) or 1
    return {
        'n_tasks': len(records),
        'task_success_rate': round(sum(r['success'] for r in records) / n, 4),
        'tool_accuracy': round(sum(bool(r['tool_accuracy']) for r in records) / n, 4),
        'expected_tool_coverage': round(sum(bool(r['tool_accuracy']) for r in records) / n, 4),
        'avg_steps': round(sum(r['n_steps'] for r in records) / n, 2),
        'max_steps_reached_rate': round(sum(bool(r['max_steps_reached']) for r in records) / n, 4),
        'forbidden_tool_violation_rate': round(sum(bool(r.get('forbidden_tool_violation')) for r in records) / n, 4),
    }


def source_fingerprint(root=ROOT):
    digest = hashlib.sha256()
    files = [p for folder in ('agents','api','config','core','models','scripts') for p in (root/folder).rglob('*.py')]
    files += [root/'mcp_server.py']
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        if '__pycache__' not in path.parts and path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b'\0')
            digest.update(_canonical_text_bytes(path))
    return digest.hexdigest()


def build_provenance(*, model='', config=None, root=ROOT, corpus_path=None, index_manifest_path=None):
    data = _canonical_text_bytes(root/'data/agent_testset.json')
    return {'source_sha256':source_fingerprint(root), 'dataset_sha256':hashlib.sha256(data).hexdigest(),
            'fingerprint_format':FINGERPRINT_FORMAT,
            'created_at':datetime.now(timezone.utc).isoformat(), 'model':model, 'config':config or {},
            'evaluation_contract_version':EVALUATION_CONTRACT_VERSION,
            'environment':environment_provenance(),
            'artifacts':{'corpus':artifact_fingerprint(corpus_path),'index_manifest':artifact_fingerprint(index_manifest_path)},
            'model_revisions':{name:(config or {}).get(name) or 'unavailable' for name in
                               ('EMBEDDING_MODEL_REVISION','CROSS_ENCODER_MODEL_REVISION')},
            'cloud_model_revision':{'status':'unavailable','reason':'provider_alias_has_no_verified_snapshot'},
            'judge_method':'llm_binary_v2'}


def validate_payload(payload, *, require_current_revision=True):
    failures = []
    if not isinstance(payload, dict):
        return {}, ['report must be an object']
    records = payload.get('records')
    if not isinstance(records, list) or not records:
        return {}, ['records must be a nonempty list']
    ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f'record {index} must be an object')
            continue
        identity = record.get('id')
        if not isinstance(identity,str) or not identity or identity in ids:
            failures.append(f'record {index} has missing/duplicate id')
        else:
            ids.add(identity)
        if not isinstance(record.get('trace_id'),str) or not record['trace_id']:
            failures.append(f'record {index} missing trace_id')
        if type(record.get('success')) is not int or record['success'] not in (0,1):
            failures.append(f'record {index} invalid success')
        if type(record.get('judge_success')) is not int or record['judge_success'] not in (0,1):
            failures.append(f'record {index} invalid or missing judge_success; re-evaluate old reports')
        scoring_status, scoring_method = record.get('scoring_status'),record.get('scoring_method')
        if (not isinstance(scoring_status,str) or not isinstance(scoring_method,str)
                or (scoring_status in {'ok','error'} and scoring_method!='llm_binary_v2')
                or (scoring_status=='not_run' and scoring_method!='unavailable')
                or scoring_status not in {'ok','error','not_run'}):
            failures.append(f'record {index} invalid or missing scoring provenance; re-evaluate old reports')
        if scoring_status!='ok' and record.get('judge_success')!=0:
            failures.append(f'record {index} claims a judge pass without successful scoring')
        for field in ('tool_accuracy','max_steps_reached','forbidden_tool_violation'):
            if type(record.get(field)) is not bool:
                failures.append(f'record {index} invalid {field}')
        if type(record.get('n_steps')) is not int or record['n_steps'] < 0:
            failures.append(f'record {index} invalid n_steps')
        tool_lists = [record.get(key) for key in ('actual_tools','expected_tools','forbidden_tools')]
        if any(not isinstance(values,list) or any(not isinstance(v,str) for v in values) for values in tool_lists):
            failures.append(f'record {index} invalid tool lists')
            continue
        actual, expected, forbidden = tool_lists
        if record.get('tool_accuracy') != set(expected).issubset(actual):
            failures.append(f'record {index} tool_accuracy does not match tool lists')
        if record.get('forbidden_tool_violation') != bool(set(forbidden).intersection(actual)):
            failures.append(f'record {index} forbidden_tool_violation does not match tool lists')
        steps = record.get('steps')
        if (not isinstance(steps,list) or any(not isinstance(s,dict) for s in steps)
                or [s.get('tool') for s in steps] != actual or record.get('n_steps') != len(steps)):
            failures.append(f'record {index} steps do not match actual_tools/n_steps')
            continue
        contract_ok = (all(s.get('status') in {'success','approval_required','pending_approval'} for s in steps)
                       and set(expected).issubset(actual) and not set(forbidden).intersection(actual)
                       and not record.get('max_steps_reached')
                       and record.get('run_status') in {'success','approval_required'})
        try:
            checks = check_tool_assertions(record,steps)
        except ValueError:
            failures.append(f'record {index} has invalid tool assertions')
            continue
        if record.get('tool_assertions') and record.get('tool_assertion_checks') != checks:
            failures.append(f'record {index} tool assertion result does not match observations')
        contract_ok = contract_ok and checks['passed']
        if type(record.get('execution_contract_passed')) is not bool or record['execution_contract_passed'] != contract_ok:
            failures.append(f'record {index} invalid execution_contract_passed')
        if record.get('success') == 1 and not contract_ok:
            failures.append(f'record {index} claims success despite failed execution contract')
        scored_success = (type(record.get('judge_success')) is int and record['judge_success']==1
                          and scoring_status=='ok' and scoring_method=='llm_binary_v2' and contract_ok)
        if record.get('success') != int(scored_success):
            failures.append(f'record {index} success does not match judge and execution outcomes')
    if failures:
        return {}, failures
    summary = aggregate(records)
    reported = payload.get('summary')
    if not isinstance(reported,dict):
        failures.append('summary must be an object')
    else:
        for field, value in summary.items():
            if reported.get(field) != value:
                failures.append(f'summary.{field} does not match records')
    if payload.get('schema_version') != 2:
        failures.append('schema_version must be 2; regenerate historical reports')
    provenance = payload.get('provenance')
    if not isinstance(provenance,dict):
        failures.append('missing provenance')
    elif require_current_revision:
        current = build_provenance()
        for key in ('source_sha256','dataset_sha256','fingerprint_format','evaluation_contract_version'):
            if provenance.get(key) != current[key]:
                failures.append(f'stale or missing provenance.{key}')
        cases = json.loads((ROOT/'data/agent_testset.json').read_text(encoding='utf-8'))
        known_cases = {case['id']: case for case in cases}
        known_ids = set(known_cases)
        if not ids.issubset(known_ids):
            failures.append('records contain unknown dataset ids')
        # Hashing the dataset does not bind the report's per-case expectations.
        # Derive these constraints from the canonical case, never from a report
        # that could erase its own required/forbidden tools or change the task.
        for record in records:
            case = known_cases.get(record['id'])
            if case is None:
                continue
            for field, default in (('task', ''), ('success_criteria', ''),
                                   ('task_type', 'general'), ('risk_level', 'low'),
                                   ('expected_tools', []), ('forbidden_tools', [])):
                if record.get(field, default) != case.get(field, default):
                    failures.append(f"record {record['id']} {field} does not match dataset specification")
            if record.get('tool_assertions',[]) != case.get('tool_assertions',[]):
                failures.append(f"record {record['id']} tool_assertions do not match dataset specification")
    return summary, failures
