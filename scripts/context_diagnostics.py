"""Offline synthetic context diagnostics; never a semantic model evaluation.

Compare the frozen and current source trees with exactly the same --cases file.
Each runs in a fresh child process and SQLite directory with network disabled.
--write-evaluation-template creates the separate human/provider result contract;
--validate-evaluation checks its shape, never grades answers automatically.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

STAGES = ('baseline','retrieval_budget','task_state','summary','tool_compaction')
SAFE_CONFIG = {'AGENT_CONTEXT_CHAR_LIMIT':60000,'MODEL_CONTEXT_TOKENS':32000,
               'MODEL_OUTPUT_RESERVE_TOKENS':4000,'MODEL_CONTEXT_SAFETY_TOKENS':128,
               'CONTEXT_MATERIAL_CHAR_LIMIT':4000,'CONTEXT_MATERIAL_TOKEN_LIMIT':2000,
               'ENABLE_AGENT_TRACE':False,'ENABLE_QUERY_REWRITE':False}


def read_cases(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if value.get('schema_version') != 1 or value.get('data_classification') != 'synthetic_only':
        raise ValueError('requires a versioned synthetic-only fixture')
    cases = value.get('cases')
    if not isinstance(cases,list) or not 1 <= len(cases) <= 200:
        raise ValueError('invalid diagnostic cases')
    if len({row['id'] for row in cases}) != len(cases):
        raise ValueError('duplicate case identifiers')
    return value


def evaluation_template(fixture):
    return {'schema_version':1,'dataset_id':fixture['dataset_id'],
        'case_definition_sha256':hashlib.sha256(json.dumps(fixture,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'stage_configurations':{stage:{key:None for key in ('source_fingerprint','context_configuration','prompt_version','index_revision','activation_description')} for stage in STAGES},
        'warning':'No model/semantic quality results have been collected. Fill provider data and independent human review; unknown values remain null.',
        'model_config':{key:None for key in ('provider','model_id','model_revision','tokenizer_revision',
            'temperature','seed','prompt_version','index_revision','source_fingerprint','context_configuration')},
        'cases':[{'id':case['id'],'gold':case['quality_gold']} for case in fixture['cases']],
        'runs':[{'case_id':case['id'],'stage':stage,'status':'not_run','answer':None,'error':None,
            'metrics':{key:None for key in ('first_answer_ms','total_ms','input_tokens','output_tokens','actual_cost',
                'correctness','constraint_compliance','source_support','critical_omission_count','wrong_citation_count')},
            'usage_source':'unavailable','review':{'completed':False,'reviewer':None,'notes':None}}
            for case in fixture['cases'] for stage in STAGES]}


def validate_evaluation(value, fixture=None):
    errors=[]
    if not isinstance(value,dict) or value.get('schema_version') != 1:
        return ['invalid evaluation schema']
    if not isinstance(value.get('cases'),list) or not isinstance(value.get('runs'),list) or not isinstance(value.get('model_config'),dict):
        return ['cases/runs must be arrays and model_config must be an object']
    if not value['cases'] or not isinstance(value.get('stage_configurations'),dict):
        return ['nonempty cases and stage_configurations are required']
    ids={row.get('id') for row in value.get('cases',[]) if isinstance(row,dict)}
    if fixture is not None:
        expected_template=evaluation_template(fixture)
        if any(value.get(key)!=expected_template[key] for key in ('dataset_id','case_definition_sha256','cases')):
            errors.append('dataset/gold changed from the fixed fixture')
    config=value.get('model_config',{})
    seen=set()
    for index,row in enumerate(value.get('runs',[])):
        if not isinstance(row,dict):
            errors.append(f'run {index}: invalid object'); continue
        key=(row.get('case_id'),row.get('stage'))
        if key in seen: errors.append(f'run {index}: duplicate stage/case')
        seen.add(key)
        if key[0] not in ids or key[1] not in STAGES: errors.append(f'run {index}: unknown case/stage')
        if row.get('status') not in ('not_run','complete','failed','cancelled'): errors.append(f'run {index}: invalid status')
        metrics=row.get('metrics',{})
        if not isinstance(metrics,dict) or not isinstance(row.get('review'),dict):
            errors.append(f'run {index}: metrics/review must be objects');continue
        if row.get('usage_source') not in ('provider','unavailable'):
            errors.append(f'run {index}: usage_source must distinguish provider from unavailable')
        for name,number in metrics.items():
            if number is not None and (type(number) not in (int,float) or not 0 <= number < float('inf')):
                errors.append(f'run {index}: invalid metric {name}')
        if row.get('status') == 'not_run' and (row.get('answer') is not None or any(v is not None for v in metrics.values())):
            errors.append(f'run {index}: not_run cannot contain measured results')
        if row.get('status') == 'complete':
            if not isinstance(row.get('answer'),str) or not row['answer'].strip(): errors.append(f'run {index}: complete requires answer')
            for name in ('provider','model_id','model_revision','temperature'):
                if config.get(name) is None: errors.append(f'run {index}: missing fixed {name}')
            stage_config=value.get('stage_configurations',{}).get(row.get('stage'),{})
            for name in ('prompt_version','index_revision','source_fingerprint','context_configuration'):
                if stage_config.get(name) is None: errors.append(f'run {index}: missing stage-specific {name}')
        for name in ('correctness','constraint_compliance','source_support'):
            score=metrics.get(name)
            if type(score) in (int,float) and not 0 <= score <= 1:
                errors.append(f'run {index}: {name} must be in [0,1]')
        scored=any(metrics.get(name) is not None for name in ('correctness','constraint_compliance','source_support','critical_omission_count','wrong_citation_count'))
        review=row.get('review',{})
        if scored and (review.get('completed') is not True or not review.get('reviewer')):
            errors.append(f'run {index}: semantic scores require independent human review')
        if row.get('usage_source') == 'unavailable' and any(metrics.get(name) is not None for name in ('input_tokens','output_tokens','actual_cost')):
            errors.append(f'run {index}: unavailable usage cannot be actual usage')
    expected={(identifier,stage) for identifier in ids for stage in STAGES}
    if seen != expected: errors.append('all cases and five stages must remain in the failure denominator')
    return errors


def source_fingerprint(root):
    files={}
    for folder in ('core','agents','config','models','api'):
        for path in sorted((root/folder).glob('*.py')):
            files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=json.dumps(files,sort_keys=True,separators=(',',':'))
    return {'sha256':hashlib.sha256(raw.encode()).hexdigest(),'files':files}


def isolate(root, work):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root))
    import dotenv
    dotenv.load_dotenv=lambda *a,**kw:False
    dotenv.dotenv_values=lambda *a,**kw:{}
    os.environ.update(DEEPSEEK_API_KEY='synthetic-offline-placeholder',API_AUTH_TOKEN='',MCP_AUTH_TOKEN='',
        ENABLE_AGENT_TRACE='false',ENABLE_MCP_AUDIT='false',ENABLE_REAL_EMAIL_SEND='false',
        MAIL_PROVIDER='simulated',AGENT_TOOL_BACKEND='local',PYTHONDONTWRITEBYTECODE='1')
    for name in ('SESSION_STORE_PATH','JOB_STORE_PATH','TOOL_RESULT_STORE_PATH','APPROVAL_STORE_PATH',
                 'EMAIL_DATA_PATH','CHROMA_PERSIST_DIR','GMAIL_CREDENTIALS_PATH','GMAIL_TOKEN_PATH',
                 'GMAIL_READONLY_TOKEN_PATH','GMAIL_SYNC_STATE_PATH','GMAIL_SYNC_OUTPUT_PATH'):
        os.environ[name]=str(work/(name.lower()+'.isolated'))
    def guard(event,args):
        if event=='socket.connect': raise RuntimeError('diagnostic network access is disabled')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            mode,flags=args[1],args[2]
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&3!=0)
            if write and not path.is_relative_to(work): raise RuntimeError('diagnostic write escaped isolated directory')
            if not write and not path.is_relative_to(work):
                if path.name=='.env' or any(part.lower() in ('credentials','data') for part in path.parts):
                    raise RuntimeError('diagnostic private data read denied')
    sys.addaudithook(guard)
    import config.settings as cfg
    for name,value in SAFE_CONFIG.items(): setattr(cfg,name,value)
    cfg.MODEL_CONTEXT_PROFILES={}
    cfg.MODEL_REVISION=None
    cfg.CONTEXT_PURPOSE_WEIGHTS={}


def seed(repo, owner, sid, case):
    prefix=case.get('answer_prefix_repeat',['',0])
    turns=[{'turn_id':'target','query':case.get('source_query','合成用户预算要求100元'),
        'answer':prefix[0]*prefix[1]+case.get('source_answer','合成助手记录。'),
        'metadata':{'status':case.get('status','success')},'include_in_context':case.get('status','success')=='success'}]
    turns += [{'turn_id':f'filler-{i}','query':f'无关闲聊第{i}轮','answer':'天气主题占位文字。',
               'metadata':{'status':'success'},'include_in_context':True} for i in range(case.get('fillers',0))]
    repo.append_turns(owner,sid,turns,expected_revision=repo.revision(owner,sid))
    return turns


def selected(repo,owner,sid,case,matches):
    from core.memory import ConversationMemory,build_model_messages
    from core.session_context import assemble_session_context
    from agents.runtime import RunContext,use_run_context
    kwargs={'char_limit':4000,'token_limit':2000}
    facts=[]
    if hasattr(repo,'context_state'):
        from core.context_contracts import TrustedScope
        state=repo.context_state(owner,sid)
        facts=state['facts']
        kwargs.update(scope=TrustedScope(owner,sid),task_state=state.get('task_state'),summary=state.get('summary'),
                      user_events=state.get('user_events'),current_request=case['query'])
    else:
        facts=repo.task_facts(owner,sid)
    material=assemble_session_context(facts,matches,**kwargs)
    memory=ConversationMemory(max_turns=30)
    memory.load_context(repo.recent_context(owner,sid,max_turns=30))
    run=RunContext(owner_id=owner,session_id=sid,task_context=material,context_char_limit=60000)
    with use_run_context(run):
        messages=build_model_messages('合成诊断；资料不代表权限。',case['query'],memory.to_messages())
    rendered='\n'.join(row.get('content','') for row in messages)
    return rendered,material,run.context_metrics


def retrieval_case(repo,owner,sid,case):
    started=time.perf_counter()
    # Mirror each tree's API history strategy without importing the HTTP app or
    # initializing any mailbox/embedding/model services.
    if hasattr(repo,'context_state'):
        matches=repo.search_history(owner,sid,case['query'],limit=15)
        strategy='current_full_query_literal_fts'
    else:
        terms=list(dict.fromkeys(re.findall(r'[A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,8}',case['query'])))[:3]
        found={}
        for term in terms:
            for row in repo.search_history(owner,sid,term,limit=5): found[row['turn_id']]=row
        matches=list(found.values());strategy='baseline_api_three_terms'
    retrieved_at=time.perf_counter()
    rendered,material,metrics=selected(repo,owner,sid,case,matches)
    found=[row['turn_id'] for row in matches]
    checks={'retrieved_target':'target' in found,
            'visible_required_text':all(text in rendered for text in case['required_text'])}
    if case.get('max_occurrences'):
        checks['duplicate_limit']=all(rendered.count(text)<=case['max_occurrences'] for text in case['required_text'])
    if case.get('expected_status'):
        checks['historical_status_label']=any(row.get('metadata',{}).get('status')==case['expected_status'] for row in matches)
        checks['failed_turn_excluded_from_recent']=all(row['turn_id']!='target' for row in repo.recent_context(owner,sid))
    return checks,{'retrieval_strategy':strategy,'retrieved_turn_ids':found,
        'retrieval_ms':round((retrieved_at-started)*1000,3),
        'assembly_ms':round((time.perf_counter()-retrieved_at)*1000,3),
        'visible_required_text':{text:text in rendered for text in case['required_text']},
        'required_text_occurrences':{text:rendered.count(text) for text in case['required_text']},
        'selected_manifest':material.get('material_manifest',[]),'omissions':material.get('omissions',[]),
        'context_metrics':metrics,'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest()}


def feature_case(repo,owner,sid,case,turns,work):
    kind=case['kind']
    if kind in ('summary_quote','summary_twenty'):
        from core.session_summary import validate_summary
        row=repo.get_turn(owner,sid,'target')
        quote={'turn_id':'target','field':'query','start':0,'end':len(row['query']),'text':row['query']}
        value={'schema_version':1,'sections':{'constraints':[{'text':row['query'],'source_quotes':[quote]}]}}
        valid=validate_summary(value,[row])
        if kind=='summary_twenty':
            from core.session_context import assemble_session_context
            outcome=repo.generate_summary(owner,sid,generate=lambda **kwargs:value,force=True,min_turns=20,
                model_id='deterministic-fixture-callback',model_revision='not-a-model')
            stored=repo.get_semantic_summary(owner,sid)
            context=assemble_session_context(summary=stored)
            visible=json.loads(context['text']) if context['text'] else {}
            summaries=visible.get('derived_summaries',[])
            return {'stored_twenty_sources':bool(stored) and len(stored['sources'])==20,
                'summary_model_view_visible':bool(summaries) and case['required_text'][0] in context['text'],
                'default_budget_met':context['context_chars']<=4100 and context['estimated_input_tokens']<=2000,
                'backend_sources_not_duplicated':bool(summaries) and 'sources' not in summaries[0]['summary']}, {
                'generation_status':outcome['status'],'fixture_callback':'deterministic_extract_not_semantic_model',
                'stored_source_count':len(stored['sources']) if stored else 0,
                'stored_chars':len(json.dumps(stored,ensure_ascii=False)),'model_view_chars':len(context['text']),
                'visible_summary_ids':[group['summary'].get('summary_id') for group in summaries]}
        value['sections']['constraints'][0]['source_quotes'][0]['text']='tampered'
        rejected=False
        try: validate_summary(value,[row])
        except ValueError: rejected=True
        normalized=valid['sections']['constraints'][0]['source_quotes'][0]
        return {'quote_hash_exact':normalized['sha256']==hashlib.sha256(row['query'].encode()).hexdigest(),
                'tampered_quote_rejected':rejected}, {'normalized_summary':valid,'semantic_quality':'not_tested'}
    if kind=='correction':
        if not hasattr(repo,'record_current_request'): raise NotImplementedError('current_request_event_unavailable')
        repo.set_task_fact(owner,sid,'budget','100元',source_turn_id='target',expected_version=0,explicit_user=True)
        repo.record_current_request(owner,sid,text=case['query'],request_id='correction')
        state=repo.context_state(owner,sid)
        rendered,material,metrics=selected(repo,owner,sid,case,[])
        return {'old_constraint_inactive':not any(row['value']=='100元' for row in state['facts']),
                'current_correction_visible':case['query'] in rendered}, {'facts':state['facts'],'events':state['user_events'],'context_metrics':metrics}
    if kind=='task_switch':
        if not hasattr(repo,'update_task'): raise NotImplementedError('task_switch_unavailable')
        for task,value in (('a','100'),('b','200')):
            repo.update_task(owner,sid,task_id=task,goal='合成任务'+task,source_turn_id='target',
                             expected_revision=repo.revision(owner,sid),explicit_user=True)
            repo.set_task_fact(owner,sid,'budget',value,source_turn_id='target',expected_version=0,explicit_user=True)
        state=repo.context_state(owner,sid)
        return {'active_task_b':state['task_id']=='b','old_task_fact_absent':all(row['value']!='100' for row in state['facts']),
                'new_task_fact_present':any(row['value']=='200' for row in state['facts'])}, {'active_task':state['task_id'],'facts':state['facts']}
    if kind in ('tool_compaction','open_tool_boundary'):
        from core.tool_results import ToolResultStore
        from core.tool_compaction import compact_tool_messages
        from agents.runtime import RunContext
        context=RunContext(owner_id=owner,session_id=sid,run_id='synthetic-run')
        context.session_repository=repo
        context.context_epoch=repo.context_epoch(owner,sid)
        context.tool_result_store=ToolResultStore(work/(sid+'-tools.db'))
        value={'_tool_result':1,'status':'success','side_effect_state':'none','data':{'body':'payload-preserved'*500,
               'coverage':{'attachment_inventory_status':'unknown','unread_attachments':None}}}
        ref=context.tool_result_store.put(owner=owner,session=sid,run=context.run_id,epoch=context.context_epoch,
            call_id='c',tool='search_emails',argument_hash='0'*64,value=value)
        context.tool_result_refs={'c':ref}
        messages=[{'role':'system','content':'synthetic'},{'role':'user','content':case['query']},
            {'role':'assistant','tool_calls':[{'id':'c','type':'function','function':{'name':'search_emails','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'c','content':json.dumps(value)}]
        if kind=='open_tool_boundary': messages[2]['tool_calls'].append({'id':'pending','type':'function','function':{'name':'search_emails','arguments':'{}'}})
        compacted=compact_tool_messages(messages,context,keep_groups=0)
        if kind=='open_tool_boundary':
            checks={'unclosed_group_unchanged':compacted==messages}
        else:
            body=json.loads(compacted[-1]['content'])
            checks={'closed_result_compacted':body.get('material_type')=='historical_tool_result',
                'coverage_unknown_retained':any(row.get('attachment_inventory_status')=='unknown' for row in body.get('source_coverage',[]) if isinstance(row,dict)),
                'current_request_unchanged':compacted[1]==messages[1]}
        return checks,{'before_chars':len(json.dumps(messages)),'after_chars':len(json.dumps(compacted)),
                       'context_metrics':context.context_metrics}
    raise ValueError('unknown diagnostic kind')


def worker(root,work,fixture):
    before=source_fingerprint(root)
    isolate(root,work)
    from core.session_repository import SessionRepository
    repo=SessionRepository(work/'synthetic.db')
    results=[]
    for case in fixture['cases']:
        started=time.perf_counter();owner='synthetic-owner';sid=case['id']
        try:
            turns=seed(repo,owner,sid,case)
            checks,details=(retrieval_case(repo,owner,sid,case) if case['kind']=='retrieval'
                            else feature_case(repo,owner,sid,case,turns,work))
            result={'status':'pass' if all(checks.values()) else 'fail','checks':checks,'details':details}
        except (ModuleNotFoundError,NotImplementedError) as exc:
            result={'status':'unsupported','checks':{},'error_type':type(exc).__name__,'reason':str(exc)}
        except Exception as exc:
            result={'status':'error','checks':{},'error_type':type(exc).__name__,'reason':str(exc)}
        results.append({'id':case['id'],'kind':case['kind'],'tags':case['tags'],**result,
                        'latency_ms':round((time.perf_counter()-started)*1000,3),'semantic_quality':'not_tested'})
    after=source_fingerprint(root)
    try:
        from core.history_index import ANALYZER_VERSION
    except ImportError: ANALYZER_VERSION='legacy_substring'
    counts={status:sum(row['status']==status for row in results) for status in ('pass','fail','error','unsupported')}
    return {'schema_version':1,'dataset_id':fixture['dataset_id'],'data_classification':'synthetic_only',
        'case_definition_sha256':hashlib.sha256(json.dumps(fixture,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'evaluation_kind':'deterministic_retrieval_selection_and_contracts','semantic_quality':'not_tested',
        'model_calls':0,'network_enabled':False,'source_root':str(root),'source':before,
        'source_consistent_during_run':before['sha256']==after['sha256'],'sqlite_version':sqlite3.sqlite_version,
        'analyzer':ANALYZER_VERSION,'fts_available':getattr(repo,'fts_available',None),'safe_configuration':SAFE_CONFIG,
        'denominator':len(results),'counts':counts,'all_cases':results,
        'latency_definition':'Per-case latency includes synthetic seeding and deterministic evaluation; retrieval/assembly timings are separately reported. No production latency claim.',
        'interpretation':'Turn hit, visible text and contract checks are separate evidence. Failures and unsupported cases remain in denominator; no model quality gains are asserted.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--cases',type=Path,default=Path(__file__).resolve().parents[1]/'tests/fixtures/context_cases.json')
    parser.add_argument('--work-dir',type=Path,default=Path(tempfile.gettempdir())/'ctx-diagnostics')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--write-evaluation-template',type=Path)
    parser.add_argument('--validate-evaluation',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.validate_evaluation:
        errors=validate_evaluation(json.loads(args.validate_evaluation.read_text(encoding='utf-8-sig')),read_cases(args.cases))
        print(json.dumps({'valid':not errors,'errors':errors},ensure_ascii=False));return int(bool(errors))
    fixture=read_cases(args.cases)
    if args.write_evaluation_template:
        args.write_evaluation_template.parent.mkdir(parents=True,exist_ok=True)
        args.write_evaluation_template.write_text(json.dumps(evaluation_template(fixture),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return 0
    if not args.source_root or not args.output: parser.error('--source-root and --output are required')
    root=args.source_root.resolve();work=args.work_dir.resolve()
    if not (root/'core/session_repository.py').is_file(): parser.error('source root has no session repository')
    if work==root or work.is_relative_to(root): parser.error('work directory must be outside source tree')
    if len(str(work))>160: parser.error('use a short writable diagnostics work directory')
    work.mkdir(parents=True,exist_ok=True)
    if args.worker:
        report=worker(root,work,fixture)
        report['fixture_sha256']=hashlib.sha256(args.cases.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return 0
    with tempfile.TemporaryDirectory(prefix='ctx-',dir=work) as directory:
        state=Path(directory).resolve()
        if state.parent!=work: raise RuntimeError('unexpected temporary directory')
        result_path=state/'report.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--source-root',str(root),
                 '--cases',str(args.cases.resolve()),'--work-dir',str(state),'--output',str(result_path)]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:
            print(result.stderr,file=sys.stderr);return result.returncode
        report=json.loads(result_path.read_text(encoding='utf-8'))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(args.output.resolve()),'denominator':report['denominator'],'counts':report['counts'],
                      'semantic_quality':'not_tested'},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
