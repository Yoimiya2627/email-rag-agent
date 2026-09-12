"""Diagnostics are synthetic and keep failed/unsupported cases in denominator."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.context_diagnostics import (read_cases, evaluation_template, validate_evaluation,
                                        source_fingerprint, STAGES)

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'tests/fixtures/context_cases.json'
SCRIPT=ROOT/'scripts/context_diagnostics.py'


def test_fixture_has_fixed_gold_and_boundary_categories():
    fixture=read_cases(CASES)
    tags={tag for row in fixture['cases'] for tag in row['tags']}
    assert {'old_over_30','cjk_two_chars','single_character','symbol_identifier','email_address','date',
        'after_200_chars','negation','failed_assistant_claim','deduplication','task_isolation',
        'effective_correction','summary_source_quote','closed_tools','unclosed_tool_group',
        'stored_summary_20_turns'} <= tags
    assert all(row['quality_gold']['origin']=='synthetic_expected_unreviewed' for row in fixture['cases'])


def test_real_evaluation_template_never_claims_results_and_retains_all_stages():
    fixture=read_cases(CASES)
    template=evaluation_template(fixture)
    assert validate_evaluation(template)==[]
    assert len(template['runs'])==len(fixture['cases'])*len(STAGES)
    bad=copy.deepcopy(template)
    bad['runs'].pop()
    assert any('denominator' in error for error in validate_evaluation(bad))
    bad=copy.deepcopy(template)
    bad['runs'][0]['metrics']['correctness']=1
    assert any('not_run' in error for error in validate_evaluation(bad))
    assert any('human review' in error for error in validate_evaluation(bad))


def test_real_result_requires_fixed_configuration_and_actual_usage_provenance():
    template=evaluation_template(read_cases(CASES))
    row=template['runs'][0]
    row.update(status='complete',answer='a synthetic placeholder')
    row['metrics']['input_tokens']=12
    errors=validate_evaluation(template)
    assert any('missing fixed model_revision' in error for error in errors)
    assert any('unavailable usage' in error for error in errors)


def test_fingerprint_excludes_credentials_and_changes_on_source_edit(tmp_path):
    (tmp_path/'core').mkdir()
    source=tmp_path/'core/example.py'
    source.write_text('x=1\n',encoding='utf-8')
    (tmp_path/'.env').write_text('SYNTHETIC_TEST_SENTINEL=never-read\n',encoding='utf-8')
    first=source_fingerprint(tmp_path)
    assert list(first['files'])==['core/example.py']
    source.write_text('x=2\n',encoding='utf-8')
    assert source_fingerprint(tmp_path)['sha256']!=first['sha256']


def test_current_tree_diagnostic_child_report_has_separate_observations(tmp_path):
    output=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(SCRIPT),'--source-root',str(ROOT),'--cases',str(CASES),
        '--work-dir',str(tmp_path/'w'),'--output',str(output)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['model_calls']==0 and not report['network_enabled']
    assert report['semantic_quality']=='not_tested'
    assert report['denominator']==len(read_cases(CASES)['cases'])==sum(report['counts'].values())
    assert report['counts']['error']==0,[(row['id'],row.get('reason')) for row in report['all_cases'] if row['status']=='error']
    assert report['counts']['fail']==0,[row['id'] for row in report['all_cases'] if row['status']=='fail']
    old=next(row for row in report['all_cases'] if row['id']=='old_35')
    assert set(old['checks'])>={'retrieved_target','visible_required_text'}
    assert report['fixture_sha256']==hashlib.sha256(CASES.read_bytes()).hexdigest()


def test_child_guard_blocks_network_and_private_named_data_reads(tmp_path):
    probe=tmp_path/'data';probe.mkdir()
    (probe/'synthetic.txt').write_text('synthetic',encoding='utf-8')
    code='''import importlib.util, pathlib, socket, sys
spec=importlib.util.spec_from_file_location('diagnostics',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root,work,private=map(pathlib.Path,sys.argv[2:]);work.mkdir()
module.isolate(root,work)
blocked=0
try:
 socket.socket().connect(('127.0.0.1',9))
except RuntimeError:
 blocked+=1
try:
 private.read_text()
except RuntimeError:
 blocked+=1
sys.exit(0 if blocked==2 else 1)
'''
    result=subprocess.run([sys.executable,'-c',code,str(SCRIPT),str(ROOT),str(tmp_path/'isolated'),str(probe/'synthetic.txt')],
                          capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert result.returncode==0,result.stderr
