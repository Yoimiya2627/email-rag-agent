"""Deterministic reference replacement at closed, read-only tool boundaries."""
import copy
import hashlib
import json

from core.tool_results import ToolResultUnavailable, trusted_result_scope

POLICY_VERSION = 'closed-tool-refs-v1'


def task_memory_guard(context, *, summary_refs=None):
    """Fingerprint task semantics without invalidating on transcript appends.

    Only summaries actually supplied to this run are pinned. A freshly generated
    unused summary must not prevent safe recovery of an otherwise unchanged run.
    """
    trusted_result_scope(context)
    state = context.session_repository.context_state(context.owner_id, context.session_id)
    projection = {'task_id':state.get('task_id'), 'task_state':state.get('task_state'),
        'facts': [{key:fact.get(key) for key in ('key','version','status','source_turn_id','value','kind','task_id','scope')}
                  for fact in state.get('facts', [])],
        'corrections':[{key:event.get(key) for key in ('event_id','event_type','text','revision','task_id','status')}
                       for event in state.get('user_events', []) if event.get('event_type') == 'current_user_correction'
                       and event.get('status') not in ('resolved','superseded','revoked')]}
    def digest(value):
        return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    if summary_refs is None:
        try:
            material = json.loads((context.task_context or {}).get('text') or '{}')
        except (ValueError, TypeError):
            raise ToolResultUnavailable('task_material_invalid') from None
        summaries = [row.get('summary', {}) for row in material.get('derived_summaries', [])]
        summary_refs = []
        for summary in summaries:
            summary_id = summary.get('summary_id')
            current = state.get('summary')
            if not summary_id or not current or current.get('summary_id') != summary_id:
                raise ToolResultUnavailable('checkpoint_summary_invalidated')
            summary_refs.append({'summary_id':summary_id,'source_hash':digest({key:current.get(key)
                for key in ('summary_id','sources','input_sha256','epoch','source_revision','schema_version','prompt_version')})})
    if not isinstance(summary_refs,list) or len(summary_refs)>10:
        raise ToolResultUnavailable('checkpoint_summary_refs_invalid')
    for ref in summary_refs:
        current = state.get('summary')
        if not current or current.get('summary_id') != ref.get('summary_id') or ref.get('source_hash') != digest({key:current.get(key)
                for key in ('summary_id','sources','input_sha256','epoch','source_revision','schema_version','prompt_version')}):
            raise ToolResultUnavailable('checkpoint_summary_invalidated')
    trusted_result_scope(context)
    return {'state_hash':digest(projection), 'summary_refs':summary_refs}


def validate_task_memory_guard(guard, context):
    if not isinstance(guard,dict) or task_memory_guard(context,summary_refs=guard.get('summary_refs', [])) != guard:
        raise ToolResultUnavailable('checkpoint_task_memory_changed')


def revalidate_restored_evidence(context, *, loader=None):
    """Keep recovered citation ranges only if current indexed text still matches.

    This verifies existing visibility, never adds an unread range to evidence.
    Missing indexes/version changes remove citation authority, not tool counters.
    """
    if not context.visible_evidence:
        return
    from core.evidence import evidence_reference, text_hash
    if loader is None:
        from core.embedder import get_indexed_email
        loader = get_indexed_email
    cache, invalid = {}, []
    for pair, source in list(context.visible_evidence.items()):
        try:
            if pair[0] not in cache:
                cache[pair[0]] = loader(pair[0])
            row = next(item for item in cache[pair[0]].get('chunks', []) if item.get('chunk_id') == pair[1])
            current = evidence_reference(row)
            for ref in source.get('visible_ranges', [evidence_reference(source)]):
                if any(current.get(key) != ref.get(key) for key in ('source_version','source_sha256','chunk_sha256')):
                    raise ValueError('source version changed')
                start, end = ref['visible_start'], ref['visible_end']
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(row['content']) or text_hash(row['content'][start:end]) != ref['visible_hash']:
                    raise ValueError('source range changed')
        except Exception:
            invalid.append(pair)
    for pair in invalid:
        context.visible_evidence.pop(pair, None)
    context.context_metrics['restored_evidence_invalidated'] = context.context_metrics.get('restored_evidence_invalidated', 0) + len(invalid)


def compact_tool_messages(messages, context, *, history_message_count=0, keep_groups=1):
    """Never remove calls, user text, pending actions, or backend execution state.

    Only old closed groups of successful read-only results can replace payloads.
    Stored pages are historical text, not a new source of citation authority.
    """
    if not context.tool_result_store or not context.tool_result_refs:
        return messages
    groups = []
    index = 2 + history_message_count
    while index < len(messages):
        message = messages[index]
        calls = message.get('tool_calls') if message.get('role') == 'assistant' else None
        if not calls:
            index += 1
            continue
        ids = [str(call.get('id', '')) for call in calls]
        end = index + 1
        results = []
        while end < len(messages) and messages[end].get('role') == 'tool':
            results.append(messages[end]); end += 1
        if not ids or len(set(ids)) != len(ids) or [str(row.get('tool_call_id', '')) for row in results] != ids:
            # No compaction crosses an unresolved protocol boundary.
            break
        decoded = []
        try:
            decoded = [json.loads(row.get('content', '{}')) for row in results]
        except (ValueError, TypeError):
            break
        if any(row.get('status') != 'success' or row.get('side_effect_state', 'none') != 'none' for row in decoded):
            break
        # Risk classification comes from the trusted local registry, not result text.
        from agents.tool_registry import TOOL_REGISTRY
        if any(call.get('function', {}).get('name') not in TOOL_REGISTRY or
               TOOL_REGISTRY[call['function']['name']].risk_level != 'low' for call in calls):
            break
        groups.append((index, end, ids, decoded))
        index = end
    selected = groups[:-keep_groups] if keep_groups else groups
    if not selected:
        return messages
    scope = trusted_result_scope(context)
    output = copy.deepcopy(messages)
    replaced = 0
    for begin, end, ids, decoded in selected:
        if any(call_id not in context.tool_result_refs for call_id in ids):
            continue
        for call_id in ids:
            ref = context.tool_result_refs[call_id]
            context.tool_result_store.load(ref['result_id'], content_hash=ref['content_hash'], **scope)
        for offset, (call_id, result) in enumerate(zip(ids, decoded), begin + 1):
            ref = context.tool_result_refs[call_id]
            if result.get('compaction_policy') == POLICY_VERSION:
                continue
            value = {'_tool_result': 1, 'status': result['status'], 'side_effect_state': result.get('side_effect_state', 'none'),
                     'error_code': result.get('error_code'), 'result_ref': ref, 'compaction_policy': POLICY_VERSION,
                     'material_type': 'historical_tool_result', 'evidence_refs': [],
                     'coverage': 'Original output omitted; use get_tool_result to read snapshot; reread get_email for current citations.'}
            # Preserve the original coverage and attachment uncertainty at every nesting level.
            def coverage_rows(item):
                rows = []
                if isinstance(item, dict):
                    if 'coverage' in item:
                        rows.append(item['coverage'])
                    for key, child in item.items():
                        if key != 'coverage':
                            rows.extend(coverage_rows(child))
                elif isinstance(item, list):
                    for child in item:
                        rows.extend(coverage_rows(child))
                return rows
            value['source_coverage'] = coverage_rows(result)
            encoded = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            if len(encoded) < len(output[offset]['content']):
                output[offset]['content'] = encoded
                replaced += 1
    context.context_metrics['compacted_tool_results'] = context.context_metrics.get('compacted_tool_results', 0) + replaced
    return output


def validate_checkpoint_references(checkpoint, context):
    version = checkpoint.get('version', 1)
    if version == 1:
        if context.session_repository is not None:
            raise ToolResultUnavailable('legacy_context_checkpoint_requires_restart')
        if checkpoint.get('compaction_policy') or checkpoint.get('tool_result_refs'):
            raise ToolResultUnavailable('checkpoint_version_downgrade')
        return
    if version != 2 or checkpoint.get('compaction_policy') != POLICY_VERSION:
        raise ToolResultUnavailable('checkpoint_policy_mismatch')
    if (checkpoint.get('owner_id'), checkpoint.get('session_id'), checkpoint.get('context_epoch'), checkpoint.get('run_id')) != (
            context.owner_id, context.session_id, context.context_epoch, context.run_id):
        raise ToolResultUnavailable('checkpoint_scope_mismatch')
    refs = checkpoint.get('tool_result_refs', {})
    if not isinstance(refs, dict) or len(refs) > 512:
        raise ToolResultUnavailable('checkpoint_references_invalid')
    scope = trusted_result_scope(context)
    if refs and context.tool_result_store is None:
        raise ToolResultUnavailable('result_store_unavailable')
    for message in checkpoint.get('messages', []):
        if message.get('role') != 'tool':
            continue
        try:
            result = json.loads(message.get('content', '{}'))
        except (ValueError, TypeError):
            raise ToolResultUnavailable('checkpoint_tool_message_invalid') from None
        if isinstance(result, dict) and result.get('result_ref') is not None:
            if refs.get(str(message.get('tool_call_id'))) != result['result_ref']:
                raise ToolResultUnavailable('checkpoint_unregistered_reference')
    for call_id, ref in refs.items():
        if not isinstance(call_id, str) or not isinstance(ref, dict) or not any(
                step.get('tool_call_id') == call_id and step.get('result_ref') == ref
                for step in checkpoint.get('steps', [])):
            raise ToolResultUnavailable('checkpoint_call_mismatch')
        _, actual = context.tool_result_store.load(ref['result_id'], content_hash=ref['content_hash'], **scope)
        if actual != ref:
            raise ToolResultUnavailable('checkpoint_reference_mismatch')
