"""Bounded, literal-safe transcript retrieval and verifiable hit windows."""
import hashlib
import re

ANALYZER_VERSION = 'cjk-bigram-v1'
MAX_CANDIDATES = 400
MAX_FALLBACK = 1000


def tokens(text):
    parts = re.findall(r'[\u3400-\u9fff]+|[a-zA-Z0-9]+(?:[._@:/+\-][a-zA-Z0-9]+)*', text.casefold())
    result = []
    for part in parts:
        if '\u3400' <= part[0] <= '\u9fff':
            result.extend(part[i:i+2] for i in range(max(1, len(part)-1)))
        else:
            result.append(part)
    return list(dict.fromkeys(result))


def index_turn(db, row):
    db.execute('INSERT OR REPLACE INTO history_documents(seq,owner_id,session_id,analyzer) VALUES (?,?,?,?)',
               (row['seq'], row['owner_id'], row['session_id'], ANALYZER_VERSION))
    db.execute('DELETE FROM history_fts WHERE rowid=?', (row['seq'],))
    db.execute('INSERT INTO history_fts(rowid,terms) VALUES (?,?)',
               (row['seq'], ' '.join(tokens(row['query']+'\n'+row['answer']))))


def hit_windows(row, query, max_chars=320):
    words = tokens(query)
    hits = []
    for field in ('query', 'answer'):
        original = row[field]
        # lower(), unlike casefold(), normally preserves offsets, but scan the
        # original for literal case-insensitive matches to handle Unicode safely.
        match = re.search(re.escape(query.strip()), original, flags=re.IGNORECASE)
        if match is None:
            matches = [m for word in words for m in [re.search(re.escape(word), original, re.IGNORECASE)] if m]
            match = min(matches, key=lambda m:m.start()) if matches else None
        if match is None:
            continue
        start = max(0, match.start()-max_chars//3)
        end = min(len(original), max(start+max_chars, match.end()))
        text = original[start:end]
        hits.append({'field':field, 'start':start, 'end':end, 'text':text,
                     'sha256':hashlib.sha256(text.encode()).hexdigest(),
                     'source_sha256':hashlib.sha256(original.encode()).hexdigest(),
                     'offset_basis':'unicode_codepoints', 'truncated':start>0 or end<len(original)})
    return hits


def search(db, key, query, limit, fts_available=True, *, task_id=None, diagnostics=None):
    import sqlite3
    import time
    started=time.monotonic()
    deadline=started+0.5
    max_chars=2_000_000
    diagnostics=diagnostics if diagnostics is not None else {}
    terms=tokens(query)[:32]
    task_filter=' AND t.turn_id IN (SELECT turn_id FROM context_turn_tasks WHERE owner_id=? AND session_id=? AND task_id=?)' if task_id is not None else ''
    task_params=(*key,task_id) if task_id is not None else ()
    candidates={};scanned=set();chars=0;degraded=None;limited=False
    db.set_progress_handler(lambda: int(time.monotonic()>deadline),1000)
    try:
        fts_ids=[]
        if fts_available and terms:
            expression=' OR '.join('"'+word.replace('"','""')+'"' for word in terms)
            try:
                fts_ids=db.execute('SELECT t.seq,length(t.query)+length(t.answer) AS chars FROM history_fts f JOIN session_turns t ON t.seq=f.rowid WHERE history_fts MATCH ? AND t.owner_id=? AND t.session_id=?'+task_filter+' ORDER BY bm25(history_fts),t.seq DESC LIMIT ?',
                    (expression,*key,*task_params,MAX_CANDIDATES)).fetchall()
            except sqlite3.OperationalError:
                degraded='fts_query_failed_or_timed_out'
        else:
            degraded='fts_unavailable' if not fts_available else 'literal_fallback'
        for item in fts_ids:
            if time.monotonic()>deadline or chars+item['chars']>max_chars:
                limited=True;continue
            row=db.execute('SELECT * FROM session_turns WHERE seq=? AND owner_id=? AND session_id=?',(item['seq'],*key)).fetchone()
            if row is not None:candidates[row['seq']]=row;scanned.add(row['seq']);chars+=item['chars']
        fallback=[]
        if time.monotonic()<deadline:
            fallback=db.execute('SELECT t.seq,length(t.query)+length(t.answer) AS chars FROM session_turns t WHERE owner_id=? AND session_id=?'+task_filter+' ORDER BY seq DESC LIMIT ?',(*key,*task_params,MAX_FALLBACK+1)).fetchall()
        for item in fallback[:MAX_FALLBACK]:
            if item['seq'] in scanned:continue
            if time.monotonic()>deadline or chars+item['chars']>max_chars:
                limited=True;break
            row=db.execute('SELECT * FROM session_turns WHERE seq=? AND owner_id=? AND session_id=?',(item['seq'],*key)).fetchone()
            chars+=item['chars'];scanned.add(item['seq'])
            if row is not None and query.casefold().strip() in (row['query']+'\n'+row['answer']).casefold():candidates[row['seq']]=row
        exact=bool(re.fullmatch(r'[A-Za-z0-9]+(?:[._@:/+\-][A-Za-z0-9]+)+',query.strip()))
        ranked=[]
        for row in candidates.values():
            combined=(row['query']+'\n'+row['answer']).casefold()
            literal=query.casefold().strip() in combined
            if exact and not re.search(r'(?<![A-Za-z0-9._@:/+\-])'+re.escape(query.strip())+r'(?![A-Za-z0-9._@:/+\-])',row['query']+'\n'+row['answer'],re.IGNORECASE):continue
            score=100*literal+sum(word in combined for word in terms)
            if score:ranked.append((score,row['seq'],row))
        ranked.sort(key=lambda item:(item[0],item[1]),reverse=True)
        diagnostics.update(analyzer_version=ANALYZER_VERSION,degraded_reason=degraded,
            fallback_truncated=len(fallback)>MAX_FALLBACK or limited,scan_chars=chars,
            scan_char_limit=max_chars,time_limit_ms=500,elapsed_ms=round((time.monotonic()-started)*1000,3),candidate_limit=MAX_CANDIDATES)
        return [(row,{**diagnostics,'score':score}) for score,seq,row in ranked[:limit]]
    except sqlite3.OperationalError as exc:
        if 'interrupt' not in str(exc).lower():raise
        diagnostics.update(degraded_reason='search_time_budget_exhausted',fallback_truncated=True,scan_chars=chars,time_limit_ms=500)
        return []
    finally:
        db.set_progress_handler(None,0)
