"""Subject-only table headers must lead to bounded, versioned body evidence."""
import hashlib
import pytest
from models.schemas import EmailChunk, SearchResult
from core import embedder, pipeline
from tests.test_index_generations import store


def indexed_receipt(store):
    header='Subject: Receipt\n\n'
    body='Table receipt: subscription total 27.00. Payment received.'
    text=header+body
    common={'subject':'Receipt','sender':'billing@example.invalid','source_length':len(text),
            'source_sha256':hashlib.sha256(text.encode()).hexdigest()}
    chunks=[EmailChunk(chunk_id='header-custom-id',email_id='receipt',chunk_index=0,content=header,
                      metadata={**common,'source_start':0,'source_end':len(header)}),
            EmailChunk(chunk_id='body-custom-id',email_id='receipt',chunk_index=1,content=body,
                      metadata={**common,'source_start':len(header),'source_end':len(text)})]
    embedder.index_chunks(chunks,replace=True)
    return [SearchResult(**row,score=.9) for row in
            (embedder.get_email_chunk('receipt',0),embedder.get_email_chunk('receipt',1))]


def search(monkeypatch, results, **kwargs):
    monkeypatch.setattr(pipeline,'hybrid_search',lambda *a,**k:results)
    monkeypatch.setattr(pipeline,'rerank',lambda q,rows,top_n:rows[:top_n])
    return pipeline.retrieve('Receipt',filters=kwargs.pop('filters',{}),**kwargs)


def test_table_header_replaced_before_top_n_with_exact_body_citation(store,monkeypatch):
    head,body=indexed_receipt(store)
    result=search(monkeypatch,[head,body],top_n=1)
    assert result==[body]
    assert result[0].chunk_id=='body-custom-id'
    assert result[0].metadata['source_start']==len(head.content)


def test_expansion_deduplicates_existing_body_and_keeps_result_bound(store,monkeypatch):
    head,body=indexed_receipt(store)
    assert search(monkeypatch,[head,body],top_n=5)==[body]


@pytest.mark.parametrize('change',[{'index_generation':'other'}, {'source_sha256':'0'*64},
                                   {'source_length':999}, {'source_start':999}])
def test_expansion_rejects_mixed_generation_or_source(store,monkeypatch,change):
    head,body=indexed_receipt(store)
    monkeypatch.setattr(pipeline,'get_email_chunk',lambda *a:{**body.model_dump(exclude={'score'}),
                        'metadata':{**body.metadata,**change}})
    with pytest.raises(ValueError,match='sources do not match'):
        search(monkeypatch,[head])


def test_genuinely_empty_or_non_header_mail_does_not_lookup(store,monkeypatch):
    head,body=indexed_receipt(store)
    empty=head.model_copy(deep=True);empty.metadata['source_length']=len(empty.content)
    monkeypatch.setattr(pipeline,'get_email_chunk',lambda *a:pytest.fail('No body lookup expected'))
    assert search(monkeypatch,[empty,body])==[empty,body]


def test_missing_next_chunk_preserves_head_without_inventing_evidence(store,monkeypatch):
    head,_=indexed_receipt(store)
    monkeypatch.setattr(pipeline,'get_email_chunk',lambda *a:None)
    assert search(monkeypatch,[head])==[head]


def test_expanded_body_must_still_match_explicit_filter(store,monkeypatch):
    head,body=indexed_receipt(store)
    monkeypatch.setattr(pipeline,'get_email_chunk',lambda *a:{**body.model_dump(exclude={'score'}),
                        'metadata':{**body.metadata,'sender':'other@example.invalid'}})
    assert search(monkeypatch,[head],filters={'sender':'billing@example.invalid'})==[head]


def test_bounded_lookup_does_not_cross_email_or_invent_chunk_ids(store):
    indexed_receipt(store)
    assert embedder.get_email_chunk('other',1) is None
    assert embedder.get_email_chunk('receipt',9) is None
    assert embedder.get_email_chunk('receipt',1)['chunk_id']=='body-custom-id'
    with pytest.raises(ValueError):embedder.get_email_chunk('receipt',True)


def test_blank_table_separators_do_not_displace_body_at_rerank(store,monkeypatch):
    head,body=indexed_receipt(store)
    blank=body.model_copy(update={'chunk_id':'separator','content':'\n\n','score':1.0})
    assert search(monkeypatch,[blank,head,body],top_n=1)==[body]
    assert search(monkeypatch,[blank])==[]
