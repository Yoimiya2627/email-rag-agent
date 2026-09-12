"""Authenticated, bounded evidence browsing; does not invoke a model or mailbox."""
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.security import require_identity
from core.evidence_pages import read_email_page, read_thread_evidence, reread_evidence


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    email_id: str = Field(min_length=1, max_length=500)
    chunk_id: str = Field(min_length=1, max_length=500)
    source_version: str = Field(min_length=1, max_length=512)
    source_sha256: Optional[str] = Field(None, pattern=r'^[0-9a-f]{64}$')
    chunk_sha256: Optional[str] = Field(None, pattern=r'^[0-9a-f]{64}$')
    visible_start: int = Field(ge=0)
    visible_end: int = Field(ge=1)
    visible_hash: str = Field(pattern=r'^[0-9a-f]{64}$')


def evidence_router(admission, safe_error):
    router = APIRouter(prefix='/evidence', dependencies=[Depends(require_identity)])

    def invoke(function, *args, **kwargs):
        try:
            with admission.slot():
                result = function(*args, **kwargs)
            if result.get('error_code') == 'evidence_not_found':
                raise HTTPException(404, 'Indexed evidence not found')
            return result
        except Exception as exc:
            raise safe_error(exc) from None

    @router.get('/email')
    def email(email_id: str = Query(...,min_length=1,max_length=500),
              chunk_id: Optional[str] = Query(None,min_length=1,max_length=500),
              start: int = Query(0,ge=0), limit: int = Query(1200,ge=1,le=4000),
              source_version: Optional[str] = Query(None,max_length=512),
              source_sha256: Optional[str] = Query(None,pattern=r'^[0-9a-f]{64}$')):
        return invoke(read_email_page,email_id,chunk_id=chunk_id,start=start,limit=limit,
                      source_version=source_version,source_sha256=source_sha256)

    @router.post('/reread')
    def reread(reference: EvidenceReference = Body(...),
               start: Optional[int] = Query(None,ge=0),limit: int = Query(1200,ge=1,le=4000)):
        return invoke(reread_evidence,reference.model_dump(),start=start,limit=limit)

    @router.get('/thread')
    def thread(thread_id: str = Query(...,min_length=1,max_length=500),
               start: int = Query(0,ge=0),limit: int = Query(10,ge=1,le=50),
               source_version: Optional[str] = Query(None,max_length=512)):
        return invoke(read_thread_evidence,thread_id,start=start,limit=limit,source_version=source_version)

    return router
