"""Installed Gmail SDK construction and execution through an in-memory transport."""
import json

import pytest

from agents.mail_providers import GmailDraftProvider
from tests.mail_binding_helpers import write_gmail_token, bound_payload


def test_real_sdk_draft_transport_has_budget_and_never_refreshes(tmp_path,monkeypatch):
    httplib2=pytest.importorskip('httplib2')
    pytest.importorskip('googleapiclient.discovery')
    pytest.importorskip('google_auth_httplib2')
    token=write_gmail_token(tmp_path/'synthetic-token.json')
    provider=GmailDraftProvider(token_path=token,timeout_seconds=2)
    service=provider._build_service(timeout_seconds=2)
    transport=service._http
    assert transport.http.timeout==2 and transport.credentials.refresh_token is None
    calls=[]
    def request(uri,method='GET',body=None,headers=None,**kwargs):
        calls.append((uri,method,json.loads(body) if body else None))
        if method == 'GET':
            assert '/profile' in uri
            return httplib2.Response({'status':'200','content-type':'application/json'}), b'{"emailAddress":"owner@example.test"}'
        return httplib2.Response({'status':'200','content-type':'application/json'}),b'{"id":"synthetic-draft","message":{"id":"synthetic-message"}}'
    monkeypatch.setattr(transport.http,'request',request)
    monkeypatch.setattr(transport.credentials,'refresh',lambda *a,**k:pytest.fail('refresh forbidden'))
    provider._service=service
    try:
        result=provider.execute_approval({'approval_id':'synthetic-approval','action_type':'send_email',
             'payload':bound_payload(provider, {'to':['fixture@example.test'],'subject':'Synthetic','body':'Offline fixture only'})})
        assert result['draft_id']=='synthetic-draft' and result['sent'] is False
        assert len(calls)==2 and calls[0][1]=='GET' and calls[1][1]=='POST' and '/drafts' in calls[1][0]
        assert all('/send' not in call[0] for call in calls) and 0<transport.http.timeout<=2
        assert 'raw' in calls[1][2]['message']
    finally: service.close()
