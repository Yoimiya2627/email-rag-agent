"""Synthetic Gmail authorization data; never usable against a real provider."""
import json
from pathlib import Path


def write_gmail_token(path, *, account='owner@example.test', grant='a' * 32, token='synthetic-access-token'):
    path = Path(path)
    info = {'token': token, 'refresh_token': 'synthetic-refresh-token', 'client_id': 'synthetic-client',
            'client_secret': 'synthetic-secret', 'token_uri': 'https://oauth.invalid/token',
            'expiry': '2099-01-01T00:00:00Z',
            '_email_agent_binding': {'version': 1, 'account_id': account, 'authorization_id': grant}}
    path.write_text(json.dumps(info), encoding='utf-8')
    return path


def bound_payload(provider, payload):
    return {**payload, 'execution_binding': provider.approval_binding()}
