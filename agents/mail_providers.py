"""Mail-provider execution for approved high-risk email actions.

The agent still cannot send email directly.  Providers only run after a human
approves a pending action through the approval API.  The default provider is a
simulation; Gmail creates a real draft and deliberately does not send it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
import threading
import time
import uuid
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any

import config.settings as cfg
from agents.approvals import ApprovalPreconditionError


class MailProviderError(RuntimeError):
    """Raised when an approved mail action cannot be executed."""


class MailProviderPreconditionError(MailProviderError, ApprovalPreconditionError):
    """Invalid mail content; no provider mutation has been submitted."""


class SimulatedMailProvider:
    """Local provider used by default for tests and offline demos."""

    provider = "simulated"

    def approval_binding(self) -> dict[str, Any]:
        return {"version": 1, "provider": self.provider}

    def execute_approval(self, approval: dict[str, Any]) -> dict[str, Any]:
        _validate_action(approval)
        build_rfc822_message(approval.get("payload"))
        binding = approval.get("payload", {}).get("execution_binding")
        # Legacy simulated actions remain harmless, but a bound Gmail approval
        # must never silently become a different action after settings change.
        if binding is not None and binding != self.approval_binding():
            raise MailProviderPreconditionError('Approval provider changed; create and review a new request')
        return {
            "mode": "simulated_send",
            "provider": self.provider,
            "sent": False,
            "simulated": True,
            "approval_id": approval.get("approval_id", ""),
            "request_id": approval.get("request_id", ""),
        }


class GmailDraftProvider:
    """Create Gmail drafts after human approval.

    The service object is injectable so tests can use a fake Gmail client.  If
    omitted, the provider builds a real Gmail API service lazily from settings.
    """

    provider = "gmail"

    def __init__(
        self,
        service: Any | None = None,
        user_id: str | None = None,
        credentials_path: str | Path | None = None,
        token_path: str | Path | None = None,
        scopes: list[str] | None = None,
        timeout_seconds: float | None = None,
    ):
        self._service = service
        self.user_id = user_id or cfg.GMAIL_USER_ID
        self.credentials_path = Path(credentials_path or cfg.GMAIL_CREDENTIALS_PATH)
        self.token_path = Path(token_path or cfg.GMAIL_TOKEN_PATH)
        self.scopes = scopes or list(cfg.GMAIL_SCOPES)
        self.timeout_seconds = (getattr(cfg, 'MAIL_APPROVAL_TIMEOUT_SECONDS', 30)
                                if timeout_seconds is None else timeout_seconds)
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise MailProviderPreconditionError('approval timeout must be a finite positive number')

    def _authorization_snapshot(self):
        info, digest = _read_token_info(self.token_path)
        grant = info.get('_email_agent_binding')
        if (not isinstance(grant, dict) or type(grant.get('version')) is not int or grant['version'] != 1
                or not isinstance(grant.get('authorization_id'), str)
                or len(grant['authorization_id']) != 32
                or any(c not in '0123456789abcdef' for c in grant['authorization_id'])):
            raise MailProviderPreconditionError('Gmail account binding is missing; run local authorization before creating a new approval')
        account = _account_id(grant.get('account_id'))
        if self.user_id != 'me' and _account_id(self.user_id) != account:
            raise MailProviderPreconditionError('Configured Gmail user differs from the authorized account')
        return info, {"version": 1, "provider": self.provider, "account_id": account,
                      "user_id": self.user_id, "authorization_id": grant['authorization_id'],
                      "authorization_sha256": digest, "scopes": sorted(self.scopes)}

    def approval_binding(self) -> dict[str, Any]:
        # Identity was verified during explicit local authorization. This read
        # neither contacts Gmail nor refreshes credentials from a model tool.
        return self._authorization_snapshot()[1]

    def execute_approval(self, approval: dict[str, Any]) -> dict[str, Any]:
        _validate_action(approval)
        payload = approval.get("payload")
        request_id = approval.get("request_id") or approval.get("approval_id")
        try:
            # Approval IDs are globally unique; request IDs are only owner-scoped.
            raw = build_gmail_raw_message(payload, request_id=approval.get("approval_id") or request_id)
        except MailProviderPreconditionError:
            raise
        except (TypeError, ValueError, UnicodeError) as exc:
            raise MailProviderPreconditionError("email content cannot be encoded as a draft") from exc
        binding = payload.get('execution_binding')
        if not isinstance(binding, dict) or binding.get('provider') != self.provider:
            raise MailProviderPreconditionError('Gmail approval has no matching account binding; create and review a new request')
        # A timed-out worker may still be returning from a blocking SDK call.
        # The lock prevents a worker from starting the mutation after the caller
        # has timed out. Once execute starts, timeout means unknown, never retry.
        deadline = time.monotonic() + self.timeout_seconds
        state = {'submitted': False, 'cancelled': False}
        lock, finished = threading.Lock(), threading.Event()

        def execute():
            service = None
            try:
                try:
                    token_info, current_binding = self._authorization_snapshot()
                    if binding != current_binding:
                        raise MailProviderPreconditionError('Gmail authorization changed; create and review a new request')
                    # Build from these exact bytes, never re-open a potentially
                    # replaced token file to construct the authenticated client.
                    service = self._service or self._build_service(timeout_seconds=self.timeout_seconds, token_info=token_info)
                    account = _profile_account(service, timeout_seconds=max(0, min(self.timeout_seconds, deadline-time.monotonic())))
                    if account != binding['account_id']:
                        raise MailProviderPreconditionError('Authenticated Gmail account differs from the approval')
                    if self.approval_binding() != binding:
                        raise MailProviderPreconditionError('Gmail authorization changed during account verification')
                    request = service.users().drafts().create(userId=self.user_id, body={"message": {"raw": raw}})
                except MailProviderPreconditionError:
                    raise
                except Exception as exc:
                    raise MailProviderPreconditionError('Gmail draft setup failed; complete local authorization and configuration first') from exc
                with lock:
                    remaining = min(self.timeout_seconds, deadline - time.monotonic())
                    if state['cancelled'] or remaining <= 0:
                        raise MailProviderPreconditionError('approval timed out before submission')
                    # Real googleapiclient requests expose their httplib2 client.
                    transport = getattr(request, 'http', None)
                    transport = getattr(transport, 'http', transport)  # AuthorizedHttp wraps httplib2.Http.
                    if transport is not None and hasattr(transport, 'timeout'):
                        transport.timeout = remaining
                    state['submitted'] = True
                state['result'] = request.execute()  # SDK defaults to zero retries.
            except BaseException as exc:
                state['error'] = (exc if state['submitted'] or isinstance(exc, MailProviderPreconditionError)
                                  else MailProviderPreconditionError('Gmail draft failed before submission'))
            finally:
                finished.set()
                if self._service is None and service is not None:
                    close = getattr(service, 'close', None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass  # Cleanup must not replace the observed provider outcome.

        try:
            threading.Thread(target=execute, daemon=True, name='gmail-draft-approval').start()
        except Exception as exc:
            raise MailProviderPreconditionError('approval worker could not start; no draft was submitted') from exc
        if not finished.wait(max(0, min(self.timeout_seconds, deadline - time.monotonic()))):
            with lock:
                state['cancelled'] = True
                submitted = state['submitted']
            if submitted:
                raise MailProviderError('Gmail draft timed out after submission; reconcile before any new action')
            raise MailProviderPreconditionError('approval timed out before submission')
        if 'error' in state:
            raise state['error']
        draft = state['result']
        if not isinstance(draft, dict) or not draft.get("id"):
            raise MailProviderError("Gmail returned no draft identifier; reconcile the action before retrying")
        message = draft.get("message") or {}
        return {
            "mode": "gmail_draft",
            "provider": self.provider,
            "draft_id": draft.get("id", ""),
            "message_id": message.get("id", ""),
            "sent": False,
            "approval_id": approval.get("approval_id", ""),
            "request_id": request_id or "",
            "account_id": binding['account_id'],
        }

    def _build_service(self, *, timeout_seconds: float, token_info: dict | None = None) -> Any:
        try:
            import httplib2
            from google.oauth2.credentials import Credentials
            from google_auth_httplib2 import AuthorizedHttp
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise MailProviderPreconditionError(
                "Gmail provider requires google-api-python-client and google-auth-httplib2"
            ) from exc
        info = _read_token_info(self.token_path)[0] if token_info is None else token_info
        creds = Credentials.from_authorized_user_info(info, self.scopes)
        if not creds.valid:
            raise MailProviderPreconditionError('Gmail authorization expired; refresh locally before approving')
        # Never open a browser, refresh/persist credentials or fetch discovery in
        # the approval path. The installed SDK supplies its static discovery doc.
        ready_token = Credentials(token=creds.token, scopes=self.scopes)
        http = AuthorizedHttp(ready_token, http=httplib2.Http(timeout=timeout_seconds), max_refresh_attempts=0)
        return build('gmail', 'v1', http=http, cache_discovery=False, static_discovery=True)


def _read_token_info(path):
    try:
        with Path(path).open('rb') as source:
            raw = source.read(65_537)
        if len(raw) > 65_536:
            raise ValueError('token too large')
        info = json.loads(raw)
        if not isinstance(info, dict):
            raise ValueError('token must be an object')
        return info, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, UnicodeError) as exc:
        raise MailProviderPreconditionError('Gmail authorization is unavailable or invalid; authorize locally') from exc


def _account_id(value):
    if (not isinstance(value, str) or len(value) > 254 or value.count('@') != 1
            or not all(value.split('@')) or any(c.isspace() or c in '<>' for c in value)):
        raise MailProviderPreconditionError('Gmail did not provide a valid account identity')
    return value.casefold()


def _profile_account(service, *, timeout_seconds):
    if timeout_seconds <= 0:
        raise MailProviderPreconditionError('approval timed out before submission')
    request = service.users().getProfile(userId='me')
    transport = getattr(request, 'http', None)
    transport = getattr(transport, 'http', transport)
    if transport is not None and hasattr(transport, 'timeout'):
        transport.timeout = timeout_seconds
    profile = request.execute()  # Read only, no retries or token refresh.
    return _account_id(profile.get('emailAddress') if isinstance(profile, dict) else None)


def _validate_action(approval: dict[str, Any]) -> None:
    if approval.get("action_type", "send_email") != "send_email":
        raise MailProviderPreconditionError("mail provider does not support this action type")


def build_rfc822_message(payload: dict[str, Any], *, request_id: str | None = None) -> EmailMessage:
    if not isinstance(payload, dict):
        raise MailProviderPreconditionError("email payload must be an object")
    recipients = payload.get("to")
    if not isinstance(recipients, list) or not recipients:
        raise MailProviderPreconditionError("email recipient must be a nonempty list")
    for recipient in recipients:
        if not isinstance(recipient, str) or not recipient.strip() or "\r" in recipient or "\n" in recipient:
            raise MailProviderPreconditionError("email recipient is invalid")
        address = parseaddr(recipient)[1]
        if address.count("@") != 1 or any(char.isspace() for char in address) or not all(address.split("@")):
            raise MailProviderPreconditionError("email recipient is invalid")
    subject = payload.get("subject")
    if not isinstance(subject, str) or not subject.strip() or "\r" in subject or "\n" in subject:
        raise MailProviderPreconditionError("email subject is required and must not contain newlines")
    body = payload.get("body")
    if not isinstance(body, str) or not body.strip():
        raise MailProviderPreconditionError("email body is required")

    message = EmailMessage()
    message["To"] = ", ".join(item.strip() for item in recipients)
    message["Subject"] = subject.strip()
    if request_id:
        # A lookup marker for reconciliation, NOT Gmail's idempotency guarantee.
        marker = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
        message["Message-ID"] = f"<approval-{marker}@email-rag.local>"
    message.set_content(body)
    return message


def build_gmail_raw_message(payload: dict[str, Any], *, request_id: str | None = None) -> str:
    message = build_rfc822_message(payload, request_id=request_id)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")


def create_mail_provider_from_settings() -> SimulatedMailProvider | GmailDraftProvider:
    provider = cfg.MAIL_PROVIDER.lower()
    if provider == "simulated":
        return SimulatedMailProvider()
    if provider == "gmail":
        return GmailDraftProvider()
    raise MailProviderError(f"unknown MAIL_PROVIDER: {cfg.MAIL_PROVIDER!r}")


def authorize_gmail_drafts_locally() -> None:
    """Explicit setup command only; never called by approval or tool execution."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = Path(cfg.GMAIL_TOKEN_PATH)
    credentials_path = Path(cfg.GMAIL_CREDENTIALS_PATH)
    scopes = list(cfg.GMAIL_SCOPES)
    creds = Credentials.from_authorized_user_file(str(token_path), scopes) if token_path.is_file() else None
    if creds and creds.valid:
        pass
    elif creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not credentials_path.is_file():
            raise MailProviderPreconditionError('Gmail OAuth client configuration is missing')
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
        creds = flow.run_local_server(port=0)
    info = json.loads(creds.to_json())
    provider = GmailDraftProvider()
    service = provider._build_service(timeout_seconds=provider.timeout_seconds, token_info=info)
    try:
        account = _profile_account(service, timeout_seconds=provider.timeout_seconds)
        if provider.user_id != 'me' and _account_id(provider.user_id) != account:
            raise MailProviderPreconditionError('Configured Gmail user differs from the authorized account')
    finally:
        close = getattr(service, 'close', None)
        if callable(close):
            close()
    # A successful explicit authorization issues a new grant version even when
    # the access token was still valid. Old pending approvals must be recreated.
    info['_email_agent_binding'] = {'version': 1, 'account_id': account, 'authorization_id': uuid.uuid4().hex}
    token_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=token_path.parent,
                                         prefix='.gmail-token-', delete=False) as target:
            temporary = Path(target.name)
            target.write(json.dumps(info, ensure_ascii=False, sort_keys=True))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, token_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Explicit local Gmail draft OAuth setup')
    parser.add_argument('command', choices=['authorize'])
    parser.parse_args()
    authorize_gmail_drafts_locally()
    print('Gmail draft authorization is ready. No draft was created or sent.')
