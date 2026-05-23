"""Mail-provider execution for approved high-risk email actions.

The agent still cannot send email directly.  Providers only run after a human
approves a pending action through the approval API.  The default provider is a
simulation; Gmail creates a real draft and deliberately does not send it.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import config.settings as cfg


class MailProviderError(RuntimeError):
    """Raised when an approved mail action cannot be executed."""


class SimulatedMailProvider:
    """Local provider used by default for tests and offline demos."""

    provider = "simulated"

    def execute_approval(self, approval: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": "simulated_send",
            "provider": self.provider,
            "sent": True,
            "approval_id": approval.get("approval_id", ""),
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
    ):
        self._service = service
        self.user_id = user_id or cfg.GMAIL_USER_ID
        self.credentials_path = Path(credentials_path or cfg.GMAIL_CREDENTIALS_PATH)
        self.token_path = Path(token_path or cfg.GMAIL_TOKEN_PATH)
        self.scopes = scopes or list(cfg.GMAIL_SCOPES)

    def execute_approval(self, approval: dict[str, Any]) -> dict[str, Any]:
        payload = dict(approval.get("payload") or {})
        raw = build_gmail_raw_message(payload)
        service = self._service or self._build_service()
        draft = (
            service.users()
            .drafts()
            .create(userId=self.user_id, body={"message": {"raw": raw}})
            .execute()
        )
        message = draft.get("message") or {}
        return {
            "mode": "gmail_draft",
            "provider": self.provider,
            "draft_id": draft.get("id", ""),
            "message_id": message.get("id", ""),
            "sent": False,
            "approval_id": approval.get("approval_id", ""),
        }

    def _build_service(self) -> Any:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise MailProviderError(
                "Gmail provider requires google-api-python-client and google-auth-oauthlib"
            ) from exc

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), self.scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise MailProviderError(
                        f"Gmail credentials file not found: {self.credentials_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path),
                    self.scopes,
                )
                creds = flow.run_local_server(port=0)
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")
        return build("gmail", "v1", credentials=creds)


def build_rfc822_message(payload: dict[str, Any]) -> EmailMessage:
    recipients = [str(item).strip() for item in payload.get("to", []) if str(item).strip()]
    if not recipients:
        raise MailProviderError("email recipient is required")
    subject = str(payload.get("subject") or "").strip()
    if not subject:
        raise MailProviderError("email subject is required")
    body = str(payload.get("body") or "")
    if not body.strip():
        raise MailProviderError("email body is required")

    message = EmailMessage()
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)
    return message


def build_gmail_raw_message(payload: dict[str, Any]) -> str:
    message = build_rfc822_message(payload)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")


def create_mail_provider_from_settings() -> SimulatedMailProvider | GmailDraftProvider:
    provider = cfg.MAIL_PROVIDER.lower()
    if provider == "simulated":
        return SimulatedMailProvider()
    if provider == "gmail":
        return GmailDraftProvider()
    raise MailProviderError(f"unknown MAIL_PROVIDER: {cfg.MAIL_PROVIDER!r}")
