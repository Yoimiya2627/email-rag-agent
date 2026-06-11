"""Gmail read-only ingestion helpers.

This module is intentionally separate from ``mail_providers.py``.  The draft
provider runs only after human approval and uses compose scope; this reader only
needs Gmail read-only scope and converts Gmail messages into the project's
existing ``Email`` schema for indexing/evaluation.
"""

from __future__ import annotations

import base64
import html
import re
from datetime import datetime, timezone
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import config.settings as cfg
from agents.mail_providers import MailProviderError
from models.schemas import Email


def _decode_body(data: str | None, charset: str | None = None) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    payload = base64.urlsafe_b64decode(padded.encode("utf-8"))
    candidates = [charset, "utf-8", "gb18030", "gbk", "big5", "latin-1"]
    tried: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        encoding = candidate.strip().lower()
        if not encoding or encoding in tried:
            continue
        tried.add(encoding)
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _headers_map(message: dict[str, Any]) -> dict[str, str]:
    headers = (message.get("payload") or {}).get("headers") or []
    return {str(item.get("name", "")).lower(): str(item.get("value", "")) for item in headers}


def _part_charset(part: dict[str, Any]) -> str | None:
    headers = {
        str(item.get("name", "")).lower(): str(item.get("value", ""))
        for item in part.get("headers") or []
    }
    content_type = headers.get("content-type", "")
    if not content_type:
        return None
    message = Message()
    message["content-type"] = content_type
    return message.get_content_charset()


def _first_email(value: str) -> str:
    parsed = getaddresses([value])
    return parsed[0][1] if parsed and parsed[0][1] else value.strip()


def _email_list(value: str) -> list[str]:
    return [email for _, email in getaddresses([value]) if email]


def _date_to_iso(value: str, internal_date_ms: int) -> str:
    if value:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            pass
    return datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc).isoformat()


def _walk_payload(part: dict[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    mime_type = str(part.get("mimeType") or "")
    body = part.get("body") or {}
    decoded = _decode_body(body.get("data"), charset=_part_charset(part))
    if decoded:
        found.append((mime_type, decoded))
    for child in part.get("parts") or []:
        found.extend(_walk_payload(child))
    return found


def _body_from_payload(payload: dict[str, Any]) -> str:
    bodies = _walk_payload(payload)
    for mime_type, text in bodies:
        if mime_type == "text/plain" and text.strip():
            return text.strip()
    for mime_type, text in bodies:
        if mime_type == "text/html" and text.strip():
            return _strip_html(text)
    return ""


def message_internal_date_ms(message: dict[str, Any]) -> int:
    try:
        return int(message.get("internalDate") or 0)
    except (TypeError, ValueError):
        return 0


def gmail_message_to_email(message: dict[str, Any]) -> Email:
    message_id = str(message.get("id") or "").strip()
    if not message_id:
        raise MailProviderError("Gmail message id is required")
    payload = message.get("payload") or {}
    headers = _headers_map(message)
    internal_date_ms = message_internal_date_ms(message)

    subject = headers.get("subject", "(no subject)").strip() or "(no subject)"
    sender = _first_email(headers.get("from", ""))
    recipients = _email_list(headers.get("to", ""))
    date = _date_to_iso(headers.get("date", ""), internal_date_ms)
    body = _body_from_payload(payload)

    return Email(
        id=f"gmail_{message_id}",
        subject=subject,
        sender=sender,
        recipients=recipients,
        date=date,
        body=body,
        labels=list(message.get("labelIds") or []),
        thread_id=str(message.get("threadId") or ""),
    )


class GmailReadOnlyProvider:
    """Read Gmail messages and map them into the local Email schema."""

    provider = "gmail_readonly"

    def __init__(
        self,
        service: Any | None = None,
        user_id: str | None = None,
        credentials_path: str | Path | None = None,
        token_path: str | Path | None = None,
        scopes: list[str] | None = None,
        session: Any | None = None,
        request_timeout: int = 60,
    ):
        self._service = service
        self._session = session
        self.user_id = user_id or cfg.GMAIL_USER_ID
        self.credentials_path = Path(credentials_path or cfg.GMAIL_CREDENTIALS_PATH)
        self.token_path = Path(token_path or cfg.GMAIL_READONLY_TOKEN_PATH)
        self.scopes = scopes or list(cfg.GMAIL_READONLY_SCOPES)
        self.request_timeout = request_timeout

    def list_message_ids(
        self,
        query: str = "",
        max_results: int = 100,
        include_spam_trash: bool = False,
    ) -> list[str]:
        if not self._service:
            return self._list_message_ids_via_session(
                query=query,
                max_results=max_results,
                include_spam_trash=include_spam_trash,
            )

        service = self._service
        message_ids: list[str] = []
        page_token = None

        while True:
            kwargs = {
                "userId": self.user_id,
                "q": query,
                "maxResults": max_results,
                "includeSpamTrash": include_spam_trash,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = service.users().messages().list(**kwargs).execute()
            message_ids.extend(item["id"] for item in response.get("messages") or [])
            page_token = response.get("nextPageToken")
            if not page_token or len(message_ids) >= max_results:
                return message_ids[:max_results]

    def get_message(self, message_id: str) -> dict[str, Any]:
        if not self._service:
            return self._get_message_via_session(message_id)

        service = self._service
        return (
            service.users()
            .messages()
            .get(userId=self.user_id, id=message_id, format="full")
            .execute()
        )

    def get_email(self, message_id: str) -> Email:
        return gmail_message_to_email(self.get_message(message_id))

    def _gmail_user_url(self, suffix: str) -> str:
        user = quote(self.user_id, safe="")
        return f"https://gmail.googleapis.com/gmail/v1/users/{user}/{suffix}"

    def _list_message_ids_via_session(
        self,
        query: str,
        max_results: int,
        include_spam_trash: bool,
    ) -> list[str]:
        session = self._session or self._build_authorized_session()
        message_ids: list[str] = []
        page_token = None
        url = self._gmail_user_url("messages")

        while True:
            params: dict[str, Any] = {
                "q": query,
                "maxResults": max_results,
                "includeSpamTrash": include_spam_trash,
            }
            if page_token:
                params["pageToken"] = page_token
            response = session.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            payload = response.json()
            message_ids.extend(item["id"] for item in payload.get("messages") or [])
            page_token = payload.get("nextPageToken")
            if not page_token or len(message_ids) >= max_results:
                return message_ids[:max_results]

    def _get_message_via_session(self, message_id: str) -> dict[str, Any]:
        session = self._session or self._build_authorized_session()
        safe_message_id = quote(message_id, safe="")
        response = session.get(
            self._gmail_user_url(f"messages/{safe_message_id}"),
            params={"format": "full"},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _build_authorized_session(self) -> Any:
        try:
            from google.auth.transport.requests import Request
            from google.auth.transport.requests import AuthorizedSession
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise MailProviderError(
                "Gmail read-only sync requires google-auth and google-auth-oauthlib"
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
        self._session = AuthorizedSession(creds)
        return self._session
