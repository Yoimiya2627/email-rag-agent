"""Gmail read-only ingestion helpers.

This module is intentionally separate from ``mail_providers.py``.  The draft
provider runs only after human approval and uses compose scope; this reader only
needs Gmail read-only scope and converts Gmail messages into the project's
existing ``Email`` schema for indexing/evaluation.
"""

from __future__ import annotations

import base64
import binascii
import time
import hashlib
import json
import re
from contextvars import ContextVar
from core.mail_quality import MimeBudget, MimeBudgetExceeded, text_quality
from agents.runtime import remaining_timeout, RunCancelled, RunDeadlineExceeded
from email.message import Message
from email.header import decode_header, make_header
from email.errors import HeaderParseError
from datetime import datetime, timezone
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import config.settings as cfg
from agents.mail_providers import MailProviderError
from models.schemas import Email
from core.cleaner import html_to_text, html_to_structured_text, HTML_INPUT_LIMIT


class MailContentError(MailProviderError):
    """A single message cannot be normalized; safe to quarantine and continue."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class GmailPageTokenError(MailProviderError):
    """A continuation request was rejected; callers may restart it once."""


def parser_version():
    from core import mail_quality
    digest = hashlib.sha256(Path(__file__).read_bytes() + Path(mail_quality.__file__).read_bytes()).hexdigest()
    return "gmail-mime-v2:" + digest


def is_missing_message_error(exc: Exception) -> bool:
    """Only use for messages.get: 404 is not an authentication failure."""
    return getattr(getattr(exc, "resp", None), "status", None) == 404


_MIME_BUDGET = ContextVar("gmail_mime_budget", default=None)


def _decode_body(data: str | None, charset: str = "utf-8") -> str:
    if data is not None and not isinstance(data, str):
        raise MailContentError("invalid_body_encoding")
    if not data:
        return ""
    budget = _MIME_BUDGET.get()
    if budget:
        budget.before_decode(data)
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        if budget:
            budget.decoded(len(raw))
        return raw.decode(charset)
    except (ValueError, UnicodeError, LookupError, binascii.Error) as exc:
        raise MailContentError("invalid_body_encoding") from exc


def _strip_html(text: str) -> str:
    return html_to_text(text)


def _headers_map(message: dict[str, Any]) -> dict[str, str]:
    headers = (message.get("payload") or {}).get("headers") or []
    return {str(item.get("name", "")).lower(): str(item.get("value", "")) for item in headers}


def _decode_subject(value: str) -> str:
    """Decode RFC 2047 words; preserve an undecodable header as evidence."""
    try:
        parts = decode_header(value)
        if all(isinstance(text, str) and charset is None for text, charset in parts):
            return value
        return str(make_header(parts)) or value
    except (ValueError, UnicodeError, LookupError, HeaderParseError):
        return value


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


def _part_headers(part: dict[str, Any]) -> Message:
    headers = Message()
    for item in part.get("headers") or []:
        headers[str(item.get("name", ""))] = str(item.get("value", ""))
    return headers


def _attachment_inventory(payload):
    inventory, stack = [], [(payload, "0")]
    while stack:
        part, part_id = stack.pop()
        headers = _part_headers(part)
        filename = str(part.get("filename") or headers.get_filename() or "")
        mime = str(part.get("mimeType") or headers.get_content_type())
        disposition = headers.get_content_disposition()
        if filename or disposition == "attachment" or mime == "message/rfc822" or (not mime.startswith("multipart/") and mime not in {"text/plain", "text/html"}):
            inventory.append({"part_id": part_id, "filename": _decode_subject(filename)[:512],
                "mime_type": mime[:128], "size": int((part.get("body") or {}).get("size") or 0),
                "inline": disposition == "inline", "status": "not_read",
                "reason": "attachment_content_not_ingested"})
        stack.extend((child, f"{part_id}.{i}") for i, child in reversed(list(enumerate(part.get("parts") or []))))
    return inventory


def _walk_payload(part: dict[str, Any],
                  attachment_loader: Callable[[str], str] | None = None) -> list[tuple[str, str]]:
    """Choose alternatives within their MIME scope; mixed bodies keep order."""
    headers = _part_headers(part)
    if (part.get("filename") or headers.get_filename()
            or headers.get_content_disposition() == "attachment"):
        return []
    mime_type = str(part.get("mimeType") or headers.get_content_type()).lower()
    children = part.get("parts") or []
    if mime_type == "multipart/related":
        # Related siblings are resources, even if a resource happens to be text.
        start = headers.get_param("start")
        if start:
            children = [child for child in children
                        if _part_headers(child).get("Content-ID") == start]
            if not children:
                raise MailContentError("missing_related_root")
        return _walk_payload(children[0], attachment_loader) if children else []
    if mime_type == "multipart/alternative":
        # Prefer plain text but do not flatten alternatives into mixed siblings.
        candidates = sorted(children, key=lambda child: str(child.get("mimeType", "")).lower() != "text/plain")
        fallback = []
        for child in candidates:
            found = _walk_payload(child, attachment_loader)
            if any(kind == "text/plain" and text.strip() for kind, text in found):
                # Plain alternatives often omit the HTML quote/pricing table.
                # Keep the chosen plain prose and supplement only HTML tables;
                # never replace plain facts or duplicate all HTML prose.
                extras = []
                for alternative in candidates:
                    if alternative is child:
                        continue
                    for kind, text in _walk_payload(alternative, attachment_loader):
                        if kind in {"text/html", "text/html-tables"}:
                            extras.append(("text/html-tables", text))
                return found + extras
            if found and not fallback:
                fallback = found
        return fallback
    if mime_type.startswith("multipart/"):
        return [body for child in children for body in _walk_payload(child, attachment_loader)]
    # message/rfc822 is a forwarded message, not the parent email's body.
    if mime_type not in {"text/plain", "text/html"}:
        return []
    body = part.get("body") or {}
    data = body.get("data")
    budget = _MIME_BUDGET.get()
    if budget:
        budget.before_decode(data or "", int(body.get("size") or 0))
    if not data and not body.get("attachmentId") and int(body.get("size") or 0) > 0:
        raise MailContentError("missing_body_data")
    if not data and body.get("attachmentId"):
        if attachment_loader is None:
            raise MailContentError("missing_body_data")
        data = attachment_loader(str(body["attachmentId"]))
        if not data:
            raise MailContentError("missing_body_data")
    decoded = _decode_body(data, headers.get_content_charset() or "utf-8")
    return [(mime_type, decoded)] if decoded.strip() else []


def _body_from_payload(payload: dict[str, Any],
                       attachment_loader: Callable[[str], str] | None = None) -> str:
    return _body_and_rows(payload, attachment_loader)[0]


def _body_and_rows(payload: dict[str, Any],
                   attachment_loader: Callable[[str], str] | None = None) -> tuple[str, list[dict]]:
    parts, rows = [], []
    length = 0
    budget = MimeBudget()
    budget.inspect(payload)
    token = _MIME_BUDGET.set(budget)
    try:
        selected_parts = _walk_payload(payload, attachment_loader)
    finally:
        _MIME_BUDGET.reset(token)
    for number, (mime_type, source) in enumerate(selected_parts, 1):
        budget.check()
        if mime_type == "text/plain":
            text, table_rows = source.strip(), []
        else:
            text, table_rows = html_to_structured_text(source, table_prefix=f"p{number}t")
            if mime_type == "text/html-tables":
                incomplete = len(source) > HTML_INPUT_LIMIT
                if not table_rows and not incomplete:
                    continue
                # Mark the alternative explicitly: HTML and plain may disagree.
                selected = "[Tables from HTML alternative; plain-text version retained separately]"
                selected_rows = []
                for row in table_rows:
                    line = text[row["start"]:row["end"]]
                    start = len(selected) + 2
                    selected += "\n\n" + line
                    selected_rows.append({**row, "start": start, "end": len(selected),
                                          "origin": "html_alternative"})
                if incomplete:
                    selected += "\n\n[HTML extraction incomplete: input_limit]"
                text, table_rows = selected, selected_rows
        if not text:
            continue
        offset = length + (2 if parts else 0)
        rows.extend({**row, "start": row["start"] + offset, "end": row["end"] + offset}
                    for row in table_rows)
        parts.append(text)
        length = offset + len(text)
        if length > int(getattr(cfg, "MIME_OUTPUT_CHAR_LIMIT", 8000000)):
            raise MimeBudgetExceeded("mime_output_limit")
    return "\n\n".join(parts), rows


def message_internal_date_ms(message: dict[str, Any]) -> int:
    try:
        return int(message.get("internalDate") or 0)
    except (TypeError, ValueError):
        return 0


def _gmail_message_to_email(message: dict[str, Any],
                            attachment_loader: Callable[[str], str] | None = None) -> Email:
    message_id = str(message.get("id") or "").strip()
    if not message_id:
        raise MailContentError("missing_message_id")
    payload = message.get("payload") or {}
    MimeBudget().inspect(payload)  # Validate headers/depth before header decoding.
    headers = _headers_map(message)
    internal_date_ms = message_internal_date_ms(message)

    subject = _decode_subject(headers.get("subject", "(no subject)").strip()) or "(no subject)"
    sender = _first_email(headers.get("from", ""))
    recipients = _email_list(headers.get("to", ""))
    date = _date_to_iso(headers.get("date", ""), internal_date_ms)
    body, table_rows = _body_and_rows(payload, attachment_loader)

    return Email(
        id=f"gmail_{message_id}",
        subject=subject,
        sender=sender,
        recipients=recipients,
        date=date,
        body=body,
        body_format="plain",
        table_rows=table_rows,
        labels=list(message.get("labelIds") or []),
        thread_id=str(message.get("threadId") or ""),
        sender_name=_decode_subject(getaddresses([headers.get("from", "")])[0][0]) if getaddresses([headers.get("from", "")]) else "",
        cc=_email_list(headers.get("cc", "")),
        message_id=headers.get("message-id", ""), in_reply_to=headers.get("in-reply-to", ""),
        references=re.findall(r"<[^<>]+>", headers.get("references", "")),
        attachments=_attachment_inventory(payload),
        decode_quality={"subject": text_quality(subject), "body": text_quality(body)},
        source={"provider": "gmail", "format": "gmail-full-v1", "message_id": message_id,
                "internal_date_ms": internal_date_ms, "parser_version": parser_version()},
    )


def gmail_message_to_email(message: dict[str, Any],
                           attachment_loader: Callable[[str], str] | None = None) -> Email:
    try:
        return _gmail_message_to_email(message, attachment_loader)
    except MimeBudgetExceeded as exc:
        raise MailContentError(str(exc)) from None
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        # Do not copy original message contents into logs or sync failure state.
        raise MailContentError("invalid_message_structure") from exc


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
        max_attempts: int = 3,
    ):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self._service = service
        self.user_id = user_id or cfg.GMAIL_USER_ID
        self.credentials_path = Path(credentials_path or cfg.GMAIL_CREDENTIALS_PATH)
        self.token_path = Path(token_path or cfg.GMAIL_READONLY_TOKEN_PATH)
        self.scopes = scopes or list(cfg.GMAIL_READONLY_SCOPES)
        self.max_attempts = max_attempts

    def _get_service(self) -> Any:
        # This reader is serial. Parallel workers must each own their client.
        if self._service is None:
            self._service = self._build_service()
        return self._service

    def _execute_read(self, make_request: Callable[[], Any]) -> dict[str, Any]:
        """Bounded retries for read requests; permission failures remain fatal."""
        deadline = time.monotonic() + float(getattr(cfg, "GMAIL_READ_TIMEOUT", 30))
        mime_budget = _MIME_BUDGET.get()
        if mime_budget is not None:
            deadline = min(deadline, mime_budget.started + float(getattr(cfg, "MIME_PARSE_SECONDS", 15)))
        for attempt in range(self.max_attempts):
            left = min(deadline - time.monotonic(), remaining_timeout(60))
            if left <= 0:
                raise RunDeadlineExceeded("Gmail read budget exhausted")
            try:
                request = make_request()
                transport = getattr(request, "http", None)
                if transport is not None and hasattr(transport, "timeout"):
                    transport.timeout = left
                result = request.execute()
                remaining_timeout(60)
                if time.monotonic() > deadline:
                    raise RunDeadlineExceeded("Gmail read budget exhausted")
                return result
            except (RunCancelled, RunDeadlineExceeded):
                raise
            except Exception as exc:
                status = getattr(getattr(exc, "resp", None), "status", None)
                retryable = status in {429, 500, 502, 503, 504} or isinstance(exc, (TimeoutError, ConnectionError))
                if not retryable or attempt + 1 == self.max_attempts:
                    raise
                retry_after = 0.0
                response = getattr(exc, "resp", None)
                if hasattr(response, "get"):
                    try:
                        retry_after = max(0., float(response.get("retry-after", 0)))
                    except (TypeError, ValueError):
                        pass
                delay = max(0.5 * 2 ** attempt, min(retry_after, 30.))
                left = min(deadline - time.monotonic(), remaining_timeout(60))
                if delay >= left:
                    raise RunDeadlineExceeded("Gmail retry exceeds read budget") from None
                from agents.runtime import current_run
                run = current_run()
                if run is not None and run.cancel_event is not None:
                    if run.cancel_event.wait(delay):
                        raise RunCancelled("Gmail retry cancelled")
                else:
                    time.sleep(delay)
                remaining_timeout(60)
        raise AssertionError("unreachable")

    def list_message_page(self, *, query: str = "", page_token: str | None = None,
                          page_size: int = 100, include_spam_trash: bool = False) -> dict:
        """Read one bounded page so sync can persist the continuation cursor."""
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 500:
            raise ValueError("page_size must be between 1 and 500")
        if page_token is not None and (not isinstance(page_token, str) or not page_token):
            raise ValueError("page_token must be a nonempty string or None")
        service = self._get_service()
        options = {"userId": self.user_id, "q": query, "maxResults": page_size,
                   "includeSpamTrash": include_spam_trash}
        if page_token is not None:
            options["pageToken"] = page_token
        try:
            response = self._execute_read(lambda: service.users().messages().list(**options))
        except Exception as exc:
            # A fresh request with the same query will distinguish a stale
            # cursor from another bad argument. Never retry 401/403 this way.
            if page_token is not None and getattr(getattr(exc, "resp", None), "status", None) == 400:
                raise GmailPageTokenError("Gmail continuation cursor was rejected") from None
            raise
        if not isinstance(response, dict):
            raise MailProviderError("Invalid Gmail message-list response")
        messages = response.get("messages") or []
        if not isinstance(messages, list) or len(messages) > page_size:
            raise MailProviderError("Invalid Gmail message-list page")
        identifiers = []
        for item in messages:
            identifier = item.get("id") if isinstance(item, dict) else None
            if not isinstance(identifier, str) or not identifier.strip():
                raise MailProviderError("Invalid Gmail message-list entry")
            identifiers.append(identifier)
        next_token = response.get("nextPageToken") or None
        if next_token is not None and (not isinstance(next_token, str) or next_token == page_token):
            raise MailProviderError("Gmail pagination did not advance")
        return {"message_ids": identifiers, "next_page_token": next_token}

    def list_message_ids(
        self,
        query: str = "",
        max_results: int = 100,
        include_spam_trash: bool = False,
        page_size: int = 500,
    ) -> list[str]:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 0:
            raise ValueError("max_results must be a nonnegative integer total limit")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 500:
            raise ValueError("page_size must be between 1 and 500")
        if max_results == 0:
            return []
        service = self._get_service()
        message_ids: list[str] = []
        seen_ids: set[str] = set()
        seen_tokens: set[str] = set()
        page_token = None

        while True:
            kwargs = {
                "userId": self.user_id,
                "q": query,
                "maxResults": min(page_size, max_results - len(message_ids)),
                "includeSpamTrash": include_spam_trash,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            response = self._execute_read(lambda: service.users().messages().list(**kwargs))
            for item in response.get("messages") or []:
                identifier = item.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise MailProviderError("Invalid Gmail message-list entry")
                if identifier not in seen_ids:
                    seen_ids.add(identifier)
                    message_ids.append(identifier)
            page_token = response.get("nextPageToken")
            if not page_token or len(message_ids) >= max_results:
                return message_ids[:max_results]
            if not isinstance(page_token, str) or page_token in seen_tokens:
                raise MailProviderError("Gmail pagination did not advance")
            seen_tokens.add(page_token)

    def get_message(self, message_id: str) -> dict[str, Any]:
        service = self._get_service()
        return self._execute_read(lambda: service.users().messages().get(
            userId=self.user_id, id=message_id, format="full"))

    def describe_account(self):
        service = self._get_service()
        profile = self._execute_read(lambda: service.users().getProfile(userId=self.user_id))
        account = profile.get("emailAddress")
        if not isinstance(account, str) or "@" not in account:
            raise MailProviderError("Gmail profile did not identify the account")
        self.account_id = account.casefold()
        labels = self._execute_read(lambda: service.users().labels().list(userId=self.user_id))
        self.label_map = {row["id"]: row["name"] for row in labels.get("labels", [])
                          if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("name"), str)}
        return {"account_id": self.account_id, "provider": "gmail",
                "authorization": getattr(self, "authorization", {"scope_status": "injected_transport_unverified"})}

    def get_message_metadata(self, message_id):
        service = self._get_service()
        return self._execute_read(lambda: service.users().messages().get(
            userId=self.user_id, id=message_id, format="metadata", metadataHeaders=[]))

    def get_email(self, message_id: str) -> Email:
        return self.message_to_email(self.get_message(message_id))

    def message_to_email(self, message: dict[str, Any]) -> Email:
        """Normalize a fetched message, lazily retrieving separated body data."""
        def load_body(attachment_id: str) -> str:
            service = self._get_service()
            response = self._execute_read(lambda: service.users().messages().attachments().get(
                userId=self.user_id, messageId=str(message.get("id") or ""), id=attachment_id,
            ))
            return response.get("data") or ""

        return gmail_message_to_email(message, attachment_loader=load_body)

    def capture_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Capture full Gmail JSON and selected external body data for replay.

        This is not an RFC822 export and does not download generic attachments.
        A content error is recorded without destroying the original payload.
        Network/auth failures propagate so the sync can checkpoint and pause.
        """
        envelope = {"format": "gmail-full-v1", "message": message,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "account_id": getattr(self, "account_id", None), "parser_version": parser_version(),
                    "body_data": {}, "body_errors": {}}

        def load_body(attachment_id: str) -> str:
            if attachment_id not in envelope["body_data"]:
                service = self._get_service()
                response = self._execute_read(lambda: service.users().messages().attachments().get(
                    userId=self.user_id, messageId=str(message.get("id") or ""), id=attachment_id))
                envelope["body_data"][attachment_id] = response.get("data") or ""
            return envelope["body_data"][attachment_id]

        try:
            gmail_message_to_email(message, attachment_loader=load_body)
        except MailContentError as exc:
            envelope["body_errors"]["content"] = exc.code
        envelope["raw_sha256"] = hashlib.sha256(json.dumps({"message": message, "body_data": envelope["body_data"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return envelope

    def email_from_capture(self, envelope: dict[str, Any]) -> Email:
        """Normalize an archived capture without accessing Gmail."""
        if envelope.get("format") != "gmail-full-v1":
            raise MailContentError("invalid_capture_format")
        def load_body(attachment_id: str) -> str:
            data = envelope.get("body_data") or {}
            if attachment_id not in data:
                raise MailContentError("missing_captured_body_data")
            return data[attachment_id]
        email = gmail_message_to_email(envelope.get("message"), attachment_loader=load_body)
        raw_hash = hashlib.sha256(json.dumps({"message": envelope.get("message"), "body_data": envelope.get("body_data") or {}}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if envelope.get("raw_sha256") and raw_hash != envelope["raw_sha256"]:
            raise MailContentError("capture_integrity_failed")
        email.source.update({"account_id": envelope.get("account_id"), "captured_at": envelope.get("captured_at"),
                             "raw_sha256": raw_hash, "original_format": "gmail-full-json-selected-body-parts"})
        email.label_names = [self.label_map.get(label, label) for label in email.labels] if getattr(self, "label_map", None) else []
        return email

    def _build_service(self) -> Any:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise MailProviderError(
                "Gmail read-only sync requires google-api-python-client and google-auth-oauthlib"
            ) from exc

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path))
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
        granted = getattr(creds, "granted_scopes", None)
        stored = getattr(creds, "scopes", None)
        effective = set(granted if granted is not None else (stored or []))
        if not set(self.scopes).issubset(effective):
            raise MailProviderError("Gmail token does not record all requested read scopes; authorize this account again")
        self.authorization = {"scope_status": "oauth_grant" if granted is not None else "token_declared",
                              "scopes": sorted(effective), "requested_scopes": sorted(self.scopes)}
        return build("gmail", "v1", credentials=creds)
