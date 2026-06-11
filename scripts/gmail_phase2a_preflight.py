"""Local readiness checks for the Phase 2A real Gmail data workflow.

This script intentionally does not contact Gmail. It verifies whether the
configured local OAuth artifacts and real-data outputs are present before a
developer claims the project has been run against real mailbox data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg

READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
BROAD_GMAIL_SCOPES = {
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.insert",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://mail.google.com/",
}


@dataclass(frozen=True)
class GmailPhase2AConfig:
    credentials_path: Path
    readonly_token_path: Path
    readonly_scopes: list[str]
    sync_output_path: Path
    sync_state_path: Path
    gold_template_path: Path


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.path:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class ReadinessReport:
    credentials_path: str
    readonly_token_path: str
    readonly_scopes: list[str]
    sync_output_path: str
    sync_state_path: str
    gold_template_path: str
    ready_for_interactive_sync: bool
    ready_for_noninteractive_sync: bool
    ready_for_real_eval: bool
    blockers: list[ReadinessIssue]
    warnings: list[ReadinessIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "credentials_path": self.credentials_path,
            "readonly_token_path": self.readonly_token_path,
            "readonly_scopes": self.readonly_scopes,
            "sync_output_path": self.sync_output_path,
            "sync_state_path": self.sync_state_path,
            "gold_template_path": self.gold_template_path,
            "ready_for_interactive_sync": self.ready_for_interactive_sync,
            "ready_for_noninteractive_sync": self.ready_for_noninteractive_sync,
            "ready_for_real_eval": self.ready_for_real_eval,
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
        }


def _path_issue(code: str, message: str, path: Path) -> ReadinessIssue:
    return ReadinessIssue(code=code, message=message, path=str(path))


def _has_broad_gmail_scope(scopes: Iterable[str]) -> bool:
    normalized = {scope.strip() for scope in scopes if scope.strip()}
    return bool(normalized & BROAD_GMAIL_SCOPES)


def _has_labeled_gold_case(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        ground_truth = str(item.get("ground_truth") or "").strip()
        gold_chunk_ids = [value for value in item.get("gold_chunk_ids") or [] if str(value)]
        if question and ground_truth and gold_chunk_ids:
            return True
    return False


def check_phase2a_readiness(config: GmailPhase2AConfig) -> ReadinessReport:
    blockers: list[ReadinessIssue] = []
    warnings: list[ReadinessIssue] = []

    credentials_exists = config.credentials_path.exists()
    token_exists = config.readonly_token_path.exists()
    corpus_exists = config.sync_output_path.exists()
    state_exists = config.sync_state_path.exists()
    gold_exists = config.gold_template_path.exists()
    scope_set = {scope.strip() for scope in config.readonly_scopes if scope.strip()}
    readonly_scope_ok = READONLY_SCOPE in scope_set

    if not credentials_exists:
        blockers.append(
            _path_issue(
                "gmail_credentials_missing",
                "Gmail OAuth client file is missing.",
                config.credentials_path,
            )
        )
    if not token_exists:
        blockers.append(
            _path_issue(
                "gmail_readonly_token_missing",
                "Gmail read-only OAuth token is missing. Run sync once with OAuth consent.",
                config.readonly_token_path,
            )
        )
    if not readonly_scope_ok:
        blockers.append(
            ReadinessIssue(
                code="gmail_readonly_scope_missing",
                message=f"Gmail read-only scope is required: {READONLY_SCOPE}",
            )
        )
    if _has_broad_gmail_scope(scope_set):
        warnings.append(
            ReadinessIssue(
                code="gmail_read_scope_too_broad",
                message="Read-only ingestion is configured with Gmail compose/send/modify scope.",
            )
        )
    if not corpus_exists:
        blockers.append(
            _path_issue(
                "real_email_corpus_missing",
                "Synced real Gmail JSON corpus is missing.",
                config.sync_output_path,
            )
        )
    if not state_exists:
        blockers.append(
            _path_issue(
                "gmail_sync_state_missing",
                "Gmail sync state file is missing.",
                config.sync_state_path,
            )
        )
    if not gold_exists:
        blockers.append(
            _path_issue(
                "real_gold_template_missing",
                "Real-mail gold chunk template is missing.",
                config.gold_template_path,
            )
        )
    elif not _has_labeled_gold_case(config.gold_template_path):
        blockers.append(
            _path_issue(
                "real_gold_template_unlabeled",
                "Real-mail gold template exists but has no labeled case with question, ground_truth, and gold_chunk_ids.",
                config.gold_template_path,
            )
        )

    ready_for_interactive_sync = credentials_exists and readonly_scope_ok
    ready_for_noninteractive_sync = ready_for_interactive_sync and token_exists
    ready_for_real_eval = (
        ready_for_noninteractive_sync
        and corpus_exists
        and state_exists
        and gold_exists
        and _has_labeled_gold_case(config.gold_template_path)
    )

    return ReadinessReport(
        credentials_path=str(config.credentials_path),
        readonly_token_path=str(config.readonly_token_path),
        readonly_scopes=sorted(scope_set),
        sync_output_path=str(config.sync_output_path),
        sync_state_path=str(config.sync_state_path),
        gold_template_path=str(config.gold_template_path),
        ready_for_interactive_sync=ready_for_interactive_sync,
        ready_for_noninteractive_sync=ready_for_noninteractive_sync,
        ready_for_real_eval=ready_for_real_eval,
        blockers=blockers,
        warnings=warnings,
    )


def default_config() -> GmailPhase2AConfig:
    gold_path = getattr(
        cfg,
        "GMAIL_REAL_GOLD_PATH",
        str(cfg.BASE_DIR / "data" / "real_emails" / "gold_chunks.real.json"),
    )
    return GmailPhase2AConfig(
        credentials_path=Path(cfg.GMAIL_CREDENTIALS_PATH),
        readonly_token_path=Path(cfg.GMAIL_READONLY_TOKEN_PATH),
        readonly_scopes=list(cfg.GMAIL_READONLY_SCOPES),
        sync_output_path=Path(cfg.GMAIL_SYNC_OUTPUT_PATH),
        sync_state_path=Path(cfg.GMAIL_SYNC_STATE_PATH),
        gold_template_path=Path(gold_path),
    )


def _build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser(
        description="Check local readiness for real Gmail sync, indexing, and eval."
    )
    parser.add_argument("--credentials", default=str(defaults.credentials_path))
    parser.add_argument("--readonly-token", default=str(defaults.readonly_token_path))
    parser.add_argument("--sync-output", default=str(defaults.sync_output_path))
    parser.add_argument("--sync-state", default=str(defaults.sync_state_path))
    parser.add_argument("--gold-template", default=str(defaults.gold_template_path))
    parser.add_argument(
        "--scope",
        action="append",
        dest="scopes",
        default=None,
        help="Gmail read scope. Repeat for multiple scopes.",
    )
    parser.add_argument(
        "--allow-interactive",
        action="store_true",
        help="Exit 0 when credentials exist and OAuth can be started interactively.",
    )
    parser.add_argument(
        "--require-real-eval",
        action="store_true",
        help="Exit 0 only when real corpus, sync state, and real gold template exist.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _print_human(report: ReadinessReport) -> None:
    print("Phase 2A Gmail real-data readiness")
    print(f"ready_for_interactive_sync: {report.ready_for_interactive_sync}")
    print(f"ready_for_noninteractive_sync: {report.ready_for_noninteractive_sync}")
    print(f"ready_for_real_eval: {report.ready_for_real_eval}")
    if report.blockers:
        print("")
        print("Blockers:")
        for issue in report.blockers:
            suffix = f" ({issue.path})" if issue.path else ""
            print(f"- {issue.code}: {issue.message}{suffix}")
    if report.warnings:
        print("")
        print("Warnings:")
        for issue in report.warnings:
            suffix = f" ({issue.path})" if issue.path else ""
            print(f"- {issue.code}: {issue.message}{suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    scopes = args.scopes if args.scopes is not None else default_config().readonly_scopes
    config = GmailPhase2AConfig(
        credentials_path=Path(args.credentials),
        readonly_token_path=Path(args.readonly_token),
        readonly_scopes=scopes,
        sync_output_path=Path(args.sync_output),
        sync_state_path=Path(args.sync_state),
        gold_template_path=Path(args.gold_template),
    )
    report = check_phase2a_readiness(config)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(report)

    if args.require_real_eval:
        return 0 if report.ready_for_real_eval else 1
    if args.allow_interactive:
        return 0 if report.ready_for_interactive_sync else 1
    return 0 if report.ready_for_noninteractive_sync else 1


if __name__ == "__main__":
    raise SystemExit(main())
