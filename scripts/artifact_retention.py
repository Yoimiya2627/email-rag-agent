"""Explicit offline retention plans. Raw mail/index inventory never deletes data."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

try:
    from scripts.bundle_files import reject_links, member_path, digest, validate_artifacts
    from scripts.state_maintenance import _open, _validate, TABLES
except ModuleNotFoundError:
    from bundle_files import reject_links, member_path, digest, validate_artifacts
    from state_maintenance import _open, _validate, TABLES


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _json(path, maximum=64 * 1024**2):
    path = reject_links(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Metadata exceeds its read budget")
    return json.loads(path.read_text(encoding="utf-8"))


def _root(path):
    root = reject_links(path)
    if not root.is_dir() or root == root.parent:
        raise ValueError("An explicit non-filesystem-root directory is required")
    return root


def _attest(value):
    if value is not True:
        raise ValueError("Stop API, UI, MCP and workers; explicitly attest service_stopped=True")


def _timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Artifact timestamp must include timezone")
    return parsed.timestamp()


def _inventory(root, max_files, max_bytes, max_directories):
    files, directories, pending = [], [], [root]
    total = 0
    while pending:
        parent = pending.pop()
        # Never materialize a whole directory before applying the budgets.
        # Stable ordering is applied only to the bounded inventory below.
        with os.scandir(parent) as entries:
            for entry in entries:
                child = reject_links(Path(entry.path))
                if not child.is_relative_to(root):
                    raise ValueError("Inventory member escapes its explicit root")
                relative = child.relative_to(root).as_posix()
                member_path(relative)
                if child.is_dir():
                    directories.append(relative)
                    if len(directories) > max_directories or len(child.relative_to(root).parts) > 64:
                        raise ValueError("Retention inventory exceeds its directory/depth budget")
                    pending.append(child)
                elif child.is_file():
                    info = child.stat()
                    total += info.st_size
                    if len(files) >= max_files or total > max_bytes:
                        raise ValueError("Retention inventory exceeds its file/byte budget")
                    checksum = digest(child)
                    after = child.stat()
                    if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("Artifact changed during retention inventory")
                    files.append({"path": relative, "bytes": info.st_size, "sha256": checksum,
                                  "mtime_ns": info.st_mtime_ns})
                else:
                    raise ValueError("Retention inventory requires regular files and directories")
    return {"files": sorted(files, key=lambda row: row["path"]), "directories": sorted(directories), "bytes": total}


def _backup(directory, rows, budgets):
    manifest = _json(directory / "manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("format") != 1 or manifest.get("service_stopped_attested") is not True
            or not isinstance(manifest.get("states"), dict) or not isinstance(manifest.get("artifacts"), dict)
            or not (manifest["states"] or manifest["artifacts"]) or not set(manifest["states"]).issubset(TABLES)):
        raise ValueError("Not a complete state_maintenance backup")
    plans = validate_artifacts(directory, manifest["artifacts"], max_files=budgets["max_files"], max_bytes=budgets["max_bytes"])
    expected = {"manifest.json"}
    expected.update(f"artifacts/{kind}/{item['path']}" for kind, plan in plans.items() for item in plan["files"])
    for kind, item in manifest["states"].items():
        if not isinstance(item, dict) or item.get("file") != kind + ".sqlite3":
            raise ValueError("Invalid backup state member")
        path = directory / item["file"]
        if path.stat().st_size != item.get("bytes") or digest(path) != item.get("sha256"):
            raise ValueError("Backup state checksum mismatch")
        with closing(_open(path, readonly=True, snapshot=True)) as db:
            _validate(db, kind)
        expected.add(item["file"])
    if {row["path"] for row in rows} != expected:
        raise ValueError("Backup contains unmanifested or missing files")
    return _timestamp(manifest["created_at"])


def _evaluation(directory, rows):
    names = {row["path"] for row in rows}
    allowed = all(name in {"agent_eval.json", "approvals.sqlite3"} or re.fullmatch(r"trace\.jsonl(?:\.\d+)?", name) for name in names)
    if "agent_eval.json" not in names or not allowed:
        raise ValueError("Evaluation directory has unrecognized members")
    report = _json(directory / "agent_eval.json")
    provenance = report.get("provenance", {}) if isinstance(report, dict) else {}
    if (report.get("schema_version") != 2 or not isinstance(report.get("records"), list)
            or not isinstance(report.get("summary"), dict) or not isinstance(provenance, dict)
            or not isinstance(provenance.get("evaluation_owner"), str)
            or not provenance["evaluation_owner"].startswith("eval:")
            or Path(provenance.get("run_dir", "")).resolve() != directory):
        raise ValueError("Not a complete isolated agent evaluation directory")
    if "approvals.sqlite3" in names:
        with closing(_open(directory / "approvals.sqlite3", readonly=True, snapshot=True)) as db:
            _validate(db, "approvals")
            owners = {row[0] for row in db.execute("SELECT DISTINCT owner_id FROM approvals")}
            if owners - {provenance["evaluation_owner"]}:
                raise ValueError("Evaluation directory contains non-evaluation approvals")
    return _timestamp(provenance["created_at"])


def _reference_ids(path):
    if path is None:
        return {"generations": [], "raw_sha256": []}, None
    data = _json(path)
    if not isinstance(data, dict) or data.get("format") != 1 or set(data) - {"format", "generations", "raw_sha256"}:
        raise ValueError("Reference inventory has an unsupported format")
    for key in ("generations", "raw_sha256"):
        values = data.get(key, [])
        if not isinstance(values, list) or len(values) > 100000 or any(not isinstance(value, str) or len(value) > 128 for value in values):
            raise ValueError("Reference inventory exceeds its identifier budget")
    return data, digest(path)


def plan_retention(root, kind, *, before, service_stopped=False, reference_manifest=None,
                   max_files=100000, max_bytes=10 * 1024**3, max_directories=10000):
    _attest(service_stopped)
    root = _root(root)
    if kind not in {"backup", "eval", "raw", "index"}:
        raise ValueError("Unknown retention kind")
    if type(before) not in (int, float) or not math.isfinite(before) or not 0 <= before <= time.time():
        raise ValueError("before must be a past Unix timestamp")
    budgets = {"max_files": max_files, "max_bytes": max_bytes, "max_directories": max_directories}
    if any(type(value) is not int or value < 1 for value in budgets.values()):
        raise ValueError("Retention budgets must be positive integers")
    tree = _inventory(root, **budgets)
    references, reference_hash = _reference_ids(reference_manifest)
    decisions = []
    if kind in {"backup", "eval"}:
        names = sorted({row["path"].split("/")[0] for row in tree["files"]} | {name.split("/")[0] for name in tree["directories"]})
        for name in names:
            directory = root / name
            rows = [{**row, "path": row["path"][len(name) + 1:]} for row in tree["files"] if row["path"].startswith(name + "/")]
            decision = {"path": name, "status": "unknown", "reason": "unrecognized_directory_or_file", "bytes": sum(row["bytes"] for row in rows)}
            if directory.is_dir():
                try:
                    created = _backup(directory, rows, budgets) if kind == "backup" else _evaluation(directory, rows)
                    old = created < before and all(row["mtime_ns"] < before * 1_000_000_000 for row in rows)
                    decision.update(status="eligible" if old else "keep", reason="verified_expired_complete_directory" if old else "not_expired_or_recently_modified")
                except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
                    decision.update(reason="validation_failed_" + type(exc).__name__)
            decisions.append(decision)
    elif kind == "raw":
        for row in tree["files"]:
            decision = {"path": row["path"], "status": "unknown", "reason": "reference_graph_incomplete_keep", "bytes": row["bytes"]}
            if row["path"].endswith(".json"):
                try:
                    capture = _json(root / row["path"])
                    if capture.get("format") == "gmail-full-v1" and capture.get("raw_sha256") in references.get("raw_sha256", []):
                        decision.update(status="referenced", reason="supplied_raw_hash_reference_keep")
                except (ValueError, TypeError, AttributeError, OSError):
                    pass
            decisions.append(decision)
    else:
        for row in tree["files"]:
            if re.fullmatch(r"index_manifests/[a-f0-9]{16}/generations/[a-f0-9]{32}\.json", row["path"]):
                decision = {"path": row["path"], "status": "unknown", "reason": "reference_graph_incomplete_keep", "bytes": row["bytes"]}
                try:
                    path = root / row["path"]
                    manifest = _json(path)
                    pointer = _json(path.parent.parent / "active.json")
                    generation = manifest.get("generation")
                    if generation == pointer.get("generation") or generation in references.get("generations", []):
                        decision.update(status="referenced", reason="active_or_supplied_generation_reference_keep")
                    elif manifest.get("status") in {"building", "paused"}:
                        decision.update(status="keep", reason="resumable_generation_keep")
                except (ValueError, TypeError, AttributeError, OSError):
                    pass
                decisions.append(decision)
    plan = {"format": 1, "root": str(root), "kind": kind, "before": before, "budgets": budgets,
            "reference_manifest": str(Path(reference_manifest).resolve()) if reference_manifest else None,
            "reference_sha256": reference_hash, "inventory": tree, "decisions": decisions,
            "deletion_supported": kind in {"backup", "eval"},
            "reference_coverage": "supplied_positive_references_only_unknowns_retained"}
    plan["plan_sha256"] = _hash(plan)
    return plan


def apply_retention(plan, *, root, service_stopped=False):
    _attest(service_stopped)
    root = _root(root)
    if not isinstance(plan, dict) or plan.get("format") != 1 or plan.get("root") != str(root):
        raise ValueError("Retention plan does not match the explicitly requested root")
    if plan.get("kind") not in {"backup", "eval"}:
        raise ValueError("Raw mail and index retention is inventory-only; deletion is disabled")
    supplied_hash = plan.get("plan_sha256")
    if supplied_hash != _hash({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("Retention plan hash mismatch")
    fresh = plan_retention(root, plan["kind"], before=plan["before"], service_stopped=True,
                           reference_manifest=plan.get("reference_manifest"), **plan["budgets"])
    if fresh["plan_sha256"] != supplied_hash:
        raise ValueError("Artifacts changed after planning; generate and review a new plan")
    removed = []
    for decision in fresh["decisions"]:
        if decision["status"] != "eligible":
            continue
        original = reject_links(root / decision["path"])
        if original.parent != root:
            raise ValueError("Retention deletion must target a direct child of its explicit root")
        quarantine = root / (".retention-" + uuid.uuid4().hex)
        if quarantine.exists():
            raise ValueError("Retention quarantine already exists")
        original.rename(quarantine)
        prefix = decision["path"] + "/"
        files = [row for row in fresh["inventory"]["files"] if row["path"].startswith(prefix)]
        directories = [name[len(prefix):] for name in fresh["inventory"]["directories"] if name.startswith(prefix)]
        for row in files:
            path = reject_links(quarantine / row["path"][len(prefix):])
            if not path.is_relative_to(quarantine) or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Artifact changed during retention; remaining quarantined files were retained")
            path.unlink()
        for relative in sorted(directories, key=lambda name: (name.count("/"), name), reverse=True):
            path = reject_links(quarantine / relative)
            if not path.is_relative_to(quarantine):
                raise ValueError("Retention directory escapes quarantine")
            path.rmdir()
        quarantine.rmdir()
        removed.append(decision["path"])
    return {"format": 1, "root": str(root), "kind": fresh["kind"], "applied": True,
            "plan_sha256": supplied_hash, "removed_directories": removed,
            "kept_count": len(fresh["decisions"]) - len(removed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--kind", choices=["backup", "eval", "raw", "index"])
    parser.add_argument("--before", type=float)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--plan-output")
    parser.add_argument("--apply-plan", help="Explicitly apply a previously reviewed plan; default is read-only")
    parser.add_argument("--service-stopped", action="store_true")
    parser.add_argument("--max-files", type=int, default=100000)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--max-directories", type=int, default=10000)
    args = parser.parse_args(argv)
    output = None
    if args.plan_output:
        output, requested_root = Path(args.plan_output).absolute(), _root(args.root)
        if output == requested_root or requested_root in output.parents:
            raise ValueError("Write the retention plan outside the inventoried root")
        reject_links(output.parent)
        if output.exists() or output.is_symlink():
            raise ValueError("Retention plan output must be a new file")
    if args.apply_plan:
        result = apply_retention(_json(args.apply_plan), root=args.root, service_stopped=args.service_stopped)
    else:
        if args.kind is None or args.before is None:
            parser.error("planning requires --kind and --before")
        result = plan_retention(args.root, args.kind, before=args.before, service_stopped=args.service_stopped,
            reference_manifest=args.reference_manifest, max_files=args.max_files, max_bytes=args.max_bytes,
            max_directories=args.max_directories)
    if output is not None:
        with output.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "inventory"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
