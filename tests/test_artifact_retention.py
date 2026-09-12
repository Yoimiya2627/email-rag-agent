import copy
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from scripts.artifact_retention import plan_retention, apply_retention
from scripts.state_maintenance import backup_states


def age_tree(path, age=1000):
    stamp = time.time() - age
    for member in path.rglob("*"):
        os.utime(member, (stamp, stamp))
    os.utime(path, (stamp, stamp))


def backup_fixture(tmp_path):
    root = tmp_path / "backups"
    root.mkdir()
    source = tmp_path / "source.json"
    source.write_text('[{"id":"synthetic"}]')
    target = root / "old"
    backup_states({}, target, service_stopped=True, artifacts={"corpus": source})
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["created_at"] = datetime.fromtimestamp(time.time() - 2000, timezone.utc).isoformat()
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    age_tree(target)
    return root, source, target


def test_default_plan_is_read_only_and_apply_removes_only_verified_whole_backup(tmp_path):
    root, source, old = backup_fixture(tmp_path)
    (root / "unknown").mkdir()
    (root / "unknown" / "private.txt").write_text("keep")
    before = source.read_bytes()
    plan = plan_retention(root, "backup", before=time.time() - 100, service_stopped=True)
    assert old.exists() and plan["decisions"][0]["status"] == "eligible"
    result = apply_retention(plan, root=root, service_stopped=True)
    assert result["removed_directories"] == ["old"] and not old.exists()
    assert (root / "unknown" / "private.txt").read_text() == "keep" and source.read_bytes() == before


def test_recent_or_unmanifested_backup_members_are_kept(tmp_path):
    root, _, old = backup_fixture(tmp_path)
    file = old / "artifacts" / "corpus" / "data.json"
    os.utime(file, None)
    plan = plan_retention(root, "backup", before=time.time() - 100, service_stopped=True)
    assert plan["decisions"][0]["status"] == "keep"
    age_tree(old)
    (old / "unmanifested.txt").write_text("do not silently delete")
    plan = plan_retention(root, "backup", before=time.time() - 100, service_stopped=True)
    assert plan["decisions"][0]["status"] == "unknown"
    assert apply_retention(plan, root=root, service_stopped=True)["removed_directories"] == []


def test_tampered_plan_changed_artifact_and_wrong_root_fail_before_delete(tmp_path):
    root, _, old = backup_fixture(tmp_path)
    plan = plan_retention(root, "backup", before=time.time() - 100, service_stopped=True)
    changed = copy.deepcopy(plan)
    changed["decisions"][0]["path"] = "../outside"
    with pytest.raises(ValueError, match="hash"):
        apply_retention(changed, root=root, service_stopped=True)
    with pytest.raises(ValueError, match="root"):
        apply_retention(plan, root=tmp_path, service_stopped=True)
    (old / "artifacts" / "corpus" / "data.json").write_text("changed")
    with pytest.raises(ValueError, match="changed after"):
        apply_retention(plan, root=root, service_stopped=True)
    assert old.exists()


def test_reparse_and_explicit_inventory_budgets_fail_closed(tmp_path):
    root, source, _ = backup_fixture(tmp_path)
    with pytest.raises(ValueError, match="file/byte budget"):
        plan_retention(root, "backup", before=time.time() - 100, service_stopped=True, max_bytes=1)
    with pytest.raises(ValueError, match="directory/depth"):
        plan_retention(root, "backup", before=time.time() - 100, service_stopped=True, max_directories=1)
    try:
        (root / "link").symlink_to(source)
    except OSError:
        pytest.skip("Windows account cannot create symlinks")
    with pytest.raises(ValueError, match="symlinks"):
        plan_retention(root, "backup", before=time.time() - 100, service_stopped=True)


def test_only_explicit_isolated_eval_format_is_eligible(tmp_path):
    root = tmp_path / "evals"
    run = root / "run-1"
    run.mkdir(parents=True)
    report = {"schema_version": 2, "summary": {}, "records": [], "provenance": {
        "created_at": datetime.fromtimestamp(time.time() - 2000, timezone.utc).isoformat(),
        "evaluation_owner": "eval:synthetic", "run_dir": str(run.resolve())}}
    (run / "agent_eval.json").write_text(json.dumps(report))
    (run / "trace.jsonl").write_text('{}\n')
    age_tree(run)
    plan = plan_retention(root, "eval", before=time.time() - 100, service_stopped=True)
    assert plan["decisions"][0]["status"] == "eligible"
    assert apply_retention(plan, root=root, service_stopped=True)["removed_directories"] == ["run-1"]


def test_raw_reference_inventory_never_enables_deletion(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "capture.json").write_text(json.dumps({"format": "gmail-full-v1", "raw_sha256": "a" * 64}))
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps({"format": 1, "raw_sha256": ["a" * 64]}))
    plan = plan_retention(root, "raw", before=time.time() - 100, reference_manifest=refs, service_stopped=True)
    assert plan["decisions"][0]["status"] == "referenced" and plan["deletion_supported"] is False
    with pytest.raises(ValueError, match="inventory-only"):
        apply_retention(plan, root=root, service_stopped=True)
    assert (root / "capture.json").exists()


def test_index_inventory_keeps_active_and_paused_and_marks_other_refs_unknown(tmp_path):
    root = tmp_path / "index"
    manifests = root / "index_manifests" / ("b" * 16)
    generations = manifests / "generations"
    generations.mkdir(parents=True)
    active, paused, old = "a" * 32, "b" * 32, "c" * 32
    (manifests / "active.json").write_text(json.dumps({"generation": active}))
    for generation, status in ((active, "ready"), (paused, "paused"), (old, "ready")):
        (generations / (generation + ".json")).write_text(json.dumps({"generation": generation, "status": status}))
    plan = plan_retention(root, "index", before=time.time() - 100, service_stopped=True)
    assert [row["status"] for row in plan["decisions"]] == ["referenced", "keep", "unknown"]
    with pytest.raises(ValueError, match="inventory-only"):
        apply_retention(plan, root=root, service_stopped=True)


def test_stop_attestation_is_required_even_for_explicit_apply(tmp_path):
    root, _, old = backup_fixture(tmp_path)
    with pytest.raises(ValueError, match="Stop API"):
        plan_retention(root, "backup", before=time.time() - 100)
    plan = plan_retention(root, "backup", before=time.time() - 100, service_stopped=True)
    with pytest.raises(ValueError, match="Stop API"):
        apply_retention(plan, root=root)
    assert old.exists()


def test_file_budget_stops_directory_iterator_before_materializing_all_entries(tmp_path, monkeypatch):
    root = tmp_path / "many-files"
    root.mkdir()
    for number in range(2):
        (root / f"f{number}").write_text("x")
    visited = []
    original = os.scandir
    @contextmanager
    def entries(path):
        if not isinstance(path, (str, Path)) or Path(path) != root:
            with original(path) as actual:
                yield actual
            return
        def many():
            for number in range(1000000):
                visited.append(number)
                yield SimpleNamespace(path=str(root / f"f{number}"))
        yield many()
    monkeypatch.setattr(os, "scandir", entries)
    with pytest.raises(ValueError, match="file/byte budget"):
        plan_retention(root, "backup", before=time.time() - 100, service_stopped=True, max_files=1)
    assert visited == [0, 1]
