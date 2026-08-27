from __future__ import annotations

import json

import pytest

from censure.storage import RunStore, deterministic_shard, session_key


def test_session_key_covers_scientific_fields() -> None:
    base = {"actor_revision": "a", "guard": "strict", "payload": "x", "seed": 1}
    key = session_key(base)
    for field, value in (("actor_revision", "b"), ("guard", "none"), ("payload", "y"), ("seed", 2)):
        changed = dict(base)
        changed[field] = value
        assert session_key(changed) != key


def test_resume_is_idempotent_and_corruption_is_not_complete(tmp_path) -> None:
    store = RunStore(tmp_path, "exp")
    store.write_trajectory(session_id="abc", role="behavior", summary={"x": 1}, trace=[{"a": 2}])
    assert store.is_complete(session_id="abc", role="behavior")
    with pytest.raises(FileExistsError):
        store.write_trajectory(session_id="abc", role="behavior", summary={"x": 1}, trace=[])
    summary_path = store.root / "behavior" / "abc" / "summary.json"
    summary_path.write_text(json.dumps({"x": 2}), encoding="utf-8")
    assert not store.is_complete(session_id="abc", role="behavior")


def test_oracle_directory_requires_evaluation_capability(tmp_path) -> None:
    store = RunStore(tmp_path, "exp")
    store.write_trajectory(session_id="abc", role="target", summary={"harm": True}, trace=[])
    with pytest.raises(PermissionError):
        store.read_oracle_summary("abc")
    with pytest.raises(PermissionError):
        store.evaluation_view()
    assert store.evaluation_view(evaluation=True).read_oracle_summary("abc")["harm"] is True


def test_hash_sharding_is_deterministic() -> None:
    assert deterministic_shard("same", num_shards=7) == deterministic_shard("same", num_shards=7)
    assert 0 <= deterministic_shard("x", num_shards=7) < 7
