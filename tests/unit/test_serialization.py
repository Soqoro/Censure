from __future__ import annotations

import hashlib
import unittest

from censure.schemas import EnvironmentLayer, ScenarioIdentity
from censure.serialization import (
    CANONICAL_SERIALIZATION_VERSION,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    make_state_snapshot,
    verify_state_snapshot,
)


class CanonicalSerializationTests(unittest.TestCase):
    def test_mapping_order_does_not_change_json_or_hash(self) -> None:
        left = {"z": [3, 2, 1], "a": {"β": True, "x": None}}
        right = {"a": {"x": None, "β": True}, "z": [3, 2, 1]}

        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))
        self.assertEqual(
            canonical_json(left),
            '{"a":{"x":null,"β":true},"z":[3,2,1]}',
        )

    def test_sha256_is_over_exact_canonical_utf8_bytes(self) -> None:
        value = {"message": "héllo", "count": 2}
        expected = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        self.assertEqual(canonical_sha256(value), expected)

    def test_pydantic_models_and_enums_are_canonical(self) -> None:
        identity = ScenarioIdentity(
            environment_layer=EnvironmentLayer.CONTROL,
            suite_or_domain="payments",
            user_task_id="user-1",
            injection_task_id=None,
            rendered_attack_id=None,
            actor_id="scripted",
            actor_revision="v1",
            decoding_seed=7,
            environment_seed=11,
            behavior_guard_id="strict",
            target_guard_id="none",
        )
        decoded = canonical_json(identity)
        self.assertIn('"environment_layer":"control"', decoded)
        self.assertEqual(len(canonical_sha256(identity)), 64)

    def test_state_snapshot_round_trip_verification(self) -> None:
        snapshot = make_state_snapshot({"balance": 10, "events": []})
        self.assertEqual(snapshot.serialization_version, CANONICAL_SERIALIZATION_VERSION)
        self.assertTrue(verify_state_snapshot(snapshot))
        tampered = snapshot.model_copy(update={"state": {"balance": 9, "events": []}})
        self.assertFalse(verify_state_snapshot(tampered))

    def test_unsupported_or_ambiguous_values_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN"):
            canonical_json({"score": float("nan")})
        with self.assertRaisesRegex(TypeError, "string keys"):
            canonical_json({1: "ambiguous"})
        with self.assertRaisesRegex(TypeError, "unsupported"):
            canonical_json(object())


if __name__ == "__main__":
    unittest.main()
