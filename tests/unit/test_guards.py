from __future__ import annotations

import unittest

from censure.guards import NEUTRAL_DENIAL_TEXT, make_guard
from censure.schemas import (
    AuthorizationPolicy,
    AuthorizationRule,
    GuardAction,
    GuardKind,
    RuleEffect,
    ToolCall,
)


class SequenceRng:
    def __init__(self, *draws: float) -> None:
        self._draws = iter(draws)

    def random(self) -> float:
        return next(self._draws)


def policy() -> AuthorizationPolicy:
    return AuthorizationPolicy(
        policy_id="payment-policy-v1",
        rules=(
            AuthorizationRule(
                rule_id="deny-untrusted-recipient",
                effect=RuleEffect.DENY,
                tool_name="send_payment",
                reason="The recipient is not trusted.",
                argument_equals={"recipient": "attacker"},
            ),
            AuthorizationRule(
                rule_id="allow-payment",
                effect=RuleEffect.ALLOW,
                tool_name="send_payment",
                reason="The user authorized this payment tool.",
                required_arguments=("recipient", "amount"),
                allowed_argument_names=("recipient", "amount"),
            ),
        ),
        default_rule_id="outside-payment-envelope",
        weak_ignored_rule_ids=("deny-untrusted-recipient",),
    )


def call(*, recipient: str = "vendor", call_id: str = "call-1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="send_payment",
        arguments={"recipient": recipient, "amount": 25},
        index=0,
    )


def decide(guard, proposed_call: ToolCall):
    return guard.decide(
        user_request="Pay the approved vendor 25 credits.",
        policy=policy(),
        observable_state={"balance": 100},
        history=(),
        proposed_call=proposed_call,
    )


class GuardTests(unittest.TestCase):
    def test_strict_allows_authorized_call_and_is_not_deny_all(self) -> None:
        proposed = call()
        decision = decide(make_guard("strict"), proposed)

        self.assertEqual(decision.action, GuardAction.ALLOW)
        self.assertEqual(decision.policy_rule_id, "allow-payment")
        self.assertEqual(decision.operation_supplied_to_environment, proposed)
        self.assertIsNone(decision.actor_visible_response)

    def test_strict_blocks_denied_and_outside_envelope_calls(self) -> None:
        denied = decide(make_guard(GuardKind.STRICT), call(recipient="attacker"))
        unknown = ToolCall(id="call-2", name="delete_account", arguments={}, index=0)
        outside = decide(make_guard("strict"), unknown)

        self.assertEqual(denied.action, GuardAction.BLOCK)
        self.assertEqual(denied.policy_rule_id, "deny-untrusted-recipient")
        self.assertEqual(outside.policy_rule_id, "outside-payment-envelope")
        denied_response = denied.actor_visible_response
        assert denied_response is not None
        self.assertEqual(denied_response.content, NEUTRAL_DENIAL_TEXT)
        self.assertEqual(denied_response.tool_call_id, "call-1")
        self.assertFalse(denied_response.is_error)

    def test_weak_removes_only_preregistered_deny_rule(self) -> None:
        relaxed = decide(make_guard("weak"), call(recipient="attacker"))
        unknown = ToolCall(id="call-2", name="delete_account", arguments={}, index=0)
        outside = decide(make_guard("weak"), unknown)

        self.assertEqual(relaxed.action, GuardAction.ALLOW)
        self.assertEqual(relaxed.policy_rule_id, "allow-payment")
        self.assertEqual(outside.action, GuardAction.BLOCK)

    def test_none_allows_every_normalized_call(self) -> None:
        proposed = ToolCall(id="call-x", name="delete_account", arguments={}, index=3)
        decision = decide(make_guard("none"), proposed)
        self.assertEqual(decision.action, GuardAction.ALLOW)
        self.assertEqual(decision.operation_supplied_to_environment, proposed)

    def test_degraded_strict_uses_supplied_rng_only_at_block_points(self) -> None:
        bypass = make_guard("degraded_strict", rho=0.25, rng=SequenceRng(0.10))
        retained = make_guard("degraded_strict", rho=0.25, rng=SequenceRng(0.90))

        bypassed = decide(bypass, call(recipient="attacker"))
        blocked = decide(retained, call(recipient="attacker"))

        self.assertEqual(bypassed.action, GuardAction.ALLOW)
        self.assertEqual(bypassed.intervention_probability, 0.25)
        self.assertEqual(bypassed.rng_draw, 0.10)
        self.assertEqual(blocked.action, GuardAction.BLOCK)
        self.assertEqual(blocked.intervention_probability, 0.75)
        self.assertEqual(blocked.rng_draw, 0.90)

        # No draw is requested when strict already allows the call.
        allowed = decide(
            make_guard("degraded_strict", rho=0.5, rng=SequenceRng()),
            call(),
        )
        self.assertEqual(allowed.action, GuardAction.ALLOW)
        self.assertIsNone(allowed.rng_draw)

    def test_degraded_guard_requires_explicit_valid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires rho"):
            make_guard("degraded_strict", rng=SequenceRng(0.1))
        with self.assertRaisesRegex(ValueError, "caller-supplied"):
            make_guard("degraded_strict", rho=0.5)
        with self.assertRaisesRegex(ValueError, "one of"):
            make_guard("degraded_strict", rho=1.1, rng=SequenceRng(0.1))

    def test_same_guard_is_an_identical_negative_control(self) -> None:
        first = decide(make_guard("same_guard", same_guard_base="strict"), call())
        second = decide(make_guard("same_guard", same_guard_base="strict"), call())

        self.assertEqual(first, second)
        self.assertEqual(first.guard_id, "same_guard")
        self.assertEqual(len(first.guard_configuration_hash), 64)

    def test_neutral_denial_text_does_not_depend_on_rule_reason(self) -> None:
        explicit = decide(make_guard("strict"), call(recipient="attacker", call_id="a"))
        outside_call = ToolCall(id="b", name="unknown", arguments={}, index=1)
        outside = decide(make_guard("strict"), outside_call)

        explicit_response = explicit.actor_visible_response
        outside_response = outside.actor_visible_response
        assert explicit_response is not None
        assert outside_response is not None
        self.assertEqual(explicit_response.content, NEUTRAL_DENIAL_TEXT)
        self.assertEqual(outside_response.content, NEUTRAL_DENIAL_TEXT)
        self.assertEqual(explicit_response.tool_call_id, "a")
        self.assertEqual(outside_response.tool_call_id, "b")


if __name__ == "__main__":
    unittest.main()
