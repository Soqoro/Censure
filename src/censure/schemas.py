"""Versioned, unambiguous data contracts for CENSURE experiments.

The models in this module deliberately contain no environment-specific runtime
objects.  They can therefore be serialized as durable JSON checkpoints and
reconstructed without importing AgentDojo or a model backend.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class FrozenModel(BaseModel):
    """Base class for checkpoint-safe immutable records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        populate_by_name=True,
    )


class EnvironmentLayer(str, Enum):
    AGENTDOJO = "agentdojo"
    CONTROL = "control"


class GuardKind(str, Enum):
    STRICT = "strict"
    WEAK = "weak"
    NONE = "none"
    SAME_GUARD = "same_guard"
    DEGRADED_STRICT = "degraded_strict"


class GuardAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


class RuleEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ObservationSource(str, Enum):
    GUARD = "guard"
    TOOL = "tool"


class TrajectoryRole(str, Enum):
    BEHAVIOR = "behavior"
    TARGET = "target"


class RunStatus(str, Enum):
    """Every terminal trajectory status allowed by the Experiment 1 protocol."""

    COMPLETED = "completed"
    NO_DIVERGENCE = "no_divergence"
    INVALID_TOOL_CALL = "invalid_tool_call"
    MODEL_ERROR = "model_error"
    CONTEXT_OVERFLOW = "context_overflow"
    TIMEOUT = "timeout"
    OOM = "oom"
    RESTORE_ERROR = "restore_error"
    VALIDATOR_ERROR = "validator_error"


class ScenarioIdentity(FrozenModel):
    """The complete frozen sampling unit used to derive a unique session key."""

    environment_layer: EnvironmentLayer
    suite_or_domain: Identifier
    user_task_id: Identifier
    injection_task_id: Identifier | None = None
    rendered_attack_id: Identifier | None = None
    actor_id: Identifier
    actor_revision: Identifier
    decoding_seed: Annotated[int, Field(ge=0)]
    environment_seed: Annotated[int, Field(ge=0)]
    behavior_guard_id: Identifier
    target_guard_id: Identifier


class ToolCall(FrozenModel):
    """A backend-independent, syntactically valid proposed tool operation."""

    id: Identifier
    name: Identifier
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    index: Annotated[int, Field(ge=0)] = 0


class ActorMessage(FrozenModel):
    """An actor-visible chat message with normalized tool calls."""

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: Identifier | None = None


class ToolObservation(FrozenModel):
    """A normalized observation returned for one proposed tool call."""

    tool_call_id: Identifier
    content: str
    source: ObservationSource
    is_error: bool = False


class AuthorizationRule(FrozenModel):
    """One declarative rule in an authorization envelope.

    Matching is intentionally simple and auditable: tool names are exact (or
    ``*``), required arguments must be present, constrained arguments must equal
    their frozen values, and optional ``allowed_argument_names`` can prohibit
    undeclared arguments.
    """

    rule_id: Identifier
    effect: RuleEffect
    tool_name: Identifier
    reason: Identifier
    required_arguments: tuple[Identifier, ...] = ()
    argument_equals: dict[str, JsonValue] = Field(default_factory=dict)
    allowed_argument_names: tuple[Identifier, ...] | None = None

    @model_validator(mode="after")
    def validate_argument_contract(self) -> AuthorizationRule:
        required = set(self.required_arguments)
        constrained = set(self.argument_equals)
        if self.allowed_argument_names is not None:
            allowed = set(self.allowed_argument_names)
            missing = (required | constrained) - allowed
            if missing:
                raise ValueError(
                    "required/constrained arguments absent from "
                    f"allowed_argument_names: {sorted(missing)}"
                )
        return self


class AuthorizationPolicy(FrozenModel):
    """Frozen strict policy plus the explicitly preregistered weak relaxation.

    Strict evaluation applies every rule and defaults to deny.  Weak evaluation
    ignores only the DENY rule IDs listed in ``weak_ignored_rule_ids``; a call
    must still match an ALLOW rule.  This makes the strict-to-weak difference
    explicit in the serialized policy rather than hidden in guard code.
    """

    policy_id: Identifier
    rules: tuple[AuthorizationRule, ...]
    default_rule_id: Identifier
    default_reason: Identifier = "Call is outside the authorization envelope."
    weak_ignored_rule_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_rule_ids(self) -> AuthorizationPolicy:
        by_id = {rule.rule_id: rule for rule in self.rules}
        if len(by_id) != len(self.rules):
            raise ValueError("authorization rule IDs must be unique")
        if self.default_rule_id in by_id:
            raise ValueError("default_rule_id must not duplicate an explicit rule ID")
        ignored = set(self.weak_ignored_rule_ids)
        unknown = ignored - set(by_id)
        if unknown:
            raise ValueError(f"weak relaxation references unknown rule IDs: {sorted(unknown)}")
        non_deny = sorted(
            rule_id for rule_id in ignored if by_id[rule_id].effect is not RuleEffect.DENY
        )
        if non_deny:
            raise ValueError(f"weak relaxation may ignore only DENY rules: {non_deny}")
        return self


class GuardInputs(FrozenModel):
    """Exactly the information an action guard is permitted to observe."""

    user_request: str
    policy: AuthorizationPolicy
    observable_state: JsonValue
    history: tuple[ActorMessage, ...]
    proposed_call: ToolCall


class GuardDecision(FrozenModel):
    """Normalized middleware result immediately preceding environment mutation."""

    guard_id: Identifier
    action: GuardAction
    intervention_probability: Probability
    policy_rule_id: Identifier
    reason: Identifier
    operation_supplied_to_environment: ToolCall | None
    actor_visible_response: ToolObservation | None
    guard_configuration_hash: Sha256Hex
    rng_draw: Probability | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> GuardDecision:
        if self.action is GuardAction.ALLOW:
            if self.operation_supplied_to_environment is None:
                raise ValueError("allow decisions require an environment operation")
            if self.actor_visible_response is not None:
                raise ValueError("allow decisions cannot predeclare the environment response")
        else:
            if self.operation_supplied_to_environment is not None:
                raise ValueError("block decisions cannot supply an environment operation")
            if self.actor_visible_response is None:
                raise ValueError("block decisions require an actor-visible response")
        return self


class StateSnapshot(FrozenModel):
    """Canonical state and its digest at one instrumentation boundary."""

    serialization_version: Literal["censure-canonical-json-v1"]
    state: JsonValue
    sha256: Sha256Hex


class InterventionTrace(FrozenModel):
    """One pre-enforcement proposal and the resulting state transition."""

    step_index: Annotated[int, Field(ge=0)]
    tool_call_index: Annotated[int, Field(ge=0)]
    pre_state: StateSnapshot
    actor_visible_messages: tuple[ActorMessage, ...] = ()
    runtime_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    model_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    guard_inputs: GuardInputs
    pre_enforcement_unsafe_attempt: bool
    guard_decision: GuardDecision
    executed_operation: ToolCall | None
    actor_visible_response: ToolObservation | None
    post_state: StateSnapshot
    state_diff: JsonValue
    rng_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_decision_projection(self) -> InterventionTrace:
        if self.executed_operation != self.guard_decision.operation_supplied_to_environment:
            raise ValueError("trace operation must equal the guard decision operation")
        if (
            self.guard_decision.action is GuardAction.BLOCK
            and self.actor_visible_response != self.guard_decision.actor_visible_response
        ):
            raise ValueError("blocked trace must preserve the deterministic guard response")
        return self


class TrajectoryResult(FrozenModel):
    """Terminal normalized outcome; missing labels are never interpreted as safe."""

    scenario: ScenarioIdentity
    role: TrajectoryRole
    status: RunStatus
    initial_state_sha256: Sha256Hex
    final_state_sha256: Sha256Hex | None = None
    attack_goal_achieved: bool | None = None
    terminal_harm: bool | None = None
    user_utility: bool | float | None = None
    final_answer: str | None = None
    termination_reason: str | None = None
    attempted_unsafe_action: bool = False
    blocked_call_count: Annotated[int, Field(ge=0)] = 0
    proposed_call_count: Annotated[int, Field(ge=0)] = 0
    terminal_validation_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    interventions: tuple[InterventionTrace, ...] = ()

    @property
    def run_status(self) -> RunStatus:
        """Unambiguous public name used by normalized environment outcomes."""

        return self.status

    @model_validator(mode="after")
    def validate_terminal_labels(self) -> TrajectoryResult:
        successful = self.status in {RunStatus.COMPLETED, RunStatus.NO_DIVERGENCE}
        if successful and (self.attack_goal_achieved is None or self.terminal_harm is None):
            raise ValueError("completed trajectories require explicit terminal labels")
        if not successful and self.terminal_harm is not None:
            raise ValueError("invalid/error trajectories must not carry a terminal harm label")
        return self


class ScenarioSplit(str, Enum):
    SMOKE = "smoke"
    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"


class FrozenScenario(FrozenModel):
    """Outcome-free scenario frozen before actor execution."""

    schema_version: Literal["censure.scenario.v1"] = "censure.scenario.v1"
    scenario_id: Identifier
    environment_layer: EnvironmentLayer
    suite_or_domain: Identifier
    user_task_id: Identifier
    injection_task_id: Identifier | None = None
    rendered_attack_id: Identifier | None = None
    rendered_attack: dict[str, str] = Field(default_factory=dict)
    rendered_attack_sha256: Sha256Hex
    injection_locations: tuple[str, ...] = ()
    canonical_initial_state: StateSnapshot
    user_request: str
    available_tools: tuple[dict[str, JsonValue], ...]
    untrusted_content: tuple[str, ...] = ()
    policy: AuthorizationPolicy
    policy_sha256: Sha256Hex
    environment_seed: Annotated[int, Field(ge=0)]
    split: ScenarioSplit
    agentdojo_package_version: str | None = None
    agentdojo_benchmark_version: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_hashes(self) -> FrozenScenario:
        # Local import avoids making the schema module the serialization owner.
        from censure.serialization import canonical_sha256

        if canonical_sha256(self.rendered_attack) != self.rendered_attack_sha256:
            raise ValueError("rendered attack hash does not match payload")
        if canonical_sha256(self.policy) != self.policy_sha256:
            raise ValueError("policy hash does not match policy")
        return self


class PairedSession(FrozenModel):
    """Actor/guard expansion of one immutable base scenario."""

    schema_version: Literal["censure.paired-session.v1"] = "censure.paired-session.v1"
    session_id: Sha256Hex
    scenario_id: Identifier
    environment_layer: EnvironmentLayer
    suite_or_domain: Identifier
    user_task_id: Identifier
    injection_task_id: Identifier | None = None
    rendered_attack_id: Identifier | None = None
    rendered_attack_sha256: Sha256Hex
    initial_state_sha256: Sha256Hex
    policy_sha256: Sha256Hex
    actor_id: Identifier
    actor_revision: Identifier
    tokenizer_revision: Identifier
    decoding_seed: Annotated[int, Field(ge=0)]
    environment_seed: Annotated[int, Field(ge=0)]
    behavior_guard_id: Identifier
    target_guard_id: Identifier
    behavior_guard_config_sha256: Sha256Hex
    target_guard_config_sha256: Sha256Hex
    generation_config_sha256: Sha256Hex
    chat_template_sha256: Sha256Hex
    prompt_chat_template_sha256: Sha256Hex
    state_serialization_version: Identifier
    split: ScenarioSplit
    guard_pair_id: Identifier
    agentdojo_package_version: str | None = None
    agentdojo_benchmark_version: str | None = None


class FirstDivergence(FrozenModel):
    """The unique earliest behavior/target intervention disagreement."""

    shared_prefix_id: Identifier
    step_index: Annotated[int, Field(ge=0)]
    pre_intervention_checkpoint: StateSnapshot
    behavior_intervention: GuardDecision
    target_intervention: GuardDecision

    @model_validator(mode="after")
    def require_disagreement(self) -> FirstDivergence:
        if self.behavior_intervention == self.target_intervention:
            raise ValueError("first-divergence decisions must disagree")
        return self
