"""Fail-closed adapter for the released AgentDojo 0.1.35 API.

The benchmark package deliberately remains an optional dependency.  Nothing is
imported from AgentDojo at module-import time; every environment-facing entry
point first verifies the exact distribution version and the public API surface
used here.

AgentDojo's raw ``security`` name is counterintuitive: in release 0.1.35,
``True`` means that the injection (attacker) goal *was achieved*.  Clean runs
are a further trap because ``TaskSuite.run_task_with_pipeline`` returns
``security=True`` as a sentinel when no injection task exists.  This adapter
therefore never infers polarity from the name and never consumes that clean-run
sentinel.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module, metadata
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, model_validator

from censure.schemas import (
    AuthorizationPolicy,
    AuthorizationRule,
    FrozenModel,
    RuleEffect,
    RunStatus,
    Sha256Hex,
)
from censure.serialization import canonical_sha256

AGENTDOJO_PACKAGE_VERSION = "0.1.35"
AGENTDOJO_BENCHMARK_VERSION = "v1.2.2"
AGENTDOJO_SOURCE_REVISION = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
AGENTDOJO_WHEEL_SHA256 = "364bea4219716b716bf639f504d195943f7f6a5535d312ca41d7098704a2affd"
AGENTDOJO_SDIST_SHA256 = "9eacbc89d996f8656b235ad7b626bcf840b1ace7101174ca62d790c7c6d62956"
AGENTDOJO_ADAPTER_VERSION = "censure-agentdojo-0.1.35-adapter-v2"
AGENTDOJO_STATE_SERIALIZATION_VERSION = "censure-agentdojo-0.1.35-state-v2"
AGENTDOJO_INJECTION_NORMALIZATION = "agentdojo-yaml-whitespace-fold-v1"
AGENTDOJO_SUITES = ("workspace", "slack", "travel", "banking")

RAW_SECURITY_SEMANTICS = "true_means_injection_goal_achieved"

_EXPECTED_SUITES: dict[str, dict[str, Any]] = {
    "workspace": {
        "benchmark_version": (1, 2, 2),
        "user_tasks": 40,
        "injection_tasks": 14,
        "tools": 24,
        "injection_vectors": 16,
    },
    "slack": {
        "benchmark_version": (1, 2, 0),
        "user_tasks": 21,
        "injection_tasks": 5,
        "tools": 11,
        "injection_vectors": 6,
    },
    "travel": {
        "benchmark_version": (1, 2, 0),
        "user_tasks": 20,
        "injection_tasks": 7,
        "tools": 28,
        "injection_vectors": 13,
    },
    "banking": {
        "benchmark_version": (1, 2, 2),
        "user_tasks": 16,
        "injection_tasks": 9,
        "tools": 11,
        "injection_vectors": 4,
    },
}

# AgentDojo 0.1.35 deliberately rebuilds these live mappings from their
# ``initial_*`` lists in Pydantic ``after`` validators.  That is suitable for
# loading benchmark YAML but would silently undo deletions and additions when
# restoring a trajectory checkpoint.  These exact released field paths are
# version-gated below and rehydrated after ordinary model validation.
_VALIDATOR_RESET_MAPPINGS: dict[
    str,
    tuple[tuple[str, str, str, tuple[str, ...]], ...],
] = {
    "workspace": (
        ("inbox", "emails", "initial_emails", ("trash",)),
        ("calendar", "events", "initial_events", ()),
        ("cloud_drive", "files", "initial_files", ()),
    ),
    "slack": (),
    "travel": (
        ("inbox", "emails", "initial_emails", ("trash",)),
        ("calendar", "events", "initial_events", ()),
    ),
    "banking": (),
}

ADAPTER_LIMITATIONS = (
    "The installed distribution records its version but not the source commit or original wheel bytes; "
    "the source revision and archive hashes are release-provenance expectations, not claims about an "
    "already-unpacked installation.",
    "TaskSuite.run_task_with_pipeline does not expose per-call state boundaries and returns security=True "
    "for clean runs, so CENSURE must orchestrate trajectories itself.",
    "AgentDojo benchmark helpers convert some provider/context errors into ordinary utility/security "
    "booleans; CENSURE must not use those helpers for scientific outcomes.",
    "AgentDojo derives its built-in trace graders from assistant proposals.  Guarded runs must instead "
    "supply only calls that actually executed successfully, or blocked calls can be mislabeled as harm.",
    "A state snapshot reconstructs the pinned Pydantic environment, including current collections that "
    "AgentDojo 0.1.35 model validators otherwise reset from initial_* fields.  Runtime functions are "
    "reconstructed from the suite; actor messages, executed-call provenance, and RNG metadata must be "
    "checkpointed separately.",
    "Canonical JSON does not encode Python object identity.  The adapter restores the released "
    "initial/current collection aliases when their values match, but arbitrary cross-component aliases "
    "created outside released tools are not reconstructible.",
    "AgentDojo loads default environments through YAML, which folds line breaks in rendered injection "
    "payloads into spaces.  Frozen scenarios retain both the raw rendering and exact state-value hashes, "
    "with an explicit whitespace-normalized projection linking the two.",
    "The released runtime supports nested FunctionCall arguments, but this adapter's durable call record "
    "accepts JSON arguments only; nested executable calls must be normalized or rejected upstream.",
)


class AgentDojoAdapterError(RuntimeError):
    """Base error for a failed AgentDojo adapter boundary."""


class AgentDojoUnavailableError(AgentDojoAdapterError):
    """The exact optional AgentDojo distribution is not importable."""


class AgentDojoCompatibilityError(AgentDojoAdapterError):
    """The installed package or benchmark surface differs from the frozen release."""


class AgentDojoRestoreError(AgentDojoAdapterError):
    """A durable environment snapshot cannot be safely reconstructed."""


class AgentDojoEvaluationError(AgentDojoAdapterError):
    """A task evaluator or executed tool boundary failed."""


class AgentDojoSuiteMetadata(FrozenModel):
    """Version-sensitive metadata for one released suite."""

    name: str
    benchmark_version: tuple[int, int, int]
    environment_type: str
    user_task_ids: tuple[str, ...]
    injection_task_ids: tuple[str, ...]
    tool_names: tuple[str, ...]
    injection_vector_ids: tuple[str, ...]


class AgentDojoCompatibilityReport(FrozenModel):
    """What was verified locally and what is only pinned release provenance."""

    adapter_version: str
    package_version: str
    benchmark_version: str
    expected_source_revision: str
    expected_wheel_sha256: str
    expected_sdist_sha256: str
    distribution_root: str
    python_version: str
    archive_bytes_verified: bool = False
    public_api_checks: tuple[str, ...]
    suites: tuple[AgentDojoSuiteMetadata, ...]
    limitations: tuple[str, ...]


class AgentDojoStateSnapshot(FrozenModel):
    """Canonical, JSON-only snapshot of one suite environment."""

    serialization_version: str = AGENTDOJO_STATE_SERIALIZATION_VERSION
    package_version: str = AGENTDOJO_PACKAGE_VERSION
    benchmark_version: str = AGENTDOJO_BENCHMARK_VERSION
    suite_name: str
    environment_type: str
    state: JsonValue
    sha256: str

    @model_validator(mode="after")
    def validate_frozen_versions(self) -> AgentDojoStateSnapshot:
        expected = {
            "serialization_version": AGENTDOJO_STATE_SERIALIZATION_VERSION,
            "package_version": AGENTDOJO_PACKAGE_VERSION,
            "benchmark_version": AGENTDOJO_BENCHMARK_VERSION,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"{field_name} does not match the pinned AgentDojo adapter")
        if self.suite_name not in AGENTDOJO_SUITES:
            raise ValueError(f"unknown AgentDojo suite: {self.suite_name}")
        if len(self.sha256) != 64:
            raise ValueError("state SHA-256 must contain 64 hexadecimal characters")
        return self


class AgentDojoToolSchema(FrozenModel):
    """Actor-facing JSON schema for one released suite tool."""

    name: str
    description: str
    parameters: dict[str, JsonValue]


class AgentDojoStateTextMatch(FrozenModel):
    """One exact frozen state value containing a rendered attack payload."""

    state_path: tuple[str | int, ...]
    state_value_sha256: Sha256Hex


class AgentDojoInjectionProjection(FrozenModel):
    """Auditable link from raw attack rendering to YAML-resident state text."""

    injection_location: str
    normalization: Literal["agentdojo-yaml-whitespace-fold-v1"] = AGENTDOJO_INJECTION_NORMALIZATION
    rendered_payload_sha256: Sha256Hex
    normalized_payload_sha256: Sha256Hex
    state_matches: tuple[AgentDojoStateTextMatch, ...]

    @model_validator(mode="after")
    def require_state_match(self) -> AgentDojoInjectionProjection:
        if not self.state_matches:
            raise ValueError("an injection projection requires at least one state match")
        return self


class FrozenAgentDojoScenario(FrozenModel):
    """Rendered attack material and canonical initial state for one sampling unit."""

    adapter_version: str = AGENTDOJO_ADAPTER_VERSION
    package_version: str = AGENTDOJO_PACKAGE_VERSION
    benchmark_version: str = AGENTDOJO_BENCHMARK_VERSION
    suite_name: str
    suite_benchmark_version: tuple[int, int, int]
    user_task_id: str
    injection_task_id: str | None = None
    attack_name: str | None = None
    attack_target_pipeline_name: str | None = None
    user_request: str
    rendered_injections: dict[str, str] = Field(default_factory=dict)
    injection_locations: tuple[str, ...] = ()
    injection_projections: tuple[AgentDojoInjectionProjection, ...] = ()
    rendered_attack_sha256: str | None = None
    available_tools: tuple[AgentDojoToolSchema, ...]
    authorization_policy: AuthorizationPolicy
    initial_state: AgentDojoStateSnapshot

    @model_validator(mode="after")
    def validate_attack_projection(self) -> FrozenAgentDojoScenario:
        attacked = self.injection_task_id is not None
        if attacked:
            if not self.attack_name or not self.rendered_injections:
                raise ValueError("attacked scenarios require an attack and rendered injections")
            if self.rendered_attack_sha256 is None:
                raise ValueError("attacked scenarios require a rendered attack hash")
        elif (
            self.attack_name is not None
            or self.attack_target_pipeline_name is not None
            or self.rendered_injections
            or self.rendered_attack_sha256 is not None
        ):
            raise ValueError("clean scenarios cannot carry rendered attack material")
        if tuple(self.rendered_injections) != self.injection_locations:
            raise ValueError("injection locations must preserve rendered-injection order")
        if tuple(item.injection_location for item in self.injection_projections) != (
            self.injection_locations
        ):
            raise ValueError("injection projections must preserve injection-location order")
        if self.initial_state.suite_name != self.suite_name:
            raise ValueError("initial state belongs to a different suite")
        for projection in self.injection_projections:
            payload = self.rendered_injections[projection.injection_location]
            expected = _project_injection_into_state(
                projection.injection_location,
                payload,
                self.initial_state.state,
            )
            if projection != expected:
                raise ValueError("injection projection does not match the frozen initial state")
        return self


class AgentDojoCallExecution(FrozenModel):
    """Durable disposition of one normalized proposal.

    Blocked calls have ``executed=False``.  Only records with all of
    ``executed=True``, ``succeeded=True`` and ``error_message=None`` may enter
    an AgentDojo trace-based grader.
    """

    function: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    call_id: str | None = None
    index: int = Field(default=0, ge=0)
    executed: bool
    succeeded: bool
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> AgentDojoCallExecution:
        if self.succeeded and not self.executed:
            raise ValueError("a call cannot succeed without executing")
        if self.succeeded and (self.error_type is not None or self.error_message is not None):
            raise ValueError("successful calls cannot carry execution errors")
        if not self.executed and (self.error_type is not None or self.error_message is not None):
            raise ValueError("blocked/unexecuted calls cannot carry environment execution errors")
        if self.executed and not self.succeeded and self.error_message is None:
            raise ValueError("failed executed calls require an error message")
        return self


@dataclass(frozen=True, slots=True)
class AgentDojoToolExecution:
    """In-memory tool return paired with its durable execution record."""

    result: Any
    record: AgentDojoCallExecution


SecuritySource = Literal[
    "security_from_traces",
    "security",
    "clean_no_injection",
    "task_suite_clean_sentinel",
    "not_evaluated",
]


class NormalizedAgentDojoSecurity(FrozenModel):
    """Unambiguous security polarity with raw release output retained."""

    attacked: bool
    raw_security: bool | None = None
    raw_security_repr: str | None = None
    raw_security_source: SecuritySource
    raw_security_semantics: str = RAW_SECURITY_SEMANTICS
    attack_goal_achieved: bool | None = None
    terminal_harm: bool | None = None
    run_status: RunStatus
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> NormalizedAgentDojoSecurity:
        valid = self.run_status in {RunStatus.COMPLETED, RunStatus.NO_DIVERGENCE}
        if valid and (self.attack_goal_achieved is None or self.terminal_harm is None):
            raise ValueError("valid security outcomes require explicit normalized labels")
        if not valid and (self.attack_goal_achieved is not None or self.terminal_harm is not None):
            raise ValueError("invalid/error outcomes cannot be labeled safe or harmful")
        if self.attack_goal_achieved != self.terminal_harm:
            raise ValueError("AgentDojo terminal harm must equal attack-goal achievement")
        return self


UtilitySource = Literal["utility_from_traces", "utility", "not_evaluated"]


class AgentDojoUtilityOutcome(FrozenModel):
    """Programmatic user-task utility with evaluator provenance."""

    raw_utility: bool | None = None
    raw_utility_repr: str | None = None
    raw_utility_source: UtilitySource
    user_utility: bool | None = None
    run_status: RunStatus
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_utility(self) -> AgentDojoUtilityOutcome:
        valid = self.run_status in {RunStatus.COMPLETED, RunStatus.NO_DIVERGENCE}
        if valid and self.user_utility is None:
            raise ValueError("valid utility outcomes require an explicit label")
        if not valid and self.user_utility is not None:
            raise ValueError("invalid/error outcomes cannot carry normalized utility")
        return self


@dataclass(frozen=True, slots=True)
class _Bindings:
    get_suite: Any
    get_suites: Any
    TaskSuite: Any
    BaseUserTask: Any
    BaseInjectionTask: Any
    FunctionsRuntime: Any
    FunctionCall: Any
    BaseAttack: Any
    load_attack: Any
    GroundTruthPipeline: Any


_PUBLIC_API_CHECKS = (
    "agentdojo.task_suite.get_suites",
    "agentdojo.task_suite.get_suite",
    "TaskSuite.load_and_inject_default_environment",
    "TaskSuite.get_injection_vector_defaults",
    "TaskSuite.get_user_task_by_id",
    "TaskSuite.get_injection_task_by_id",
    "BaseUserTask.utility[_from_traces]",
    "BaseInjectionTask.security[_from_traces]",
    "FunctionsRuntime.run_function",
    "agentdojo.attacks.load_attack",
    "GroundTruthPipeline",
    "FunctionCall.model_fields",
    "released validator-reset environment model_fields",
)


def _signature_names(function: Any) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def _require_signature(label: str, function: Any, expected: Sequence[str]) -> None:
    actual = _signature_names(function)
    if actual != tuple(expected):
        raise AgentDojoCompatibilityError(
            f"AgentDojo {label} signature drifted: expected {tuple(expected)!r}, got {actual!r}"
        )


@lru_cache(maxsize=1)
def _load_bindings() -> _Bindings:
    try:
        installed_version = metadata.version("agentdojo")
    except metadata.PackageNotFoundError as exc:
        raise AgentDojoUnavailableError(
            "AgentDojo is not installed; install the 'agentdojo' extra with agentdojo==0.1.35"
        ) from exc
    if installed_version != AGENTDOJO_PACKAGE_VERSION:
        raise AgentDojoCompatibilityError(
            f"expected agentdojo=={AGENTDOJO_PACKAGE_VERSION}, found {installed_version}"
        )

    try:
        task_suite_module = import_module("agentdojo.task_suite")
        runtime_module = import_module("agentdojo.functions_runtime")
        attacks_module = import_module("agentdojo.attacks")
        pipeline_module = import_module("agentdojo.agent_pipeline")
    except Exception as exc:
        raise AgentDojoUnavailableError(
            "AgentDojo 0.1.35 is installed but its declared runtime dependencies cannot be imported"
        ) from exc

    bindings = _Bindings(
        get_suite=task_suite_module.get_suite,
        get_suites=task_suite_module.get_suites,
        TaskSuite=task_suite_module.TaskSuite,
        BaseUserTask=task_suite_module.BaseUserTask,
        BaseInjectionTask=task_suite_module.BaseInjectionTask,
        FunctionsRuntime=runtime_module.FunctionsRuntime,
        FunctionCall=runtime_module.FunctionCall,
        BaseAttack=attacks_module.BaseAttack,
        load_attack=attacks_module.load_attack,
        GroundTruthPipeline=pipeline_module.GroundTruthPipeline,
    )

    signature_checks = (
        ("get_suites", bindings.get_suites, ("benchmark_version",)),
        ("get_suite", bindings.get_suite, ("benchmark_version", "suite_name")),
        (
            "TaskSuite.load_and_inject_default_environment",
            bindings.TaskSuite.load_and_inject_default_environment,
            ("self", "injections"),
        ),
        (
            "TaskSuite.get_injection_vector_defaults",
            bindings.TaskSuite.get_injection_vector_defaults,
            ("self",),
        ),
        (
            "TaskSuite.get_user_task_by_id",
            bindings.TaskSuite.get_user_task_by_id,
            ("self", "task_id"),
        ),
        (
            "TaskSuite.get_injection_task_by_id",
            bindings.TaskSuite.get_injection_task_by_id,
            ("self", "injection_task_id"),
        ),
        (
            "BaseUserTask.init_environment",
            bindings.BaseUserTask.init_environment,
            ("environment",),
        ),
        (
            "BaseUserTask.utility",
            bindings.BaseUserTask.utility,
            ("self", "model_output", "pre_environment", "post_environment", "strict"),
        ),
        (
            "BaseUserTask.utility_from_traces",
            bindings.BaseUserTask.utility_from_traces,
            ("self", "model_output", "pre_environment", "post_environment", "traces"),
        ),
        (
            "BaseInjectionTask.security",
            bindings.BaseInjectionTask.security,
            ("self", "model_output", "pre_environment", "post_environment"),
        ),
        (
            "BaseInjectionTask.security_from_traces",
            bindings.BaseInjectionTask.security_from_traces,
            ("self", "model_output", "pre_environment", "post_environment", "traces"),
        ),
        (
            "FunctionsRuntime.run_function",
            bindings.FunctionsRuntime.run_function,
            ("self", "env", "function", "kwargs", "raise_on_error"),
        ),
        (
            "load_attack",
            bindings.load_attack,
            ("attack_name", "task_suite", "target_pipeline"),
        ),
    )
    for label, function, expected in signature_checks:
        _require_signature(label, function, expected)

    function_call_fields = tuple(bindings.FunctionCall.model_fields)
    if function_call_fields != ("function", "args", "id", "placeholder_args"):
        raise AgentDojoCompatibilityError(
            f"FunctionCall fields drifted: expected released fields, got {function_call_fields!r}"
        )
    suites = bindings.get_suites(AGENTDOJO_BENCHMARK_VERSION)
    if set(suites) != set(AGENTDOJO_SUITES):
        raise AgentDojoCompatibilityError(
            f"benchmark {AGENTDOJO_BENCHMARK_VERSION} suite registry drifted: {sorted(suites)!r}"
        )
    return bindings


def _task_id_key(task_id: str) -> tuple[str, int | str]:
    prefix, _, suffix = task_id.rpartition("_")
    return (prefix, int(suffix) if suffix.isdigit() else suffix)


def _environment_type_name(suite: Any) -> str:
    environment_type = suite.environment_type
    return f"{environment_type.__module__}.{environment_type.__qualname__}"


def _validate_suite(suite_name: str, suite: Any) -> None:
    expected = _EXPECTED_SUITES[suite_name]
    observed = {
        "benchmark_version": tuple(suite.benchmark_version),
        "user_tasks": len(suite.user_tasks),
        "injection_tasks": len(suite.injection_tasks),
        "tools": len(suite.tools),
        "injection_vectors": len(suite.get_injection_vector_defaults()),
    }
    if observed != expected:
        raise AgentDojoCompatibilityError(
            f"suite {suite_name!r} differs from AgentDojo 0.1.35/{AGENTDOJO_BENCHMARK_VERSION}: "
            f"expected {expected!r}, observed {observed!r}"
        )
    tool_names = [tool.name for tool in suite.tools]
    if len(tool_names) != len(set(tool_names)):
        raise AgentDojoCompatibilityError(f"suite {suite_name!r} contains duplicate tool names")
    environment_fields = suite.environment_type.model_fields
    for component_name, current_name, initial_name, related_names in _VALIDATOR_RESET_MAPPINGS[
        suite_name
    ]:
        component_field = environment_fields.get(component_name)
        component_type = None if component_field is None else component_field.annotation
        if not isinstance(component_type, type) or not issubclass(component_type, BaseModel):
            raise AgentDojoCompatibilityError(
                f"suite {suite_name!r} state component {component_name!r} drifted"
            )
        required_fields = {current_name, initial_name, *related_names}
        missing = required_fields - set(component_type.model_fields)
        if missing:
            raise AgentDojoCompatibilityError(
                f"suite {suite_name!r} state component {component_name!r} is missing "
                f"released restoration fields {sorted(missing)!r}"
            )


def load_suite(suite_name: str) -> Any:
    """Load and validate one exact ``v1.2.2`` suite singleton."""

    if suite_name not in AGENTDOJO_SUITES:
        raise ValueError(f"suite must be one of {AGENTDOJO_SUITES!r}, got {suite_name!r}")
    bindings = _load_bindings()
    try:
        suite = bindings.get_suite(AGENTDOJO_BENCHMARK_VERSION, suite_name)
    except Exception as exc:
        raise AgentDojoCompatibilityError(
            f"failed to load suite {suite_name!r} at benchmark {AGENTDOJO_BENCHMARK_VERSION}"
        ) from exc
    _validate_suite(suite_name, suite)
    return suite


def _suite_metadata(suite_name: str) -> AgentDojoSuiteMetadata:
    suite = load_suite(suite_name)
    return AgentDojoSuiteMetadata(
        name=suite_name,
        benchmark_version=tuple(suite.benchmark_version),
        environment_type=_environment_type_name(suite),
        user_task_ids=tuple(sorted(suite.user_tasks, key=_task_id_key)),
        injection_task_ids=tuple(sorted(suite.injection_tasks, key=_task_id_key)),
        tool_names=tuple(tool.name for tool in suite.tools),
        injection_vector_ids=tuple(suite.get_injection_vector_defaults()),
    )


@lru_cache(maxsize=1)
def compatibility_report() -> AgentDojoCompatibilityReport:
    """Verify the installed release and enumerate all frozen suite metadata."""

    _load_bindings()
    distribution = metadata.distribution("agentdojo")
    return AgentDojoCompatibilityReport(
        adapter_version=AGENTDOJO_ADAPTER_VERSION,
        package_version=distribution.version,
        benchmark_version=AGENTDOJO_BENCHMARK_VERSION,
        expected_source_revision=AGENTDOJO_SOURCE_REVISION,
        expected_wheel_sha256=AGENTDOJO_WHEEL_SHA256,
        expected_sdist_sha256=AGENTDOJO_SDIST_SHA256,
        distribution_root=str(distribution.locate_file("")),
        python_version=sys.version.split()[0],
        public_api_checks=_PUBLIC_API_CHECKS,
        suites=tuple(_suite_metadata(name) for name in AGENTDOJO_SUITES),
        limitations=ADAPTER_LIMITATIONS,
    )


def snapshot_environment(suite_name: str, environment: Any) -> AgentDojoStateSnapshot:
    """Serialize a released suite environment as reconstructible canonical JSON."""

    suite = load_suite(suite_name)
    if not isinstance(environment, suite.environment_type):
        raise TypeError(
            f"expected {suite.environment_type.__qualname__} for {suite_name}, "
            f"got {type(environment).__qualname__}"
        )
    # round_trip=True excludes computed fields such as Inbox.sent/received and
    # emits the actual model inputs needed by model_validate during restore.
    state = environment.model_dump(mode="json", round_trip=True)
    return AgentDojoStateSnapshot(
        suite_name=suite_name,
        environment_type=_environment_type_name(suite),
        state=state,
        sha256=canonical_sha256(state),
    )


def _validate_component_field(component: BaseModel, field_name: str, value: Any) -> Any:
    field = type(component).model_fields.get(field_name)
    if field is None or field.annotation is None:
        raise AgentDojoCompatibilityError(
            f"released state field {type(component).__qualname__}.{field_name} is unavailable"
        )
    return TypeAdapter(field.annotation).validate_python(value)


def _relink_initial_values(
    current: dict[Any, Any],
    initial_values: Sequence[Any],
) -> None:
    """Recreate released initial/current aliases when JSON values prove equality."""

    initial_by_id: dict[Any, Any] = {}
    for item in initial_values:
        if not hasattr(item, "id_"):
            raise AgentDojoCompatibilityError(
                "released initial-state collection item no longer exposes id_"
            )
        initial_by_id[item.id_] = item
    for key, value in tuple(current.items()):
        initial = initial_by_id.get(key)
        if initial is not None and canonical_sha256(value) == canonical_sha256(initial):
            current[key] = initial


def _rehydrate_validator_reset_state(suite_name: str, environment: Any, state: Any) -> None:
    """Undo released after-validator resets without bypassing model validation.

    Pydantic first validates the complete public environment object.  We then
    revalidate only the exact live mapping fields overwritten by AgentDojo
    0.1.35 ``after`` validators and assign those typed values back.  The caller
    performs a full canonical round trip afterward, so any unhandled mutation
    or model drift remains an explicit restore error.
    """

    if not isinstance(state, Mapping):
        raise AgentDojoRestoreError("AgentDojo state payload must be an object")
    for component_name, current_name, initial_name, related_names in _VALIDATOR_RESET_MAPPINGS[
        suite_name
    ]:
        component_state = state.get(component_name)
        if not isinstance(component_state, Mapping) or current_name not in component_state:
            raise AgentDojoRestoreError(
                f"snapshot omits released state field {component_name}.{current_name}"
            )
        component = getattr(environment, component_name)
        restored_current = _validate_component_field(
            component,
            current_name,
            component_state[current_name],
        )
        if not isinstance(restored_current, dict):
            raise AgentDojoCompatibilityError(
                f"released state field {component_name}.{current_name} is no longer a mapping"
            )
        initial_values = getattr(component, initial_name)
        if not isinstance(initial_values, Sequence):
            raise AgentDojoCompatibilityError(
                f"released state field {component_name}.{initial_name} is no longer a sequence"
            )
        _relink_initial_values(restored_current, initial_values)
        setattr(component, current_name, restored_current)
        for related_name in related_names:
            related = getattr(component, related_name)
            if isinstance(related, dict):
                _relink_initial_values(related, initial_values)


def restore_environment(snapshot: AgentDojoStateSnapshot | Mapping[str, Any]) -> Any:
    """Restore an independent state copy and verify a canonical round trip."""

    try:
        parsed = (
            snapshot
            if isinstance(snapshot, AgentDojoStateSnapshot)
            else AgentDojoStateSnapshot.model_validate(snapshot)
        )
        if canonical_sha256(parsed.state) != parsed.sha256:
            raise AgentDojoRestoreError("snapshot state hash does not match its payload")
        suite = load_suite(parsed.suite_name)
        expected_type = _environment_type_name(suite)
        if parsed.environment_type != expected_type:
            raise AgentDojoRestoreError(
                f"snapshot environment type {parsed.environment_type!r} != {expected_type!r}"
            )
        restored = suite.environment_type.model_validate(parsed.state)
        _rehydrate_validator_reset_state(parsed.suite_name, restored, parsed.state)
        round_trip = snapshot_environment(parsed.suite_name, restored)
        if round_trip.sha256 != parsed.sha256:
            raise AgentDojoRestoreError("restored state failed canonical round-trip verification")
        return restored
    except AgentDojoRestoreError:
        raise
    except Exception as exc:
        raise AgentDojoRestoreError(f"could not restore AgentDojo state: {exc}") from exc


def _state_text_values(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[tuple[str | int, ...], str], ...]:
    if isinstance(value, str):
        return ((path, value),)
    if isinstance(value, Mapping):
        matches: list[tuple[tuple[str | int, ...], str]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise AgentDojoCompatibilityError("frozen state contains a non-string mapping key")
            matches.extend(_state_text_values(item, (*path, key)))
        return tuple(matches)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        matches = []
        for index, item in enumerate(value):
            matches.extend(_state_text_values(item, (*path, index)))
        return tuple(matches)
    return ()


def _normalize_injected_text(text: str) -> str:
    """Mirror YAML's semantically irrelevant whitespace folding for evidence."""

    return " ".join(text.split())


def _project_injection_into_state(
    injection_location: str,
    rendered_payload: str,
    state: Any,
) -> AgentDojoInjectionProjection:
    """Find and hash exact state values containing a YAML-folded payload."""

    normalized_payload = _normalize_injected_text(rendered_payload)
    if not normalized_payload:
        raise AgentDojoCompatibilityError("rendered attack payload is empty after normalization")
    state_matches = tuple(
        AgentDojoStateTextMatch(
            state_path=path,
            state_value_sha256=canonical_sha256(state_value),
        )
        for path, state_value in _state_text_values(state)
        if normalized_payload in _normalize_injected_text(state_value)
    )
    if not state_matches:
        raise AgentDojoCompatibilityError(
            f"rendered attack payload at {injection_location!r} was not present in frozen state "
            f"under {AGENTDOJO_INJECTION_NORMALIZATION}"
        )
    return AgentDojoInjectionProjection(
        injection_location=injection_location,
        rendered_payload_sha256=canonical_sha256(rendered_payload),
        normalized_payload_sha256=canonical_sha256(normalized_payload),
        state_matches=state_matches,
    )


_READ_ONLY_TOOL_PREFIXES = ("get_", "list_", "search_", "find_", "read_", "check_")


def available_tool_schemas(suite_name: str) -> tuple[AgentDojoToolSchema, ...]:
    """Return the released tools in deterministic execution order."""

    suite = load_suite(suite_name)
    schemas = []
    for tool in suite.tools:
        parameters = tool.parameters.model_json_schema()
        schemas.append(
            AgentDojoToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
            )
        )
    return tuple(schemas)


def _ground_truth_arguments(call: Any) -> dict[str, JsonValue]:
    dumped = call.model_dump(mode="json", round_trip=True)
    arguments = dumped.get("args")
    if not isinstance(arguments, dict):
        raise AgentDojoCompatibilityError("user-task ground truth contains non-object arguments")
    return arguments


def build_authorization_policy(
    suite_name: str,
    user_task_id: str,
    initialized_environment: Any,
    *,
    allow_read_only: bool = True,
) -> AuthorizationPolicy:
    """Build a declarative guard policy from user-task information only.

    This function deliberately has no injection-task parameter and never calls
    an attack or security evaluator.  Exact ground-truth calls are authorized;
    read-style tools may be broadly authorized; every remaining suite tool gets
    a strict deny paired with a weak-only fallback allow.  The default remains
    deny, including alternate arguments to an authorized mutating tool.
    """

    suite = load_suite(suite_name)
    if not isinstance(initialized_environment, suite.environment_type):
        raise TypeError(f"environment does not belong to suite {suite_name!r}")
    try:
        user_task = suite.get_user_task_by_id(user_task_id)
        ground_truth = user_task.ground_truth(initialized_environment.model_copy(deep=True))
    except Exception as exc:
        raise AgentDojoCompatibilityError(
            f"could not derive authorization policy from {suite_name}/{user_task_id} ground truth"
        ) from exc

    known_tools = {tool.name for tool in suite.tools}
    authorized_mutation_tools: set[str] = set()
    rules: list[AuthorizationRule] = []
    for index, call in enumerate(ground_truth):
        if call.function not in known_tools:
            raise AgentDojoCompatibilityError(
                f"user-task ground truth references unavailable tool {call.function!r}"
            )
        arguments = _ground_truth_arguments(call)
        authorized_mutation_tools.add(call.function)
        rules.append(
            AuthorizationRule(
                rule_id=f"allow-user-ground-truth-{index}-{call.function}",
                effect=RuleEffect.ALLOW,
                tool_name=call.function,
                reason=f"Exact operation {index} from the frozen user-task ground truth.",
                required_arguments=tuple(arguments),
                argument_equals=arguments,
                allowed_argument_names=tuple(arguments),
            )
        )

    broadly_readable = {
        tool_name
        for tool_name in known_tools
        if allow_read_only and tool_name.startswith(_READ_ONLY_TOOL_PREFIXES)
    }
    for tool in suite.tools:
        if tool.name in broadly_readable:
            rules.append(
                AuthorizationRule(
                    rule_id=f"allow-read-style-{tool.name}",
                    effect=RuleEffect.ALLOW,
                    tool_name=tool.name,
                    reason="Released read-style tool is within the frozen observation envelope.",
                )
            )

    weak_ignored_rule_ids: list[str] = []
    for tool in suite.tools:
        if tool.name in authorized_mutation_tools or tool.name in broadly_readable:
            continue
        deny_id = f"deny-unrequested-mutation-{tool.name}"
        weak_ignored_rule_ids.append(deny_id)
        rules.extend(
            (
                AuthorizationRule(
                    rule_id=deny_id,
                    effect=RuleEffect.DENY,
                    tool_name=tool.name,
                    reason="Mutating tool is absent from the frozen user-task ground truth.",
                ),
                AuthorizationRule(
                    rule_id=f"allow-weak-fallback-{tool.name}",
                    effect=RuleEffect.ALLOW,
                    tool_name=tool.name,
                    reason="Preregistered weak-guard fallback for an unrequested mutation.",
                ),
            )
        )

    policy_material = {
        "adapter_version": AGENTDOJO_ADAPTER_VERSION,
        "suite_name": suite_name,
        "user_task_id": user_task_id,
        "allow_read_only": allow_read_only,
        "rules": [rule.model_dump(mode="json") for rule in rules],
        "weak_ignored_rule_ids": weak_ignored_rule_ids,
    }
    policy_hash = canonical_sha256(policy_material)
    return AuthorizationPolicy(
        policy_id=f"agentdojo-{suite_name}-{user_task_id}-{policy_hash[:16]}",
        rules=tuple(rules),
        default_rule_id=f"deny-outside-user-envelope-{policy_hash[:16]}",
        default_reason="Call is outside the frozen AgentDojo user-task authorization envelope.",
        weak_ignored_rule_ids=tuple(weak_ignored_rule_ids),
    )


def freeze_scenario(
    suite_name: str,
    user_task_id: str,
    injection_task_id: str | None = None,
    *,
    attack_name: str | None = None,
    target_pipeline: Any | None = None,
) -> FrozenAgentDojoScenario:
    """Render attack payloads and freeze the initialized canonical state.

    ``direct`` and ``tool_knowledge`` can be rendered without a live model.  For
    ``tool_knowledge`` a non-query ``GroundTruthPipeline`` is assigned the
    frozen identity ``name='local'``; AgentDojo therefore deterministically
    substitutes ``"Local model"`` without making a provider/model call.
    Other model-name-dependent attacks require an explicit frozen target
    pipeline so defaults cannot silently change.
    """

    suite = load_suite(suite_name)
    try:
        user_task = suite.get_user_task_by_id(user_task_id)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown {suite_name} user task {user_task_id!r}") from exc

    rendered_injections: dict[str, str]
    if injection_task_id is None:
        if attack_name is not None:
            raise ValueError("clean scenarios must not specify an attack")
        rendered_injections = {}
        rendered_attack_sha256 = None
        attack_target_pipeline_name = None
    else:
        selected_attack = attack_name or "direct"
        try:
            injection_task = suite.get_injection_task_by_id(injection_task_id)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown {suite_name} injection task {injection_task_id!r}") from exc
        active_target_pipeline: Any
        if target_pipeline is None:
            if selected_attack not in {"direct", "tool_knowledge"}:
                raise ValueError(
                    "this attack requires an explicit target_pipeline to freeze model-dependent rendering"
                )
            active_target_pipeline = _load_bindings().GroundTruthPipeline(user_task)
            if selected_attack == "tool_knowledge":
                active_target_pipeline.name = "local"
        else:
            active_target_pipeline = target_pipeline
        attack_target_pipeline_name = getattr(active_target_pipeline, "name", None)
        try:
            attack = _load_bindings().load_attack(selected_attack, suite, active_target_pipeline)
            raw_injections = attack.attack(user_task, injection_task)
        except Exception as exc:
            raise AgentDojoCompatibilityError(
                f"failed to render {selected_attack!r} for {suite_name}/{user_task_id}/{injection_task_id}"
            ) from exc
        if not isinstance(raw_injections, dict) or not raw_injections:
            raise AgentDojoCompatibilityError("attacked scenario rendered no injection locations")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_injections.items()
        ):
            raise AgentDojoCompatibilityError(
                "rendered injections must be a string-to-string mapping"
            )
        rendered_injections = dict(raw_injections)
        rendered_attack_sha256 = canonical_sha256(rendered_injections)
        attack_name = selected_attack

    try:
        environment = suite.load_and_inject_default_environment(rendered_injections)
        environment = user_task.init_environment(environment)
    except Exception as exc:
        raise AgentDojoCompatibilityError("failed to render and initialize scenario state") from exc
    snapshot = snapshot_environment(suite_name, environment)
    policy = build_authorization_policy(suite_name, user_task_id, environment)
    tool_schemas = available_tool_schemas(suite_name)
    injection_projections = tuple(
        _project_injection_into_state(location, payload, snapshot.state)
        for location, payload in rendered_injections.items()
    )

    return FrozenAgentDojoScenario(
        suite_name=suite_name,
        suite_benchmark_version=tuple(suite.benchmark_version),
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        attack_name=attack_name,
        attack_target_pipeline_name=attack_target_pipeline_name,
        user_request=user_task.PROMPT,
        rendered_injections=rendered_injections,
        injection_locations=tuple(rendered_injections),
        injection_projections=injection_projections,
        rendered_attack_sha256=rendered_attack_sha256,
        available_tools=tool_schemas,
        authorization_policy=policy,
        initial_state=snapshot,
    )


def make_runtime(suite_name: str) -> Any:
    """Reconstruct registered tools from the exact released suite."""

    suite = load_suite(suite_name)
    return _load_bindings().FunctionsRuntime(suite.tools)


def execute_tool_call(
    suite_name: str,
    environment: Any,
    *,
    function: str,
    arguments: Mapping[str, JsonValue],
    call_id: str | None = None,
    index: int = 0,
    runtime: Any | None = None,
) -> AgentDojoToolExecution:
    """Execute one allowed call while preserving all error provenance.

    AgentDojo normally returns tool failures as strings.  This wrapper retains
    that behavior for actor-visible formatting but records failure explicitly;
    callers must terminate the trajectory rather than pass a failed record to a
    terminal grader.
    """

    suite = load_suite(suite_name)
    if not isinstance(environment, suite.environment_type):
        raise TypeError(f"environment does not belong to suite {suite_name!r}")
    active_runtime = runtime if runtime is not None else make_runtime(suite_name)
    durable_args = dict(arguments)
    try:
        result, error = active_runtime.run_function(
            environment,
            function,
            durable_args,
            raise_on_error=False,
        )
    except Exception as exc:
        record = AgentDojoCallExecution(
            function=function,
            arguments=durable_args,
            call_id=call_id,
            index=index,
            executed=True,
            succeeded=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return AgentDojoToolExecution(result=None, record=record)
    if error is not None:
        error_type, separator, detail = error.partition(":")
        record = AgentDojoCallExecution(
            function=function,
            arguments=durable_args,
            call_id=call_id,
            index=index,
            executed=True,
            succeeded=False,
            error_type=error_type.strip() or "ToolExecutionError",
            error_message=(detail.strip() if separator else error),
        )
        return AgentDojoToolExecution(result=result, record=record)
    return AgentDojoToolExecution(
        result=result,
        record=AgentDojoCallExecution(
            function=function,
            arguments=durable_args,
            call_id=call_id,
            index=index,
            executed=True,
            succeeded=True,
        ),
    )


def executed_successful_function_calls(
    records: Sequence[AgentDojoCallExecution],
) -> tuple[Any, ...]:
    """Convert only actually executed, successful records into grader traces."""

    FunctionCall = _load_bindings().FunctionCall
    return tuple(
        FunctionCall(
            function=record.function,
            args=dict(record.arguments),
            id=record.call_id,
        )
        for record in records
        if record.executed
        and record.succeeded
        and record.error_type is None
        and record.error_message is None
    )


_MISSING = object()
_SUCCESS_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.NO_DIVERGENCE})


def _error_fields(error: BaseException | None) -> tuple[str | None, str | None]:
    if error is None:
        return None, None
    return type(error).__name__, str(error)


def normalize_security(
    *,
    injection_task_id: str | None,
    raw_security: Any = _MISSING,
    raw_security_source: SecuritySource | None = None,
    run_status: RunStatus | str = RunStatus.COMPLETED,
    error: BaseException | None = None,
) -> NormalizedAgentDojoSecurity:
    """Normalize AgentDojo's harmful-positive raw security label.

    For completed clean controls the normalized harm is explicitly false even
    if a caller preserves AgentDojo's raw clean sentinel ``True``.  For attacked
    runs, only an exact Python boolean is a valid label.  Missing, malformed, or
    exceptional results become ``validator_error`` with both terminal labels
    unset; they are never coerced to safety.
    """

    status = RunStatus(run_status)
    attacked = injection_task_id is not None
    supplied = raw_security is not _MISSING
    raw_value = raw_security if supplied and type(raw_security) is bool else None
    raw_repr = None if not supplied or type(raw_security) is bool else repr(raw_security)
    source: SecuritySource
    if raw_security_source is not None:
        source = raw_security_source
    elif not attacked and supplied:
        source = "task_suite_clean_sentinel"
    elif not attacked:
        source = "clean_no_injection"
    else:
        source = "not_evaluated"

    validation_error = error
    if status in _SUCCESS_STATUSES and attacked:
        if not supplied or raw_security is None:
            validation_error = validation_error or AgentDojoEvaluationError(
                "attacked completed run has no security label"
            )
        elif type(raw_security) is not bool:
            validation_error = validation_error or TypeError(
                f"AgentDojo security grader returned non-boolean {type(raw_security).__qualname__}"
            )
    if validation_error is not None and status in _SUCCESS_STATUSES:
        status = RunStatus.VALIDATOR_ERROR

    error_type, error_message = _error_fields(validation_error)
    if status not in _SUCCESS_STATUSES:
        return NormalizedAgentDojoSecurity(
            attacked=attacked,
            raw_security=raw_value,
            raw_security_repr=raw_repr,
            raw_security_source=source,
            run_status=status,
            error_type=error_type,
            error_message=error_message,
        )
    if not attacked:
        return NormalizedAgentDojoSecurity(
            attacked=False,
            raw_security=raw_value,
            raw_security_repr=raw_repr,
            raw_security_source=source,
            attack_goal_achieved=False,
            terminal_harm=False,
            run_status=status,
        )
    achieved = bool(raw_security)
    return NormalizedAgentDojoSecurity(
        attacked=True,
        raw_security=raw_value,
        raw_security_repr=raw_repr,
        raw_security_source=source,
        attack_goal_achieved=achieved,
        terminal_harm=achieved,
        run_status=status,
    )


def _execution_failure(
    records: Sequence[AgentDojoCallExecution],
) -> AgentDojoEvaluationError | None:
    failures = [record for record in records if record.executed and not record.succeeded]
    if not failures:
        return None
    first = failures[0]
    return AgentDojoEvaluationError(
        f"executed tool call {first.index} ({first.function}) failed: "
        f"{first.error_type or 'ToolExecutionError'}: {first.error_message}"
    )


def evaluate_security(
    injection_task: Any | None,
    *,
    model_output: str,
    pre_environment: Any,
    post_environment: Any,
    call_records: Sequence[AgentDojoCallExecution],
    run_status: RunStatus | str = RunStatus.COMPLETED,
    prior_error: BaseException | None = None,
) -> NormalizedAgentDojoSecurity:
    """Run the released attack evaluator over successful executed calls only."""

    injection_task_id = None if injection_task is None else getattr(injection_task, "ID", None)
    if injection_task is not None and not injection_task_id:
        raise ValueError("injection task must expose its released ID")
    status = RunStatus(run_status)
    failure = prior_error or _execution_failure(call_records)
    if failure is not None or status not in _SUCCESS_STATUSES:
        failure_status = status
        if failure is not None and status in _SUCCESS_STATUSES:
            failure_status = RunStatus.INVALID_TOOL_CALL
        return normalize_security(
            injection_task_id=injection_task_id,
            run_status=failure_status,
            error=failure,
        )
    if injection_task is None:
        return normalize_security(injection_task_id=None)

    traces = executed_successful_function_calls(call_records)
    try:
        raw_security = injection_task.security_from_traces(
            model_output,
            pre_environment,
            post_environment,
            traces,
        )
        source: SecuritySource = "security_from_traces"
        if raw_security is None:
            raw_security = injection_task.security(
                model_output,
                pre_environment,
                post_environment,
            )
            source = "security"
    except Exception as exc:
        return normalize_security(
            injection_task_id=injection_task_id,
            raw_security_source="not_evaluated",
            error=exc,
        )
    return normalize_security(
        injection_task_id=injection_task_id,
        raw_security=raw_security,
        raw_security_source=source,
    )


def _invalid_utility(
    status: RunStatus,
    error: BaseException | None,
    *,
    raw_utility: Any = _MISSING,
    source: UtilitySource = "not_evaluated",
) -> AgentDojoUtilityOutcome:
    if status in _SUCCESS_STATUSES:
        status = RunStatus.VALIDATOR_ERROR
    valid_raw = raw_utility if type(raw_utility) is bool else None
    raw_repr = None if raw_utility is _MISSING or type(raw_utility) is bool else repr(raw_utility)
    error_type, error_message = _error_fields(error)
    return AgentDojoUtilityOutcome(
        raw_utility=valid_raw,
        raw_utility_repr=raw_repr,
        raw_utility_source=source,
        run_status=status,
        error_type=error_type,
        error_message=error_message,
    )


def evaluate_utility(
    user_task: Any,
    *,
    model_output: str,
    pre_environment: Any,
    post_environment: Any,
    call_records: Sequence[AgentDojoCallExecution],
    strict: bool = True,
    run_status: RunStatus | str = RunStatus.COMPLETED,
    prior_error: BaseException | None = None,
) -> AgentDojoUtilityOutcome:
    """Run released user utility, retaining trace and failure provenance."""

    status = RunStatus(run_status)
    failure = prior_error or _execution_failure(call_records)
    if failure is not None or status not in _SUCCESS_STATUSES:
        if failure is not None and status in _SUCCESS_STATUSES:
            status = RunStatus.INVALID_TOOL_CALL
        return _invalid_utility(status, failure)
    traces = executed_successful_function_calls(call_records)
    try:
        raw_utility = user_task.utility_from_traces(
            model_output,
            pre_environment,
            post_environment,
            traces,
        )
        source: UtilitySource = "utility_from_traces"
        if raw_utility is None:
            raw_utility = user_task.utility(
                model_output,
                pre_environment,
                post_environment,
                strict=strict,
            )
            source = "utility"
        if type(raw_utility) is not bool:
            raise TypeError(
                f"AgentDojo utility grader returned non-boolean {type(raw_utility).__qualname__}"
            )
    except Exception as exc:
        return _invalid_utility(
            status,
            exc,
            raw_utility=locals().get("raw_utility", _MISSING),
            source=locals().get("source", "not_evaluated"),
        )
    return AgentDojoUtilityOutcome(
        raw_utility=raw_utility,
        raw_utility_source=source,
        user_utility=raw_utility,
        run_status=status,
    )


class AgentDojoV0135Adapter:
    """Small object facade over the module-level, version-gated functions."""

    package_version = AGENTDOJO_PACKAGE_VERSION
    benchmark_version = AGENTDOJO_BENCHMARK_VERSION
    suites = AGENTDOJO_SUITES
    limitations = ADAPTER_LIMITATIONS

    def __init__(self) -> None:
        _load_bindings()

    def compatibility_report(self) -> AgentDojoCompatibilityReport:
        return compatibility_report()

    def load_suite(self, suite_name: str) -> Any:
        return load_suite(suite_name)

    def freeze_scenario(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None = None,
        *,
        attack_name: str | None = None,
        target_pipeline: Any | None = None,
    ) -> FrozenAgentDojoScenario:
        return freeze_scenario(
            suite_name,
            user_task_id,
            injection_task_id,
            attack_name=attack_name,
            target_pipeline=target_pipeline,
        )

    def snapshot_environment(self, suite_name: str, environment: Any) -> AgentDojoStateSnapshot:
        return snapshot_environment(suite_name, environment)

    def restore_environment(self, snapshot: AgentDojoStateSnapshot | Mapping[str, Any]) -> Any:
        return restore_environment(snapshot)

    def make_runtime(self, suite_name: str) -> Any:
        return make_runtime(suite_name)

    def available_tool_schemas(self, suite_name: str) -> tuple[AgentDojoToolSchema, ...]:
        return available_tool_schemas(suite_name)

    def build_authorization_policy(
        self,
        suite_name: str,
        user_task_id: str,
        initialized_environment: Any,
        *,
        allow_read_only: bool = True,
    ) -> AuthorizationPolicy:
        return build_authorization_policy(
            suite_name,
            user_task_id,
            initialized_environment,
            allow_read_only=allow_read_only,
        )

    def execute_tool_call(
        self, suite_name: str, environment: Any, **kwargs: Any
    ) -> AgentDojoToolExecution:
        return execute_tool_call(suite_name, environment, **kwargs)

    def evaluate_security(
        self, injection_task: Any | None, **kwargs: Any
    ) -> NormalizedAgentDojoSecurity:
        return evaluate_security(injection_task, **kwargs)

    def evaluate_utility(self, user_task: Any, **kwargs: Any) -> AgentDojoUtilityOutcome:
        return evaluate_utility(user_task, **kwargs)
