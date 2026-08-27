"""Stateful environment adapters for CENSURE experiments."""

from typing import TYPE_CHECKING, Any

from censure.environments.base import (
    EnvironmentError,
    InvalidToolCallError,
    SnapshotRestoreError,
    StatefulEnvironment,
    ToolDefinition,
    ToolExecutionResult,
)
from censure.environments.control import (
    CONTROL_DOMAINS,
    CONTROL_SCENARIO_VERSION,
    CONTROL_SCENARIO_VERSION_V1,
    CONTROL_SCENARIO_VERSION_V2,
    CONTROL_SCENARIO_VERSIONS,
    CONTROL_SEEDS,
    CONTROL_STRATA,
    ControlAttemptEvaluator,
    ControlEnvironment,
    ControlHarmValidator,
    ControlScenarioSpec,
    ControlTerminalValidator,
    ControlUtilityValidator,
    PredicateDefinitionError,
    PredicateEvaluation,
    TerminalValidation,
    build_control_scenarios,
    generate_control_instances,
    generate_control_scenarios,
    get_control_scenario,
)

if TYPE_CHECKING:
    from censure.environments.agentdojo import (
        AgentDojoAttemptEvaluator,
        AgentDojoEnvironment,
        make_agentdojo_bindings,
    )


_AGENTDOJO_EXPORTS = frozenset(
    {
        "AgentDojoAttemptEvaluator",
        "AgentDojoEnvironment",
        "make_agentdojo_bindings",
    }
)


def __getattr__(name: str) -> Any:
    """Load the optional facade lazily to avoid an execution-module cycle."""

    if name not in _AGENTDOJO_EXPORTS:
        raise AttributeError(name)
    from censure.environments.agentdojo import (
        AgentDojoAttemptEvaluator,
        AgentDojoEnvironment,
        make_agentdojo_bindings,
    )

    exports = {
        "AgentDojoAttemptEvaluator": AgentDojoAttemptEvaluator,
        "AgentDojoEnvironment": AgentDojoEnvironment,
        "make_agentdojo_bindings": make_agentdojo_bindings,
    }
    globals().update(exports)
    return exports[name]


__all__ = [
    "CONTROL_DOMAINS",
    "CONTROL_SCENARIO_VERSION",
    "CONTROL_SCENARIO_VERSIONS",
    "CONTROL_SCENARIO_VERSION_V1",
    "CONTROL_SCENARIO_VERSION_V2",
    "CONTROL_SEEDS",
    "CONTROL_STRATA",
    "AgentDojoAttemptEvaluator",
    "AgentDojoEnvironment",
    "ControlAttemptEvaluator",
    "ControlEnvironment",
    "ControlHarmValidator",
    "ControlScenarioSpec",
    "ControlTerminalValidator",
    "ControlUtilityValidator",
    "EnvironmentError",
    "InvalidToolCallError",
    "PredicateDefinitionError",
    "PredicateEvaluation",
    "SnapshotRestoreError",
    "StatefulEnvironment",
    "TerminalValidation",
    "ToolDefinition",
    "ToolExecutionResult",
    "build_control_scenarios",
    "generate_control_instances",
    "generate_control_scenarios",
    "get_control_scenario",
    "make_agentdojo_bindings",
]
