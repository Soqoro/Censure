"""Outcome-free freezing and sampling-unit expansion for Experiment 1.

Manifest construction is intentionally unable to accept trajectory results.
It uses only experiment configuration, released benchmark metadata, rendered
attack material, and deterministic hashes.  Base scenarios are frozen first;
actor and guard-pair expansion happens only after the complete scenario catalog
exists, which keeps the distinction between 320 environments and the larger
paired-session matrix explicit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import Field, JsonValue, model_validator

from censure.environments.control import (
    CONTROL_DOMAINS,
    CONTROL_SCENARIO_VERSION_V1,
    CONTROL_STRATA,
    ControlScenarioSpec,
    generate_control_scenarios,
)
from censure.guards import GUARD_IMPLEMENTATION_VERSION, NEUTRAL_DENIAL_TEXT
from censure.schemas import (
    AuthorizationPolicy,
    EnvironmentLayer,
    FrozenModel,
    FrozenScenario,
    PairedSession,
    ScenarioSplit,
)
from censure.serialization import canonical_sha256, make_state_snapshot

MANIFEST_SCHEMA_VERSION = "censure.exp1-manifest.v1"
SESSION_KEY_VERSION = "censure.exp1-session-key.v1"
PROMPT_CONTRACT_VERSION = "censure.exp1-tool-agent-prompt.v1"
_SPLIT_ORDER = (
    ScenarioSplit.SMOKE,
    ScenarioSplit.DEVELOPMENT,
    ScenarioSplit.CONFIRMATORY,
)
_FORBIDDEN_OUTCOME_FIELDS = frozenset(
    {
        "attack_goal_achieved",
        "h_b",
        "h_star",
        "run_status",
        "terminal_harm",
        "user_utility",
    }
)


class ManifestError(ValueError):
    """The configured frozen matrix is unavailable or scientifically ambiguous."""


@dataclass(frozen=True, slots=True)
class AgentDojoCatalog:
    """Metadata-only candidate IDs for one pinned AgentDojo suite."""

    suite_name: str
    user_task_ids: tuple[str, ...]
    injection_task_ids: tuple[str, ...]


class AgentDojoScenarioSource(Protocol):
    """Narrow source interface; importantly, it exposes no outcome API."""

    package_version: str
    benchmark_version: str

    def catalog(self, suite_name: str) -> AgentDojoCatalog: ...

    def freeze_scenario(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None,
        *,
        attack_name: str | None,
    ) -> Any: ...


class ReleasedAgentDojoSource:
    """Lazy facade over the exact AgentDojo 0.1.35 adapter."""

    def __init__(self) -> None:
        from censure.adapters.agentdojo_v0135 import AgentDojoV0135Adapter

        self._adapter = AgentDojoV0135Adapter()
        self.package_version = self._adapter.package_version
        self.benchmark_version = self._adapter.benchmark_version
        report = self._adapter.compatibility_report()
        self._catalogs = {
            suite.name: AgentDojoCatalog(
                suite_name=suite.name,
                user_task_ids=suite.user_task_ids,
                injection_task_ids=suite.injection_task_ids,
            )
            for suite in report.suites
        }

    def catalog(self, suite_name: str) -> AgentDojoCatalog:
        try:
            return self._catalogs[suite_name]
        except KeyError as exc:
            raise ManifestError(f"AgentDojo suite is unavailable: {suite_name}") from exc

    def freeze_scenario(
        self,
        suite_name: str,
        user_task_id: str,
        injection_task_id: str | None,
        *,
        attack_name: str | None,
    ) -> Any:
        # The pinned adapter gives tool_knowledge a deterministic non-querying
        # GroundTruthPipeline with name='local'.  Do not substitute another
        # attack here: the configured primary attack is part of the manifest.
        return self._adapter.freeze_scenario(
            suite_name,
            user_task_id,
            injection_task_id,
            attack_name=attack_name,
        )


class ManifestSummary(FrozenModel):
    """Dry-run-safe counts with no realized labels."""

    scenario_count: int = Field(ge=0)
    paired_session_count: int = Field(ge=0)
    trajectory_count: int = Field(ge=0)
    scenarios_by_layer: dict[str, int]
    scenarios_by_split: dict[str, int]
    scenarios_by_suite_or_domain: dict[str, int]
    sessions_by_guard_pair: dict[str, int]
    sessions_by_actor: dict[str, int]


class ExperimentManifest(FrozenModel):
    """Complete deterministic manifest persisted before actor execution."""

    schema_version: Literal["censure.exp1-manifest.v1"] = MANIFEST_SCHEMA_VERSION
    experiment_id: str
    manifest_seed: int = Field(ge=0)
    resolved_config_sha256: str
    scenarios: tuple[FrozenScenario, ...]
    sessions: tuple[PairedSession, ...]
    scenario_set_sha256: str
    session_set_sha256: str
    summary: ManifestSummary

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def validate_projection(self) -> ExperimentManifest:
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        session_ids = [session.session_id for session in self.sessions]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("manifest scenario IDs must be unique")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("manifest session IDs must be unique")
        if canonical_sha256([item.model_dump(mode="json") for item in self.scenarios]) != (
            self.scenario_set_sha256
        ):
            raise ValueError("scenario-set hash does not match manifest scenarios")
        if canonical_sha256([item.model_dump(mode="json") for item in self.sessions]) != (
            self.session_set_sha256
        ):
            raise ValueError("session-set hash does not match manifest sessions")

        by_id = {scenario.scenario_id: scenario for scenario in self.scenarios}
        for session in self.sessions:
            scenario = by_id.get(session.scenario_id)
            if scenario is None:
                raise ValueError(f"session references absent scenario {session.scenario_id!r}")
            _validate_session_projection(session, scenario)
            if derive_session_id(session, scenario) != session.session_id:
                raise ValueError(f"session key is invalid for {session.session_id}")
        if self.summary.scenario_count != len(self.scenarios):
            raise ValueError("manifest summary scenario count is inconsistent")
        if self.summary.paired_session_count != len(self.sessions):
            raise ValueError("manifest summary session count is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class _AgentDojoCandidate:
    suite_name: str
    user_task_id: str
    injection_task_id: str | None
    split: ScenarioSplit
    environment_seed: int

    @property
    def identity(self) -> str:
        return f"{self.suite_name}|{self.user_task_id}|{self.injection_task_id or 'clean'}"


def dry_run_manifest_summary(
    config: Mapping[str, Any],
    *,
    agentdojo_source: AgentDojoScenarioSource | None = None,
) -> ManifestSummary:
    """Validate metadata availability and report exact planned counts.

    No AgentDojo scenario is rendered and no stateful environment is created.
    This is the implementation behind ``manifest --dry-run``.
    """

    source = agentdojo_source or ReleasedAgentDojoSource()
    _validate_source_versions(config, source)
    manifest_seed = _manifest_seed(config)
    agent_config = _mapping(config, "agentdojo")
    suites = tuple(str(value) for value in agent_config.get("suites", ()))
    target_per_suite = _positive_int(agent_config, "target_scenarios_per_suite")
    controls_per_suite = _nonnegative_int(agent_config, "controls_per_suite")
    if controls_per_suite > target_per_suite:
        raise ManifestError("AgentDojo controls_per_suite exceeds target_scenarios_per_suite")
    for suite in suites:
        catalog = source.catalog(suite)
        _validate_catalog_capacity(catalog, target_per_suite, controls_per_suite)

    agent_split_per_suite = _agentdojo_split_counts(config, target_per_suite)
    agent_count = len(suites) * target_per_suite
    agent_split_counts = {
        split.value: count * len(suites) for split, count in agent_split_per_suite.items()
    }

    control_specs, control_split_map = _selected_control_specs(config, manifest_seed)
    control_split_counts = Counter(control_split_map.values())
    scenario_count = agent_count + len(control_specs)
    scenarios_by_layer = {
        EnvironmentLayer.AGENTDOJO.value: agent_count,
        EnvironmentLayer.CONTROL.value: len(control_specs),
    }
    scenarios_by_split = {
        split.value: agent_split_counts.get(split.value, 0) + control_split_counts[split]
        for split in _SPLIT_ORDER
    }
    suite_counts = {suite: target_per_suite for suite in suites}
    suite_counts.update(Counter(spec.domain for spec in control_specs))
    actor_configs = _actor_configs(config)
    guard_counts = _planned_guard_pair_counts(config, scenarios_by_layer)
    sessions_by_guard = {
        guard_id: count * len(actor_configs) for guard_id, count in guard_counts.items()
    }
    paired_count = sum(sessions_by_guard.values())
    per_actor = sum(guard_counts.values())
    return ManifestSummary(
        scenario_count=scenario_count,
        paired_session_count=paired_count,
        trajectory_count=paired_count * 2,
        scenarios_by_layer=scenarios_by_layer,
        scenarios_by_split=scenarios_by_split,
        scenarios_by_suite_or_domain=dict(sorted(suite_counts.items())),
        sessions_by_guard_pair=sessions_by_guard,
        sessions_by_actor={actor.actor_id: per_actor for actor in actor_configs},
    )


def build_manifest(
    config: Mapping[str, Any],
    *,
    agentdojo_source: AgentDojoScenarioSource | None = None,
) -> ExperimentManifest:
    """Freeze all scenarios, then expand deterministic paired sessions."""

    source = agentdojo_source or ReleasedAgentDojoSource()
    manifest_seed = _manifest_seed(config)
    _validate_source_versions(config, source)
    config_sha256 = _resolved_config_hash(config)

    agent_candidates = _select_agentdojo_candidates(config, source, manifest_seed)
    scenarios = [
        _freeze_agentdojo_candidate(config, source, candidate) for candidate in agent_candidates
    ]
    control_specs, control_split_map = _selected_control_specs(config, manifest_seed)
    scenarios.extend(
        _freeze_control_scenario(spec, control_split_map[spec.scenario_id])
        for spec in control_specs
    )
    scenarios.sort(
        key=lambda item: (
            item.environment_layer.value,
            item.suite_or_domain,
            item.scenario_id,
        )
    )
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ManifestError("scenario selection produced duplicate scenario IDs")

    sessions = _expand_sessions(config, tuple(scenarios), manifest_seed)
    summary = _summary_from_frozen(tuple(scenarios), sessions)
    expected = dry_run_manifest_summary(config, agentdojo_source=source)
    if summary != expected:
        raise ManifestError(
            "frozen manifest counts differ from metadata-only dry run: "
            f"planned={expected.model_dump()}, frozen={summary.model_dump()}"
        )
    scenario_set_sha256 = canonical_sha256(
        [scenario.model_dump(mode="json") for scenario in scenarios]
    )
    session_set_sha256 = canonical_sha256([session.model_dump(mode="json") for session in sessions])
    manifest = ExperimentManifest(
        experiment_id=str(config.get("experiment_id", "exp1")),
        manifest_seed=manifest_seed,
        resolved_config_sha256=config_sha256,
        scenarios=tuple(scenarios),
        sessions=sessions,
        scenario_set_sha256=scenario_set_sha256,
        session_set_sha256=session_set_sha256,
        summary=summary,
    )
    assert_outcome_free(manifest)
    return manifest


# Entry-point-friendly descriptive alias.
freeze_manifest = build_manifest


def derive_session_id(
    session_or_fields: PairedSession | Mapping[str, Any],
    scenario: FrozenScenario,
) -> str:
    """Derive a globally unique key from all durable scientific material."""

    if isinstance(session_or_fields, PairedSession):
        fields = session_or_fields.model_dump(mode="json", exclude={"session_id"})
    else:
        fields = dict(session_or_fields)
        fields.pop("session_id", None)
    return canonical_sha256(
        {
            "session_key_version": SESSION_KEY_VERSION,
            "frozen_scenario_sha256": canonical_sha256(scenario.model_dump(mode="json")),
            "paired_session_fields": fields,
        }
    )


def _select_agentdojo_candidates(
    config: Mapping[str, Any],
    source: AgentDojoScenarioSource,
    manifest_seed: int,
) -> tuple[_AgentDojoCandidate, ...]:
    agent_config = _mapping(config, "agentdojo")
    suites = tuple(str(value) for value in agent_config.get("suites", ()))
    target_per_suite = _positive_int(agent_config, "target_scenarios_per_suite")
    controls_per_suite = _nonnegative_int(agent_config, "controls_per_suite")
    attacked_per_suite = target_per_suite - controls_per_suite
    split_counts = _agentdojo_split_counts(config, target_per_suite)
    selected: list[_AgentDojoCandidate] = []

    for suite_name in suites:
        catalog = source.catalog(suite_name)
        _validate_catalog_capacity(catalog, target_per_suite, controls_per_suite)
        clean_ids = sorted(
            catalog.user_task_ids,
            key=lambda user: _rank(manifest_seed, "agentdojo-clean", suite_name, user),
        )[:controls_per_suite]
        attacked_pairs = _balanced_attack_pairs(
            catalog,
            attacked_per_suite,
            manifest_seed=manifest_seed,
        )
        raw = [(user, None) for user in clean_ids] + list(attacked_pairs)
        split_by_pair = _assign_agentdojo_splits(
            raw,
            split_counts,
            manifest_seed=manifest_seed,
            suite_name=suite_name,
        )
        for user_task_id, injection_task_id in raw:
            identity = f"{suite_name}|{user_task_id}|{injection_task_id or 'clean'}"
            selected.append(
                _AgentDojoCandidate(
                    suite_name=suite_name,
                    user_task_id=user_task_id,
                    injection_task_id=injection_task_id,
                    split=split_by_pair[(user_task_id, injection_task_id)],
                    environment_seed=_seed32(manifest_seed, "environment", identity),
                )
            )
    identities = [candidate.identity for candidate in selected]
    if len(identities) != len(set(identities)):
        raise ManifestError("AgentDojo selection duplicated a user/injection task pair")
    return tuple(
        sorted(selected, key=lambda item: (item.suite_name, item.user_task_id, item.identity))
    )


def _balanced_attack_pairs(
    catalog: AgentDojoCatalog,
    count: int,
    *,
    manifest_seed: int,
) -> tuple[tuple[str, str], ...]:
    if count == 0:
        return ()
    candidates = {
        (user, injection)
        for user in catalog.user_task_ids
        for injection in catalog.injection_task_ids
    }
    if count > len(candidates):
        raise ManifestError(
            f"suite {catalog.suite_name} has only {len(candidates)} unique attacked pairs, "
            f"cannot select {count}"
        )
    user_counts: Counter[str] = Counter()
    injection_counts: Counter[str] = Counter()
    selected: list[tuple[str, str]] = []
    while len(selected) < count:
        pair = min(
            candidates,
            key=lambda item: (
                injection_counts[item[1]],
                user_counts[item[0]],
                _rank(
                    manifest_seed,
                    "agentdojo-attacked",
                    catalog.suite_name,
                    item[0],
                    item[1],
                ),
            ),
        )
        candidates.remove(pair)
        selected.append(pair)
        user_counts[pair[0]] += 1
        injection_counts[pair[1]] += 1
    return tuple(selected)


def _assign_agentdojo_splits(
    pairs: Sequence[tuple[str, str | None]],
    split_counts: Mapping[ScenarioSplit, int],
    *,
    manifest_seed: int,
    suite_name: str,
) -> dict[tuple[str, str | None], ScenarioSplit]:
    attacked = [pair for pair in pairs if pair[1] is not None]
    clean = [pair for pair in pairs if pair[1] is None]
    attack_allocations = _category_split_allocation(len(attacked), split_counts)
    clean_allocations = {
        split: split_counts[split] - attack_allocations[split] for split in _SPLIT_ORDER
    }
    attacked.sort(
        key=lambda item: _rank(
            manifest_seed, "agentdojo-attacked-split", suite_name, item[0], cast(str, item[1])
        )
    )
    clean.sort(key=lambda item: _rank(manifest_seed, "agentdojo-clean-split", suite_name, item[0]))
    result: dict[tuple[str, str | None], ScenarioSplit] = {}
    attack_offset = 0
    clean_offset = 0
    for split in _SPLIT_ORDER:
        attack_count = attack_allocations[split]
        clean_count = clean_allocations[split]
        for pair in attacked[attack_offset : attack_offset + attack_count]:
            result[pair] = split
        for pair in clean[clean_offset : clean_offset + clean_count]:
            result[pair] = split
        attack_offset += attack_count
        clean_offset += clean_count
    if len(result) != len(pairs):
        raise AssertionError("AgentDojo split allocation lost candidate pairs")
    return result


def _category_split_allocation(
    category_count: int,
    split_counts: Mapping[ScenarioSplit, int],
) -> dict[ScenarioSplit, int]:
    total = sum(split_counts.values())
    if total == 0:
        if category_count:
            raise ManifestError("cannot allocate categories into empty splits")
        return dict.fromkeys(_SPLIT_ORDER, 0)
    ideals = {split: split_counts[split] * category_count / total for split in _SPLIT_ORDER}
    allocated = {split: min(split_counts[split], int(ideals[split])) for split in _SPLIT_ORDER}
    remaining = category_count - sum(allocated.values())
    while remaining:
        eligible = [split for split in _SPLIT_ORDER if allocated[split] < split_counts[split]]
        if not eligible:
            raise ManifestError("split capacities cannot hold requested category count")
        selected = max(
            eligible,
            key=lambda split: (ideals[split] - allocated[split], -_SPLIT_ORDER.index(split)),
        )
        allocated[selected] += 1
        remaining -= 1
    return allocated


def _freeze_agentdojo_candidate(
    config: Mapping[str, Any],
    source: AgentDojoScenarioSource,
    candidate: _AgentDojoCandidate,
) -> FrozenScenario:
    agent_config = _mapping(config, "agentdojo")
    primary_attack = str(agent_config.get("primary_attack", ""))
    attacked = candidate.injection_task_id is not None
    frozen = source.freeze_scenario(
        candidate.suite_name,
        candidate.user_task_id,
        candidate.injection_task_id,
        attack_name=primary_attack if attacked else None,
    )
    required = (
        "available_tools",
        "authorization_policy",
        "benchmark_version",
        "initial_state",
        "package_version",
        "rendered_injections",
        "user_request",
    )
    missing = [name for name in required if not hasattr(frozen, name)]
    if missing:
        raise ManifestError(
            "AgentDojo frozen scenario lacks indispensable fields: " + ", ".join(missing)
        )
    if str(frozen.package_version) != str(agent_config.get("package_version")):
        raise ManifestError("frozen AgentDojo package version differs from experiment config")
    if str(frozen.benchmark_version) != str(agent_config.get("benchmark_version")):
        raise ManifestError("frozen AgentDojo benchmark version differs from experiment config")
    if frozen.suite_name != candidate.suite_name:
        raise ManifestError("AgentDojo source returned a scenario from a different suite")
    if frozen.user_task_id != candidate.user_task_id:
        raise ManifestError("AgentDojo source returned a different user task")
    if frozen.injection_task_id != candidate.injection_task_id:
        raise ManifestError("AgentDojo source returned a different injection task")

    rendered_attack = dict(frozen.rendered_injections)
    attack_hash = canonical_sha256(rendered_attack)
    declared_attack_hash = getattr(frozen, "rendered_attack_sha256", None)
    if attacked and declared_attack_hash != attack_hash:
        raise ManifestError("AgentDojo rendered attack hash does not match frozen payload")
    locations = tuple(str(value) for value in frozen.injection_locations)
    if locations != tuple(rendered_attack):
        raise ManifestError("AgentDojo injection-location ordering was not preserved")
    initial = make_state_snapshot(frozen.initial_state.state)
    if initial.sha256 != frozen.initial_state.sha256:
        raise ManifestError("AgentDojo initial-state hash changed at manifest boundary")
    policy = frozen.authorization_policy
    if not isinstance(policy, AuthorizationPolicy):
        raise ManifestError("AgentDojo adapter did not provide an AuthorizationPolicy")
    available_tools = tuple(_tool_schema_dict(tool) for tool in frozen.available_tools)
    if not available_tools:
        raise ManifestError("AgentDojo adapter returned no available tool schemas")
    if not hasattr(frozen, "model_dump"):
        raise ManifestError("AgentDojo source did not provide a durable runtime_spec")
    runtime_spec = frozen.model_dump(mode="json")
    if not isinstance(runtime_spec, dict):
        raise ManifestError("AgentDojo runtime_spec must serialize to an object")
    scenario_id = (
        f"agentdojo:{candidate.suite_name}:{candidate.user_task_id}:"
        f"{candidate.injection_task_id or 'clean'}:{candidate.environment_seed}"
    )
    metadata: dict[str, JsonValue] = {
        "adapter_version": str(getattr(frozen, "adapter_version", "unknown")),
        "attack_name": getattr(frozen, "attack_name", None),
        "attack_target_pipeline_name": getattr(frozen, "attack_target_pipeline_name", None),
        "source_state_serialization_version": str(frozen.initial_state.serialization_version),
        "suite_benchmark_version": list(frozen.suite_benchmark_version),
        "runtime_spec": cast(JsonValue, runtime_spec),
    }
    return FrozenScenario(
        scenario_id=scenario_id,
        environment_layer=EnvironmentLayer.AGENTDOJO,
        suite_or_domain=candidate.suite_name,
        user_task_id=candidate.user_task_id,
        injection_task_id=candidate.injection_task_id,
        rendered_attack_id=(
            f"{getattr(frozen, 'attack_name', primary_attack)}:{attack_hash}" if attacked else None
        ),
        rendered_attack=rendered_attack,
        rendered_attack_sha256=attack_hash,
        injection_locations=locations,
        canonical_initial_state=initial,
        user_request=str(frozen.user_request),
        available_tools=available_tools,
        untrusted_content=tuple(rendered_attack.values()),
        policy=policy,
        policy_sha256=canonical_sha256(policy),
        environment_seed=candidate.environment_seed,
        split=candidate.split,
        agentdojo_package_version=str(frozen.package_version),
        agentdojo_benchmark_version=str(frozen.benchmark_version),
        metadata=metadata,
    )


def _tool_schema_dict(tool: Any) -> dict[str, JsonValue]:
    if hasattr(tool, "model_dump"):
        value = tool.model_dump(mode="json")
    elif isinstance(tool, Mapping):
        value = dict(tool)
    else:
        raise ManifestError(f"unsupported AgentDojo tool schema type: {type(tool).__name__}")
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ManifestError("AgentDojo tool schema lacks a name")
    if not isinstance(value.get("parameters"), dict):
        raise ManifestError(f"AgentDojo tool {value.get('name')!r} lacks a JSON parameter schema")
    return cast(dict[str, JsonValue], value)


def _selected_control_specs(
    config: Mapping[str, Any],
    manifest_seed: int,
) -> tuple[tuple[ControlScenarioSpec, ...], dict[str, ScenarioSplit]]:
    control_config = _mapping(config, "controlled")
    if not bool(control_config.get("enabled", True)):
        return (), {}
    domains = tuple(str(value) for value in control_config.get("domains", CONTROL_DOMAINS))
    strata = tuple(str(value) for value in control_config.get("strata", CONTROL_STRATA))
    scenario_version = str(control_config.get("scenario_version", CONTROL_SCENARIO_VERSION_V1))
    seeds_per_cell = _positive_int(control_config, "seeds_per_cell")
    if seeds_per_cell > 10:
        raise ManifestError("CENSURE-Control supports exactly ten deterministic seeds per cell")
    try:
        specs = generate_control_scenarios(
            domains=cast(Any, domains),
            strata=cast(Any, strata),
            seeds=tuple(range(seeds_per_cell)),
            scenario_version=scenario_version,
        )
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    split_config = config.get("splits", {})
    if not isinstance(split_config, Mapping):
        raise ManifestError("splits config must be a mapping")
    explicit = split_config.get("controlled")
    if isinstance(explicit, Mapping):
        split_by_seed: dict[int, ScenarioSplit] = {}
        keys = {
            ScenarioSplit.SMOKE: "smoke_seeds",
            ScenarioSplit.DEVELOPMENT: "development_seeds",
            ScenarioSplit.CONFIRMATORY: "confirmatory_seeds",
        }
        for split, key in keys.items():
            values = explicit.get(key, ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ManifestError(f"controlled split {key} must be a sequence")
            for raw_seed in values:
                seed = int(raw_seed)
                if seed in split_by_seed:
                    raise ManifestError(f"control seed {seed} appears in multiple splits")
                split_by_seed[seed] = split
        missing = set(range(seeds_per_cell)) - set(split_by_seed)
        if missing:
            raise ManifestError(f"selected control seeds have no frozen split: {sorted(missing)}")
        return specs, {spec.scenario_id: split_by_seed[spec.seed] for spec in specs}

    ratios = _split_ratios(split_config)
    assignments: dict[str, ScenarioSplit] = {}
    by_domain: dict[str, list[ControlScenarioSpec]] = defaultdict(list)
    for spec in specs:
        by_domain[spec.domain].append(spec)
    for domain, domain_specs in sorted(by_domain.items()):
        counts = _ratio_counts(len(domain_specs), ratios)
        ranked = sorted(
            domain_specs,
            key=lambda spec: _rank(manifest_seed, "control-split", domain, spec.scenario_id),
        )
        offset = 0
        for split in _SPLIT_ORDER:
            for spec in ranked[offset : offset + counts[split]]:
                assignments[spec.scenario_id] = split
            offset += counts[split]
    return specs, assignments


def _freeze_control_scenario(
    spec: ControlScenarioSpec,
    split: ScenarioSplit,
) -> FrozenScenario:
    rendered_attack = {
        str(payload["location"]): str(payload["content"]) for payload in spec.untrusted_content
    }
    attack_hash = canonical_sha256(rendered_attack)
    policy = spec.authorization_policy
    metadata: dict[str, JsonValue] = {
        "control_scenario_version": spec.scenario_version,
        "control_spec_sha256": spec.spec_sha256,
        "runtime_spec": cast(JsonValue, spec.to_dict()),
        "stratum": spec.stratum,
        "terminal_harm_predicate": cast(JsonValue, dict(spec.terminal_harm_predicate)),
        "unsafe_attempt_predicate": cast(JsonValue, dict(spec.unsafe_attempt_predicate)),
        "utility_predicate": cast(JsonValue, dict(spec.utility_predicate)),
    }
    return FrozenScenario(
        scenario_id=spec.scenario_id,
        environment_layer=EnvironmentLayer.CONTROL,
        suite_or_domain=spec.domain,
        user_task_id=spec.user_task_id,
        injection_task_id=(
            f"control-injection:{spec.domain}:{spec.stratum}" if rendered_attack else None
        ),
        rendered_attack_id=(f"control-rendered:{attack_hash}" if rendered_attack else None),
        rendered_attack=rendered_attack,
        rendered_attack_sha256=attack_hash,
        injection_locations=tuple(rendered_attack),
        canonical_initial_state=make_state_snapshot(spec.canonical_initial_state),
        user_request=spec.user_request,
        available_tools=tuple(
            cast(dict[str, JsonValue], tool.to_dict()) for tool in spec.available_tools
        ),
        untrusted_content=tuple(rendered_attack.values()),
        policy=policy,
        policy_sha256=canonical_sha256(policy),
        environment_seed=spec.seed,
        split=split,
        metadata=metadata,
    )


def _expand_sessions(
    config: Mapping[str, Any],
    scenarios: tuple[FrozenScenario, ...],
    manifest_seed: int,
) -> tuple[PairedSession, ...]:
    actors = _actor_configs(config)
    guard_pairs = config.get("guard_pairs")
    if not isinstance(guard_pairs, Sequence) or isinstance(guard_pairs, (str, bytes)):
        raise ManifestError("guard_pairs must be a sequence")
    subset_cache: dict[tuple[str, int | None, int | None], tuple[FrozenScenario, ...]] = {}
    sessions: list[PairedSession] = []
    for raw_pair in guard_pairs:
        if not isinstance(raw_pair, Mapping):
            raise ManifestError("guard-pair entries must be mappings")
        pair_id = _required_string(raw_pair, "id")
        behavior_guard = _required_string(raw_pair, "behavior")
        target_guard = _required_string(raw_pair, "target")
        scope = str(raw_pair.get("scope", "all"))
        max_total = _optional_positive_int(raw_pair.get("max_total"), "max_total")
        max_per_layer = _optional_positive_int(raw_pair.get("max_per_layer"), "max_per_layer")
        cache_key = (scope, max_total, max_per_layer)
        selected = subset_cache.get(cache_key)
        if selected is None:
            selected = _scenarios_for_scope(
                scenarios,
                scope=scope,
                max_total=max_total,
                max_per_layer=max_per_layer,
                manifest_seed=manifest_seed,
            )
            subset_cache[cache_key] = selected
        for scenario in selected:
            for actor in actors:
                fields = _paired_session_fields(
                    config,
                    scenario,
                    actor,
                    pair_id=pair_id,
                    behavior_guard=behavior_guard,
                    target_guard=target_guard,
                    manifest_seed=manifest_seed,
                )
                session_id = derive_session_id(fields, scenario)
                sessions.append(PairedSession(session_id=session_id, **fields))
    sessions.sort(key=lambda item: (item.session_id, item.guard_pair_id, item.actor_id))
    if len({session.session_id for session in sessions}) != len(sessions):
        raise ManifestError("actor/guard expansion produced duplicate session keys")
    return tuple(sessions)


@dataclass(frozen=True, slots=True)
class _ActorConfig:
    alias: str
    actor_id: str
    actor_revision: str
    tokenizer_revision: str
    generation_sha256: str
    chat_template_sha256: str
    prompt_sha256: str


def _actor_configs(config: Mapping[str, Any]) -> tuple[_ActorConfig, ...]:
    actors = config.get("actors")
    resolved = config.get("resolved_models")
    if not isinstance(actors, Sequence) or isinstance(actors, (str, bytes)) or not actors:
        raise ManifestError("experiment config must select at least one actor")
    if not isinstance(resolved, Mapping):
        raise ManifestError("manifest requires resolved_models with immutable revisions")
    results: list[_ActorConfig] = []
    for raw_alias in actors:
        alias = str(raw_alias)
        model = resolved.get(alias)
        if not isinstance(model, Mapping):
            raise ManifestError(f"resolved model config is missing for {alias}")
        model_id = _required_string(model, "model_id")
        revision = _frozen_revision(model, "model_revision", alias)
        tokenizer_revision = _frozen_revision(model, "tokenizer_revision", alias)
        chat_template_sha256 = str(model.get("chat_template_sha256", ""))
        if len(chat_template_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in chat_template_sha256
        ):
            raise ManifestError(f"{alias} chat_template_sha256 is not a lowercase SHA-256")
        scientific_model_config = {
            key: model.get(key)
            for key in (
                "actor_id",
                "backend",
                "chat_template_args",
                "device",
                "dtype",
                "generation",
                "limits",
                "model_id",
                "model_revision",
                "quantization",
                "schema_version",
                "thinking_mode",
                "tokenizer_revision",
                "trust_remote_code",
            )
        }
        extension_model_keys = (
            "checkpoint_load_mode",
            "history_projection",
            "model_loader",
            "native_tools",
            "native_weight_format",
            "required_package_versions",
            "response_parser_version",
            "serializer_fingerprint_sha256",
            "template_current_date",
            "tokenizer_asset_sha256",
            "tokenizer_backend",
            "tool_name_projection",
            "tool_protocol",
        )
        scientific_model_config.update(
            {key: model[key] for key in extension_model_keys if key in model}
        )
        if "tool_protocol" in model and "prompt_format_version" in model:
            scientific_model_config["prompt_format_version"] = model["prompt_format_version"]
        generation_sha256 = canonical_sha256(scientific_model_config)
        prompt_config = {
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "model_id": model_id,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_sha256": chat_template_sha256,
            "thinking_mode": model.get("thinking_mode"),
            "chat_template_args": model.get("chat_template_args", {}),
        }
        prompt_config.update({key: model[key] for key in extension_model_keys if key in model})
        if "tool_protocol" in model and "prompt_format_version" in model:
            prompt_config["prompt_format_version"] = model["prompt_format_version"]
        prompt_sha256 = canonical_sha256(prompt_config)
        results.append(
            _ActorConfig(
                alias=alias,
                actor_id=model_id,
                actor_revision=revision,
                tokenizer_revision=tokenizer_revision,
                generation_sha256=generation_sha256,
                chat_template_sha256=chat_template_sha256,
                prompt_sha256=prompt_sha256,
            )
        )
    return tuple(results)


def _paired_session_fields(
    config: Mapping[str, Any],
    scenario: FrozenScenario,
    actor: _ActorConfig,
    *,
    pair_id: str,
    behavior_guard: str,
    target_guard: str,
    manifest_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "censure.paired-session.v1",
        "scenario_id": scenario.scenario_id,
        "environment_layer": scenario.environment_layer,
        "suite_or_domain": scenario.suite_or_domain,
        "user_task_id": scenario.user_task_id,
        "injection_task_id": scenario.injection_task_id,
        "rendered_attack_id": scenario.rendered_attack_id,
        "rendered_attack_sha256": scenario.rendered_attack_sha256,
        "initial_state_sha256": scenario.canonical_initial_state.sha256,
        "policy_sha256": scenario.policy_sha256,
        "actor_id": actor.actor_id,
        "actor_revision": actor.actor_revision,
        "tokenizer_revision": actor.tokenizer_revision,
        "decoding_seed": _seed32(manifest_seed, "decoding", scenario.scenario_id, actor.actor_id),
        "environment_seed": scenario.environment_seed,
        "behavior_guard_id": behavior_guard,
        "target_guard_id": target_guard,
        "behavior_guard_config_sha256": _guard_configuration_hash(behavior_guard),
        "target_guard_config_sha256": _guard_configuration_hash(target_guard),
        "generation_config_sha256": actor.generation_sha256,
        "chat_template_sha256": actor.chat_template_sha256,
        "prompt_chat_template_sha256": actor.prompt_sha256,
        "state_serialization_version": str(
            config.get(
                "state_serialization_version",
                scenario.canonical_initial_state.serialization_version,
            )
        ),
        "split": scenario.split,
        "guard_pair_id": pair_id,
        "agentdojo_package_version": scenario.agentdojo_package_version,
        "agentdojo_benchmark_version": scenario.agentdojo_benchmark_version,
    }


def _scenarios_for_scope(
    scenarios: tuple[FrozenScenario, ...],
    *,
    scope: str,
    max_total: int | None,
    max_per_layer: int | None,
    manifest_seed: int,
) -> tuple[FrozenScenario, ...]:
    if scope == "all":
        if max_total is not None or max_per_layer is not None:
            raise ManifestError("scope=all cannot also specify a subset limit")
        return scenarios
    if scope not in {"degradation_subset", "negative_control"}:
        raise ManifestError(f"unknown guard-pair scope: {scope}")
    if max_total is not None and max_per_layer is not None:
        raise ManifestError("guard-pair subset cannot set both max_total and max_per_layer")
    label = scope
    if max_per_layer is not None:
        selected: list[FrozenScenario] = []
        by_layer: dict[EnvironmentLayer, list[FrozenScenario]] = defaultdict(list)
        for scenario in scenarios:
            by_layer[scenario.environment_layer].append(scenario)
        for layer in sorted(by_layer, key=lambda item: item.value):
            selected.extend(
                _balanced_subset(
                    by_layer[layer],
                    max_per_layer,
                    manifest_seed=manifest_seed,
                    label=f"{label}:{layer.value}",
                )
            )
        return tuple(sorted(selected, key=lambda item: item.scenario_id))
    default_total = 80 if scope == "degradation_subset" else 32
    return _balanced_subset(
        scenarios,
        max_total if max_total is not None else default_total,
        manifest_seed=manifest_seed,
        label=label,
    )


def _balanced_subset(
    scenarios: Sequence[FrozenScenario],
    total: int,
    *,
    manifest_seed: int,
    label: str,
) -> tuple[FrozenScenario, ...]:
    if total > len(scenarios):
        raise ManifestError(f"subset {label} requests {total} from only {len(scenarios)} scenarios")
    groups: dict[tuple[str, str], list[FrozenScenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[(scenario.environment_layer.value, scenario.suite_or_domain)].append(scenario)
    for key, values in groups.items():
        values.sort(
            key=lambda item: _rank(manifest_seed, "guard-subset", label, *key, item.scenario_id)
        )
    ordered_groups = sorted(groups)
    selected: list[FrozenScenario] = []
    offsets = Counter[tuple[str, str]]()
    while len(selected) < total:
        made_progress = False
        for key in ordered_groups:
            offset = offsets[key]
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                offsets[key] += 1
                made_progress = True
                if len(selected) == total:
                    break
        if not made_progress:  # pragma: no cover - guarded by total capacity.
            raise AssertionError("balanced subset exhausted unexpectedly")
    return tuple(sorted(selected, key=lambda item: item.scenario_id))


def _guard_configuration_hash(guard_id: str) -> str:
    kind = guard_id
    parameters: dict[str, JsonValue] = {}
    if guard_id.startswith("degraded_strict:"):
        kind, _, raw_rho = guard_id.partition(":")
        try:
            rho = float(raw_rho)
        except ValueError as exc:
            raise ManifestError(f"invalid degraded guard ID: {guard_id}") from exc
        if rho not in {0.25, 0.5, 0.75, 1.0}:
            raise ManifestError(f"unsupported degraded strict rho: {rho}")
        parameters["rho"] = rho
    elif guard_id not in {"strict", "weak", "none", "same_guard"}:
        raise ManifestError(f"unknown guard ID: {guard_id}")
    return canonical_sha256(
        {
            "implementation_version": GUARD_IMPLEMENTATION_VERSION,
            "guard_id": guard_id,
            "kind": kind,
            "neutral_denial_text": NEUTRAL_DENIAL_TEXT,
            **parameters,
        }
    )


def _summary_from_frozen(
    scenarios: tuple[FrozenScenario, ...],
    sessions: tuple[PairedSession, ...],
) -> ManifestSummary:
    split_counts = Counter(item.split.value for item in scenarios)
    layer_counts = Counter(item.environment_layer.value for item in scenarios)
    return ManifestSummary(
        scenario_count=len(scenarios),
        paired_session_count=len(sessions),
        trajectory_count=len(sessions) * 2,
        scenarios_by_layer={layer.value: layer_counts[layer.value] for layer in EnvironmentLayer},
        scenarios_by_split={split.value: split_counts[split.value] for split in _SPLIT_ORDER},
        scenarios_by_suite_or_domain=dict(
            sorted(Counter(item.suite_or_domain for item in scenarios).items())
        ),
        sessions_by_guard_pair=dict(
            sorted(Counter(item.guard_pair_id for item in sessions).items())
        ),
        sessions_by_actor=dict(sorted(Counter(item.actor_id for item in sessions).items())),
    )


def _planned_guard_pair_counts(
    config: Mapping[str, Any],
    scenarios_by_layer: Mapping[str, int],
) -> dict[str, int]:
    guard_pairs = config.get("guard_pairs")
    if not isinstance(guard_pairs, Sequence) or isinstance(guard_pairs, (str, bytes)):
        raise ManifestError("guard_pairs must be a sequence")
    total = sum(scenarios_by_layer.values())
    result: dict[str, int] = {}
    for raw_pair in guard_pairs:
        if not isinstance(raw_pair, Mapping):
            raise ManifestError("guard-pair entries must be mappings")
        pair_id = _required_string(raw_pair, "id")
        if pair_id in result:
            raise ManifestError(f"duplicate guard-pair ID: {pair_id}")
        _guard_configuration_hash(_required_string(raw_pair, "behavior"))
        _guard_configuration_hash(_required_string(raw_pair, "target"))
        scope = str(raw_pair.get("scope", "all"))
        if scope == "all":
            count = total
        elif scope in {"degradation_subset", "negative_control"}:
            max_total = _optional_positive_int(raw_pair.get("max_total"), "max_total")
            max_per_layer = _optional_positive_int(raw_pair.get("max_per_layer"), "max_per_layer")
            if max_total is not None and max_per_layer is not None:
                raise ManifestError("guard-pair subset cannot set both subset limits")
            if max_per_layer is not None:
                active_layers = sum(count > 0 for count in scenarios_by_layer.values())
                count = max_per_layer * active_layers
            else:
                count = max_total or (80 if scope == "degradation_subset" else 32)
            if count > total:
                raise ManifestError(f"guard-pair {pair_id} subset exceeds scenario count")
        else:
            raise ManifestError(f"unknown guard-pair scope: {scope}")
        result[pair_id] = count
    return result


def _agentdojo_split_counts(
    config: Mapping[str, Any], target_per_suite: int
) -> dict[ScenarioSplit, int]:
    split_config = config.get("splits", {})
    if not isinstance(split_config, Mapping):
        raise ManifestError("splits config must be a mapping")
    explicit = split_config.get("agentdojo")
    if isinstance(explicit, Mapping):
        mapping = {
            ScenarioSplit.SMOKE: _nonnegative_int(explicit, "smoke_per_suite"),
            ScenarioSplit.DEVELOPMENT: _nonnegative_int(explicit, "development_per_suite"),
            ScenarioSplit.CONFIRMATORY: _nonnegative_int(explicit, "confirmatory_per_suite"),
        }
        if sum(mapping.values()) != target_per_suite:
            raise ManifestError("AgentDojo per-suite split counts do not sum to target")
        return mapping
    return _ratio_counts(target_per_suite, _split_ratios(split_config))


def _split_ratios(config: Mapping[str, Any]) -> dict[ScenarioSplit, float]:
    ratios = {
        ScenarioSplit.SMOKE: float(config.get("smoke", 0.0)),
        ScenarioSplit.DEVELOPMENT: float(config.get("development", 0.0)),
        ScenarioSplit.CONFIRMATORY: float(config.get("confirmatory", 0.0)),
    }
    if any(value < 0 for value in ratios.values()) or sum(ratios.values()) <= 0:
        raise ManifestError("split ratios must be nonnegative and sum to a positive value")
    return ratios


def _ratio_counts(total: int, ratios: Mapping[ScenarioSplit, float]) -> dict[ScenarioSplit, int]:
    ratio_total = sum(ratios.values())
    ideals = {split: total * ratios[split] / ratio_total for split in _SPLIT_ORDER}
    counts = {split: int(ideals[split]) for split in _SPLIT_ORDER}
    for split in sorted(
        _SPLIT_ORDER,
        key=lambda item: (-(ideals[item] - counts[item]), _SPLIT_ORDER.index(item)),
    )[: total - sum(counts.values())]:
        counts[split] += 1
    return counts


def _validate_catalog_capacity(
    catalog: AgentDojoCatalog,
    target_per_suite: int,
    controls_per_suite: int,
) -> None:
    if controls_per_suite > len(set(catalog.user_task_ids)):
        raise ManifestError(f"suite {catalog.suite_name} has too few clean user tasks")
    attacked = target_per_suite - controls_per_suite
    capacity = len(set(catalog.user_task_ids)) * len(set(catalog.injection_task_ids))
    if attacked > capacity:
        raise ManifestError(f"suite {catalog.suite_name} has too few unique attacked pairs")
    if len(catalog.user_task_ids) != len(set(catalog.user_task_ids)):
        raise ManifestError(f"suite {catalog.suite_name} catalog duplicates user task IDs")
    if len(catalog.injection_task_ids) != len(set(catalog.injection_task_ids)):
        raise ManifestError(f"suite {catalog.suite_name} catalog duplicates injection task IDs")


def _validate_source_versions(config: Mapping[str, Any], source: AgentDojoScenarioSource) -> None:
    agent_config = _mapping(config, "agentdojo")
    if str(agent_config.get("package_version")) != str(source.package_version):
        raise ManifestError("AgentDojo source package version differs from config")
    if str(agent_config.get("benchmark_version")) != str(source.benchmark_version):
        raise ManifestError("AgentDojo source benchmark version differs from config")


def _validate_session_projection(session: PairedSession, scenario: FrozenScenario) -> None:
    expected = {
        "environment_layer": scenario.environment_layer,
        "suite_or_domain": scenario.suite_or_domain,
        "user_task_id": scenario.user_task_id,
        "injection_task_id": scenario.injection_task_id,
        "rendered_attack_id": scenario.rendered_attack_id,
        "rendered_attack_sha256": scenario.rendered_attack_sha256,
        "initial_state_sha256": scenario.canonical_initial_state.sha256,
        "policy_sha256": scenario.policy_sha256,
        "environment_seed": scenario.environment_seed,
        "split": scenario.split,
        "agentdojo_package_version": scenario.agentdojo_package_version,
        "agentdojo_benchmark_version": scenario.agentdojo_benchmark_version,
    }
    for field_name, expected_value in expected.items():
        if getattr(session, field_name) != expected_value:
            raise ValueError(f"session {field_name} differs from frozen scenario")


def _resolved_config_hash(config: Mapping[str, Any]) -> str:
    value = {key: item for key, item in config.items() if key != "resolved_config_hash"}
    return canonical_sha256(value)


def _manifest_seed(config: Mapping[str, Any]) -> int:
    value = config.get("manifest_seed")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError("manifest_seed must be a nonnegative integer")
    return value


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ManifestError(f"{key} config must be a mapping")
    return value


def _required_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{key} must be a nonnegative integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _frozen_revision(model: Mapping[str, Any], key: str, alias: str) -> str:
    value = model.get(key)
    if not isinstance(value, str) or value == "resolve_at_doctor" or len(value) != 40:
        raise ManifestError(f"{alias} {key} is not a frozen 40-character revision")
    return value


def _rank(seed: int, *parts: str) -> str:
    return canonical_sha256({"manifest_seed": seed, "parts": list(parts)})


def _seed32(seed: int, *parts: str) -> int:
    return int(_rank(seed, *parts)[:8], 16)


def assert_outcome_free(value: Any) -> None:
    """Defensive test/helper: reject realized outcome fields recursively."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        forbidden = {str(key).lower() for key in value} & _FORBIDDEN_OUTCOME_FIELDS
        if forbidden:
            raise ManifestError(
                f"realized outcome fields are forbidden in manifests: {sorted(forbidden)}"
            )
        for item in value.values():
            assert_outcome_free(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            assert_outcome_free(item)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SESSION_KEY_VERSION",
    "AgentDojoCatalog",
    "AgentDojoScenarioSource",
    "ExperimentManifest",
    "ManifestError",
    "ManifestSummary",
    "ReleasedAgentDojoSource",
    "assert_outcome_free",
    "build_manifest",
    "derive_session_id",
    "dry_run_manifest_summary",
    "freeze_manifest",
]
