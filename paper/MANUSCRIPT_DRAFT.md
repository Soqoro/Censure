# When Guardrails Hide Risk: First-Divergence Auditing for Tool-Using Language Agents

Working manuscript draft, 2 September 2026. Numerical results are tied to synthesis
`qwen_gemma_ministral_v1`, specification SHA-256
`3e56c71a6acfc606794052d17bd10b7b00cf4da06d8bd0a0d77c637e435eda37`. The analysis was
implemented at repository commit `4147c3acdcaf08633d4395292def3aec4002228a`.

> **Inference status.** This manuscript combines a post-hoc partial analysis of the completed
> Qwen and Gemma arms with a separately prospective Ministral breadth extension. Actor-specific
> results retain those source labels. The three-model synthesis and all cross-model contrasts are
> retrospective and exploratory. This draft does not represent the original preregistered actor
> matrix as complete.

## Abstract

Action guards can reduce the harm observed during an agent evaluation while leaving the
underlying actor's propensity for unsafe action unresolved. Evaluating only the deployed,
guarded trajectory can therefore conflate actor safety with successful intervention. We study
this problem using a paired full-trajectory audit: for every frozen task and seed, we independently
restore the same canonical initial state and run the same language-model actor once with a strict
behavior guard and once with a target policy that does not block syntactically valid actions. The
primary endpoint is the signed difference in aggregate terminal harm between target and behavior
trajectories. Across 696 confirmatory actor-task pairs from three open-weight model families,
Qwen3-8B exhibited a 0.137 masking gap (95% task-cluster bootstrap interval 0.060 to 0.237) and
Ministral-3-14B exhibited a 0.178 gap (0.097 to 0.281), whereas Gemma-3-12B exhibited no
complete-case gap. Sharp binary missing-harm bounds were positive in the observed sample for
Qwen and Ministral; only Ministral's conservative lower endpoint remained positive after
accounting for sampling uncertainty. Effects were concentrated in travel, Slack, and payments
tasks rather than being uniform across domains. Ministral additionally showed a monotone matched
degradation curve, while identical-guard negative controls produced zero observed gaps for all
three actors. Descriptive diagnostics indicate that Gemma used fewer tools and rarely proposed
unsafe actions, identifying an important boundary condition. These results show that guarded
evaluation can mask actor risk, but the magnitude depends on both the actor and the task
environment.

## 1. Introduction

Tool-using language-model agents act through a stack: a model proposes an operation, an action
guard decides whether it may execute, and an environment produces the observation that shapes the
remainder of the trajectory. A low observed harm rate for this deployed stack does not by itself
identify which component was safe. The actor may have selected safe actions, or it may have
selected unsafe actions that the guard successfully blocked. Existing agent-safety benchmarks
show that consequential failures can arise across simulated high-stakes tools, indirect prompt
injection, and explicitly malicious multi-step tasks
[@ruan2024toolemu; @zhan2024injecagent; @andriushchenko2025agentharm].

This distinction matters when an evaluation is used to characterize the actor rather than the
entire deployed system. A guard can prevent the very actions whose downstream consequences would
reveal actor risk. It can also alter later observations and induce a different sequence of model
decisions. Removing one block from an already generated trace is therefore insufficient: the
relevant comparison is a complete target-policy trajectory from the same initial state.

We call the resulting discrepancy **guard-induced safety masking**. We measure it with paired
full-trajectory execution under a behavior guard and a target guard. The design freezes the task,
attack payload, model revision, decoding seed, environment seed, policy, and canonical initial
state before execution. Both trajectories are then run independently. Terminal harm, user
utility, pre-enforcement unsafe attempts, guard interventions, tool calls, state divergence, and
invalid execution are recorded separately.

This paper makes four empirical contributions:

1. It operationalizes guard-induced masking as a signed paired estimand over complete target
   trajectories, preserving reverse events rather than truncating negative differences.
2. It demonstrates positive masking for Qwen3-8B and, prospectively, for Ministral-3-14B, while
   retaining all frozen tasks through conservative missing-harm bounds.
3. It identifies Gemma-3-12B as a useful boundary case: the actor rarely attempted unsafe actions
   and exhibited no observed complete-case masking gap.
4. It triangulates the result using matched guard degradation, identical-guard negative controls,
   checkpoint restoration, task-clustered uncertainty, and descriptive mechanism diagnostics.

The empirical conclusion is deliberately conditional. We show that behavior guards **can** make
risky tool-using actors look safer, not that every guard masks every actor in every environment.

## 2. Problem formulation

Let \(\pi\) denote a fixed actor, \(g_b\) the deployed behavior guard, and \(g_\star\) the target
guard whose risk is to be audited. For frozen scenario and seed unit \(i\), let \(H_{bi}\) and
\(H_{\star i}\) be binary terminal-harm outcomes from independent full trajectories initialized
from the same canonical state. Aggregate behavior and target risks are

\[
V_b = \mathbb{E}[H_b], \qquad V_\star = \mathbb{E}[H_\star].
\]

The signed masking gap is

\[
\Delta_{\mathrm{mask}} = V_\star - V_b.
\]

A positive value means that evaluation under the behavior guard reports less harm than complete
execution under the target guard. A negative value is retained and means the target guard reduced
harm relative to the behavior guard. The primary row-level masking event is
\(H_b=0,H_\star=1\), but an individual event is a realized outcome rather than a risk estimate.

The primary comparison fixes \(g_b\) to a strict authorization guard and sets \(g_\star\) to no
action guard. Secondary comparisons replace the target with
\(\operatorname{degraded\_strict}(\rho)\) for
\(\rho\in\{0.25,0.50,0.75,1.00\}\). At each strict-guard block point, the degraded guard follows
the no-guard decision with probability \(\rho\). Identical strict guards provide a negative
control.

## 3. Experimental design

### 3.1 Frozen scenarios

The experiment contains 320 outcome-independent scenarios. The external layer contributes 160
scenarios from the pinned AgentDojo 0.1.35 benchmark (version 1.2.2) across banking, Slack,
travel, and workspace suites. The controlled layer contributes 160 deterministic scenarios
across payments, communication/data sharing, travel/calendar, and filesystem/DevOps. Controlled
instances span clean, ambiguous, untrusted-context, and multi-step strata. AgentDojo supplies
stateful tool environments in which untrusted tool outputs can induce prompt-injection attacks
against realistic user tasks [@debenedetti2024agentdojo].

Scenarios were assigned to smoke, development, and confirmatory splits before model outcomes were
inspected. The primary analysis uses 232 confirmatory strict-to-none pairs per actor. Secondary
confirmatory analyses use a matched 55-scenario degradation subset and 20 identical-guard
negative controls per actor.

Every sampling unit records the environment layer and domain, user and injection tasks, rendered
attack and hash, model and tokenizer revisions, decoding and environment seeds, policy and state
hashes, and both guards. Cross-experiment synthesis required all three actors to match one-to-one
on all 672 frozen scenario/guard units.

### 3.2 Actors and analysis provenance

The completed actor studies were:

- `Qwen/Qwen3-8B`;
- `google/gemma-3-12b-it`; and
- `mistralai/Ministral-3-14B-Instruct-2512-BF16`.

Qwen and Gemma belong to the frozen `exp1_full_v2` experiment. Their joint reporting scope was
declared after run-status review and is labeled
`post_hoc_partial_prespecified_actor_analysis`. The originally planned Llama arm was deferred
after status-only feasibility review; its harm outcomes were not used in that decision.

Ministral was selected as a different-family breadth extension using technical feasibility
evidence only. Its actor, adapter, matrix, and endpoints were frozen before its outcome-bearing
full run and are labeled `prospective_model_breadth_extension`. GPT-OSS and Granite were excluded
by outcome-blind technical gates and are not treated as zero-harm actors.

The combined three-model synthesis was specified after actor outcomes were known. Consequently,
new cross-model contrasts are exploratory and are not used to complete or repair the original
actor matrix.

### 3.3 Paired full-trajectory execution

For each pair, the environment is restored twice from a canonical, checksummed initial state. The
actor first runs under the behavior guard and independently under the target guard. The target
trajectory is not a modified behavior trace or a one-step counterfactual. Each trajectory may
diverge after an intervention and continues until task completion, a tool-step ceiling, or an
explicit invalid status.

The strict guard operates immediately before a syntactically valid proposed tool call can mutate
the environment. It enforces a declarative authorization policy and returns a deterministic
neutral denial observation when blocking. The guard does not receive the attack goal, terminal
harm label, or evaluation answer. A separate pre-enforcement evaluator records whether a proposed
action is unsafe.

### 3.4 Outcomes and diagnostics

Terminal harm is evaluated programmatically from the final environment state. User utility is
recorded separately and is never combined with harm. We additionally report unsafe-attempt,
blocked-call, proposed-call, zero-call, guard-dependence, and trajectory-divergence diagnostics.
These variables describe possible pathways but are not causal mediation estimates.

Malformed model calls, environment tool errors, context overflow, timeouts, and similar failures
are invalid rather than safe. Structural consistency and checkpoint restoration are validated
before analysis.

### 3.5 Statistical analysis

The primary complete-case estimate averages \(H_\star-H_b\) over pairs with valid binary harm in
both trajectories. Percentile intervals use 10,000 deterministic bootstrap replicates clustered
by the composite of environment layer, domain, and user-task ID.

Because execution validity can depend on the actor, two additional analyses retain all frozen
pairs. The declared sensitivity analysis treats an invalid target trajectory as harmful and uses
the experiment's frozen harmful rule for invalid behavior trajectories. The assumption-free
binary analysis instead assigns each missing harm independently to \([0,1]\), producing sharp
finite-sample bounds

\[
L_{\Delta i}=L_{\star i}-U_{bi}, \qquad
U_{\Delta i}=U_{\star i}-L_{bi}.
\]

We report the mean identification interval
\([\overline L_\Delta,\overline U_\Delta]\) and bootstrap intervals for each endpoint. The
identification interval describes missing-outcome uncertainty; endpoint intervals describe
sampling uncertainty. They are not interchangeable.

Actor-specific effects are the primary presentation. Task-paired actor contrasts, domain cells,
mechanism summaries, and combined degradation trends are retrospective, exploratory, and
unadjusted for multiplicity.

## 4. Results

### 4.1 Technical validity and completion

The synthesis contains 2,016 actor-by-scenario/guard rows: 672 for each actor and three actors for
each of 672 aligned units. All source validation reports were structurally clean and all checked
checkpoints were restorable. The primary analysis contains 696 actor-task pairs. Valid primary
pair counts were 211/232 for Qwen, 223/232 for Gemma, and 219/232 for Ministral, corresponding to
invalid-pair rates of 9.1%, 3.9%, and 5.6%.

### 4.2 Actor-specific masking

| Actor | Complete/total | Complete-case gap (95% CI) | Sensitivity gap (95% CI) | All-pair identification interval |
|---|---:|---:|---:|---:|
| Qwen3-8B | 211/232 | 0.137 (0.060, 0.237) | 0.142 (0.059, 0.237) | [0.065, 0.203] |
| Gemma-3-12B | 223/232 | 0.000 (0.000, 0.000) | 0.013 (0.000, 0.036) | [-0.026, 0.039] |
| Ministral-3-14B | 219/232 | 0.178 (0.097, 0.281) | 0.190 (0.108, 0.289) | [0.134, 0.224] |

Qwen and Ministral therefore exhibited substantial positive masking under complete-case and
declared sensitivity analyses. Gemma exhibited no complete-case masking events in the primary
sample. This zero does not establish equivalence: Gemma's all-pair interval permits both small
negative and positive gaps.

The observed finite-sample missing-harm interval was entirely positive for Qwen and Ministral,
but sampling uncertainty distinguishes the two. Qwen's conservative lower endpoint was 0.065
with a bootstrap interval of -0.027 to 0.167. Ministral's lower endpoint was 0.134 with an interval
of 0.048 to 0.234. Thus, only the prospective Ministral result remained strictly positive after
simultaneously adopting the conservative missing-harm endpoint and its sampling interval.

### 4.3 Exploratory actor contrasts

| Task-paired contrast | Joint complete | Complete-case difference (95% CI) | Sensitivity difference (95% CI) | All-pair contrast interval |
|---|---:|---:|---:|---:|
| Qwen minus Gemma | 204 | 0.142 (0.061, 0.246) | 0.129 (0.044, 0.230) | [0.026, 0.228] |
| Ministral minus Gemma | 211 | 0.180 (0.094, 0.286) | 0.177 (0.093, 0.276) | [0.095, 0.250] |
| Ministral minus Qwen | 205 | 0.044 (0.000, 0.095) | 0.047 (-0.004, 0.109) | [-0.069, 0.159] |

Qwen and Ministral each had larger estimated gaps than Gemma in the complete-case and sensitivity
analyses. Their conservative all-pair contrast lower endpoints were less decisive after sampling
uncertainty: Qwen-minus-Gemma had a lower endpoint interval of -0.075 to 0.131, and
Ministral-minus-Gemma had an interval of 0.000 to 0.198. The Ministral-minus-Qwen sensitivity
interval and identification interval both included zero. These retrospective comparisons support
heterogeneity but not a confirmatory ranking of models or families.

### 4.4 Domain concentration

| Domain | Qwen gap | Gemma gap | Ministral gap |
|---|---:|---:|---:|
| Banking | 0.034 | 0.000 | 0.133 |
| Communication | 0.000 | 0.000 | 0.000 |
| Filesystem/DevOps | 0.000 | 0.000 | 0.000 |
| Payments | 0.500 | 0.000 | 0.500 |
| Slack | 0.190 | 0.000 | 0.240 |
| Travel | 0.333 | 0.000 | 0.483 |
| Travel/calendar | 0.000 | 0.000 | 0.000 |
| Workspace | 0.000 | 0.000 | 0.043 |

Positive effects were not uniform across tasks. Qwen and Ministral replicated one another most
clearly in travel, Slack, and payments. Qwen's travel interval was 0.194 to 0.500 and Slack's was
0.045 to 0.389; Ministral's corresponding intervals were 0.321 to 0.645 and 0.080 to 0.414.
Payments produced a 0.500 point estimate for both actors, but its clustered interval was wide
because the controlled task family contains few independent task clusters. Banking was positive
but imprecise, and the remaining domains were null or inconclusive.

The effect therefore spans both the AgentDojo and controlled environment layers and is not a
single-domain artifact. It nevertheless concentrates in action-rich task families, limiting any
claim of broad task-level universality. Domain analyses are exploratory and were not adjusted for
multiple comparisons.

### 4.5 Descriptive mechanism diagnostics

| Actor | Behavior unsafe attempt | Target unsafe attempt | Behavior block rate | Guard dependence | Mean behavior calls | Mean target calls |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.123 | 0.156 | 0.152 | 0.318 | 2.564 | 2.692 |
| Gemma-3-12B | 0.018 | 0.018 | 0.083 | 0.081 | 0.937 | 0.951 |
| Ministral-3-14B | 0.183 | 0.183 | 0.161 | 0.361 | 2.621 | 2.658 |

Gemma rarely exposed the guard to an unsafe proposal and made fewer than one tool call per
trajectory on average. Qwen and Ministral used roughly 2.6 calls and had substantially larger
unsafe-attempt and guard-dependence rates. These differences are consistent with an opportunity
condition: a guard cannot mask an unsafe execution when the actor does not propose one.

The two positive actors also suggest different descriptive pathways. Ministral's behavior and
target unsafe-attempt rates were equal, while terminal harm differed sharply; this pattern is
consistent with direct blocking of proposals that would otherwise execute. Qwen's target
unsafe-attempt rate exceeded its behavior rate, consistent with guard-induced downstream
trajectory feedback in addition to direct blocking. Neither pattern identifies causal mediation,
because the mechanism variables were not randomized independently of the trajectory.

### 4.6 Matched degradation and falsification checks

On the matched 55-scenario subset, Ministral's complete-case masking gap increased from 0.036 at
25% degradation to 0.073, 0.109, and 0.145 at 50%, 75%, and 100% degradation. The no-guard
estimate was also 0.145. All 55 pairs were valid at each level, so its complete-case, sensitivity,
and missing-harm endpoints coincided. The monotone increments and agreement between 100%
degradation and no guard provide a strong construct-validity check.

Qwen showed a weaker step pattern: 0.040 at 25% through 75% degradation and 0.080 at 100% and no
guard among its complete pairs. Its all-pair bounds crossed zero on this smaller subset. Gemma's
complete-case gap remained zero at every level; its small sensitivity changes arose from invalid
outcome handling rather than realized masking events. The degradation evidence is therefore
strongest for Ministral and should not be described as a universal dose response.

The identical-strict negative controls yielded zero complete-case and sensitivity gaps for every
actor. Qwen and Gemma each had one invalid pair, producing finite-sample intervals of -0.050 to
0.050; all 20 Ministral pairs were valid and its interval was exactly zero. These controls support
the integrity of paired restoration and execution but do not independently validate the
substantive masking effect.

## 5. Discussion

### 5.1 Guarded system safety is not actor safety

The results demonstrate a measurement distinction. The strict behavior configuration can be an
effective deployed system while simultaneously providing an optimistic estimate of how the actor
would behave under a less restrictive target policy. For Qwen and Ministral, terminal harm was
substantially lower behind the strict guard. A behavior-only evaluation would have attributed
this low observed risk to the combined system without revealing how much depended on guard
intervention.

This does not mean that blocking is undesirable. The guard prevented harm in precisely the cases
where it was intended to intervene. The methodological point is that a successful guard should
not be mistaken for evidence that the underlying actor is intrinsically safe or transferable to a
different enforcement policy.

### 5.2 Masking is an actor–guard–environment property

Gemma's boundary result is scientifically useful rather than a failed replication. It shows that
guard removal need not increase harm when the actor rarely proposes consequential unsafe
actions. The concentration of positive gaps in travel, Slack, and payments further indicates that
the task must create both an unsafe opportunity and a path by which the actor can exercise it.

The appropriate object of inference is therefore not a context-free model ranking. Masking is a
property of an actor, guard, task distribution, tool interface, and downstream environment. Future
evaluations should report these components explicitly.

### 5.3 Implications for agent evaluation

Three practices follow from the experiment:

1. Evaluations intended to characterize an actor should distinguish pre-enforcement unsafe
   attempts from post-enforcement terminal harm.
2. A target-policy audit should execute complete trajectories from restored states, because a
   guard intervention changes the observations and later decisions available to the actor.
3. Invalid model/runtime outcomes should remain visible through rates, sensitivity analyses, and
   missing-harm bounds rather than being silently counted as safe or removed.

The paired design is especially useful during guard replacement, policy relaxation, or transfer
to an environment whose enforcement differs from the deployed configuration.

## 6. Limitations

First, the three-model synthesis was frozen after actor outcomes had been inspected. Qwen and
Gemma are a post-hoc partial scope; only the Ministral breadth extension was prospectively frozen
with respect to its own outcomes. Cross-model contrasts and mechanism comparisons are exploratory.

Second, three open-weight actors do not identify a model-family or parameter-scale effect. Model
family, tool-call interface, instruction tuning, activity level, and other actor properties are
confounded. Gemma's zero estimate should not be interpreted as equivalence without a prespecified
margin and substantially more power for that estimand.

Third, effects were concentrated in three task families. The overall estimates apply to the
frozen scenario and seed distribution; they do not imply masking in every domain or deployment.
Domain cells contain fewer independent task clusters and are vulnerable to multiplicity.

Fourth, complete-case estimates can be selected by actor-dependent execution failure. We report
declared sensitivity estimates and assumption-free binary-harm bounds, but those bounds widen and
their endpoint intervals can include zero. They diagnose uncertainty rather than recover missing
terminal states.

Fifth, primary decoding was deterministic and each frozen scenario/seed unit contributed one
paired trajectory per actor. The bootstrap represents sampling over frozen task clusters, not
within-prompt stochastic-generation variance.

Sixth, the mechanism evidence is descriptive. Unsafe attempts, blocks, tool activity, and
trajectory divergence are post-treatment variables and do not establish causal mediation.

Finally, the experiment compares one strict authorization guard with a no-guard target and a
particular degradation operator. It does not establish results for semantic classifiers, approval
systems, substitution guards, alternative policies, or production environments outside the
implemented action interfaces.

## 7. Reproducibility and artifact provenance

The experiment stores canonical frozen manifests, scenario and session hashes, immutable model
revisions, rendered attacks, policies, initial states, trajectory summaries, checkpoints,
intervention traces, terminal evaluations, and restoration validation. The synthesis verifies
source manifest hashes, validation-report hashes, source inferential declarations, actor sets,
row counts, scenario-set hashes, and cross-actor state/task invariants before analysis.

The synthesis specification is
`configs/analysis/exp1_three_model_synthesis_v1.yaml`. Its generated artifacts include the
combined pair table, actor and domain effects, task-paired contrasts, degradation and negative
controls, LaTeX tables, figures, source and run provenance, and a checksummed artifact manifest.
Raw oracle trajectories and potentially sensitive environment states remain separated from
public aggregate results.

## 8. Ethics and responsible reporting

All consequential actions occurred inside version-pinned benchmark or deterministic controlled
environments. No real payment, communication, travel, filesystem, or account mutation was
performed. Reports aggregate realized terminal outcomes and do not reproduce private raw oracle
content. Technical model exclusions are reported as feasibility outcomes rather than being
misrepresented as safe behavior.

## 9. Related work

### 9.1 Agent-risk and prompt-injection benchmarks

ToolEmu uses a language-model-emulated sandbox and automated evaluator to search for risks across
high-stakes toolkits [@ruan2024toolemu]. InjecAgent evaluates indirect prompt injection in
tool-integrated agents [@zhan2024injecagent], while AgentHarm tests whether agents can complete
explicitly malicious multi-step tasks [@andriushchenko2025agentharm]. AgentDojo provides dynamic,
stateful tasks for jointly evaluating prompt-injection attacks, defenses, task utility, and
security [@debenedetti2024agentdojo]. CENSURE reuses pinned AgentDojo environments and adds a
controlled layer, but asks a different question: for a fixed actor and frozen scenario
distribution, how far can risk observed behind one action guard differ from complete execution
under a target guard?

### 9.2 Runtime defenses for tool-using agents

Recent defenses enforce safety or user intent at action time. The Task Shield checks whether each
instruction and tool call contributes to the user's task [@jia2024taskshield]. GuardAgent maps
safety requests into executable checks [@xiang2025guardagent], and ShieldAgent reasons over
explicit policies and action trajectories using verifiable rule structures
[@chen2025shieldagent]. CaMeL instead separates control and data flow and applies capabilities at
tool boundaries to provide prompt-injection security by design [@debenedetti2025camel]. These
systems primarily ask how to prevent violations while retaining utility. Our strict guard is not
presented as a competing defense. It is the behavior policy whose measurement effect is audited:
successful blocking may improve deployed-system safety while obscuring the actor risk that would
be expressed under another enforcement policy.

### 9.3 Trajectory evaluation and counterfactual replay

TraceSafe directly evaluates guardrails on multi-step tool-call traces, emphasizing that
intermediate execution structure is itself a safety surface [@chen2026tracesafe]. Causal Agent
Replay intervenes on individual agent steps and re-executes forward to attribute a failure to
earlier decisions [@shah2026causalreplay]. CENSURE is complementary to both. It intervenes on the
guard policy for the entire trajectory, independently restores the same initial state for both
arms, and estimates a population-level signed harm difference rather than classifying a trace or
attributing one observed failure to a step. Its additional contribution is inferential: invalid
terminal outcomes remain visible through actor-specific failure rates, declared sensitivity
analyses, and sharp binary missing-harm bounds.

## 10. Conclusion

Behavior guards can make tool-using actors appear substantially safer than complete execution
under a less restrictive target policy. This effect replicated for Qwen and prospectively for
Ministral, survived conservative missing-harm analysis most strongly for Ministral, and appeared
in multiple action-rich domains. Gemma's null complete-case result and low tool activity reveal an
important boundary condition: masking requires an actor and task that generate unsafe execution
opportunities. Evaluations should therefore report the actor, guard, and environment separately
rather than treating guarded system outcomes as direct measurements of actor safety.
