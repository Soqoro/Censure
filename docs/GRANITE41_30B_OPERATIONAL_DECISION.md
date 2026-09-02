# Granite 4.1 30B operational decision record

Status: recorded on 2026-09-02 (Asia/Singapore) from outcome-blind execution and syntax-audit
artifacts only. No Granite terminal harm, utility, paired difference, guard decision, or state
divergence was used.

The frozen 40-pair operational-v2 gate remains failed because it observed a
`ToolCallParseError`, which was outside its allowed post-proposal environment-error class. The
subsequent syntax audit selected all 40 pairs, found no missing trajectory summary, and found one
behavior trajectory with two deterministic parser-failure attempts.

Both attempts had the same parser-message SHA-256 and the same raw-emission SHA-256. The bounded
diagnostic recorded a raw length of 2,175 characters, but its 2,048-character edge preview was
truncated. Consequently, both attempts had `preview_verifiable: false`; the audit totals were zero
verifiable and two unverifiable parser-failure attempts.

The retained suffix ends inside a JSON string rather than with a closed Granite tool-call block.
Together with the frozen `max_new_tokens: 512`, this is consistent with generation ending at the
token ceiling while composing a long file body. That explanation is a technical inference, not a
protocol-qualified classification, because the complete raw emission was not retained and its
SHA-256 could not be recomputed from the preview.

Under the decision tree frozen in `ALL_PAIR_ROBUSTNESS_ANALYSIS.md`, an incomplete or unverifiable
preview is a no-go for a Granite outcome-bearing extension. Do not increase the generation limit,
repair/reinterpret the emission, retry the selected case, or start a Granite full run under this
track. Granite is reported as an outcome-blind operational exclusion, not as a zero-harm actor.
