"""
Statify — Analysis Plan (Step 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Turns candidates.py's honest, flat Candidate Analyses list into a
structured, bounded sequence of plan steps: which tool plays which
role (primary / diagnostic / alternative / supporting / exploratory),
why it's there, and how steps relate to each other.

This is a PLANNING layer only. It decides what belongs in the plan
and how steps relate to each other — it never runs a calculation,
never inspects a p-value, never decides whether an assumption is
met, and never picks between primary/alternative based on data.
Those are validation's job (not yet built) and stats_engine.py's
job (already built, untouched by this file).

    Dataset Capability Map  = what the dataset can potentially support
    User Objective           = what the user wants
    Candidate Analyses        = what is relevant to consider given both
    Analysis Plan              = what Statify PROPOSES to do, and how
                                  the proposed steps relate            <- this file
    Validation (future)         = whether each planned step is actually
                                    ready given the real data
    Calculation                 = deterministic execution (unchanged)

────────────────────────────────────────────────────────────────
Declared relevance vs. structural compatibility
────────────────────────────────────────────────────────────────
A diagnostic or effect-size tool is added to the plan ONLY if it
passes BOTH of these independent checks, in this order:

  1. Declared relevance (_DIAGNOSTIC_RELEVANCE / _EFFECT_SIZE_RELEVANCE
     below): is this specific diagnostic/effect-size tool one that is
     actually known to be paired with this specific primary tool in
     real statistical practice? This is a fact about a relationship
     between two tools. Nothing in tool_registry.py encodes this today
     (checked directly: every tool's fields are just {desc, graphs,
     assumptions, roles, category, params} — no cross-references
     exist), so this small, explicit, hand-curated, and DELIBERATELY
     INCOMPLETE table lives here, local to this module, rather than
     being invented as new tool_registry.py fields before a validation
     layer exists to actually consume them.

  2. Structural compatibility (reusing the same role-matching logic
     candidates.py already established): given that a tool is
     declared-relevant, can its role contract actually be satisfied by
     the primary step's specific matched variables?

A tool absent from the relevance table is NEVER added, no matter how
well its role contract happens to match — this is what prevents, for
example, Pearson Correlation from picking up Shapiro-Wilk Normality
just because both of Pearson's variables happen to be numeric and
Shapiro-Wilk's role also just needs "a numeric variable". Structural
fit alone is not evidence of statistical relevance; both checks are
required, always in that order (relevance first, since a tool that
isn't relevant is never even structurally checked).
"""

from tool_registry import STAT_TOOLS


# ============================================================
# Declared relevance tables (local to Step 4, NOT tool_registry.py)
# ============================================================
# tool name -> list of diagnostic/effect-size tool names known to be
# actually relevant to assessing/quantifying that tool's result.
# Absence of an entry means "no diagnostic/effect size is currently
# known to be relevant" — not "none exists". These tables are
# intentionally small and will grow deliberately, one hand-verified
# pairing at a time, not by inference from role shape.
_DIAGNOSTIC_RELEVANCE = {
    "Independent t-Test": ["Shapiro-Wilk Normality", "Levene's Test"],
    "Paired t-Test": ["Shapiro-Wilk Normality"],
    "One-Sample t-Test": ["Shapiro-Wilk Normality"],
    "One-Way ANOVA": ["Shapiro-Wilk Normality", "Levene's Test"],
    "Two-Way ANOVA": ["Shapiro-Wilk Normality", "Levene's Test"],
    "Repeated Measures ANOVA": ["Shapiro-Wilk Normality"],
    "ANCOVA": ["Shapiro-Wilk Normality", "Levene's Test"],
    "Stationarity (ADF Test)": [],  # placeholder-safe: not a primary that needs its own diagnostic here
    "ARIMA Forecast": ["Stationarity (ADF Test)"],
    "Exponential Smoothing": ["Stationarity (ADF Test)"],
    # Deliberately NOT mapped: Pearson Correlation, Spearman Correlation,
    # Kendall's Tau, Simple/Multiple Linear Regression, Chi-Square family,
    # clustering tools. See module docstring — an accurate empty slot
    # beats an inaccurate populated one. These may gain real entries
    # later, but only once a specific, correctly-scoped diagnostic is
    # identified for them (e.g. Pearson's real assumption is bivariate
    # linearity/joint distribution shape, not univariate normality of
    # each variable separately — nothing in the current registry
    # checks that, so nothing is claimed here).
}

_EFFECT_SIZE_RELEVANCE = {
    "Independent t-Test": ["Cohen's d (Effect Size)"],
    "Mann-Whitney U Test": ["Cohen's d (Effect Size)"],
    "Paired t-Test": ["Cohen's d (Effect Size)"],
    "One-Sample t-Test": ["Cohen's d (Effect Size)"],
    "Chi-Square Test of Independence": ["Cramér's V"],
    "Fisher's Exact Test": ["Cramér's V"],
    # Deliberately NOT mapped: One-Way/Two-Way ANOVA (a real effect
    # size for ANOVA — eta-squared — is not currently a registered
    # tool; adding a false pairing to Cohen's d, which is a two-group
    # measure, would misrepresent a 3+-group comparison). This is a
    # named registry gap, not something this table should paper over.
}

# Supporting steps for 'description' intent under comprehensive scope.
# These are NOT relevance-table-gated (per the approved design,
# constraint 12) — they are a fixed, small, always-considered set for
# this one intent only, found by the same structural role-match
# mechanism, independent of _DIAGNOSTIC_RELEVANCE/_EFFECT_SIZE_RELEVANCE.
_DESCRIPTION_COMPREHENSIVE_SUPPORTING = ["Skewness & Kurtosis", "Outlier Detection (IQR)"]


def _variable_types_from_primary(primary_tool: str, matched_variables: dict) -> dict:
    """
    Map each column name in matched_variables to its semantic type(s),
    using the PRIMARY tool's own role definitions in tool_registry.py
    (the only place that type information is available at this point
    — candidates.py's output carries column names, not their types).
    Returns {column_name: set_of_types}. A column can only have the
    type(s) declared by whichever role it filled for the primary.
    """
    if primary_tool not in STAT_TOOLS:
        return {}
    primary_roles = STAT_TOOLS[primary_tool]["roles"]
    col_types = {}
    for role_name, value in matched_variables.items():
        role_def = primary_roles.get(role_name)
        if role_def is None:
            continue
        vals = value if isinstance(value, list) else [value]
        for v in vals:
            col_types[v] = set(role_def["types"])
    return col_types


def _tool_role_matches(tool_name: str, matched_variables: dict,
                        col_types: dict) -> dict | None:
    """
    Structural compatibility check: does tool_name's role contract
    accept these specific variable(s), INCLUDING their actual type(s)?
    Reuses the matching idea already established in candidates.py
    (bucket/type + min/max/multi/optional).

    col_types maps each column name (as it appears somewhere in
    matched_variables) to the set of semantic types it was already
    established to have, sourced from the PRIMARY tool's own role
    definitions (see _variable_types_from_primary). This is required
    — without it, this function could not tell that e.g. a
    categorical grouping column doesn't satisfy a diagnostic tool
    that needs a numeric variable, which would silently defeat the
    whole point of "structural compatibility" as a real check.

    Tries the same role-name conventions candidates.py already
    established (value/group, x/y, var1/var2, variables) so this
    works regardless of which naming convention the target tool uses,
    without hardcoding a translation per tool. When the target tool
    only declares a single-variable role and the primary has multiple
    matched variables, each individual value is tried in turn — this
    is a mechanical fit check only (does at least one of the primary's
    variables have a type this tool's role accepts), not a judgment
    about which variable is statistically "the right one."

    Returns the new tool's own matched_variables dict on success,
    None otherwise.
    """
    if tool_name not in STAT_TOOLS:
        return None
    roles = STAT_TOOLS[tool_name]["roles"]
    role_names = set(roles.keys())

    values = list(matched_variables.values())
    flat_values = []
    for v in values:
        if isinstance(v, list):
            flat_values.extend(v)
        else:
            flat_values.append(v)

    candidate_mappings = []
    if len(flat_values) == 1:
        for key in ("variable", "value", "x"):
            if key in role_names:
                candidate_mappings.append({key: flat_values[0]})
        if "variables" in role_names:
            candidate_mappings.append({"variables": flat_values})
    elif len(flat_values) >= 2:
        pairs = [("value", "group"), ("x", "y"), ("var1", "var2")]
        for key_a, key_b in pairs:
            if key_a in role_names and key_b in role_names:
                candidate_mappings.append({key_a: flat_values[0], key_b: flat_values[1]})
        if "variables" in role_names:
            candidate_mappings.append({"variables": flat_values})
        # Purely structural fallback (no statistical judgment involved):
        # if the target tool only declares a SINGLE-variable role (e.g.
        # Shapiro-Wilk's "variable"), and the primary has multiple
        # matched variables, try each one individually. Needed because
        # e.g. Independent t-Test's matched_variables is
        # {value: numeric, group: categorical} — two values — while
        # Shapiro-Wilk only ever accepts one.
        for key in ("variable", "value", "x"):
            if key in role_names:
                for v in flat_values:
                    candidate_mappings.append({key: v})

    for mapping in candidate_mappings:
        ok = True
        for role_name, role_def in roles.items():
            val = mapping.get(role_name)
            if val is None:
                if not role_def.get("optional"):
                    ok = False
                    break
                continue
            vals = val if isinstance(val, list) else [val]

            # Type check: every value assigned to this role must have
            # at least one type in common with what this role accepts.
            # Unknown columns (not in col_types) fail closed, rather
            # than being silently assumed compatible.
            for v in vals:
                v_types = col_types.get(v)
                if not v_types or not (v_types & role_def["types"]):
                    ok = False
                    break
            if not ok:
                break

            if role_def.get("multi"):
                if len(vals) < role_def.get("min", 1):
                    ok = False
                    break
                max_n = role_def.get("max")
                if max_n and len(vals) > max_n:
                    ok = False
                    break
            else:
                if len(vals) != 1:
                    ok = False
                    break
        if ok:
            return mapping

    return None


def _find_relevant_and_compatible(primary_tool: str, matched_variables: dict,
                                    relevance_table: dict) -> list[tuple[str, dict]]:
    """
    Apply BOTH checks, in order: declared relevance first (lookup in
    relevance_table), then structural compatibility (role-match,
    including real type-checking against the primary's own role
    definitions) only for tools that passed the relevance check.
    Returns a list of (tool_name, its_own_matched_variables) for every
    tool that passes both. This is the single shared mechanism used
    for both diagnostics and effect sizes — the two tables passed in
    are the only thing that differs between the two use sites.
    """
    declared = relevance_table.get(primary_tool, [])
    if not declared:
        return []
    col_types = _variable_types_from_primary(primary_tool, matched_variables)
    passing = []
    for candidate_tool in declared:
        matched = _tool_role_matches(candidate_tool, matched_variables, col_types)
        if matched is not None:
            passing.append((candidate_tool, matched))
    return passing


def _known_diagnostic_tools() -> set:
    """Every tool name that appears as a value anywhere in
    _DIAGNOSTIC_RELEVANCE. A tool that is itself declared as someone
    else's diagnostic (e.g. Levene's Test) should never be treated as
    its own primary candidate — this is a principled exclusion (it's
    real data already declared above), not a per-tool name hack."""
    known = set()
    for diag_list in _DIAGNOSTIC_RELEVANCE.values():
        known.update(diag_list)
    return known


def _next_step_id(counter: list[int]) -> str:
    counter[0] += 1
    return f"s{counter[0]}"


def _blank_plan(objective_intent) -> dict:
    return {
        "objective_intent": objective_intent,
        "possible_in_principle": False,
        "blocking_issue": None,
        "scope": "standard",
        "plan": [],
        "notes": None,
    }


def build_analysis_plan(candidate_result: dict, objective: dict) -> dict:
    """
    Public entry point. Takes candidates.py's find_candidate_analyses()
    output and the same objective dict from objectives.py, and returns
    a structured Analysis Plan.

    Does not touch the dataset, the capability map, or run anything —
    candidate_result already carries everything needed (matched
    variables were already resolved against real columns by Step 3).
    """
    intent = candidate_result.get("objective_intent")
    scope = objective.get("scope", "standard")
    result = _blank_plan(intent)
    result["scope"] = scope

    # Propagate ambiguous objectives without guessing — Step 2's own
    # clarification flag is checked FIRST and takes priority, since
    # it's the most specific, most upstream signal that the objective
    # itself is incomplete. (Checked before Step 3's possible_in_principle
    # below because for some intents, e.g. group_comparison with no
    # variables named at all, candidates.py's own early-return also
    # produces a blocking_issue for the same underlying reason — but
    # objectives.py's needs_clarification is the more authoritative,
    # earlier signal and should be surfaced, not silently superseded.)
    if objective.get("needs_clarification"):
        result["possible_in_principle"] = False
        result["blocking_issue"] = objective.get("clarification_reason") or (
            "This objective needs clarification before a plan can be built."
        )
        return result

    # Propagate blocked / unsupported objectives without guessing.
    if not candidate_result.get("possible_in_principle"):
        result["possible_in_principle"] = False
        result["blocking_issue"] = candidate_result.get("blocking_issue")
        return result

    candidates = candidate_result.get("candidates", [])
    if not candidates:
        # NOTE: Step 3 can report possible_in_principle=True here (the
        # OBJECTIVE TYPE was valid — e.g. group_comparison is a coherent
        # thing to ask for) while still returning zero candidates,
        # because the SPECIFIC data was insufficient (e.g. a grouping
        # variable with only one distinct value). For Step 4's own
        # output, possible_in_principle must reflect "is there anything
        # to actually propose" — an empty candidate list always means
        # False here, regardless of what Step 3's flag says, since
        # Step 3's flag answers a different question (objective-type
        # validity) than Step 4's flag needs to (plan buildability).
        result["possible_in_principle"] = False
        result["blocking_issue"] = candidate_result.get("blocking_issue") or "No candidate analyses were found."
        return result

    result["possible_in_principle"] = True
    step_counter = [0]

    # ---------- broad_exploratory: category-level areas only ----------
    if intent == "broad_exploratory":
        for cand in candidates:
            step_id = _next_step_id(step_counter)
            result["plan"].append({
                "step_id": step_id,
                "tool": None,
                "category": cand["category"],
                "role": "exploratory",
                "matched_variables": {},
                "eligible_tools": list(cand.get("eligible_tools", [])),
                "depends_on": None,
                "relevance_basis": None,
                "why": cand.get("why", f"This dataset's structure could support {cand['category'].lower()} analysis."),
            })
        return result

    # ---------- every other intent: primary + scaffolding ----------
    # NOTE on Step 3's candidate shape (confirmed directly, not
    # assumed): candidates.py tags every tool it finds within the
    # routed categories as priority="primary" — it does not single
    # out "the" primary among siblings (this matches how Step 3 was
    # deliberately built: honest, flat, un-ranked). For an intent like
    # group_comparison this can include several genuinely different
    # primaries (e.g. Independent t-Test, One-Way ANOVA, Mann-Whitney
    # U, Tukey HSD Post-Hoc all structurally fit the same two
    # variables). Each is treated here as its own separate, complete
    # primary candidate with its own scaffolding — nothing is silently
    # dropped or picked as "the" one. The one exception: a tool that
    # is ITSELF a known diagnostic (declared as a value somewhere in
    # _DIAGNOSTIC_RELEVANCE, e.g. Levene's Test) is excluded from
    # being treated as its own primary — this is a principled
    # exclusion using data already declared above, not a name-based
    # special case for one tool.
    known_diagnostic_tools = _known_diagnostic_tools()
    primary_candidates = [
        c for c in candidates
        if c.get("priority") == "primary" and c["tool"] not in known_diagnostic_tools
    ]
    alt_candidates = [c for c in candidates if c.get("priority") == "alternative"]

    if not primary_candidates:
        result["possible_in_principle"] = False
        result["blocking_issue"] = "No primary candidate analysis was found for this objective."
        return result

    primary_step_ids_by_tool = {}
    relevant_diagnostics = {}   # tool_name -> {"matched": dict, "for_primaries": [names]}
    relevant_effect_sizes = {}  # tool_name -> {"matched": dict, "for_primaries": [names]}

    for primary in primary_candidates:
        # NOTE: today, only explicit_method's handler in candidates.py
        # ever produces priority="alternative" candidates, and it only
        # ever produces exactly one primary alongside them — so
        # attaching the full alt_candidates list to every primary here
        # is safe in practice. If a future candidates.py change ever
        # produces multiple distinct primaries together with
        # alternatives, this would need each alternative paired to
        # its own specific primary rather than broadcast to all of
        # them; flagging this coupling explicitly rather than silently
        # assuming it always holds.
        primary_id = _next_step_id(step_counter)
        result["plan"].append({
            "step_id": primary_id,
            "tool": primary["tool"],
            "category": primary["category"],
            "role": "primary",
            "matched_variables": primary["matched_variables"],
            "eligible_tools": None,
            "depends_on": None,
            "relevance_basis": None,
            "why": primary.get("why", f"Directly addresses the stated objective using '{primary['tool']}'."),
        })
        primary_step_ids_by_tool[primary["tool"]] = primary_id

        for alt in alt_candidates:
            alt_id = _next_step_id(step_counter)
            result["plan"].append({
                "step_id": alt_id,
                "tool": alt["tool"],
                "category": alt["category"],
                "role": "alternative",
                "matched_variables": alt["matched_variables"],
                "eligible_tools": None,
                "depends_on": [primary_id],
                "relevance_basis": None,
                "why": (
                    f"A structurally valid alternative to '{primary['tool']}' — considered only if "
                    f"a readiness check on the primary analysis indicates it may not be appropriate."
                ),
            })

        if scope in ("standard", "comprehensive"):
            for diag_tool, diag_matched in _find_relevant_and_compatible(
                primary["tool"], primary["matched_variables"], _DIAGNOSTIC_RELEVANCE
            ):
                relevant_diagnostics.setdefault(diag_tool, {"matched": diag_matched, "for_primaries": []})
                relevant_diagnostics[diag_tool]["for_primaries"].append(primary["tool"])

            for es_tool, es_matched in _find_relevant_and_compatible(
                primary["tool"], primary["matched_variables"], _EFFECT_SIZE_RELEVANCE
            ):
                relevant_effect_sizes.setdefault(es_tool, {"matched": es_matched, "for_primaries": []})
                relevant_effect_sizes[es_tool]["for_primaries"].append(primary["tool"])

    # Diagnostic/effect-size steps are added ONCE per distinct tool,
    # even if several primaries in this plan happen to declare the
    # same one relevant (e.g. both Independent t-Test and One-Way
    # ANOVA declare Levene's Test relevant) — this keeps the plan
    # bounded rather than duplicating the same diagnostic once per
    # primary that references it. The "why" text names every primary
    # it's relevant to, so nothing is hidden.
    for diag_tool, info in relevant_diagnostics.items():
        diag_id = _next_step_id(step_counter)
        primaries_text = "', '".join(info["for_primaries"])
        result["plan"].append({
            "step_id": diag_id,
            "tool": diag_tool,
            "category": STAT_TOOLS[diag_tool]["category"],
            "role": "diagnostic",
            "matched_variables": info["matched"],
            "eligible_tools": None,
            "depends_on": None,
            "relevance_basis": "declared",
            "why": f"A diagnostic relevant to assessing: '{primaries_text}'.",
        })

    for es_tool, info in relevant_effect_sizes.items():
        es_id = _next_step_id(step_counter)
        primaries_text = "', '".join(info["for_primaries"])
        result["plan"].append({
            "step_id": es_id,
            "tool": es_tool,
            "category": STAT_TOOLS[es_tool]["category"],
            "role": "supporting",
            "matched_variables": info["matched"],
            "eligible_tools": None,
            "depends_on": None,
            "relevance_basis": "declared",
            "why": f"Quantifies the magnitude associated with: '{primaries_text}', beyond significance alone.",
        })

        if intent == "description" and scope == "comprehensive":
            desc_col_types = _variable_types_from_primary(primary["tool"], primary["matched_variables"])
            for supp_tool in _DESCRIPTION_COMPREHENSIVE_SUPPORTING:
                matched = _tool_role_matches(supp_tool, primary["matched_variables"], desc_col_types)
                if matched is not None:
                    supp_id = _next_step_id(step_counter)
                    result["plan"].append({
                        "step_id": supp_id,
                        "tool": supp_tool,
                        "category": STAT_TOOLS[supp_tool]["category"],
                        "role": "supporting",
                        "matched_variables": matched,
                        "eligible_tools": None,
                        "depends_on": None,
                        "relevance_basis": None,
                        "why": f"Additional context alongside the descriptive summary.",
                    })

    if scope == "comprehensive":
        result["notes"] = (
            "Comprehensive scope: this plan includes available diagnostics and effect-size "
            "steps where a relevant, structurally-compatible tool is known. It does not "
            "include every tool in the registry that happens to share a variable type."
        )

    return result
