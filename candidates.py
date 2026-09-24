"""
Statify — Candidate Analyses (Capability x Objective)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines profiler.py's Dataset Capability Map with objectives.py's
User Objective to produce a list of Candidate Analyses: tools that
are actually relevant to consider given BOTH what this dataset can
support AND what this user wants.

This is deliberately the narrow middle layer described in the
architecture:

    Dataset Capability Map  = what the dataset can potentially support
    User Objective           = what the user wants
    Candidate Analyses        = what is relevant given both  <- this file
    Analysis Plan              = what Statify should actually perform
                                  (a later, separate step)

What this file does NOT do, on purpose:
  - No ranking or scoring of candidates against each other.
  - No statistical readiness/assumption checking (sample size,
    normality, missingness thresholds) — that belongs to a future,
    deeper validator.py.
  - No expansion into a multi-step plan (diagnostics, primary test,
    alternative, effect size) — that is the future Analysis Plan.
  - No AI, no LLM, no network calls, no UI.

Key design point (confirmed against the actual codebase before
writing this): the Dataset Capability Map's per-category
"supported" flag is a coarse, honest, but sometimes misleading
signal. For example, Kaplan-Meier Survival's role contract only
requires "any numeric column" (duration) and "any categorical/
boolean column" (event) — so on an ordinary dataset like
marks+gender, the capability map reports Survival as "supported"
even though the data has nothing to do with survival analysis.
This file NEVER treats capability_map[category]["supported"] as
sufficient on its own. It is used only as a cheap early exit
(skip a category immediately if the map says it's impossible) and,
for objectives with named variables, the objective's actual named
columns are always what gets checked against each tool's specific
role contract — never "does the dataset contain a column of this
type somewhere." For broad_exploratory (no named variables to check
against), an extra safeguard is applied for the categories most
prone to this false-positive pattern (Survival, Time Series,
Forecasting): they are only considered genuinely applicable if a
real datetime column, or a plausible duration+event column pair,
is actually present — not merely "the role types matched something."
"""

from tool_registry import STAT_TOOLS


# ============================================================
# Intent -> candidate category routing
# ============================================================
# Which tool_registry.py categories are even worth searching for a
# given objective intent. This is a coarse routing table, not the
# final filter — every tool found this way still has its role
# contract checked against the objective's actual named variables
# (or, for broad_exploratory, against the dataset's buckets the same
# way profiler.py's own per-tool check already works).
_INTENT_CATEGORIES = {
    "group_comparison":       ["Hypothesis Testing", "ANOVA", "Non-Parametric"],
    "relationship":           ["Correlation", "Regression"],
    "description":            ["Descriptive"],
    "categorical_association": ["Hypothesis Testing", "Non-Parametric", "Effect Size"],
    "prediction":              ["Regression", "Machine Learning"],
    "time_pattern":            ["Time Series", "Forecasting"],
    "clustering":              ["Clustering", "Dimensionality"],
}

# Categories that require special handling in broad_exploratory mode
# because their role contracts alone are prone to false positives
# (see module docstring). Only included if the extra check in
# _exploratory_category_genuinely_applicable() passes.
_TIME_SHAPED_CATEGORIES = {"Time Series", "Forecasting"}
_SURVIVAL_CATEGORIES = {"Survival"}

# Hard cap on how many categories broad_exploratory can surface.
# This is a bound on breadth, not a ranking — every category that
# passes the applicability check up to this cap is included as-is,
# nothing is picked as "the" representative within a category.
_MAX_EXPLORATORY_CATEGORIES = 13  # = the full current category count;
                                    # this exists as an explicit ceiling
                                    # so growth in tool_registry.py can't
                                    # silently make exploratory output
                                    # unbounded later.


def _blank_result(objective_intent: str) -> dict:
    return {
        "objective_intent": objective_intent,
        "possible_in_principle": False,
        "blocking_issue": None,
        "candidates": [],
    }


def _tool_matches_named_variables(role_spec: dict, buckets: dict,
                                   named_vars: dict) -> dict | None:
    """
    Check whether a tool's role contract can be satisfied using
    SPECIFIC named variables (not "any column of the right type").
    named_vars maps role-name-ish keys the caller has already decided
    on (e.g. {"outcome": "marks", "grouping": "gender"} conceptually)
    to actual column names — see the per-intent handlers below for how
    each intent maps its objective fields onto a tool's actual role
    names. Returns the matched_variables dict (role_name -> column
    name/list) if every non-optional role is satisfied, else None.
    """
    matched = {}
    col_to_type = {c: t for t, cols in buckets.items() for c in cols}

    for role_name, role_def in role_spec.items():
        candidate_value = named_vars.get(role_name)
        if candidate_value is None:
            if role_def.get("optional"):
                continue
            return None

        values = candidate_value if isinstance(candidate_value, list) else [candidate_value]

        if role_def.get("multi"):
            if len(values) < role_def.get("min", 1):
                return None
            max_n = role_def.get("max")
            if max_n and len(values) > max_n:
                return None
        else:
            if len(values) != 1:
                return None

        for col in values:
            col_type = col_to_type.get(col)
            if col_type not in role_def["types"]:
                return None

        matched[role_name] = candidate_value

    return matched


def _find_tools_for_named_pair(category_names: list[str], buckets: dict,
                                 role_key_a: str, value_a: str,
                                 role_key_b: str | None, value_b) -> list[dict]:
    """
    Search the given categories for tools whose roles can be
    satisfied by mapping (value_a -> some role named role_key_a-ish,
    value_b -> some role named role_key_b-ish). Because different
    tools use different role-name conventions (value/group, x/y,
    var1/var2, variables), this tries the conventions actually present
    in tool_registry.py rather than assuming one fixed naming scheme.
    Returns a list of {"tool", "category", "matched_variables"} dicts.
    """
    found = []
    role_name_pairs = [
        ("value", "group"), ("x", "y"), ("var1", "var2"),
        ("y", "x"),  # some regression tools declare y before x
    ]

    for cat in category_names:
        for tool_name, spec in STAT_TOOLS.items():
            if spec["category"] != cat:
                continue
            roles = spec["roles"]
            role_names = set(roles.keys())

            matched = None
            for key_a, key_b in role_name_pairs:
                if key_a in role_names and (value_b is None or key_b in role_names):
                    named_vars = {key_a: value_a}
                    if value_b is not None:
                        named_vars[key_b] = value_b
                    matched = _tool_matches_named_variables(roles, buckets, named_vars)
                    if matched:
                        break

            # "variables" multi-role tools (e.g. Correlation Matrix,
            # K-Means): try passing both/all named values as the list.
            if matched is None and "variables" in role_names:
                values_list = [v for v in [value_a, value_b] if v is not None]
                matched = _tool_matches_named_variables(roles, buckets, {"variables": values_list})

            if matched:
                found.append({"tool": tool_name, "category": cat, "matched_variables": matched})

    return found


def _find_tools_for_single_var(category_names: list[str], buckets: dict,
                                 value: str) -> list[dict]:
    """Same idea as _find_tools_for_named_pair but for a single named
    variable (used by 'description')."""
    found = []
    for cat in category_names:
        for tool_name, spec in STAT_TOOLS.items():
            if spec["category"] != cat:
                continue
            roles = spec["roles"]
            role_names = set(roles.keys())
            matched = None
            for key in ("variable", "variables", "value"):
                if key in role_names:
                    candidate = [value] if key == "variables" else value
                    matched = _tool_matches_named_variables(roles, buckets, {key: candidate})
                    if matched:
                        break
            if matched:
                found.append({"tool": tool_name, "category": cat, "matched_variables": matched})
    return found


def _usable_categorical_columns(df, buckets: dict) -> list[str]:
    """Same cardinality rule as profiler.py's Capability Map: a
    categorical/boolean column needs 2+ distinct non-null values to
    be usable for grouping. Re-implemented here (not imported) to
    keep this file's only dependency on profiler.py limited to data
    it's already given (buckets), per the instruction to modify
    nothing else — this mirrors, not calls into, profiler.py."""
    usable = []
    for t in ("categorical", "boolean"):
        for col in buckets.get(t, []):
            if col in df.columns and df[col].dropna().nunique() >= 2:
                usable.append(col)
    return usable


# ============================================================
# Per-intent handlers
# ============================================================

def _handle_group_comparison(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("group_comparison")
    outcome = objective.get("outcome_variable")
    group = objective.get("grouping_variable")

    if not outcome or not group:
        result["blocking_issue"] = "Both an outcome variable and a grouping variable are needed to compare groups."
        return result

    result["possible_in_principle"] = True

    if group not in df.columns or df[group].dropna().nunique() < 2:
        result["blocking_issue"] = (
            f"The grouping variable '{group}' has only one distinct value in this dataset — "
            "at least two groups are needed to compare."
        )
        return result

    if outcome not in df.columns:
        result["blocking_issue"] = f"The outcome variable '{outcome}' was not found in this dataset."
        return result

    cats = [c for c in _INTENT_CATEGORIES["group_comparison"] if capability_map.get(c, {}).get("supported")]
    found = _find_tools_for_named_pair(cats, buckets, "value", outcome, "group", group)

    if not found:
        result["blocking_issue"] = (
            f"No registered tool's requirements are met by comparing '{outcome}' across '{group}'."
        )
        return result

    for f in found:
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": f"Compares '{outcome}' across the groups defined by '{group}'.",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_relationship(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("relationship")
    vars_ = objective.get("variables_of_interest") or []

    if len(vars_) < 2:
        result["blocking_issue"] = "At least two variables are needed to examine a relationship."
        return result

    result["possible_in_principle"] = True
    missing = [v for v in vars_ if v not in df.columns]
    if missing:
        result["blocking_issue"] = f"Variable(s) not found in this dataset: {', '.join(missing)}."
        return result

    cats = [c for c in _INTENT_CATEGORIES["relationship"] if capability_map.get(c, {}).get("supported")]
    x, y = vars_[0], vars_[1]
    found = _find_tools_for_named_pair(cats, buckets, "x", x, "y", y)

    if not found:
        result["blocking_issue"] = f"No registered tool's requirements are met by relating '{x}' and '{y}'."
        return result

    for f in found:
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": f"Examines the relationship between '{x}' and '{y}'.",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_description(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("description")
    vars_ = list(objective.get("variables_of_interest") or [])
    if objective.get("outcome_variable"):
        vars_.append(objective["outcome_variable"])

    if not vars_:
        if objective.get("scope") == "comprehensive":
            vars_ = buckets.get("numeric", [])
            if not vars_:
                result["blocking_issue"] = "No numeric variables are available to describe."
                return result
        else:
            result["blocking_issue"] = "At least one variable needs to be named to describe it."
            return result

    result["possible_in_principle"] = True
    missing = [v for v in vars_ if v not in df.columns]
    if missing:
        result["blocking_issue"] = f"Variable(s) not found in this dataset: {', '.join(missing)}."
        return result

    cats = [c for c in _INTENT_CATEGORIES["description"] if capability_map.get(c, {}).get("supported")]
    found = _find_tools_for_single_var(cats, buckets, vars_ if len(vars_) > 1 else vars_[0])
    # also try each variable individually in case only single-var roles exist
    if not found:
        for v in vars_:
            found.extend(_find_tools_for_single_var(cats, buckets, v))

    if not found:
        result["blocking_issue"] = "No registered descriptive tool's requirements are met by the named variable(s)."
        return result

    # de-duplicate by tool name (a variable-by-variable search can repeat a tool)
    seen = set()
    for f in found:
        if f["tool"] in seen:
            continue
        seen.add(f["tool"])
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": "Summarizes the named variable(s).",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_categorical_association(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("categorical_association")
    vars_ = objective.get("variables_of_interest") or []
    v1 = vars_[0] if len(vars_) > 0 else None
    v2 = vars_[1] if len(vars_) > 1 else None

    if not v1 or not v2:
        result["blocking_issue"] = "Two categorical variables are needed to examine an association between them."
        return result

    result["possible_in_principle"] = True
    missing = [v for v in (v1, v2) if v not in df.columns]
    if missing:
        result["blocking_issue"] = f"Variable(s) not found in this dataset: {', '.join(missing)}."
        return result

    col_to_type = {c: t for t, cols in buckets.items() for c in cols}
    if col_to_type.get(v1) not in ("categorical", "boolean") or col_to_type.get(v2) not in ("categorical", "boolean"):
        result["blocking_issue"] = f"Both '{v1}' and '{v2}' need to be categorical (or boolean) variables."
        return result

    # Search across ALL categories by role-shape (var1/var2, both
    # categorical/boolean) rather than assuming a dedicated category
    # exists — confirmed the registry keeps these tools inside
    # "Hypothesis Testing" / "Effect Size", not a category of their own.
    all_cats = sorted({spec["category"] for spec in STAT_TOOLS.values()})
    found = _find_tools_for_named_pair(all_cats, buckets, "var1", v1, "var2", v2, )

    if not found:
        result["blocking_issue"] = f"No registered tool's requirements are met by associating '{v1}' and '{v2}'."
        return result

    for f in found:
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": f"Examines the association between '{v1}' and '{v2}'.",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_prediction(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("prediction")
    outcome = objective.get("outcome_variable")
    predictors = objective.get("variables_of_interest") or []

    if not outcome or not predictors:
        result["blocking_issue"] = "An outcome variable and at least one predictor variable are needed."
        return result

    result["possible_in_principle"] = True
    missing = [v for v in [outcome] + predictors if v not in df.columns]
    if missing:
        result["blocking_issue"] = f"Variable(s) not found in this dataset: {', '.join(missing)}."
        return result

    cats = [c for c in _INTENT_CATEGORIES["prediction"] if capability_map.get(c, {}).get("supported")]
    x = predictors[0] if len(predictors) == 1 else predictors
    found = _find_tools_for_named_pair(cats, buckets, "y", outcome, "x", x)

    if not found:
        result["blocking_issue"] = f"No registered tool's requirements are met by predicting '{outcome}'."
        return result

    for f in found:
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": f"Models '{outcome}' using the named predictor(s).",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_time_pattern(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("time_pattern")
    value = objective.get("outcome_variable") or (
        objective["variables_of_interest"][0] if objective.get("variables_of_interest") else None
    )

    if not value:
        result["blocking_issue"] = "A variable to analyze over time is needed."
        return result

    if not buckets.get("datetime"):
        result["possible_in_principle"] = False
        result["blocking_issue"] = "No datetime column was found in this dataset — time-pattern analysis needs one."
        return result

    result["possible_in_principle"] = True
    if value not in df.columns:
        result["blocking_issue"] = f"Variable '{value}' was not found in this dataset."
        return result

    cats = [c for c in _INTENT_CATEGORIES["time_pattern"] if capability_map.get(c, {}).get("supported")]
    found = _find_tools_for_single_var(cats, buckets, value)

    if not found:
        result["blocking_issue"] = f"No registered time-pattern tool's requirements are met by '{value}'."
        return result

    for f in found:
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": f"Analyzes '{value}' over time.",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_clustering(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("clustering")
    vars_ = objective.get("variables_of_interest") or []

    if not vars_:
        vars_ = buckets.get("numeric", [])

    if len(vars_) < 2:
        result["blocking_issue"] = "At least two numeric variables are needed for clustering."
        return result

    result["possible_in_principle"] = True
    missing = [v for v in vars_ if v not in df.columns]
    if missing:
        result["blocking_issue"] = f"Variable(s) not found in this dataset: {', '.join(missing)}."
        return result

    cats = [c for c in _INTENT_CATEGORIES["clustering"] if capability_map.get(c, {}).get("supported")]
    found = []
    for cat in cats:
        for tool_name, spec in STAT_TOOLS.items():
            if spec["category"] != cat:
                continue
            matched = _tool_matches_named_variables(spec["roles"], buckets, {"variables": vars_})
            if matched:
                found.append({"tool": tool_name, "category": cat, "matched_variables": matched})

    if not found:
        result["blocking_issue"] = "No registered clustering/dimensionality tool's requirements are met by the named variables."
        return result

    for f in found:
        result["candidates"].append({
            "tool": f["tool"], "category": f["category"],
            "matched_variables": f["matched_variables"],
            "why": "Groups observations based on the named variables.",
            "priority": "primary",
            "relationship_to_primary": None,
        })
    return result


def _handle_explicit_method(df, buckets, capability_map, objective) -> dict:
    result = _blank_result("explicit_method")
    tool_name = objective.get("explicit_method")

    if not tool_name or tool_name not in STAT_TOOLS:
        result["blocking_issue"] = f"'{tool_name}' is not a registered tool."
        return result

    result["possible_in_principle"] = True
    spec = STAT_TOOLS[tool_name]
    roles = spec["roles"]
    role_names = set(roles.keys())

    named_vars = {}
    if objective.get("outcome_variable") and "value" in role_names:
        named_vars["value"] = objective["outcome_variable"]
    if objective.get("grouping_variable") and "group" in role_names:
        named_vars["group"] = objective["grouping_variable"]
    vars_of_interest = objective.get("variables_of_interest") or []
    if vars_of_interest:
        if "variables" in role_names:
            named_vars["variables"] = vars_of_interest
        elif "x" in role_names and "y" in role_names and len(vars_of_interest) >= 2:
            named_vars["x"], named_vars["y"] = vars_of_interest[0], vars_of_interest[1]
        elif "var1" in role_names and "var2" in role_names and len(vars_of_interest) >= 2:
            named_vars["var1"], named_vars["var2"] = vars_of_interest[0], vars_of_interest[1]

    matched = _tool_matches_named_variables(roles, buckets, named_vars) if named_vars else {}
    if named_vars and matched is None:
        result["blocking_issue"] = (
            f"'{tool_name}' was requested, but the named variable(s) don't satisfy its requirements."
        )
        return result

    result["candidates"].append({
        "tool": tool_name, "category": spec["category"],
        "matched_variables": matched or {},
        "why": "This is the method you selected.",
        "priority": "primary",
        "relationship_to_primary": None,
    })

    # Look for structurally-fitting alternatives in the same category,
    # using the same named variables — never replacing the primary,
    # always additional and clearly labeled.
    if named_vars:
        for other_name, other_spec in STAT_TOOLS.items():
            if other_name == tool_name or other_spec["category"] != spec["category"]:
                continue
            other_matched = _tool_matches_named_variables(other_spec["roles"], buckets, named_vars)
            if other_matched:
                result["candidates"].append({
                    "tool": other_name, "category": other_spec["category"],
                    "matched_variables": other_matched,
                    "why": f"Also structurally fits the same variables as '{tool_name}' — worth considering as an alternative.",
                    "priority": "alternative",
                    "relationship_to_primary": "alternative",
                })

    return result


def _exploratory_category_genuinely_applicable(category: str, df, buckets: dict) -> bool:
    """Extra safeguard for broad_exploratory mode specifically, where
    there are no named variables to check against (see module
    docstring). Categories prone to the "any numeric + any categorical
    satisfies a semantically unrelated role" false positive get an
    additional, honest check here rather than being trusted on role-
    type-matching alone.

    Time Series / Forecasting: a real datetime column is a genuine,
    reliable structural signal — required here.

    Survival: NOT included in broad_exploratory at all. Tested this
    directly: a "2 distinct categorical values" check (the only
    structural proxy available for "looks like an event column") is
    satisfied by completely unrelated binary columns too (e.g. a
    'gender' column on an exam-marks dataset) — it is not a
    meaningfully stronger signal than the base role-type match that
    already produces the false positive this function exists to
    prevent. Rather than keep a heuristic that doesn't actually
    discriminate, Survival is excluded from exploratory mode
    entirely until either better registry metadata or an explicit
    survival-related objective is available (see Step 3 design notes
    on this exact limitation)."""
    if category in _TIME_SHAPED_CATEGORIES:
        return bool(buckets.get("datetime"))
    if category in _SURVIVAL_CATEGORIES:
        return False
    return True


def _handle_broad_exploratory(df, buckets, capability_map) -> dict:
    result = _blank_result("broad_exploratory")
    result["possible_in_principle"] = True

    applicable_categories = []
    for category, info in capability_map.items():
        if not info.get("supported"):
            continue
        if not _exploratory_category_genuinely_applicable(category, df, buckets):
            continue
        applicable_categories.append(category)

    applicable_categories = sorted(applicable_categories)[:_MAX_EXPLORATORY_CATEGORIES]

    if not applicable_categories:
        result["blocking_issue"] = "No analysis category is genuinely applicable to this dataset."
        return result

    for category in applicable_categories:
        eligible_tools = capability_map[category]["supporting_tools"]
        result["candidates"].append({
            "category": category,
            "eligible_tools": list(eligible_tools),
            "why": f"This dataset's structure could support {category.lower()} analysis.",
            "priority": "exploratory",
        })

    return result


# ============================================================
# Public entry point
# ============================================================

_HANDLERS = {
    "group_comparison": _handle_group_comparison,
    "relationship": _handle_relationship,
    "description": _handle_description,
    "categorical_association": _handle_categorical_association,
    "prediction": _handle_prediction,
    "time_pattern": _handle_time_pattern,
    "clustering": _handle_clustering,
    "explicit_method": _handle_explicit_method,
}


def find_candidate_analyses(df, capability_map: dict, buckets: dict, objective: dict) -> dict:
    """
    Combine a Dataset Capability Map (profiler.py), a bucket map
    (profiler.py's get_column_types), and a User Objective
    (objectives.py) into a Candidate Analyses result.

    df is required (not just buckets) because several checks need
    actual column values, not just their type — e.g. confirming a
    grouping variable has 2+ distinct values, matching the same
    cardinality rule profiler.py's Capability Map already applies.

    Returns the schema described in the Step 3 design: a dict with
    objective_intent, possible_in_principle, blocking_issue, and a
    candidates list. For broad_exploratory specifically, candidates
    are grouped by category (each with an eligible_tools list) rather
    than naming one tool per category — no tool within an applicable
    category is treated as preferred over another at this stage.
    """
    intent = objective.get("intent")

    if intent == "broad_exploratory":
        return _handle_broad_exploratory(df, buckets, capability_map)

    if intent not in _HANDLERS:
        result = _blank_result(intent)
        result["blocking_issue"] = f"Objective intent '{intent}' is not yet handled by this step."
        return result

    return _HANDLERS[intent](df, buckets, capability_map, objective)
