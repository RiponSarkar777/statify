"""
Statify — User Objective Representation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Defines the structured shape Statify uses to represent "what the
user wants," independent of the dataset. This is deliberately kept
separate from profiler.py's Dataset Capability Map (dataset.py's
job is "what can this data support," this file's job is "what does
this user want") — the two are combined in a later step, not here.

This step is fully deterministic: no AI, no LLM, no network calls.
Only two ways exist to build an objective right now:
  - objective_from_tool_selection(): for a user who already picked
    a specific tool and variables (the existing Builder page flow).
  - objective_from_example_choice(): for a user who picked one of a
    small fixed set of plain-language example choices.
Free-text natural-language parsing (via AI) is a later, separate
step — nothing here calls out to anything external.
"""

# Fixed, intentionally small taxonomy for this step. Not meant to be
# exhaustive yet — new intents are added deliberately, in a later
# step, once there's a concrete reason (e.g. a new example choice or
# a real free-text case that doesn't fit).
VALID_INTENTS = {
    "group_comparison",
    "relationship",
    "description",
    "categorical_association",
    "prediction",
    "time_pattern",
    "clustering",
    "broad_exploratory",
    "explicit_method",
}

VALID_SCOPES = {"quick", "standard", "comprehensive"}

VALID_SOURCES = {"direct_tool_selection", "example_choice", "free_text", None}

# Fixed menu of example choices for objective_from_example_choice().
# Each entry declares the intent it produces and whether that intent,
# on its own, still needs specific variables named before an analysis
# could proceed (needs_clarification=True) or not (e.g. "explore"
# needs no variables to get started).
_EXAMPLE_CHOICES = {
    "compare_groups": {
        "intent": "group_comparison",
        "needs_clarification": True,
        "clarification_reason": "Which column is the outcome, and which column defines the groups?",
    },
    "relationship": {
        "intent": "relationship",
        "needs_clarification": True,
        "clarification_reason": "Which two (or more) variables do you want to relate?",
    },
    "describe": {
        "intent": "description",
        "needs_clarification": True,
        "clarification_reason": "Which variable(s) do you want summarized?",
    },
    "explore": {
        "intent": "broad_exploratory",
        "needs_clarification": False,
        "clarification_reason": None,
    },
}


def _blank_objective() -> dict:
    """The full schema, every field present, all defaults set.
    Every constructor below builds from this so the shape is always
    complete — no function ever returns a partial dict."""
    return {
        "intent": None,
        "outcome_variable": None,
        "grouping_variable": None,
        "variables_of_interest": [],
        "explicit_method": None,
        "scope": "standard",
        "source": None,
        "raw_user_text": None,
        "needs_clarification": False,
        "clarification_reason": None,
    }


def is_valid_objective(obj) -> bool:
    """Structural check only: right shape, right key set, allowed
    values where the field is constrained. Does NOT check whether
    the objective makes statistical sense or fits any dataset —
    that's a later step's job."""
    if not isinstance(obj, dict):
        return False

    expected_keys = set(_blank_objective().keys())
    if set(obj.keys()) != expected_keys:
        return False

    if obj["intent"] not in VALID_INTENTS:
        return False
    if obj["scope"] not in VALID_SCOPES:
        return False
    if obj["source"] not in VALID_SOURCES:
        return False
    if not isinstance(obj["variables_of_interest"], list):
        return False
    if not isinstance(obj["needs_clarification"], bool):
        return False
    for key in ("outcome_variable", "grouping_variable", "explicit_method",
                "raw_user_text", "clarification_reason"):
        if obj[key] is not None and not isinstance(obj[key], str):
            return False

    return True


def objective_from_tool_selection(tool_name: str, variables: dict) -> dict:
    """
    Build an objective from a direct tool + variables selection —
    the path an existing Builder-page user (who already knows what
    they want) takes today. This does not change or replace that
    existing flow; it's a new, optional function nothing currently
    calls.

    variables is the same shape the Builder page already collects:
    a dict mapping a tool's role names (from tool_registry.py) to
    the chosen column name(s). Recognised role-name conventions used
    across the existing tool registry — "value"/"group" for group
    comparisons, "x"/"y" for two-variable relationships — are mapped
    into outcome_variable/grouping_variable or
    variables_of_interest accordingly. Anything that doesn't match
    those conventions is still captured, in variables_of_interest,
    so no information is dropped even for tools with different role
    names.
    """
    obj = _blank_objective()
    obj["intent"] = "explicit_method"
    obj["explicit_method"] = tool_name
    obj["source"] = "direct_tool_selection"

    variables = variables or {}
    recognised = {"value", "group"}

    if "value" in variables:
        obj["outcome_variable"] = variables["value"]
    if "group" in variables:
        obj["grouping_variable"] = variables["group"]

    leftover = []
    for role_name, value in variables.items():
        if role_name in recognised:
            continue
        if isinstance(value, list):
            leftover.extend(value)
        else:
            leftover.append(value)
    obj["variables_of_interest"] = leftover

    return obj


def objective_from_example_choice(choice_key: str, outcome_variable: str = None,
                                   grouping_variable: str = None,
                                   variables_of_interest: list = None,
                                   scope: str = "standard") -> dict:
    """
    Build an objective from one of the fixed example choices (e.g.
    "compare_groups", "relationship", "describe", "explore"). If the
    caller already knows specific variables (e.g. a later UI step
    collected them), pass them in and needs_clarification is
    resolved to False automatically; otherwise the choice's default
    clarification state is used.

    Raises ValueError for an unrecognised choice_key — this is a
    programming-time check (the fixed menu is closed by design),
    not a runtime user-input validation concern.
    """
    if choice_key not in _EXAMPLE_CHOICES:
        raise ValueError(
            f"Unrecognised example choice '{choice_key}'. "
            f"Valid choices: {sorted(_EXAMPLE_CHOICES.keys())}"
        )
    if scope not in VALID_SCOPES:
        raise ValueError(f"Invalid scope '{scope}'. Valid scopes: {sorted(VALID_SCOPES)}")

    template = _EXAMPLE_CHOICES[choice_key]
    obj = _blank_objective()
    obj["intent"] = template["intent"]
    obj["source"] = "example_choice"
    obj["scope"] = scope
    obj["outcome_variable"] = outcome_variable
    obj["grouping_variable"] = grouping_variable
    obj["variables_of_interest"] = list(variables_of_interest) if variables_of_interest else []

    have_enough_info = bool(
        obj["outcome_variable"] or obj["grouping_variable"] or obj["variables_of_interest"]
    )
    if template["needs_clarification"] and not have_enough_info:
        obj["needs_clarification"] = True
        obj["clarification_reason"] = template["clarification_reason"]
    else:
        obj["needs_clarification"] = False
        obj["clarification_reason"] = None

    return obj
