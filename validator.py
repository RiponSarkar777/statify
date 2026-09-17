"""
Statify — Test-Specific Validator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates that a drafted analysis (chosen tool + assigned variables)
satisfies that tool's role contracts from tool_registry.py — correct
variable types, required roles filled, and cardinality (min/max
variable counts) respected.

This checks structural compliance only (does the selection match the
tool's declared contract) — it does not check statistical assumptions
(e.g. normality) or sample-size adequacy. That is a planned future
enhancement, not part of this module today.

Moved out of app.py — behavior unchanged.
"""
from tool_registry import get_tool


def _validate(analysis: dict, buckets: dict) -> tuple[bool, list[str]]:
    """Validate role contracts and parameters. Return (ok, errors)."""
    errors: list[str] = []
    if not analysis.get("title", "").strip():
        errors.append("Provide a title.")
    if not analysis.get("tool"):
        errors.append("Select a statistical tool.")
        return False, errors

    spec = get_tool(analysis["tool"])
    for role_name, role_def in spec["roles"].items():
        sel = analysis["variables"].get(role_name)
        allowed_cols = []
        for t in role_def["types"]:
            allowed_cols += buckets.get(t, [])
        if not sel:
            if not role_def.get("optional"):
                errors.append(f"Select **{role_def.get('label') or role_name}**.")
            continue
        # Coerce to list
        sel_list = sel if isinstance(sel, list) else [sel]
        # Cardinality
        if role_def.get("multi"):
            if len(sel_list) < role_def.get("min", 1):
                errors.append(f"`{role_name}` needs at least {role_def.get('min',1)} variable(s).")
            if role_def.get("max") and len(sel_list) > role_def["max"]:
                errors.append(f"`{role_name}` allows at most {role_def['max']} variable(s).")
        # Type compliance
        for c in sel_list:
            if c not in allowed_cols:
                errors.append(f"`{c}` is not a valid type for **{role_def.get('label') or role_name}**.")
    return len(errors) == 0, errors
