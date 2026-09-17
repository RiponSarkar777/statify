"""
Statify — Recommendation Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given the semantic-type buckets of a dataset (from profiler.py),
determine which registered tools (tool_registry.py) could apply.

Today this is a coarse compatibility filter: it checks whether the
dataset has ANY column of each required type — not whether the
user's specific chosen variables fit. Upgrading this to true
ranked recommendation is planned for a later phase.

Moved out of app.py — behavior unchanged.
"""
from tool_registry import STAT_TOOLS


def compatible_tools_for(buckets: dict) -> list[str]:
    """Return tools whose role contracts can be satisfied by the dataset's buckets."""
    avail = {t: bool(cols) for t, cols in buckets.items()}
    compat = []
    for name, spec in STAT_TOOLS.items():
        ok = True
        for r in spec["roles"].values():
            if r.get("optional"):
                continue
            if not any(avail.get(t, False) for t in r["types"]):
                ok = False
                break
        if ok:
            compat.append(name)
    return compat
