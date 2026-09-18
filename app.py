
"""
Statify v2.0 — AI-Era Statistical Analysis Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-file version. Copy this entire file to your GitHub repo as app.py.
No other files needed.
"""
from __future__ import annotations
import io, uuid, json, zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sp_stats


from theme import inject_global_css
from io_layer import load_file
from profiler import get_column_types, summarize_dataframe
from cleaning import handle_missing, convert_dtype, rename_column, delete_row, delete_column, update_cell
from tool_registry import STAT_TOOLS, get_tool, list_tools_by_category
from recommender import compatible_tools_for
from validator import _validate
from stats_engine import run_analysis
from graph_engine import GRAPH_CATALOG, list_graphs_for_tool, build_graph
from interpreter import interpret_result, interpret_graph
from export import export_pdf, export_excel, export_html, export_png_zip





APP_CONFIG = {
    "name": "Statify", "version": "2.0.0", "icon": "📊",
    "tagline": "AI-Era Statistical Analysis Platform",
    "max_analyses": 8,
}
PAGES = {
    1: {"title": "Upload",    "icon": "📤"},
    2: {"title": "Editor",    "icon": "✏️"},
    3: {"title": "Builder",   "icon": "🧪"},
    4: {"title": "Dashboard", "icon": "🗂"},
    5: {"title": "Report",    "icon": "📑"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOADER
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN TYPES
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# STATS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

"""
Statistical-tool registry.

Each entry declares:
  - category
  - description
  - variable role contracts (which roles, which semantic types are accepted, cardinality)
  - parameters (with defaults & types)
  - assumptions (for reporting)
  - recommended graphs (for Page 4 graph picker)
"""



# ═══════════════════════════════════════════════════════════════════════════════
# STATS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Statistical execution engine.

Each tool dispatches to a small isolated function returning a dict:
{
    "summary": dict[str, scalar],
    "tables":  dict[str, pandas.DataFrame],
    "extras":  dict (anything graph engine might need),
    "p_value": float | None,
    "alpha":   float,
    "ok": bool,
    "error": str (optional),
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


        
# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════


        

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

"""Page 1 — Upload"""


def page1_render():
    st.markdown(
        '<div class="hero-title">Upload your <span class="grad">dataset</span></div>'
        '<div class="hero-sub">CSV · XLSX · XLS · TSV · JSON · SPSS — Statify auto-detects format and types.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_up, col_meta = st.columns([1.2, 1])

    with col_up:
        st.markdown('<div class="sec-title">📤 Drop your file</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose dataset",
            type=["csv","xlsx","xls","tsv","json","sav"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            df, status, msg = load_file(uploaded)
            if status == "success":
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.session_state.filename = uploaded.name
                st.session_state.file_meta = summarize_dataframe(df)
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    with col_meta:
        if st.session_state.df is not None:
            meta = st.session_state.file_meta
            st.markdown('<div class="sec-title">📋 Dataset Snapshot</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="metric"><div class="label">Rows</div><div class="value">{meta["rows"]:,}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric"><div class="label">Columns</div><div class="value">{meta["cols"]}</div></div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            c3.markdown(f'<div class="metric"><div class="label">Missing</div><div class="value">{meta["missing"]:,}</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric"><div class="label">Memory KB</div><div class="value">{meta["memory_kb"]}</div></div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">👁️ Preview · first 5 rows</div>', unsafe_allow_html=True)
        prev = st.session_state.df.head(5).copy()
        prev.index = range(1, len(prev)+1)
        st.dataframe(prev, use_container_width=True)

        # Type pills
        st.markdown('<div class="sec-title" style="margin-top:1.2rem;">🧩 Variable Types</div>', unsafe_allow_html=True)
        b = st.session_state.file_meta["buckets"]
        type_html = ""
        for kind, color in [("numeric","accent"),("categorical","good"),
                            ("datetime","warn"),("boolean","accent"),("text","")]:
            for col in b.get(kind, []):
                cls = f"pill pill-{color}" if color else "pill"
                type_html += f"<span class='{cls}' title='{kind}'>{col}</span>"
        st.markdown(type_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button("⚡ Statify It →", type="primary", use_container_width=True):
                goto(2)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EDITOR
# ═══════════════════════════════════════════════════════════════════════════════



def _refresh_meta():
    st.session_state.file_meta = summarize_dataframe(st.session_state.df)


def page2_render():
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    st.markdown(
        '<div class="hero-title">Edit & <span class="grad">clean</span> your data</div>'
        '<div class="hero-sub">Spreadsheet-like editing with type conversion, missing-value handling, and instant preview.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_edit, tab_clean, tab_struct = st.tabs(["✏️ Cells", "🧼 Missing & Types", "🏗 Structure"])

    df = st.session_state.df

    # ============ CELLS ============
    with tab_edit:
        st.markdown('<div class="sec-title">Inline editor — changes saved instantly</div>', unsafe_allow_html=True)
        edited = st.data_editor(
            df,
            use_container_width=True,
            height=480,
            num_rows="dynamic",
            key="data_editor",
        )
        col_save, col_reset = st.columns([1,1])
        with col_save:
            changed = not edited.equals(df)
            if st.button("💾 Save Changes", type="primary", disabled=not changed, use_container_width=True):
                st.session_state.df = edited.reset_index(drop=True)
                _refresh_meta()
                st.success("Changes saved.")
                st.rerun()
        with col_reset:
            if st.button("↺ Reset to Original", use_container_width=True):
                st.session_state.df = st.session_state.df_original.copy()
                _refresh_meta()
                st.rerun()

    # ============ MISSING & TYPES ============
    with tab_clean:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-title">Missing-value handling</div>', unsafe_allow_html=True)
            col = st.selectbox("Column", df.columns, key="miss_col")
            strategy = st.selectbox(
                "Strategy",
                ["drop","mean","median","mode","zero","ffill","bfill","custom"],
                key="miss_strategy",
            )
            custom_val = st.text_input("Custom value", key="miss_val") if strategy == "custom" else None
            n_missing = int(df[col].isna().sum()) if col in df.columns else 0
            st.caption(f"{n_missing} missing in `{col}`")
            if st.button("Apply", type="primary", key="apply_miss", use_container_width=True):
                st.session_state.df = handle_missing(df, col, strategy, custom_val)
                _refresh_meta()
                st.success(f"Applied {strategy} to {col}.")
                st.rerun()

        with c2:
            st.markdown('<div class="sec-title">Type conversion</div>', unsafe_allow_html=True)
            col2 = st.selectbox("Column ", df.columns, key="dtype_col")
            target = st.selectbox("Target type", ["int","float","str","bool","datetime","category"], key="dtype_t")
            st.caption(f"Current: `{df[col2].dtype}`")
            if st.button("Convert", type="primary", key="apply_dtype", use_container_width=True):
                st.session_state.df = convert_dtype(df, col2, target)
                _refresh_meta()
                st.success(f"`{col2}` → {target}")
                st.rerun()

    # ============ STRUCTURE ============
    with tab_struct:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="sec-title">Rename column</div>', unsafe_allow_html=True)
            old = st.selectbox("Column to rename", df.columns, key="ren_old")
            new = st.text_input("New name", key="ren_new")
            if st.button("Rename", type="primary", key="apply_ren", use_container_width=True):
                if new.strip() and new != old:
                    st.session_state.df = rename_column(df, old, new.strip())
                    _refresh_meta()
                    st.success(f"`{old}` → `{new}`")
                    st.rerun()

        with c2:
            st.markdown('<div class="sec-title">Delete column</div>', unsafe_allow_html=True)
            dc = st.selectbox("Column", df.columns, key="del_col")
            if st.button("🗑 Delete column", key="apply_dc", use_container_width=True):
                st.session_state.df = delete_column(df, dc)
                _refresh_meta()
                st.warning(f"Deleted `{dc}`")
                st.rerun()

        with c3:
            st.markdown('<div class="sec-title">Delete row</div>', unsafe_allow_html=True)
            ri = st.number_input("Row index (0-based)", min_value=0, max_value=max(len(df)-1,0), step=1, key="del_row")
            if st.button("🗑 Delete row", key="apply_dr", use_container_width=True):
                st.session_state.df = delete_row(df, int(ri))
                _refresh_meta()
                st.warning(f"Row {ri} removed")
                st.rerun()

    # Quick stats
    st.markdown("<br>", unsafe_allow_html=True)
    meta = st.session_state.file_meta
    cs = st.columns(4)
    cs[0].markdown(f'<div class="metric"><div class="label">Rows</div><div class="value">{meta["rows"]:,}</div></div>', unsafe_allow_html=True)
    cs[1].markdown(f'<div class="metric"><div class="label">Columns</div><div class="value">{meta["cols"]}</div></div>', unsafe_allow_html=True)
    cs[2].markdown(f'<div class="metric"><div class="label">Missing</div><div class="value">{meta["missing"]:,}</div></div>', unsafe_allow_html=True)
    cs[3].markdown(f'<div class="metric"><div class="label">Duplicates</div><div class="value">{meta["duplicates"]:,}</div></div>', unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

"""
Page 3 — Statistical Analysis Builder (REDESIGNED).

Fixes from previous version:
  • Dynamic role-based variable filtering (numeric/cat/datetime/bool)
  • Tool dropdown filtered by dataset compatibility
  • Per-role multi-select with cardinality validation
  • Parameter forms generated dynamically from registry
  • Up to 8 saved analyses with proper edit/delete
  • Validation messages, AI placeholder slots
"""


def _new_analysis_template():
    return {
        "id": uuid.uuid4().hex[:8],
        "title": "",
        "tool": None,
        "variables": {},
        "params": {},
    }




def _render_role_selector(role_name, role_def, buckets, current_value, key_prefix):
    allowed = []
    for t in role_def["types"]:
        allowed += buckets.get(t, [])
    label = role_def.get("label") or role_name.replace("_", " ").title()
    type_hint = "/".join(sorted(role_def["types"]))
    help_text = f"Allowed types: {type_hint}"

    if role_def.get("multi"):
        default = current_value if isinstance(current_value, list) else []
        return st.multiselect(label, options=allowed, default=[v for v in default if v in allowed],
                                help=help_text, key=f"{key_prefix}_{role_name}")
    else:
        idx = 0
        if current_value in allowed:
            idx = allowed.index(current_value) + 1
        opts = ["— select —"] + allowed
        choice = st.selectbox(label, options=opts, index=idx,
                                help=help_text, key=f"{key_prefix}_{role_name}")
        return None if choice == "— select —" else choice


def _render_param_form(spec, current_params, key_prefix):
    out = {}
    if not spec["params"]:
        st.caption("This tool has no extra parameters.")
        return out
    cols = st.columns(min(3, len(spec["params"])))
    for i, (pname, pdef) in enumerate(spec["params"].items()):
        with cols[i % len(cols)]:
            label = pdef.get("label") or pname.replace("_", " ").title()
            cur = current_params.get(pname, pdef.get("default"))
            ptype = pdef.get("type", "text")
            if ptype == "number":
                out[pname] = st.number_input(label, value=float(cur), key=f"{key_prefix}_p_{pname}", format="%.4f")
            elif ptype == "select":
                opts = pdef["options"]
                out[pname] = st.selectbox(label, options=opts, index=opts.index(cur) if cur in opts else 0, key=f"{key_prefix}_p_{pname}")
            elif ptype == "boolean":
                out[pname] = st.checkbox(label, value=bool(cur), key=f"{key_prefix}_p_{pname}")
            else:
                out[pname] = st.text_input(label, value=str(cur), key=f"{key_prefix}_p_{pname}")
    return out


def page3_render():
    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
        return

    st.markdown(
        '<div class="hero-title">Build your <span class="grad">analysis</span></div>'
        '<div class="hero-sub">Choose a statistical tool, assign variables to roles, and set parameters. Save up to 8 analyses.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    df = st.session_state.df
    buckets = get_column_types(df)

    # AI placeholder
    with st.expander("🤖 AI Recommendations (Coming Soon)", expanded=False):
        st.info(
            "Statify's AI engine will soon recommend the best test based on your dataset, "
            "auto-select compatible variables, and explain trade-offs. "
            "Slot is reserved for `core.ai.recommendations` integration."
        )

    # Layout: builder on left, saved cards on right
    col_build, col_saved = st.columns([1.4, 1])

    # =================== BUILDER ===================
    with col_build:
        st.markdown('<div class="sec-title">🧪 New Analysis</div>', unsafe_allow_html=True)

        if "p3_draft" not in st.session_state:
            st.session_state.p3_draft = _new_analysis_template()
        draft = st.session_state.p3_draft

        # Title
        draft["title"] = st.text_input("Analysis Title", value=draft.get("title",""),
                                          placeholder="e.g. Income vs Spending Score",
                                          key=f"title_{draft['id']}")

        # Tool picker — categorised + compatibility-filtered
        compat = set(compatible_tools_for(buckets))
        cats = list_tools_by_category()
        cat_choice = st.selectbox("Category", ["All"] + sorted(cats.keys()), key=f"cat_{draft['id']}")
        if cat_choice == "All":
            tool_list = sorted(STAT_TOOLS.keys())
        else:
            tool_list = sorted(cats[cat_choice])
        # Annotate compatibility
        annotated = [(t, "✓" if t in compat else "✕") for t in tool_list]
        labels = [f"{ok}  {t}" for t, ok in annotated]
        cur_tool = draft.get("tool")
        cur_idx = 0
        if cur_tool:
            for i, t in enumerate(tool_list):
                if t == cur_tool: cur_idx = i; break
        chosen_label = st.selectbox("Statistical Tool", labels, index=cur_idx, key=f"tool_{draft['id']}")
        chosen_tool = tool_list[labels.index(chosen_label)]

        if chosen_tool != draft.get("tool"):
            draft["tool"] = chosen_tool
            draft["variables"] = {}
            draft["params"] = {pn: pd.get("default") for pn, pd in get_tool(chosen_tool)["params"].items()}

        spec = get_tool(chosen_tool)
        if chosen_tool not in compat:
            st.warning(f"⚠ This tool's variable requirements aren't fully met by your dataset. Missing types: " +
                        ", ".join(sorted({t for r in spec["roles"].values() for t in r["types"]})))
        st.markdown(f"<div class='card'><b>{chosen_tool}</b><div style='color:#9aa0b4;font-size:.85rem;margin-top:6px;'>{spec['desc']}</div></div>", unsafe_allow_html=True)

        # Roles
        st.markdown('<div class="sec-title" style="margin-top:1rem;">📐 Variables</div>', unsafe_allow_html=True)
        for rname, rdef in spec["roles"].items():
            cur = draft["variables"].get(rname)
            sel = _render_role_selector(rname, rdef, buckets, cur, key_prefix=f"v_{draft['id']}")
            draft["variables"][rname] = sel

        # Params
        st.markdown('<div class="sec-title" style="margin-top:1rem;">⚙️ Parameters</div>', unsafe_allow_html=True)
        draft["params"] = _render_param_form(spec, draft["params"], key_prefix=f"par_{draft['id']}")

        # Assumptions
        if spec["assumptions"]:
            st.markdown('<div class="sec-title" style="margin-top:1rem;">📋 Assumptions</div>', unsafe_allow_html=True)
            html = "".join(f"<span class='pill'>{a}</span>" for a in spec["assumptions"])
            st.markdown(html, unsafe_allow_html=True)

        # Recommended graphs
        if spec.get("graphs"):
            names = [GRAPH_CATALOG[g][0] for g in spec["graphs"] if g in GRAPH_CATALOG]
            st.markdown(
                '<div class="sec-title" style="margin-top:1rem;">📊 Recommended Graphs</div>'
                + "".join(f"<span class='pill pill-accent'>{n}</span>" for n in names),
                unsafe_allow_html=True,
            )

        # Validate + save
        ok, errs = _validate(draft, buckets)
        st.markdown("<br>", unsafe_allow_html=True)
        if errs:
            for e in errs:
                st.error(e)

        cap = len(st.session_state.analyses) >= APP_CONFIG["max_analyses"]
        c_save, c_clear = st.columns([2,1])
        with c_save:
            if st.button("💾 Save Analysis", type="primary", disabled=not ok or cap, use_container_width=True):
                st.session_state.analyses.append(dict(draft))
                st.session_state.p3_draft = _new_analysis_template()
                st.success("Analysis saved.")
                st.rerun()
        with c_clear:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.p3_draft = _new_analysis_template()
                st.rerun()

        if cap:
            st.warning(f"Maximum of {APP_CONFIG['max_analyses']} analyses reached. Delete one to add more.")

    # =================== SAVED ===================
    with col_saved:
        st.markdown(f'<div class="sec-title">📂 Saved ({len(st.session_state.analyses)}/{APP_CONFIG["max_analyses"]})</div>', unsafe_allow_html=True)

        if not st.session_state.analyses:
            st.markdown("<div class='card' style='text-align:center;color:#9aa0b4;'>No analyses saved yet.</div>", unsafe_allow_html=True)
        else:
            for i, a in enumerate(st.session_state.analyses):
                vars_str = " · ".join(
                    f"{k}: {', '.join(v) if isinstance(v,list) else v}"
                    for k, v in a["variables"].items() if v
                )
                st.markdown(
                    f"""
                    <div class='analysis-card'>
                        <h4>{i+1}. {a.get('title','Untitled')}</h4>
                        <div class='meta'><b>Tool:</b> {a.get('tool')}</div>
                        <div class='meta'>{vars_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✏️ Edit", key=f"edit_{a['id']}", use_container_width=True):
                        st.session_state.p3_draft = dict(a)
                        st.session_state.analyses.pop(i)
                        st.rerun()
                with cc2:
                    if st.button("🗑", key=f"del_{a['id']}", use_container_width=True):
                        st.session_state.analyses.pop(i)
                        st.rerun()
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

"""Page 4 — Analysis Selection Dashboard"""


def page4_render():
    if not st.session_state.analyses:
        st.warning("Save at least one analysis on Page 3 first.")
        return

    st.markdown(
        '<div class="hero-title">Run your <span class="grad">analyses</span></div>'
        '<div class="hero-sub">Pick graphs and launch the dashboard for any saved analysis.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, a in enumerate(st.session_state.analyses):
        with cols[idx % 2]:
            spec = get_tool(a["tool"])
            vars_str = " · ".join(
                f"<b>{k}</b>: {', '.join(v) if isinstance(v,list) else v}"
                for k, v in a["variables"].items() if v
            )
            params_str = " · ".join(f"{k}={v}" for k, v in a["params"].items()) or "default"

            st.markdown(
                f"""
                <div class='analysis-card'>
                    <h4>📌 {a.get('title','Untitled')}</h4>
                    <div class='meta'>🧪 <b>{a.get('tool')}</b></div>
                    <div class='meta'>{vars_str}</div>
                    <div class='meta'>⚙️ {params_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Graph picker
            graph_opts = list_graphs_for_tool(spec)
            graph_id_map = {label: gid for gid, label in graph_opts}
            cur = st.session_state.graph_selections.get(a["id"], [label for _, label in graph_opts])
            chosen_labels = st.multiselect(
                "Graphs to include",
                options=list(graph_id_map.keys()),
                default=[c for c in cur if c in graph_id_map],
                key=f"graphs_{a['id']}",
            )
            st.session_state.graph_selections[a["id"]] = chosen_labels

            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("▶ Run Analysis", key=f"run_{a['id']}", type="primary", use_container_width=True):
                    st.session_state.active_analysis = a["id"]
                    goto(5)
            with cc2:
                if st.button("📊 Graphs Only", key=f"graphs_only_{a['id']}", use_container_width=True):
                    st.session_state.active_analysis = a["id"]
                    goto(5)
            st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORT
# ═══════════════════════════════════════════════════════════════════════════════

"""Page 5 — Professional Dashboard & Report"""


def _find_analysis(aid):
    for a in st.session_state.analyses:
        if a["id"] == aid:
            return a
    return None


def page5_render():
    aid = st.session_state.active_analysis
    if aid is None:
        st.warning("Select an analysis from Page 4 first.")
        return
    analysis = _find_analysis(aid)
    if analysis is None:
        st.error("Analysis not found.")
        return

    # Compute (cached)
    if aid not in st.session_state.analysis_results:
        with st.spinner("Computing..."):
            st.session_state.analysis_results[aid] = run_analysis(
                analysis["tool"], st.session_state.df,
                analysis["variables"], analysis["params"],
            )
    result = st.session_state.analysis_results[aid]
    interp = interpret_result(analysis["tool"], result)

    # ---------- TOP NAV ----------
    top1, top2, top3 = st.columns([1, 4, 2])
    with top1:
        if st.button("← Back", key="report_back", use_container_width=True):
            goto(4)
    with top2:
        st.markdown(
            f"<div style='font-weight:700;font-size:1.4rem;'>📑 {analysis.get('title','Report')}</div>"
            f"<div class='meta' style='color:#9aa0b4;font-size:.85rem;'>{analysis['tool']}  ·  generated {datetime.now():%Y-%m-%d %H:%M}</div>",
            unsafe_allow_html=True,
        )
    with top3:
        if st.button("🔄 Recompute", use_container_width=True):
            st.session_state.analysis_results.pop(aid, None)
            st.rerun()

    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    if not result.get("ok", True):
        st.error(f"❌ {result.get('error','Analysis failed')}")
        return

    # ---------- HEADLINE CARD ----------
    st.markdown(
        f"""
        <div class='glass'>
            <div class='sec-title'>🎯 Headline</div>
            <div style='font-size:1.15rem;font-weight:600;color:#ECEEF6;'>{interp['headline']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- KEY METRICS ----------
    s = result.get("summary", {})
    if s:
        keys = list(s.keys())[:4]
        cs = st.columns(max(len(keys), 1))
        for i, k in enumerate(keys):
            v = s[k]
            try:
                v_str = f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)
            except Exception:
                v_str = str(v)
            cs[i].markdown(
                f"<div class='metric'><div class='label'>{k.replace('_',' ')}</div><div class='value'>{v_str}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # ---------- TABS ----------
    tab_overview, tab_tables, tab_graphs, tab_interpret, tab_export = st.tabs(
        ["🎯 Overview", "📋 Tables", "📊 Graphs", "🧠 Interpretation", "📤 Export"]
    )

    # === Overview ===
    with tab_overview:
        st.markdown(f"<div class='card'><b>Explanation</b><br>{interp['explanation']}</div>", unsafe_allow_html=True)
        if interp.get("statistical"):
            st.markdown(f"<div class='card' style='margin-top:.8rem;'><b>Statistical Meaning</b><br>{interp['statistical']}</div>", unsafe_allow_html=True)
        if interp.get("conclusion"):
            st.markdown(f"<div class='card' style='margin-top:.8rem;'><b>Conclusion</b><br>{interp['conclusion']}</div>", unsafe_allow_html=True)
        if interp.get("assumptions") and interp["assumptions"] != "—":
            st.markdown(f"<div class='card' style='margin-top:.8rem;'><b>Assumptions</b><br>{interp['assumptions']}</div>", unsafe_allow_html=True)

        # Significance pill
        p = result.get("p_value"); a = result.get("alpha", 0.05)
        if p is not None:
            sig = p < a
            cls = "pill-good" if sig else "pill-bad"
            txt = "Significant" if sig else "Not Significant"
            st.markdown(f"<div style='margin-top:1rem;'><span class='pill {cls}'>{txt} · p = {p:.4f} · α = {a}</span></div>", unsafe_allow_html=True)

    # === Tables ===
    with tab_tables:
        for tname, tdf in result.get("tables", {}).items():
            st.markdown(f"<div class='sec-title'>{tname}</div>", unsafe_allow_html=True)
            st.dataframe(tdf, use_container_width=True)

    # === Graphs ===
    figs_for_export = []
    with tab_graphs:
        spec = get_tool(analysis["tool"])
        graph_opts = list_graphs_for_tool(spec)
        chosen_labels = st.session_state.graph_selections.get(aid)
        if chosen_labels is None:
            chosen_labels = [label for _, label in graph_opts]
        chosen_ids = [gid for gid, label in graph_opts if label in chosen_labels]

        if not chosen_ids:
            st.info("No graphs selected. Go to Page 4 to choose visualizations.")
        for gid in chosen_ids:
            label = GRAPH_CATALOG[gid][0]
            try:
                fig = build_graph(gid, result)
            except Exception as e:
                st.warning(f"Could not render {label}: {e}")
                continue
            if fig is None:
                continue
            # Could be list (e.g. histograms per variable)
            figs = fig if isinstance(fig, list) else [fig]
            for j, f in enumerate(figs):
                title = f"{label}" + (f" #{j+1}" if len(figs) > 1 else "")
                if isinstance(f, dict) and "_img" in f:
                    st.markdown(f"<div class='sec-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<img src='{f['_img']}' style='width:100%;border-radius:12px;'>", unsafe_allow_html=True)
                else:
                    st.plotly_chart(f, use_container_width=True)
                with st.expander(f"💡 What does this {label} mean?"):
                    st.write(interpret_graph(gid))
                figs_for_export.append((title, f))

    # === Interpretation tab ===
    with tab_interpret:
        st.markdown(f"### {interp['headline']}")
        st.write(f"**Plain-language summary**\n\n{interp['explanation']}")
        if interp.get("statistical"):
            st.write(f"**Statistical reasoning**\n\n{interp['statistical']}")
        if interp.get("assumptions") and interp["assumptions"] != "—":
            st.write(f"**Assumptions**\n\n{interp['assumptions']}")
        if interp.get("conclusion"):
            st.write(f"**Bottom line**\n\n{interp['conclusion']}")

    # === Export ===
    with tab_export:
        st.markdown('<div class="sec-title">Download your report</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        safe_title = "".join(c if c.isalnum() else "_" for c in analysis.get("title","report"))[:40] or "report"

        with c1:
            try:
                pdf_bytes = export_pdf(analysis, result, interp, figs_for_export)
                st.download_button("📕 PDF", pdf_bytes, file_name=f"{safe_title}.pdf",
                                     mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF: {e}")

        with c2:
            try:
                xls = export_excel(analysis, result, interp)
                st.download_button("📗 Excel", xls, file_name=f"{safe_title}.xlsx",
                                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                     use_container_width=True)
            except Exception as e:
                st.error(f"Excel: {e}")

        with c3:
            try:
                html = export_html(analysis, result, interp, figs_for_export)
                st.download_button("🌐 HTML", html, file_name=f"{safe_title}.html",
                                     mime="text/html", use_container_width=True)
            except Exception as e:
                st.error(f"HTML: {e}")

        with c4:
            try:
                if figs_for_export:
                    z = export_png_zip(figs_for_export)
                    st.download_button("🖼 PNG ZIP", z, file_name=f"{safe_title}_charts.zip",
                                         mime="application/zip", use_container_width=True)
                else:
                    st.button("🖼 PNG ZIP", disabled=True, use_container_width=True)
            except Exception as e:
                st.error(f"PNG: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def goto(page: int):
    st.session_state.page = max(1, min(5, page))
    st.rerun()


def render_progress(current: int):
    parts = ['<div class="progress-wrap">']
    for i in range(1, 6):
        cls = "progress-step"
        if i == current:  cls += " active"
        elif i < current: cls += " done"
        meta = PAGES[i]
        parts.append(f'<div class="{cls}">{meta["icon"]} &nbsp; {i}. {meta["title"]}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_footer_nav():
    page         = st.session_state.page
    df_loaded    = st.session_state.df is not None
    has_analysis = len(st.session_state.analyses) > 0
    next_enabled = {
        1: df_loaded, 2: df_loaded, 3: has_analysis,
        4: st.session_state.active_analysis is not None, 5: False,
    }[page]
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    cols = st.columns([1, 4, 1])
    with cols[0]:
        if page > 1:
            if st.button("← Back", key=f"back_{page}", use_container_width=True):
                goto(page - 1)
    with cols[2]:
        if page < 5:
            if st.button("Next →", key=f"next_{page}", type="primary",
                         disabled=not next_enabled, use_container_width=True):
                goto(page + 1)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div class="brand">{APP_CONFIG["name"]}</div>'
            f'<div class="brand-sub">v{APP_CONFIG["version"]} · {APP_CONFIG["tagline"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='div' style='margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Pages</div>', unsafe_allow_html=True)
        for i in range(1, 6):
            label    = f"{PAGES[i]['icon']}  {i}. {PAGES[i]['title']}"
            disabled = (i > 1 and st.session_state.df is None) or \
                       (i >= 4 and not st.session_state.analyses)
            btn_type = "primary" if st.session_state.page == i else "secondary"
            if st.button(label, key=f"nav_{i}", disabled=disabled,
                         use_container_width=True, type=btn_type):
                goto(i)
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown('<div class="sec-title">Dataset</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card" style="padding:0.8rem;">'
                f'<div style="font-weight:600;font-size:0.9rem;">{st.session_state.filename or "data"}</div>'
                f'<div style="color:#9aa0b4;font-size:0.78rem;margin-top:4px;">'
                f'{df.shape[0]:,} rows × {df.shape[1]} cols</div></div>',
                unsafe_allow_html=True,
            )
        if st.session_state.analyses:
            st.markdown('<div class="sec-title" style="margin-top:1rem;">Analyses</div>', unsafe_allow_html=True)
            for a in st.session_state.analyses:
                st.markdown(f"<div class='pill pill-accent'>{a.get('title','Untitled')}</div>", unsafe_allow_html=True)
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        if st.button("🔄 Reset Session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "page": 1, "df": None, "df_original": None,
        "filename": "", "file_meta": {}, "edit_history": [],
        "analyses": [], "active_analysis": None,
        "analysis_results": {}, "graph_selections": {},
        "ai_suggestions": [], "theme": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(
        page_title=APP_CONFIG["name"],
        page_icon=APP_CONFIG["icon"],
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    inject_global_css()
    render_sidebar()
    render_progress(st.session_state.page)

    {
        1: page1_render,
        2: page2_render,
        3: page3_render,
        4: page4_render,
        5: page5_render,
    }[st.session_state.page]()

    render_footer_nav()


if __name__ == "__main__":
    main()


