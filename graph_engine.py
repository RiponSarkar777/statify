"""
Statify — Graph Builder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plotly-based graph builder. Each function returns a
plotly.graph_objects.Figure (except _dendrogram, which returns a
dict with a base64-encoded PNG, since dendrograms are drawn with
matplotlib/scipy rather than Plotly).

Given a tool's result dict (from stats_engine.py) and a chosen
graph_id, build_graph() renders the corresponding visualization.
list_graphs_for_tool() tells the UI which graph options are valid
for a given tool spec (from tool_registry.py).

Moved out of app.py — behavior unchanged. No rendering logic altered.

Notes:
- GRAPH_CATALOG is used by app.py directly (Builder and Results
  pages), so it is imported there, not just used internally.
- Several individual graph-building functions do their own local
  imports (scipy, sklearn, matplotlib, io, base64) inside the
  function body — unchanged from the original, intentional
  lazy-loading, not an omission.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Modern dark theme template
TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#11131b",
        plot_bgcolor="#11131b",
        font=dict(family="Inter, sans-serif", color="#ECEEF6", size=12),
        xaxis=dict(gridcolor="#232636", zerolinecolor="#232636"),
        yaxis=dict(gridcolor="#232636", zerolinecolor="#232636"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=["#7C5CFF","#4FD1C5","#F687B3","#F6AD55","#34d399","#60a5fa","#a78bfa","#f472b6"],
    )
)


def _apply(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=dict(text=title, x=0.02, font=dict(size=14)))
    return fig


GRAPH_CATALOG = {
    "histogram":       ("Histogram",          "_hist"),
    "box":             ("Box Plot",           "_box"),
    "violin":          ("Violin Plot",        "_violin"),
    "kde":             ("Density (KDE)",      "_kde"),
    "qq":              ("Q-Q Plot",           "_qq"),
    "scatter":         ("Scatter Plot",       "_scatter"),
    "regression":      ("Regression Line",    "_regression"),
    "residual":        ("Residual Plot",      "_residual"),
    "bar":             ("Bar Chart",          "_bar"),
    "stacked_bar":     ("Stacked Bar",        "_stacked_bar"),
    "pie":             ("Pie Chart",          "_pie"),
    "line":            ("Line Chart",         "_line"),
    "heatmap":         ("Heatmap",            "_heatmap"),
    "pair":            ("Pair Plot",          "_pair"),
    "cm":              ("Confusion Matrix",   "_cm"),
    "roc":             ("ROC Curve",          "_roc"),
    "dendrogram":      ("Dendrogram",         "_dendrogram"),
}


def list_graphs_for_tool(tool_spec: dict) -> list[tuple[str,str]]:
    ids = tool_spec.get("graphs", [])
    return [(g, GRAPH_CATALOG[g][0]) for g in ids if g in GRAPH_CATALOG]


def _hist(result):
    figs = []
    for c in result["extras"]["data"].columns:
        fig = px.histogram(result["extras"]["data"], x=c, nbins=30)
        figs.append(_apply(fig, f"Histogram — {c}"))
    return figs[0] if len(figs) == 1 else figs


def _box(result):
    ex = result["extras"]
    if "g1" in ex:
        df = pd.concat([
            pd.DataFrame({"Value": ex["g1"], "Group": ex.get("groups",["A","B"])[0]}),
            pd.DataFrame({"Value": ex["g2"], "Group": ex.get("groups",["A","B"])[1]}),
        ])
        fig = px.box(df, x="Group", y="Value")
        return _apply(fig, f"Box Plot — {ex['value']} by {ex['group']}")
    data = ex["data"]
    fig = px.box(data)
    return _apply(fig, "Box Plot")


def _violin(result):
    ex = result["extras"]
    if "g1" in ex:
        df = pd.concat([
            pd.DataFrame({"Value": ex["g1"], "Group": ex.get("groups",["A","B"])[0]}),
            pd.DataFrame({"Value": ex["g2"], "Group": ex.get("groups",["A","B"])[1]}),
        ])
        fig = px.violin(df, x="Group", y="Value", box=True)
        return _apply(fig, f"Violin — {ex['value']} by {ex['group']}")
    data = ex["data"]
    fig = px.violin(data, box=True)
    return _apply(fig, "Violin Plot")


def _kde(result):
    from scipy.stats import gaussian_kde
    data = result["extras"]["data"]
    col = data.columns[0] if hasattr(data, "columns") else None
    s = data[col] if col else data
    kde = gaussian_kde(s)
    xs = np.linspace(s.min(), s.max(), 200)
    ys = kde(xs)
    fig = go.Figure(go.Scatter(x=xs, y=ys, fill="tozeroy", mode="lines"))
    return _apply(fig, "Density (KDE)")


def _qq(result):
    from scipy import stats as sp
    data = result["extras"]["data"]
    col = data.columns[0] if hasattr(data, "columns") else None
    s = (data[col] if col else data).dropna()
    osm, osr = sp.probplot(s, dist="norm", fit=False)
    fig = go.Figure()
    fig.add_scatter(x=osm[0], y=osm[1], mode="markers", name="Data")
    fig.add_scatter(x=osm[0], y=osm[0], mode="lines", name="Reference")
    return _apply(fig, f"Q-Q Plot — {col}")


def _scatter(result):
    ex = result["extras"]
    if "x" in ex and "y" in ex:
        fig = px.scatter(ex["data"], x=ex["x"], y=ex["y"])
        return _apply(fig, f"Scatter — {ex['y']} vs {ex['x']}")
    data = ex["data"]
    fig = px.scatter(data, x=data.columns[0], y=data.columns[1])
    return _apply(fig, "Scatter")


def _regression(result):
    ex = result["extras"]
    if "pred" in ex and "x" in ex:
        fig = go.Figure()
        fig.add_scatter(x=ex["data"][ex["x"]], y=ex["data"][ex["y"]], mode="markers", name="Data")
        fig.add_scatter(x=ex["data"][ex["x"]], y=ex["pred"], mode="lines", name="Fit")
        return _apply(fig, f"Regression — {ex['y']} ~ {ex['x']}")
    if "pred" in ex:
        fig = go.Figure()
        fig.add_scatter(y=ex["data"][ex["y"]].values, mode="markers", name="Actual")
        fig.add_scatter(y=ex["pred"].values if hasattr(ex["pred"],"values") else ex["pred"], mode="lines", name="Predicted")
        return _apply(fig, "Predicted vs Actual")
    return None


def _residual(result):
    ex = result["extras"]
    resid = ex.get("resid")
    pred = ex.get("pred")
    fig = go.Figure(go.Scatter(x=pred, y=resid, mode="markers"))
    fig.add_hline(y=0, line_dash="dash")
    return _apply(fig, "Residual Plot")


def _bar(result):
    for tname, tbl in result["tables"].items():
        if tbl.shape[1] >= 2 and tbl.shape[0] <= 50:
            cols = tbl.columns
            try:
                fig = px.bar(tbl, x=cols[0], y=cols[1])
                return _apply(fig, tname)
            except Exception:
                continue
    return None


def _stacked_bar(result):
    ex = result["extras"]
    if "matrix" in ex:
        ct = ex["matrix"]
        df = ct.reset_index().melt(id_vars=ct.index.name or "index")
        fig = px.bar(df, x=df.columns[0], y="value", color=df.columns[1], barmode="stack")
        return _apply(fig, "Stacked Bar")
    return None


def _pie(result):
    for tname, tbl in result["tables"].items():
        if tbl.shape[1] >= 2 and tbl.shape[0] <= 20:
            cols = tbl.columns
            try:
                fig = px.pie(tbl, names=cols[0], values=cols[1])
                return _apply(fig, "Pie Chart")
            except Exception:
                continue
    return None


def _line(result):
    ex = result["extras"]
    if "forecast" in ex:
        hist = ex["data"]
        fc = ex["forecast"]
        fig = go.Figure()
        fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Observed")
        fig.add_scatter(x=fc.index, y=fc.values, mode="lines", name="Forecast", line=dict(dash="dash"))
        return _apply(fig, "Forecast")
    if "result" in ex:
        r = ex["result"]
        fig = go.Figure()
        fig.add_scatter(x=r.observed.index, y=r.observed.values, mode="lines", name="Observed")
        fig.add_scatter(x=r.trend.index, y=r.trend.values, mode="lines", name="Trend")
        return _apply(fig, "Time Series Decomposition")
    if "kmf" in ex:
        kmf = ex["kmf"]
        sf = kmf.survival_function_.reset_index()
        fig = px.line(sf, x=sf.columns[0], y=sf.columns[1])
        return _apply(fig, "Kaplan-Meier Survival")
    data = ex.get("data")
    if data is not None and hasattr(data, "columns") and len(data.columns) >= 1:
        fig = px.line(data)
        return _apply(fig, "Line Chart")
    return None


def _heatmap(result):
    ex = result["extras"]
    if "matrix" in ex:
        m = ex["matrix"]
        fig = px.imshow(m, text_auto=True, aspect="auto")
        return _apply(fig, "Heatmap")
    return None


def _pair(result):
    data = result["extras"]["data"]
    fig = px.scatter_matrix(data)
    return _apply(fig, "Pair Plot")


def _cm(result):
    ex = result["extras"]
    cm_tbl = result["tables"].get("Confusion Matrix")
    if cm_tbl is None:
        return None
    fig = px.imshow(cm_tbl.set_index(cm_tbl.columns[0]) if cm_tbl.columns[0]=="" else cm_tbl, text_auto=True)
    return _apply(fig, "Confusion Matrix")


def _roc(result):
    from sklearn.metrics import roc_curve, auc
    ex = result["extras"]
    if "y_true" not in ex or "y_proba" not in ex:
        return None
    fpr, tpr, _ = roc_curve(ex["y_true"], ex["y_proba"])
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})")
    fig.add_scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Chance")
    return _apply(fig, "ROC Curve")


def _dendrogram(result):
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64
    ex = result["extras"]
    fig, ax = plt.subplots(figsize=(8,5))
    fig.patch.set_facecolor("#11131b")
    ax.set_facecolor("#11131b")
    dendrogram(ex["linkage"], ax=ax, color_threshold=0)
    ax.tick_params(colors="#ECEEF6"); ax.spines[:].set_color("#232636")
    ax.set_title("Dendrogram", color="#ECEEF6")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#11131b"); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode()
    return {"_img": f"data:image/png;base64,{img}"}


_BUILDERS = {
    "_hist": _hist, "_box": _box, "_violin": _violin, "_kde": _kde, "_qq": _qq,
    "_scatter": _scatter, "_regression": _regression, "_residual": _residual,
    "_bar": _bar, "_stacked_bar": _stacked_bar, "_pie": _pie, "_line": _line,
    "_heatmap": _heatmap, "_pair": _pair, "_cm": _cm, "_roc": _roc,
    "_dendrogram": _dendrogram,
}


def build_graph(graph_id: str, result: dict):
    if graph_id not in GRAPH_CATALOG: return None
    fn_name = GRAPH_CATALOG[graph_id][1]
    return _BUILDERS[fn_name](result)
