"""
Statify — Theme / Styling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Global CSS injection for the Streamlit UI.
Moved out of app.py — behavior unchanged.
"""
import streamlit as st


def inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {
            --bg:        #0a0b10;
            --bg-2:      #11131b;
            --bg-3:      #161924;
            --line:      #232636;
            --line-2:    #2a2e42;
            --text:      #ECEEF6;
            --text-2:    #9aa0b4;
            --text-3:    #6b6f82;
            --accent:    #7C5CFF;
            --accent-2:  #4FD1C5;
            --good:      #34d399;
            --warn:      #f59e0b;
            --bad:       #ef4444;
            --grad: linear-gradient(135deg,#7C5CFF 0%,#4FD1C5 100%);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
        }
        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            max-width: 1400px !important;
            padding: 1.2rem 2rem 6rem 2rem !important;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--bg-2) !important;
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] * { color: var(--text) !important; }

        /* ---------- Brand ---------- */
        .brand {
            font-weight: 800;
            font-size: 1.6rem;
            letter-spacing: -0.02em;
            background: var(--grad);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .brand-sub {
            font-size: 0.75rem;
            color: var(--text-3);
            margin-top: -4px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* ---------- Glass cards ---------- */
        .glass {
            background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005));
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            backdrop-filter: blur(14px);
            transition: border-color .2s, transform .2s;
        }
        .glass:hover { border-color: var(--line-2); }

        .card {
            background: var(--bg-2);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
        }

        /* ---------- Hero ---------- */
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            line-height: 1.1;
            margin: 0;
        }
        .hero-title .grad {
            background: var(--grad);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            color: var(--text-2);
            font-size: 1rem;
            margin-top: 0.4rem;
            font-weight: 400;
        }

        /* ---------- Pills / Badges ---------- */
        .pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            background: var(--bg-3);
            border: 1px solid var(--line);
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-2);
            margin: 2px 4px 2px 0;
        }
        .pill-accent  { color: var(--accent-2); border-color: rgba(79,209,197,0.3); background: rgba(79,209,197,0.07); }
        .pill-good    { color: var(--good);     border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.07); }
        .pill-warn    { color: var(--warn);     border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.07); }
        .pill-bad     { color: var(--bad);      border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.07); }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: var(--bg-3);
            color: var(--text) !important;
            border: 1px solid var(--line);
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.1rem !important;
            transition: all .15s ease;
        }
        .stButton > button:hover {
            border-color: var(--accent);
            transform: translateY(-1px);
        }
        .stButton > button[kind="primary"] {
            background: var(--grad) !important;
            color: #0a0b10 !important;
            border: none !important;
            font-weight: 700 !important;
        }
        .stButton > button:disabled {
            background: var(--bg-2) !important;
            color: var(--text-3) !important;
            border-color: var(--line) !important;
        }

        /* ---------- Inputs ---------- */
        .stTextInput input, .stNumberInput input, .stSelectbox > div > div,
        .stMultiSelect > div > div, .stTextArea textarea {
            background: var(--bg-3) !important;
            color: var(--text) !important;
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
        }
        .stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px 8px 0 0;
            color: var(--text-2);
            padding: 0.5rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--bg-3) !important;
            color: var(--text) !important;
            border-bottom: 2px solid var(--accent) !important;
        }

        /* ---------- Metrics ---------- */
        [data-testid="stMetric"] {
            background: var(--bg-2);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem 1.2rem;
        }
        [data-testid="stMetricLabel"] { color: var(--text-2) !important; }
        [data-testid="stMetricValue"] { color: var(--text) !important; }

        /* ---------- Expander ---------- */
        .streamlit-expanderHeader {
            background: var(--bg-2) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
        }

        /* ---------- Dataframe ---------- */
        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

        /* ---------- Progress steps ---------- */
        .step-track { display: flex; align-items: center; gap: 0; margin: 0.5rem 0 1.6rem 0; }
        .step-dot {
            width: 30px; height: 30px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.78rem; font-weight: 700;
            background: var(--bg-3); border: 1px solid var(--line); color: var(--text-3);
            flex-shrink: 0;
        }
        .step-dot.active { background: var(--grad); color: #0a0b10; border: none; }
        .step-dot.done { background: var(--good); color: #0a0b10; border: none; }
        .step-line { flex: 1; height: 2px; background: var(--line); margin: 0 2px; }
        .step-line.done { background: var(--good); }
        .step-label { font-size: 0.7rem; color: var(--text-3); text-align: center; margin-top: 4px; }

        /* ---------- Misc text ---------- */
        .muted { color: var(--text-3); font-size: 0.85rem; }
        .small { font-size: 0.82rem; color: var(--text-2); }
        code { color: var(--accent-2); background: var(--bg-3); padding: 1px 6px; border-radius: 5px; }

        /* ---------- Divider ---------- */
        hr.div { border: none; border-top: 1px solid var(--line); margin: 1.4rem 0; }

        /* ---------- Scrollbars ---------- */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )