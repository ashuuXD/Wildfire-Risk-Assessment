"""
theme.py
=========
"Ember Command" visual theme for the dashboard -- a dark, glowing
fire/night aesthetic (deep charcoal background with drifting ember glow,
condensed bold headers, glowing amber card highlights on hover). Inspired
by premium dark-gaming-UI conventions but reskinned in fire colours and
built from clean typography + colour rather than icons/emoji, so it stays
tonally serious and professional for a wildfire safety tool.

Kept in its own module so app.py stays readable. inject_theme() is called
once near the top of app.py; everything else here is CSS plus small HTML/
Python helpers (section headers, risk badges, info cards) used throughout
the four tabs.

This only changes CSS/HTML presentation -- no data, model, or simulation
logic lives in this file.
"""
from __future__ import annotations
import streamlit as st

# Colours also used from Python (e.g. building an inline badge's style
# attribute), kept in sync with the CSS custom properties below by hand.
RISK_COLORS_GLOW = {
    "Low": "#3DDC7A",
    "Moderate": "#F5C542",
    "High": "#FF8A3D",
    "Extreme": "#FF4141",
}

_FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');"
)

_CSS = f"""
<style>
{_FONTS}

:root {{
    --bg-base: #0A0C0F;
    --bg-panel: #14171C;
    --bg-panel-2: #191D23;
    --border-soft: rgba(255,255,255,0.10);
    --ember-orange: #FF7A3D;
    --ember-red: #FF4141;
    --ember-amber: #F5C542;
    --text-main: #F2F1ED;
    --text-muted: #9AA0A8;
    --font-head: 'Rajdhani', 'Segoe UI', sans-serif;
    --font-body: 'Inter', 'Segoe UI', sans-serif;
}}

html, body {{ background: #0A0C0F; }}

/* ---------- Background: set directly on Streamlit's own app container ----------
   (a real CSS `background` property on an element ALWAYS paints behind that
   element's own content -- this is guaranteed by how backgrounds work, unlike
   a separately-injected fixed div, which depends on Streamlit's internal DOM
   stacking context and can silently fail to show through.) */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse 1000px 700px at 10% 0%, rgba(255,122,61,0.22), transparent 62%),
        radial-gradient(ellipse 800px 600px at 92% 10%, rgba(255,65,65,0.14), transparent 60%),
        radial-gradient(ellipse 900px 800px at 50% 105%, rgba(245,197,66,0.10), transparent 58%),
        var(--bg-base) !important;
    background-attachment: fixed, fixed, fixed, fixed;
    background-size: 160% 160%, 160% 160%, 160% 160%, 100% 100%;
    animation: emberDrift 24s ease-in-out infinite alternate;
}}
@keyframes emberDrift {{
    0%   {{ background-position: 0% 0%, 100% 0%, 50% 100%, 0 0; }}
    100% {{ background-position: 8% 12%, 88% 18%, 46% 92%, 0 0; }}
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container {{ padding-top: 2rem; max-width: 1400px; }}

/* ---------- Typography ---------- */
h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: var(--text-main) !important;
}}
h1 {{ font-size: 2.3rem !important; }}
p, span, label, div {{ font-family: var(--font-body); }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--text-muted) !important; }}

h1::after {{
    content: ""; display: block; width: 84px; height: 3px; margin-top: 10px;
    background: linear-gradient(90deg, var(--ember-orange), var(--ember-red));
    border-radius: 2px; box-shadow: 0 0 12px rgba(255,122,61,0.6);
}}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-family: var(--font-head); font-weight: 600; font-size: 1.02rem;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted);
    padding: 10px 18px;
}}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text-main) !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background: linear-gradient(90deg, var(--ember-orange), var(--ember-red)) !important;
    height: 3px !important; box-shadow: 0 0 10px rgba(255,122,61,0.7);
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ background: var(--border-soft) !important; }}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {{
    background: linear-gradient(145deg, var(--bg-panel), var(--bg-panel-2));
    border: 1px solid var(--border-soft);
    border-radius: 12px; padding: 14px 16px 10px 16px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    border-color: var(--ember-orange) !important;
    box-shadow: 0 0 0 1px rgba(255,122,61,0.35), 0 10px 26px rgba(255,122,61,0.28);
}}
[data-testid="stMetricLabel"] {{
    font-family: var(--font-head); text-transform: uppercase; font-size: 0.78rem !important;
    letter-spacing: 0.05em; color: var(--text-muted) !important;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--font-head); font-weight: 700 !important; color: var(--text-main) !important;
}}

/* ---------- Bordered containers (st.container(border=True)) as glow cards ----------
   Multiple selector variants targeted together since the exact internal
   testid Streamlit uses for this has changed across versions -- covering
   several increases the odds of a match, and the effect is intentionally
   strong (solid colour change + real shadow, not a subtle tint) so it's
   unmistakable if it does. */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stContainer"],
div[data-testid="stExpander"] {{
    background: linear-gradient(160deg, var(--bg-panel), var(--bg-panel-2)) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover,
div[data-testid="stContainer"]:hover {{
    border: 1px solid var(--ember-orange) !important;
    box-shadow: 0 0 0 1px rgba(255,122,61,0.4), 0 12px 34px rgba(255,122,61,0.30) !important;
    background: linear-gradient(160deg, #1A1E24, #20242B) !important;
}}

/* ---------- Buttons ---------- */
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {{
    font-family: var(--font-head); font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; border-radius: 8px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
[data-testid="stButton"] button[kind="primary"] {{
    background: linear-gradient(135deg, var(--ember-orange), var(--ember-red)) !important;
    border: none !important; box-shadow: 0 4px 16px rgba(255,90,61,0.35);
}}
[data-testid="stButton"] button[kind="primary"]:hover {{
    transform: translateY(-1px); box-shadow: 0 6px 22px rgba(255,90,61,0.55);
}}
[data-testid="stButton"] button:not([kind="primary"]):hover {{
    border-color: var(--ember-orange) !important; color: var(--ember-orange) !important;
    box-shadow: 0 0 0 1px rgba(255,122,61,0.3);
}}

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background: var(--bg-panel) !important; border-color: var(--border-soft) !important;
    border-radius: 8px !important; transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
[data-testid="stTextInput"] input:hover, [data-testid="stNumberInput"] input:hover,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{
    border-color: rgba(255,122,61,0.55) !important;
}}
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {{
    border-color: var(--ember-orange) !important;
    box-shadow: 0 0 0 1px rgba(255,122,61,0.5) !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background: var(--ember-orange) !important;
    box-shadow: 0 0 10px rgba(255,122,61,0.6) !important;
}}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
    background: linear-gradient(90deg, var(--ember-amber), var(--ember-orange)) !important;
}}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {{
    border-radius: 10px !important; border: 1px solid var(--border-soft) !important;
    font-family: var(--font-body);
}}

/* ---------- DataFrame ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border-soft); border-radius: 10px; overflow: hidden;
}}

hr {{ border-color: var(--border-soft) !important; }}

/* ---------- Risk badge pill ---------- */
.risk-badge {{
    display: inline-block; padding: 3px 14px; border-radius: 20px;
    font-family: var(--font-head); font-weight: 700; font-size: 0.85rem;
    letter-spacing: 0.04em; text-transform: uppercase;
}}

/* ---------- Section header: text + colour, no icons/emoji ---------- */
.ember-section-header {{
    display: flex; align-items: baseline; gap: 10px; margin: 6px 0 2px 0;
    border-left: 3px solid var(--ember-orange); padding-left: 12px;
}}
.ember-section-header .tag {{
    font-family: var(--font-head); font-weight: 700; font-size: 0.95rem;
    color: var(--ember-orange); letter-spacing: 0.08em;
}}
.ember-section-header .title {{
    font-family: var(--font-head); font-weight: 700; font-size: 1.35rem;
    text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-main);
}}
.ember-section-sub {{ color: var(--text-muted); font-size: 0.92rem; margin: 2px 0 14px 15px; }}

/* ---------- Info card (plain HTML, used where no Streamlit widgets needed inside) ---------- */
.ember-card {{
    background: linear-gradient(160deg, var(--bg-panel), var(--bg-panel-2));
    border: 1px solid var(--border-soft); border-radius: 14px;
    padding: 18px 20px; transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
    height: 100%;
}}
.ember-card:hover {{
    border: 1px solid var(--ember-orange);
    box-shadow: 0 0 0 1px rgba(255,122,61,0.4), 0 12px 34px rgba(255,122,61,0.28);
    transform: translateY(-3px);
}}
.ember-card .card-title {{
    font-family: var(--font-head); font-weight: 700; font-size: 1.02rem;
    color: var(--ember-orange); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.03em;
}}
.ember-card .card-body {{ color: var(--text-muted); font-size: 0.92rem; line-height: 1.5; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--bg-base); }}
::-webkit-scrollbar-thumb {{ background: #2A2E35; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--ember-orange); }}
</style>
"""


def inject_theme():
    """Call once near the top of app.py, right after st.set_page_config()."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", tag: str = ""):
    """
    Consistent section header used across all four tabs -- an accent bar
    plus bold uppercase title, no icons/emoji (text + colour carries the
    styling, matching a clean HUD/command-console look).
    """
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="ember-section-header">{tag_html}<span class="title">{title}</span></div>'
        + (f'<div class="ember-section-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> str:
    """Inline HTML pill for a risk category, using the glowing risk palette."""
    color = RISK_COLORS_GLOW.get(category, "#9AA0A8")
    return (f'<span class="risk-badge" style="background:{color}22; color:{color}; '
            f'border:1px solid {color}66; box-shadow:0 0 12px {color}33;">{category}</span>')


def info_card(title: str, body_html: str) -> str:
    """Returns HTML for a static info card (About tab, etc.) -- wrap in st.markdown(..., unsafe_allow_html=True)."""
    return f'<div class="ember-card"><div class="card-title">{title}</div><div class="card-body">{body_html}</div></div>'


# ---------------------------------------------------------------------------
# Matplotlib dark-theme helpers -- so charts sit naturally in the dark UI
# instead of rendering as bright white rectangles. Two genuinely distinct
# colormaps (cool blue vs warm ember) are provided so side-by-side charts
# (e.g. raw-count vs normalized confusion matrix) are visually
# distinguishable at a glance rather than looking like near-duplicates.
# ---------------------------------------------------------------------------
PANEL_BG = "#14171C"
GRID_COLOR = "#2A2E35"
TEXT_LIGHT = "#F2F1ED"
TEXT_MUTED = "#9AA0A8"


def dark_figure(figsize=(4, 3.6), dpi=140):
    """Creates a matplotlib Figure/Axes pair pre-styled to match the dashboard's dark panels."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    style_dark_axes(fig, ax)
    return fig, ax


def style_dark_axes(fig, ax):
    """Apply dark chrome to an existing Figure/Axes (or list of Axes)."""
    fig.patch.set_facecolor(PANEL_BG)
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    for a in axes:
        a.set_facecolor(PANEL_BG)
        a.tick_params(colors=TEXT_MUTED)
        a.xaxis.label.set_color(TEXT_LIGHT)
        a.yaxis.label.set_color(TEXT_LIGHT)
        a.title.set_color(TEXT_LIGHT)
        for spine in a.spines.values():
            spine.set_color(GRID_COLOR)


def get_cool_cmap():
    """Dark charcoal -> steel blue -> bright cyan. Used for 'raw count' style charts."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ember_cool", ["#14171C", "#254A6B", "#4FB4E0"])


def get_warm_cmap():
    """Dark charcoal -> ember orange -> bright amber. Used for 'normalized / performance' style charts."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ember_warm", ["#14171C", "#C1461E", "#FFC670"])
