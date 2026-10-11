"""
07_Dashboard/app.py
====================
Streamlit dashboard for the AI-Based Wildfire Risk Assessment and
Fire-Spread Decision Support System (Prototype / DEMO MODE).

Run with:
    streamlit run 07_Dashboard/app.py

No database or API key required for the core pipeline -- it runs entirely
on the locally generated synthetic demonstration dataset. Map tiles use
plain OpenStreetMap (darkened via a scoped CSS filter -- see risk_mapping.py
/ fire_spread.py), since commercial dark-tile providers like CartoDB now
require an API key. The one part of this dashboard that needs internet at
all is the optional "search by place name" box (Nominatim geocoding); if
unavailable it reports that clearly and the manual grid-ID search keeps
working exactly as before.

Visual theme lives in theme.py ("Ember Command" -- see that file's
docstring). This file is data flow + layout only.
"""

from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_Source_Code"))

import config  # noqa: E402
import theme  # noqa: E402
from dashboard_utils import (get_pipeline_bundle, latest_week_risk_table,  # noqa: E402
                              get_explainability, run_spread_simulation)
from explainability import top_reasons_for_cell  # noqa: E402
from risk_mapping import generate_risk_map_html  # noqa: E402
from fire_spread import generate_spread_map_html  # noqa: E402
from geocoding import geocode_place, is_within_study_region, nearest_grid_id  # noqa: E402
from utils import try_import  # noqa: E402

folium = try_import("folium")

st.set_page_config(page_title="Wildfire Risk & Spread DSS (Prototype)", layout="wide",
                    initial_sidebar_state="collapsed")
theme.inject_theme()


def _place_search_widget(risk_df: pd.DataFrame, query_key: str, button_key: str,
                          target_search_key: str, target_select_key: str, label: str,
                          located_point_key: str = None):
    """
    Shared 'search by place name' widget used on both the Risk Map and
    Fire-Spread tabs. Geocodes via Nominatim (needs internet -- the only
    part of this dashboard that does); on success it jumps the matching
    tab's grid-cell selector to the nearest grid cell, clearing any
    grid-ID text filter so the match is guaranteed to be a valid option.
    If located_point_key is given, the raw geocoded point is also stored
    in session_state under that key so the caller can drop a marker for
    it on a map (see the Risk Map tab). On failure / out-of-region
    results it explains why, without touching the current selection.
    """
    pc1, pc2 = st.columns([4, 1])
    with pc1:
        place_query = st.text_input(label, value="", key=query_key,
                                     placeholder="e.g. Bhopal, VIT Bhopal, Indore, Raipur",
                                     label_visibility="collapsed")
    with pc2:
        locate_clicked = st.button("LOCATE", key=button_key)

    if locate_clicked:
        if not place_query.strip():
            st.info("Type a place name first.")
        else:
            with st.spinner(f"Looking up '{place_query}'..."):
                geo = geocode_place(place_query)
            if geo is None:
                st.error(f"Couldn't find '{place_query}' or couldn't reach the geocoding service "
                         f"(needs internet). You can still search by grid ID below.")
            elif not is_within_study_region(geo["lat"], geo["lon"]):
                st.warning(f"**{geo['display_name']}** ({geo['lat']:.3f}, {geo['lon']:.3f}) is "
                           f"outside the study region ({config.STUDY_REGION['name']}). Try a "
                           f"location within Madhya Pradesh or Chhattisgarh.")
            else:
                nearest_id, dist_km = nearest_grid_id(geo["lat"], geo["lon"], risk_df)
                st.session_state[target_search_key] = ""
                st.session_state[target_select_key] = nearest_id
                if located_point_key:
                    st.session_state[located_point_key] = geo
                st.success(f"Found **{geo['display_name']}** → nearest grid cell **{nearest_id}** "
                           f"(~{dist_km:.0f} km away). Map has centred on it below.")
                st.rerun()


def _render_map_html(html: str, height: int):
    """
    Streamlit is deprecating st.components.v1.html in favour of st.iframe.
    Try the new API first; if this Streamlit version's st.iframe doesn't
    accept raw HTML the way we expect, fall back to the old (still working)
    call rather than crashing the whole page over a rendering API name.
    """
    try:
        st.iframe(srcdoc=html, height=height, scrolling=False)
    except (AttributeError, TypeError):
        st.components.v1.html(html, height=height, scrolling=False)


st.title(config.DASHBOARD_TITLE)
st.caption(
    "Prototype implementation for a B.Tech CSE-AIML capstone project. Estimates wildfire "
    "occurrence probability per grid cell and simulates how a fire might spread from an "
    "ignition point, using environmental, vegetation, terrain, and historical-fire features."
)
st.warning(config.DEMO_MODE_NOTICE)

with st.spinner("Loading demo pipeline (generating data, training models on first run)..."):
    bundle = get_pipeline_bundle()

df = bundle["df"]
comparison = bundle["comparison"]
best_model_name_ = bundle["best_model"]
risk_df = latest_week_risk_table(bundle)

tab_risk, tab_models, tab_spread, tab_about = st.tabs(
    ["WILDFIRE RISK MAP", "MODEL PERFORMANCE", "FIRE-SPREAD SIMULATION", "ABOUT & LIMITATIONS"]
)

# ---------------------------------------------------------------------------
# TAB 1: RISK MAP
# ---------------------------------------------------------------------------
with tab_risk:
    col_map, col_info = st.columns([2, 1])

    with col_map:
        with st.container(border=True):
            theme.section_header("Interactive Wildfire Risk Map",
                                  "Most recent demo week · click any cell for details", tag="//")
            located = st.session_state.get("located_point_risk")
            map_path = generate_risk_map_html(risk_df, out_name="dashboard_risk_map.html",
                                               located_point=located)
            if folium is not None and map_path is not None:
                html = Path(map_path).read_text()
                _render_map_html(html, height=520)
            else:
                st.info("folium not installed -- showing risk table instead of interactive map.")
                st.dataframe(risk_df[["grid_id", "latitude", "longitude", "risk_probability", "risk_category"]],
                             width="stretch")

            legend_row = " &nbsp; ".join(
                theme.risk_badge(cat) + f" <span style='color:#9AA0A8;font-size:0.85rem;'>{lo:.2f}\u2013{hi:.2f}</span>"
                for cat, (lo, hi) in config.RISK_THRESHOLDS.items()
            )
            st.markdown(f"<div style='margin-top:10px;'>{legend_row}</div>", unsafe_allow_html=True)
            st.caption("Prototype thresholds -- not scientifically calibrated. See About & Limitations.")

    with col_info:
        with st.container(border=True):
            theme.section_header("Selected Grid-Cell", tag="//")

            st.markdown("**Search by place name**")
            _place_search_widget(risk_df, "place_query_risk", "locate_btn_risk",
                                  "grid_search_text", "cell_select",
                                  "Search by place name (e.g. Bhopal)",
                                  located_point_key="located_point_risk")

            st.markdown("**...or search by grid ID**")
            search_col1, search_col2 = st.columns([3, 2])
            with search_col1:
                cell_search = st.text_input("Search by grid ID (e.g. R014C017)",
                                             key="grid_search_text",
                                             placeholder="Type to filter, e.g. R014",
                                             label_visibility="collapsed")
            with search_col2:
                sort_choice = st.selectbox("Sort", ["Highest risk first", "Grid ID"],
                                            label_visibility="collapsed")

            all_ids = risk_df.sort_values("risk_probability", ascending=False)["grid_id"].tolist() \
                if sort_choice == "Highest risk first" else sorted(risk_df["grid_id"].unique())
            if cell_search.strip():
                filtered_ids = [g for g in all_ids if cell_search.strip().upper() in g.upper()]
            else:
                filtered_ids = all_ids

            if not filtered_ids:
                st.warning(f"No grid cell matches '{cell_search}'. Showing all cells instead.")
                filtered_ids = all_ids
            if st.session_state.get("cell_select") not in filtered_ids:
                st.session_state.pop("cell_select", None)

            cell_id = st.selectbox(f"Choose a grid cell ({len(filtered_ids)} match(es))",
                                    options=filtered_ids, key="cell_select")
            cell = risk_df[risk_df["grid_id"] == cell_id].iloc[0]

            st.markdown(
                f"<div style='margin:10px 0;'>Risk category: {theme.risk_badge(cell['risk_category'])}</div>",
                unsafe_allow_html=True)
            st.metric("Risk probability", f"{cell['risk_probability']:.1%}")

            m1, m2 = st.columns(2)
            m1.metric("Temperature", f"{cell['temperature']:.1f} °C")
            m1.metric("Wind speed", f"{cell['wind_speed']:.1f} km/h")
            m1.metric("NDVI", f"{cell['ndvi']:.2f}")
            m2.metric("Humidity", f"{cell['humidity']:.0f} %")
            m2.metric("Rainfall (7-day)", f"{cell['rainfall_7day_sum']:.1f} mm")
            m2.metric("Land cover", cell["land_cover"])

            st.markdown("**Why this risk level (top contributing factors):**")
            explain_df = get_explainability(bundle)
            reasons = top_reasons_for_cell(explain_df, cell, top_n=5)
            for r in reasons:
                st.markdown(f"- {r}")
            st.caption("Feature importance reflects learned association, not proven causation.")

# ---------------------------------------------------------------------------
# TAB 2: MODEL COMPARISON, CONFUSION MATRIX, EXPLAINABILITY
# ---------------------------------------------------------------------------
with tab_models:
    with st.container(border=True):
        theme.section_header("Model Comparison Table",
                              "Chronological test split · wildfire occurrence is a rare/imbalanced "
                              "event -- PR-AUC and recall matter more than raw accuracy", tag="//")

        def _highlight_best(col):
            is_best = col == col.max()
            return ["background-color: rgba(255,122,61,0.18); color: #FFB27A; font-weight: 700;"
                    if v else "" for v in is_best]

        styled_comparison = comparison.style.apply(
            _highlight_best, subset=["pr_auc", "roc_auc", "balanced_accuracy", "recall", "f1_score"])
        st.dataframe(styled_comparison, width="stretch")

    with st.container(border=True):
        theme.section_header(f"Confusion Matrix -- Best Model ({best_model_name_})", tag="//")
        from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score
        best_bundle = bundle["result"]["models"][best_model_name_]
        X_test, y_test = bundle["result"]["splits"]["test"]
        if best_bundle.get("uses_scaler"):
            X_in = best_bundle["scaler"].transform(X_test)
        else:
            X_in = X_test
        proba_test = best_bundle["model"].predict_proba(X_in)[:, 1]

        threshold = st.slider(
            "Classification threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
            help="A cell is classified 'Fire' if its predicted probability is at or above this value. "
                 "0.5 is just a default, not a rule -- for a rare-event problem like this, lowering the "
                 "threshold trades some precision (more false alarms) for higher recall (fewer missed "
                 "fires), which is usually the right trade-off for a safety-oriented warning system."
        )
        y_pred = (proba_test >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = confusion_matrix(y_test, y_pred, normalize="true") * 100
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Balanced Accuracy", f"{bal_acc:.1%}",
                   help="Average of each class's own recall -- the standard accuracy metric for "
                        "imbalanced classification, since it isn't inflated by the majority 'no fire' class.")
        bm2.metric("Precision at this threshold", f"{prec:.1%}")
        bm3.metric("Recall at this threshold", f"{rec:.1%}")
        bm4.metric("Fire weeks correctly identified", f"{cm_pct[1, 1]:.1f}%")

        st.caption(
            "Raw counts (left) differ hugely in scale because only a small fraction of test weeks "
            "contain a fire event -- that scale difference IS the class imbalance, not a modelling "
            "error. The row-normalized view (right) puts both classes on the same 0-100% scale so "
            "per-class performance is directly comparable. Move the slider above to see the precision/"
            "recall trade-off directly -- this is the standard way to work with an imbalanced target, "
            "rather than a single fixed 0.5 cutoff."
        )

        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            fig, ax = theme.dark_figure(figsize=(4, 3.6), dpi=140)
            ax.imshow(cm, cmap=theme.get_cool_cmap())
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11, color="white")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["No Fire", "Fire"], fontsize=9)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["No Fire", "Fire"], fontsize=9)
            ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
            ax.set_title("Raw Counts", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig, width="content")
        with cm_col2:
            fig2, ax2 = theme.dark_figure(figsize=(4, 3.6), dpi=140)
            ax2.imshow(cm_pct, cmap=theme.get_warm_cmap(), vmin=0, vmax=100)
            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center", fontsize=11, color="white")
            ax2.set_xticks([0, 1]); ax2.set_xticklabels(["No Fire", "Fire"], fontsize=9)
            ax2.set_yticks([0, 1]); ax2.set_yticklabels(["No Fire", "Fire"], fontsize=9)
            ax2.set_xlabel("Predicted", fontsize=9); ax2.set_ylabel("Actual", fontsize=9)
            ax2.set_title("Row-Normalized (%)", fontsize=10)
            fig2.tight_layout()
            st.pyplot(fig2, width="content")

    with st.container(border=True):
        theme.section_header("Feature Importance / Explainability", tag="//")
        explain_df = get_explainability(bundle)
        st.caption(f"Method used: {explain_df['method'].iloc[0]}")
        fig3, ax3 = theme.dark_figure(figsize=(6, 4.5), dpi=140)
        top = explain_df.sort_values("importance", ascending=True).tail(12)
        ax3.barh(top["feature"], top["importance"], color="#FF7A3D")
        ax3.tick_params(labelsize=8)
        fig3.tight_layout()
        fi_col, _ = st.columns([1, 1])
        with fi_col:
            st.pyplot(fig3, width="content")
        st.caption("Feature importance shows what the model learned to associate with fire risk -- "
                   "it does not prove real-world causation.")

# ---------------------------------------------------------------------------
# TAB 3: FIRE-SPREAD SIMULATION
# ---------------------------------------------------------------------------
with tab_spread:
    with st.container(border=True):
        theme.section_header("Fire-Spread Simulation Controls",
                              "Simple, explainable cellular-automata simulation. This is a "
                              "decision-support prototype, NOT a physically exact fire-behaviour model.",
                              tag="//")

        st.markdown("**Search ignition point by place name**")
        _place_search_widget(risk_df, "place_query_spread", "locate_btn_spread",
                              "ignition_search_text", "ignition_select",
                              "Search ignition point by place name")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**...or search by grid ID**")
            ignition_search = st.text_input("Search ignition grid cell (e.g. R014C017)",
                                             key="ignition_search_text",
                                             placeholder="Type to filter, e.g. R014")
            ignition_ids_by_risk = risk_df.sort_values("risk_probability", ascending=False)["grid_id"].tolist()
            if ignition_search.strip():
                ignition_options = [g for g in ignition_ids_by_risk if ignition_search.strip().upper() in g.upper()]
                if not ignition_options:
                    st.warning(f"No grid cell matches '{ignition_search}'. Showing all cells instead.")
                    ignition_options = ignition_ids_by_risk
            else:
                ignition_options = ignition_ids_by_risk
            if st.session_state.get("ignition_select") not in ignition_options:
                st.session_state.pop("ignition_select", None)
            ignition_id = st.selectbox(f"Ignition location, sorted by risk ({len(ignition_options)} match(es))",
                                        options=ignition_options, key="ignition_select")
            ign_risk = risk_df[risk_df["grid_id"] == ignition_id].iloc[0]
            st.markdown(f"Current risk here: {theme.risk_badge(ign_risk['risk_category'])} "
                        f"<span style='color:#9AA0A8;'>({ign_risk['risk_probability']:.1%})</span>",
                        unsafe_allow_html=True)
        with c2:
            wind_speed_sim = st.number_input("Wind speed (km/h)", min_value=0, max_value=100, value=20, step=1)
            wind_direction_sim = st.number_input("Wind direction (° from North, blowing towards)",
                                                  min_value=0, max_value=359, value=90, step=5)
        with c3:
            duration_sim = st.number_input("Simulation duration (minutes)", min_value=10, max_value=240,
                                            value=60, step=10,
                                            help="Up to 240 minutes (4 hours) of simulated spread time.")
            moisture_sim = st.number_input("Fuel moisture (NDMI, higher = wetter)",
                                            min_value=-0.20, max_value=0.60, value=0.15, step=0.01,
                                            format="%.2f")

        run_clicked = st.button("RUN FIRE-SPREAD SIMULATION", type="primary")
        if run_clicked:
            with st.spinner("Simulating fire spread..."):
                sim = run_spread_simulation(bundle, ignition_id, wind_speed_sim, wind_direction_sim,
                                             duration_sim, ndmi_value=moisture_sim)
                st.session_state["sim_result"] = sim

    if "sim_result" in st.session_state:
        sim = st.session_state["sim_result"]
        with st.container(border=True):
            theme.section_header("Simulation Result", tag="//")
            m1, m2, m3 = st.columns(3)
            m1.metric("Affected cells", len(sim["affected_cells"]))
            m2.metric("Estimated affected area", f"{sim['affected_area_km2']} km²")
            m3.metric("Simulation steps", sim["n_steps"])

            map_path = generate_spread_map_html(sim, out_name="dashboard_spread_map.html")
            if folium is not None and map_path is not None:
                html = Path(map_path).read_text()
                _render_map_html(html, height=500)
            else:
                st.dataframe(sim["grid_snapshot"][sim["grid_snapshot"]["status"] != "unburned"],
                             width="stretch")
    else:
        st.info("Set parameters above and click 'Run Fire-Spread Simulation'.")

    st.caption("Spread probability depends on distance from the burning cell, wind alignment and "
               "speed, slope direction, vegetation/fuel load, and fuel moisture. Barriers (roads/"
               "rivers) are documented as a planned enhancement -- see FUTURE_WORK.md.")

# ---------------------------------------------------------------------------
# TAB 4: ABOUT / LIMITATIONS
# ---------------------------------------------------------------------------
with tab_about:
    theme.section_header("Limitations & Disclaimer", tag="//")
    lim_cards = [
        ("Demo Mode", config.DEMO_MODE_NOTICE),
        ("Simplified Region", f"Study region uses a simplified rectangular bounding box "
         f"approximating {config.STUDY_REGION['name']}, not exact administrative boundaries."),
        ("Uncalibrated Thresholds", "Risk thresholds (Low/Moderate/High/Extreme) are initial "
         "prototype values and require calibration against real historical fire data."),
        ("Simplified Fire Physics", "The fire-spread module is a simplified cellular-automata "
         "approximation, not a physically validated model (e.g. not Rothermel / FARSITE / WRF-Fire)."),
        ("Synthetic Data Only", "Model metrics reflect performance on synthetic data only and "
         "do not represent real-world predictive accuracy."),
        ("Not Operational", "This is a capstone research prototype. It is not validated, "
         "real-time, or suitable for actual emergency response or disaster-management decisions."),
    ]
    lc = st.columns(3)
    for i, (title, body) in enumerate(lim_cards):
        with lc[i % 3]:
            st.markdown(theme.info_card(title, body), unsafe_allow_html=True)
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    theme.section_header("Data Sources", "Planned for real deployment (Phase-II)", tag="//")
    src_cards = [
        ("Fire History", "NASA FIRMS -- MODIS / VIIRS active fire products.<br>"
         "<a href='https://firms.modaps.eosdis.nasa.gov/' style='color:#FF9A6A;' target='_blank'>firms.modaps.eosdis.nasa.gov</a>"),
        ("Weather", "NASA POWER / IMD ground and reanalysis weather data.<br>"
         "<a href='https://power.larc.nasa.gov/' style='color:#FF9A6A;' target='_blank'>power.larc.nasa.gov</a>"),
        ("Vegetation / Satellite", "Google Earth Engine -- MODIS NDVI/NDMI, land-surface "
         "temperature, land cover."),
        ("Terrain", "Copernicus DEM GLO-30 elevation/slope/aspect.<br>"
         "<a href='https://registry.opendata.aws/copernicus-dem/' style='color:#FF9A6A;' target='_blank'>registry.opendata.aws/copernicus-dem</a>"),
    ]
    sc = st.columns(4)
    for i, (title, body) in enumerate(src_cards):
        with sc[i]:
            st.markdown(theme.info_card(title, body), unsafe_allow_html=True)
