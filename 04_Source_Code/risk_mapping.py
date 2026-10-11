"""
risk_mapping.py
================
Converts model probabilities into prototype risk categories (Low / Moderate
/ High / Extreme, per config.RISK_THRESHOLDS) and renders an interactive
Folium risk map, saved as HTML to 06_GIS/risk_maps/.

These thresholds are INITIAL PROTOTYPE VALUES and require calibration and
validation against real historical fire data before any operational use.
"""

from __future__ import annotations
import pandas as pd

import config
from utils import get_logger, try_import

logger = get_logger(__name__)
folium = try_import("folium")

# Darkens plain OpenStreetMap tiles via a CSS filter scoped to ONLY the
# Leaflet tile pane (the raster basemap layer) -- rectangles, markers, and
# popups live in separate Leaflet panes and are unaffected, so this can't
# distort the risk colours. This avoids depending on any commercial tile
# provider's "free vs API-key-required" policy (which is exactly what broke
# with CartoDB) -- OpenStreetMap's own tiles are the one option that has
# reliably stayed free/keyless for this kind of use.
_DARK_TILE_CSS = """
<style>
.leaflet-tile-pane { filter: invert(100%) hue-rotate(200deg) brightness(80%) contrast(95%) saturate(65%); }
</style>
"""


def build_risk_table(df: pd.DataFrame, probabilities) -> pd.DataFrame:
    """Attach predicted probability + risk category columns to a cell-level dataframe."""
    out = df.copy()
    out["risk_probability"] = probabilities
    out["risk_category"] = out["risk_probability"].apply(config.classify_risk)
    return out


def generate_risk_map_html(risk_df: pd.DataFrame, out_name: str = "wildfire_risk_map.html",
                            located_point: dict = None):
    """
    Render an interactive Folium map of predicted wildfire risk per grid
    cell, colour-coded by risk category. Falls back to saving a static CSV
    + logging a warning if folium is unavailable.

    Uses plain OpenStreetMap tiles (the one Folium base layer that is
    reliably free with no API key) and darkens them with a CSS filter
    scoped to ONLY the tile layer, so the dashboard's dark theme is
    preserved without depending on a commercial tile provider's key policy
    (CartoDB's dark/light basemaps started requiring an API key -- this
    sidesteps that entirely).

    located_point (optional): {"lat", "lon", "display_name"} for a place
    the user searched for (see geocoding.py) -- if given, the map centres
    on it and drops a distinct marker there, separate from the grid-risk
    rectangles, so a place search visibly "lands" on the map.
    """
    out_path = config.RISK_MAPS_DIR / out_name

    if folium is None:
        logger.warning("folium not installed -- saving risk table CSV instead of an interactive map")
        risk_df.to_csv(config.RISK_MAPS_DIR / "wildfire_risk_table.csv", index=False)
        return None

    if located_point:
        center_lat, center_lon, zoom = located_point["lat"], located_point["lon"], 10
    else:
        center_lat = (config.STUDY_REGION["min_lat"] + config.STUDY_REGION["max_lat"]) / 2
        center_lon = (config.STUDY_REGION["min_lon"] + config.STUDY_REGION["max_lon"]) / 2
        zoom = 6
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                       tiles="OpenStreetMap", control_scale=True)
    fmap.get_root().html.add_child(folium.Element(_DARK_TILE_CSS))

    half_res = config.GRID_RESOLUTION_DEG / 2
    for _, row in risk_df.iterrows():
        color = config.RISK_COLORS[row["risk_category"]]
        bounds = [[row["latitude"] - half_res, row["longitude"] - half_res],
                  [row["latitude"] + half_res, row["longitude"] + half_res]]
        popup = (f"<div style='font-family:Inter,sans-serif;'>"
                 f"<b>Grid:</b> {row.get('grid_id', 'N/A')}<br>"
                 f"<b>Risk:</b> <span style='color:{color};font-weight:700;'>{row['risk_category']}</span> "
                 f"({row['risk_probability']:.1%})<br>"
                 f"<b>Land cover:</b> {row.get('land_cover', 'N/A')}</div>")
        folium.Rectangle(bounds=bounds, color=color, weight=1, fill=True,
                          fill_color=color, fill_opacity=0.5,
                          popup=folium.Popup(popup, max_width=250)).add_to(fmap)

    if located_point:
        folium.Marker(
            [located_point["lat"], located_point["lon"]],
            icon=folium.Icon(color="pink", icon="star", prefix="fa"),
            popup=folium.Popup(f"<b>{located_point.get('display_name', 'Searched location')}</b>", max_width=250),
            tooltip="Searched location",
        ).add_to(fmap)

    fmap.save(str(out_path))
    logger.info(f"Interactive risk map saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    from data_loader import load_dataset
    from train_models import train_all_models

    df = load_dataset(source="demo")
    result = train_all_models(df)
    bundle = result["models"]["random_forest"]
    X_test, y_test = result["splits"]["test"]
    proba = bundle["model"].predict_proba(X_test)[:, 1]

    latest_date = df["date"].max()
    latest_cells = df[df["date"] == df["date"].unique()[-2]].reset_index(drop=True)
    latest_X = latest_cells[config.FEATURE_COLUMNS]
    latest_proba = bundle["model"].predict_proba(latest_X)[:, 1]
    risk_df = build_risk_table(latest_cells, latest_proba)
    risk_df.to_csv(config.PROCESSED_DATA_DIR / "latest_risk_predictions.csv", index=False)
    generate_risk_map_html(risk_df)
