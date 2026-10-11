"""
fire_spread.py
===============
A simple, explainable, PHYSICS-INSPIRED cellular-automata fire-spread
simulator. This is a decision-support PROTOTYPE, not a physically exact
fire-behaviour model (it is not Rothermel's equations or FARSITE/WRF-Fire).
It captures the qualitative drivers real fire-behaviour models use --
wind alignment, slope, fuel load, moisture, and barriers -- in a
computationally cheap grid simulation suitable for an interactive demo.

WHY A SEPARATE FINE LOCAL GRID (IMPORTANT DESIGN NOTE)
---------------------------------------------------------
The wildfire-RISK model uses a coarse REGIONAL grid (config.GRID_RESOLUTION_DEG
= 0.4 deg, ~44 km/cell) -- correct for a weekly, regional risk map, but far
too coarse for a MINUTE-SCALE LOCAL fire-spread simulation: a real fire
advances at most a few km in 1-2 hours, not tens of kilometres. Running the
cellular automaton directly on the regional grid (as an earlier version of
this module did) produced physically nonsensical results, e.g. "450,000 km^2
burned in 2 hours" -- larger than many countries.

So run_fire_spread_simulation() below builds its OWN fine local grid
(FINE_CELL_SIZE_KM = 0.2 km/cell by default, FINE_GRID_RADIUS_CELLS cells in
each direction from the ignition point) centred on the chosen ignition point,
runs the identical cellular-automaton logic on THAT grid, then reports area
using the fine grid's (much smaller, realistic) cell size. The regional grid
is still used to look up the ignition point's location and to keep the fine
grid's large-scale terrain consistent with the regional risk map (elevation/
slope/aspect/land-cover are inherited from the nearest regional cell, with a
little added local variation since a 44 km cell obviously can't capture
kilometre-scale terrain detail). This keeps the public function signature
and the dashboard's calling code completely unchanged -- only the internals
changed.

Simulation logic (per discrete time step of config.SPREAD_CELL_STEP_MINUTES):
  For every currently-burning cell, each of its 8 neighbours has an ignition
  probability built from:
    - base probability (config.BASE_SPREAD_PROBABILITY)
    - wind alignment factor: neighbours downwind of the burning cell get a
      boost; upwind neighbours are suppressed
    - wind speed factor: stronger wind increases spread probability
    - slope factor: fire spreads faster upslope (uphill) in the aspect
      direction
    - fuel/vegetation factor: forest/grassland carries fire better than
      cropland/barren/water (config.LAND_COVER_FUEL_FACTOR)
    - moisture factor: higher NDMI (moisture) suppresses spread
    - barrier factor: cells flagged as water/barren act as partial firebreaks
  A random draw decides ignition. Burning cells become "burned out" after a
  configurable number of steps (simple fuel-consumption approximation).

A cell that survives every ignition attempt from its burning neighbours
(each attempt is a probability roll, capped well under 100%) simply stays
"unburned" -- even if fully surrounded by fire. Real fires leave exactly
this kind of unburned island/refugia due to local fuel or moisture gaps, so
these gaps in the rendered map are an intentional, realistic feature of the
stochastic simulation, not a rendering bug (see generate_spread_map_html:
only "burning"/"burned_out" cells get a coloured rectangle at all).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config
from utils import get_logger, try_import

logger = get_logger(__name__)
folium = try_import("folium")

# See risk_mapping.py for why this exists (CartoDB's dark tiles now require
# an API key; this darkens plain, reliably-free OpenStreetMap tiles instead,
# scoped to only the raster tile pane so rectangles/markers/popups are unaffected).
_DARK_TILE_CSS = """
<style>
.leaflet-tile-pane { filter: invert(100%) hue-rotate(200deg) brightness(80%) contrast(95%) saturate(65%); }
</style>
"""

NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Compass bearing (degrees, 0=N) from a cell TOWARD each neighbour offset (dr, dc)
OFFSET_BEARING = {
    (-1, 0): 0, (-1, 1): 45, (0, 1): 90, (1, 1): 135,
    (1, 0): 180, (1, -1): 225, (0, -1): 270, (-1, -1): 315,
}

# Fine local grid used ONLY for fire-spread (see module docstring above).
FINE_CELL_SIZE_KM = 0.2
FINE_GRID_RADIUS_CELLS = 45  # -> (2*45+1)=91 cells per side, ~18.2 km across


def _wind_alignment(wind_direction_deg: float, bearing_deg: float) -> float:
    """1.0 if the neighbour is directly downwind of the burning cell, -1.0 if directly upwind."""
    diff = abs((bearing_deg - wind_direction_deg + 180) % 360 - 180)
    return np.cos(np.radians(diff))  # +1 downwind aligned, -1 upwind


def _slope_alignment(aspect_deg: float, bearing_deg: float) -> float:
    """Fire spreads preferentially upslope; aspect = downhill-facing direction, so uphill = aspect+180."""
    uphill_bearing = (aspect_deg + 180) % 360
    diff = abs((bearing_deg - uphill_bearing + 180) % 360 - 180)
    return np.cos(np.radians(diff))


def build_fine_local_grid(center_lat: float, center_lon: float, regional_grid: pd.DataFrame,
                           ndmi_value: float = 0.15, seed: int = None) -> pd.DataFrame:
    """
    Build a fine (FINE_CELL_SIZE_KM-resolution) local grid centred on
    (center_lat, center_lon), for use by the fire-spread simulation only.
    Terrain (elevation/slope/aspect/land_cover_code) is inherited from the
    NEAREST regional grid cell for large-scale consistency with the risk
    map, with a small amount of added local variation (a 44 km regional
    cell cannot represent kilometre-scale terrain texture on its own).
    """
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    R = FINE_GRID_RADIUS_CELLS
    size = 2 * R + 1

    dist = np.hypot(regional_grid["latitude"] - center_lat, regional_grid["longitude"] - center_lon)
    nearest = regional_grid.loc[dist.idxmin()]
    base_elevation = float(nearest["elevation"])
    base_slope = float(nearest["slope"])
    base_aspect = float(nearest["aspect"])
    base_land_cover = int(nearest["land_cover_code"])

    lat_per_cell = FINE_CELL_SIZE_KM / 111.0
    lon_per_cell = FINE_CELL_SIZE_KM / (111.0 * max(np.cos(np.radians(center_lat)), 0.1))

    ii, jj = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    local_relief = 25 * np.sin(ii / size * 2.5 * np.pi + rng.uniform(0, 6)) * \
        np.cos(jj / size * 2.5 * np.pi + rng.uniform(0, 6))

    rows = []
    for r in range(size):
        for c in range(size):
            rows.append({
                "row": r, "col": c, "grid_id": f"F{r:03d}_{c:03d}",
                "latitude": center_lat + (r - R) * lat_per_cell,
                "longitude": center_lon + (c - R) * lon_per_cell,
                "elevation": base_elevation + float(local_relief[r, c]),
                "slope": max(0.0, base_slope + float(rng.normal(0, 1.5))),
                "aspect": base_aspect,
                "land_cover_code": base_land_cover,
                "ndmi": float(np.clip(ndmi_value + rng.normal(0, 0.03), -0.5, 0.9)),
            })
    grid = pd.DataFrame(rows)
    return grid, R


def run_fire_spread_simulation(grid_df: pd.DataFrame, ignition_grid_id: str,
                                wind_speed: float, wind_direction: float,
                                duration_minutes: int = config.DEFAULT_SPREAD_DURATION_MIN,
                                moisture_col: str = "ndmi") -> dict:
    """
    Run the cellular-automata fire-spread simulation on a FINE local grid
    built around the ignition point (see module docstring). grid_df is the
    REGIONAL grid -- used to look up the ignition point's coordinates and
    to seed the fine grid's terrain -- not simulated on directly.

    Parameters
    ----------
    grid_df : the regional DataFrame (row, col, grid_id, latitude, longitude,
        elevation, slope, aspect, land_cover_code, and a moisture column e.g. ndmi).
    ignition_grid_id : grid_id of the ignition point, from grid_df.
    wind_speed : km/h (dashboard-controlled).
    wind_direction : degrees, direction the wind is BLOWING TOWARDS (0=N,90=E).
    duration_minutes : total simulated duration.

    Returns dict with: burned_cells (list of grid_id with step-ignited),
    affected_area_km2 (approx., using the FINE grid's cell size), grid
    snapshot dataframe, and parameters used.
    """
    if ignition_grid_id not in grid_df.set_index("grid_id").index:
        raise ValueError(f"Unknown ignition_grid_id: {ignition_grid_id}")
    origin = grid_df.set_index("grid_id").loc[ignition_grid_id]
    origin_lat, origin_lon = float(origin["latitude"]), float(origin["longitude"])
    ndmi_value = float(origin.get(moisture_col, 0.15)) if moisture_col in grid_df.columns else 0.15

    fine_grid, R = build_fine_local_grid(origin_lat, origin_lon, grid_df, ndmi_value=ndmi_value)
    fine_ignition_id = f"F{R:03d}_{R:03d}"  # centre cell of the fine grid

    rng = np.random.default_rng(config.RANDOM_SEED)

    n_rows = int(fine_grid["row"].max()) + 1
    n_cols = int(fine_grid["col"].max()) + 1
    cell_lookup = fine_grid.set_index("grid_id")
    rc_lookup = {(int(r["row"]), int(r["col"])): idx for idx, r in fine_grid.set_index("grid_id").iterrows()}

    state = np.zeros((n_rows, n_cols), dtype=int)  # 0=unburned, 1=burning, 2=burned out
    burn_step = np.full((n_rows, n_cols), -1)

    ir, ic = int(cell_lookup.loc[fine_ignition_id, "row"]), int(cell_lookup.loc[fine_ignition_id, "col"])
    state[ir, ic] = 1
    burn_step[ir, ic] = 0

    n_steps = max(1, int(duration_minutes / config.SPREAD_CELL_STEP_MINUTES))
    burnout_after_steps = 3  # a cell stops actively spreading after this many steps (fuel consumed)

    wind_speed_factor = np.clip(wind_speed / 25.0, 0.2, 2.0)  # normalised, capped

    for step in range(1, n_steps + 1):
        burning_cells = np.argwhere(state == 1)
        newly_ignited = []
        for (r, c) in burning_cells:
            if step - burn_step[r, c] > burnout_after_steps:
                state[r, c] = 2  # burned out
                continue
            for (dr, dc) in NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                if state[nr, nc] != 0:
                    continue
                neighbor_id = rc_lookup.get((nr, nc))
                if neighbor_id is None:
                    continue
                cell = cell_lookup.loc[neighbor_id]

                bearing = OFFSET_BEARING[(dr, dc)]
                wind_align = _wind_alignment(wind_direction, bearing)
                slope_align = _slope_alignment(float(cell.get("aspect", 0)), bearing)
                fuel_factor = config.LAND_COVER_FUEL_FACTOR.get(int(cell.get("land_cover_code", 3)), 0.5)
                moisture = float(cell.get(moisture_col, 0.2))
                moisture_suppression = np.clip(1 - moisture, 0.1, 1.2)
                slope_deg = float(cell.get("slope", 0))
                slope_boost = 1 + 0.02 * slope_deg * max(slope_align, 0)

                prob = (
                    config.BASE_SPREAD_PROBABILITY
                    * (1 + 0.5 * wind_align * wind_speed_factor)
                    * wind_speed_factor
                    * slope_boost
                    * fuel_factor
                    * moisture_suppression
                )
                prob = float(np.clip(prob, 0.0, 0.95))

                if rng.random() < prob:
                    newly_ignited.append((nr, nc, neighbor_id, prob))

        for (nr, nc, neighbor_id, prob) in newly_ignited:
            if state[nr, nc] == 0:
                state[nr, nc] = 1
                burn_step[nr, nc] = step

    # Build result snapshot
    result_rows = []
    for idx, cell in fine_grid.iterrows():
        r, c = int(cell["row"]), int(cell["col"])
        s = state[r, c]
        result_rows.append({
            "grid_id": cell["grid_id"], "latitude": cell["latitude"], "longitude": cell["longitude"],
            "status": {0: "unburned", 1: "burning", 2: "burned_out"}[s],
            "ignited_at_step": int(burn_step[r, c]) if burn_step[r, c] >= 0 else None,
        })
    result_df = pd.DataFrame(result_rows)
    affected = result_df[result_df["status"] != "unburned"]

    cell_area_km2 = FINE_CELL_SIZE_KM ** 2
    affected_area_km2 = round(len(affected) * cell_area_km2, 2)
    cell_size_deg_lat = FINE_CELL_SIZE_KM / 111.0
    cell_size_deg_lon = FINE_CELL_SIZE_KM / (111.0 * max(np.cos(np.radians(origin_lat)), 0.1))

    logger.info(f"Fire spread simulation: ignition={ignition_grid_id} (fine grid, {FINE_CELL_SIZE_KM * 1000:.0f}m cells), "
                f"wind={wind_speed}km/h@{wind_direction}deg, duration={duration_minutes}min -> "
                f"{len(affected)} cells affected (~{affected_area_km2} km2)")

    return {
        "grid_snapshot": result_df,
        "affected_cells": affected["grid_id"].tolist(),
        "affected_area_km2": affected_area_km2,
        "n_steps": n_steps,
        "params": {"ignition_grid_id": ignition_grid_id, "wind_speed": wind_speed,
                   "wind_direction": wind_direction, "duration_minutes": duration_minutes,
                   "cell_size_deg_lat": cell_size_deg_lat, "cell_size_deg_lon": cell_size_deg_lon,
                   "cell_size_km": FINE_CELL_SIZE_KM},
    }


def generate_spread_map_html(sim_result: dict, out_name: str = "fire_spread_map.html"):
    """Render the fire-spread simulation result as an interactive Folium map."""
    out_path = config.SPREAD_MAPS_DIR / out_name
    df = sim_result["grid_snapshot"]

    if folium is None:
        logger.warning("folium not installed -- saving spread result CSV instead of an interactive map")
        df.to_csv(config.SPREAD_MAPS_DIR / "fire_spread_table.csv", index=False)
        return None

    center_lat, center_lon = df["latitude"].mean(), df["longitude"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                       tiles="OpenStreetMap", control_scale=True)
    fmap.get_root().html.add_child(folium.Element(_DARK_TILE_CSS))

    status_colors = {"unburned": None, "burning": "#FF4141", "burned_out": "#6B6F76"}
    half_res_lat = sim_result["params"].get("cell_size_deg_lat", config.GRID_RESOLUTION_DEG) / 2
    half_res_lon = sim_result["params"].get("cell_size_deg_lon", config.GRID_RESOLUTION_DEG) / 2
    for _, row in df[df["status"] != "unburned"].iterrows():
        color = status_colors[row["status"]]
        bounds = [[row["latitude"] - half_res_lat, row["longitude"] - half_res_lon],
                  [row["latitude"] + half_res_lat, row["longitude"] + half_res_lon]]
        folium.Rectangle(bounds=bounds, color=color, weight=0.5, fill=True,
                          fill_color=color, fill_opacity=0.75,
                          popup=f"{row['grid_id']} ({row['status']}, step {row['ignited_at_step']})").add_to(fmap)

    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                background: rgba(15,17,21,0.92); padding: 12px 16px; border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.12); font-size: 13px;
                font-family: 'Inter', sans-serif; color: #F2F1ED;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);">
        <b style="font-family:'Rajdhani',sans-serif; letter-spacing:0.03em;">FIRE-SPREAD SIMULATION</b><br>
        <span style="color:#FF4141; text-shadow:0 0 6px #FF4141;">&#9632;</span> Burning (active front)<br>
        <span style="color:#6B6F76;">&#9632;</span> Burned out<br>
        <span style="color:#9AA0A8;">&#9679;</span> Ignition point
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Ignition marker: centre of the fine grid == the point closest to (center_lat, center_lon)
    dist = (df["latitude"] - center_lat) ** 2 + (df["longitude"] - center_lon) ** 2
    ign_row = df.loc[dist.idxmin()]
    folium.Marker([ign_row["latitude"], ign_row["longitude"]],
                  icon=folium.Icon(color="orange", icon="fire", prefix="fa"),
                  popup=f"Ignition point ({sim_result['params']['ignition_grid_id']})").add_to(fmap)

    fmap.save(str(out_path))
    logger.info(f"Fire spread map saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    from generate_demo_data import build_spatial_grid, _assign_static_terrain_and_landcover

    rng = np.random.default_rng(config.RANDOM_SEED)
    grid = build_spatial_grid()
    grid = _assign_static_terrain_and_landcover(grid, rng)
    grid["ndmi"] = 0.15  # dry-season default for standalone demo run

    center_idx = len(grid) // 2
    ignition_id = grid.iloc[center_idx]["grid_id"]

    sim = run_fire_spread_simulation(grid, ignition_grid_id=ignition_id,
                                      wind_speed=22, wind_direction=90,
                                      duration_minutes=60)
    generate_spread_map_html(sim)
    print(f"Affected cells: {len(sim['affected_cells'])} | Area: {sim['affected_area_km2']} km2")
