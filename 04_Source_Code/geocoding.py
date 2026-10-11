"""
geocoding.py
=============
Free, no-API-key place-name search (OpenStreetMap Nominatim) so the
dashboard can accept a location name ("Bhopal", "VIT Bhopal", "Indore")
instead of requiring the user to already know a grid_id.

This is the ONLY part of the whole project that needs live internet access
at runtime (everything else -- the model, the risk map, the fire-spread
simulation -- runs entirely offline on the synthetic demo data). If the
request fails for any reason (no internet, DNS blocked on campus wifi,
Nominatim rate limiting), geocode_place() returns None and the dashboard
falls back gracefully to the existing manual grid-ID search, which is
unaffected.

Nominatim's usage policy (operations.osmfoundation.org/policies/nominatim)
requires a descriptive User-Agent and at most ~1 request/second -- both are
naturally satisfied by a single person clicking "Locate" in a dashboard.
"""
from __future__ import annotations
import json
import urllib.request
import urllib.parse
from typing import Optional

import numpy as np
import pandas as pd

import config
from utils import get_logger

logger = get_logger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "WildfireRiskCapstoneDashboard/1.0 (B.Tech CSE-AIML student capstone project)"


def geocode_place(place_name: str, timeout: float = 6.0) -> Optional[dict]:
    """
    Look up a place name (biased toward India). Returns
    {"lat": float, "lon": float, "display_name": str} or None if not found
    / the request failed for any reason (network, timeout, service down).
    """
    if not place_name or not place_name.strip():
        return None
    params = {"q": place_name.strip(), "format": "json", "limit": 1,
              "countrycodes": "in", "addressdetails": 0}
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            return None
        result = data[0]
        return {"lat": float(result["lat"]), "lon": float(result["lon"]),
                "display_name": result.get("display_name", place_name)}
    except Exception as exc:
        logger.warning(f"Geocoding failed for '{place_name}': {exc.__class__.__name__}: {exc}")
        return None


def is_within_study_region(lat: float, lon: float) -> bool:
    return (config.STUDY_REGION["min_lat"] <= lat <= config.STUDY_REGION["max_lat"]
            and config.STUDY_REGION["min_lon"] <= lon <= config.STUDY_REGION["max_lon"])


def nearest_grid_id(lat: float, lon: float, grid_df: pd.DataFrame):
    """Grid cell (from grid_df, which must have latitude/longitude/grid_id
    columns) closest to (lat, lon), plus the approximate distance in km."""
    dist_deg = np.hypot(grid_df["latitude"] - lat, grid_df["longitude"] - lon)
    idx = dist_deg.idxmin()
    approx_km = float(dist_deg.loc[idx]) * 111.0  # rough deg -> km at these latitudes
    return grid_df.loc[idx, "grid_id"], approx_km
