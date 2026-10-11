# Future Work (Phase-II and Beyond)

## Immediate Phase-II Priorities
1. **Real data integration**: implement the documented adapters in
   `data_loader.py` (NASA FIRMS, NASA POWER/IMD weather, Google Earth
   Engine NDVI/NDMI/land cover, Copernicus DEM) and re-run the full
   evaluation methodology on real historical data for Madhya Pradesh &
   Chhattisgarh.
2. **Threshold calibration**: calibrate risk-category thresholds and the
   classification decision threshold (currently 0.5) against real
   historical fire outcomes, e.g. via precision-recall curve analysis.
3. **Exact administrative boundaries**: replace the rectangular bounding
   box with the true state/district polygon boundaries using GeoPandas.
4. **Barrier-aware fire spread**: incorporate roads, rivers, and firebreaks
   as partial/complete spread barriers using OpenStreetMap or government
   GIS layers.

## Medium-Term Enhancements
5. **Finer spatio-temporal resolution**: move from weekly/0.4-degree to
   daily/finer-resolution grids once real data volume and compute budget
   allow.
6. **Deep learning / spatio-temporal models**: explore CNN or
   ConvLSTM-style architectures that can learn spatial fire-spread
   patterns directly from raster time series, compared against the
   classical-ML baseline established in Phase-I.
7. **Physically-grounded fire-spread model**: extend or replace the
   cellular-automata simulation with a Rothermel-based or
   FARSITE-inspired spread-rate calculation for greater physical fidelity.
8. **Automated model monitoring**: track model drift and retraining needs
   as new fire-season data becomes available each year.

## Long-Term / Production Architecture
9. Build the production architecture outlined in
   `SYSTEM_ARCHITECTURE.md` (scheduled ETL -> PostgreSQL/PostGIS ->
   FastAPI backend -> React/Next.js frontend) for a always-on, multi-user
   deployment.
10. **Field validation partnership**: collaborate with state forest
    departments to validate predictions against ground-truth fire reports
    and incorporate expert feedback into feature engineering.
11. **Alerting integration**: once validated, explore integration with
    existing alert channels (e.g. SMS/app notifications to forest range
    officers) -- explicitly out of scope until real-world validation is
    complete.
