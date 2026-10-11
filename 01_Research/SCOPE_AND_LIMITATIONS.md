# Scope and Limitations

## In Scope (Phase-I Prototype)
- A configurable, rectangular study-region bounding box (default: Central
  India, approximating Madhya Pradesh & Chhattisgarh).
- A fully synthetic, clearly-labelled demonstration dataset generated with
  a fixed random seed, encoding realistic spatial and seasonal patterns.
- A leak-free feature-engineering and target-generation pipeline.
- Three classical ML classifiers with a chronological train/val/test split.
- Imbalance-aware evaluation (precision, recall, F1, ROC-AUC, PR-AUC,
  confusion matrix).
- Explainability via SHAP or a documented fallback.
- A prototype spatial risk map with configurable thresholds.
- A simplified, explainable cellular-automata fire-spread simulation.
- A local Streamlit dashboard requiring no database, cloud deployment, or
  API key.
- Documented (but not executed) adapters for real NASA FIRMS / weather /
  satellite / DEM data sources.

## Explicit Simplifications
- **Study-region boundary**: a rectangular bounding box is used instead of
  the true, irregular state boundary polygons, to keep grid construction
  simple for a capstone timeline.
- **Grid resolution**: ~0.4 degrees (~44 km) per cell -- coarse, to keep
  the demo dataset small enough to train in seconds on a modest laptop.
- **Weekly time step**: the demo uses weekly (not daily) synthetic
  records to keep total row-count and training time tractable in a
  live-demo setting.
- **Land cover**: 6 simplified classes rather than a full standard land
  cover taxonomy.
- **Fire-spread simulation**: a probabilistic cellular-automata model
  capturing qualitative drivers (wind, slope, fuel, moisture), not a
  physically calibrated fire-behaviour model.

## Limitations
1. **Synthetic data only.** No real NASA FIRMS, weather, or satellite data
   has been downloaded or used. All relationships in the data are
   deliberately engineered to be learnable, not empirically observed.
2. **No causal claims.** Feature importance / SHAP values reflect
   statistical association learned by the model, not proven causal
   drivers of real wildfire ignition.
3. **Thresholds are uncalibrated.** Risk categories and fire-spread
   parameters are illustrative starting points, not scientifically
   validated values.
4. **Not for operational or emergency use.** This is a capstone research
   prototype. It must not be used for real wildfire detection,
   monitoring, or emergency response decisions.
5. **No barrier data.** Roads/rivers as fire-spread barriers are
   documented as a planned enhancement, not implemented in Phase-I.
6. **Classical ML only.** No deep learning / spatio-temporal architectures
   are used in Phase-I (see `FUTURE_WORK.md`).

## Real-World Validation is Future Work
Phase-II should replace the synthetic generator with real NASA FIRMS,
weather, and satellite-derived features for the same study region, and
re-run the full evaluation methodology on that real data before any
claim of predictive skill is made.
