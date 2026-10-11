# System Architecture

## Prototype Architecture (implemented in this repository)

```mermaid
flowchart TD
    A[Data Sources] --> B[Data Ingestion]
    B --> C[Data Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Target Generation]
    E --> F[Model Training]
    F --> G[Risk Prediction]
    G --> H[GIS Visualization]
    G --> I[Explainability]
    G --> J[Fire Spread Simulation]
    H --> K[Streamlit Dashboard]
    I --> K
    J --> K
```

### Component Mapping
| Stage | Module |
|---|---|
| Data Sources (demo) | `generate_demo_data.py` |
| Data Ingestion | `data_loader.py` |
| Data Preprocessing | `preprocessing.py` |
| Feature Engineering | `feature_engineering.py` |
| Target Generation | `feature_engineering.generate_target()` |
| Model Training | `train_models.py` |
| Risk Prediction | `evaluate_models.py`, `risk_mapping.build_risk_table()` |
| GIS Visualization | `risk_mapping.py` |
| Explainability | `explainability.py` |
| Fire Spread Simulation | `fire_spread.py` |
| Dashboard | `07_Dashboard/app.py` |

All stages read configuration (paths, study region, thresholds, model
parameters) from a single source of truth: `04_Source_Code/config.py`.

## Future Production Architecture (NOT implemented in Phase-I)

```mermaid
flowchart LR
    A[NASA FIRMS / Weather / Satellite APIs] --> B[Scheduled Data Pipeline]
    B --> C[(PostgreSQL / PostGIS)]
    C --> D[FastAPI Backend]
    D --> E[XGBoost Model Service]
    E --> F[React / Next.js Frontend]
```

This production architecture would replace the local Streamlit prototype
with a scheduled ETL pipeline, a spatial database, a REST API backend, and
a modern web frontend suitable for multi-user, always-on deployment. It is
documented here as the Phase-II/III target, not built in this capstone
(see `FUTURE_WORK.md`), because the local Streamlit prototype is the
priority within the current project timeline.
