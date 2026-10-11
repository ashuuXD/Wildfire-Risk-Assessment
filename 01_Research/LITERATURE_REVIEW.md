# Literature Review

## Overview

Wildfire risk prediction and fire-spread modelling are established research
areas spanning remote sensing, GIS, fire science, and machine learning.
This review summarizes the broad research themes relevant to this capstone.
Specific papers listed in `LITERATURE_MATRIX.csv` that are marked
"VERIFY BEFORE FINAL SUBMISSION" are research-theme placeholders and must
be checked against a citation database (Google Scholar, IEEE Xplore,
ScienceDirect) before being used in a final report, per academic-honesty
requirements.

## Key Research Themes

### 1. Satellite-Based Active Fire Detection
NASA's MODIS and VIIRS active-fire products (distributed via the Fire
Information for Resource Management System, FIRMS) are the most widely
used real-world data sources for historical wildfire occurrence, providing
latitude, longitude, acquisition date/time, brightness temperature,
detection confidence, and Fire Radiative Power (FRP).

### 2. Machine-Learning Approaches to Wildfire-Risk / Susceptibility Mapping
A large body of research applies classical ML (logistic regression, random
forests, gradient boosting) and, more recently, deep learning, to predict
wildfire occurrence or susceptibility from weather, vegetation index
(NDVI/NDMI), terrain (elevation/slope/aspect), and historical fire data.
Random Forest and gradient-boosted trees are consistently reported as
strong, robust baselines for tabular environmental data of this kind.

### 3. Explainable AI for Environmental Risk Models
SHAP (SHapley Additive exPlanations) has become a standard tool for
explaining tree-based and linear model predictions in environmental and
disaster-risk applications, letting practitioners see which factors (e.g.
temperature, dryness, wind) drove a specific prediction.

### 4. Fire-Spread / Fire-Behaviour Modelling
Physically-based fire-spread models trace back to Rothermel's (1972)
mathematical model of fire spread through wildland fuels, which underlies
operational tools such as FARSITE and BehavePlus. Cellular-automata and
probabilistic-grid approaches are a widely used simplification for
computationally cheap, explainable spread simulation, at the cost of
physical precision.

### 5. Class Imbalance in Rare-Event Environmental Prediction
Wildfire occurrence, like many rare natural-hazard events, produces a
heavily imbalanced classification target. The research consensus is that
accuracy is a poor primary metric for such problems; precision, recall,
F1, ROC-AUC, and particularly PR-AUC are recommended instead.

## Research Gap Addressed by This Project

Most published wildfire-ML studies (a) use a single ML model rather than a
transparent multi-model comparison, (b) do not combine risk *prediction*
with an explainable *spread simulation* in a single accessible tool, and
(c) are rarely packaged as an interactive, locally runnable decision-
support dashboard aimed at non-specialist stakeholders. This capstone
targets that gap for a Central Indian study region, while being explicit
that its current results are on synthetic, not real, data.

## References
See `09_Papers/references.bib` for citable entries and
`LITERATURE_MATRIX.csv` for the structured comparison table.
