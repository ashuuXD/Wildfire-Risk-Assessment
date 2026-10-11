# Research Problem

## Problem Statement

Wildfires cause significant ecological, economic, and human loss across
forested regions of India every dry season. Early identification of
high-risk areas, and an understanding of how a fire might spread once
ignited, can support faster, better-informed response planning. This
capstone addresses two related geospatial machine-learning problems:

1. **Wildfire risk assessment** -- given environmental, vegetation,
   terrain, and historical-fire information for a geographic grid cell,
   estimate the probability that a wildfire will occur in that cell during
   a future prediction window (default: 7 days).

2. **Fire-spread assessment** -- given an ignition point, wind conditions,
   terrain, vegetation/fuel, and moisture, estimate how a fire may spread
   to nearby grid cells over a configurable time horizon.

## Central Research Question

> Given location, weather, vegetation, terrain, and historical fire
> activity, what is the probability of wildfire occurrence, and if a fire
> starts there, which nearby areas may be affected?

## Why This Matters

- Forest departments and disaster-management agencies currently rely
  heavily on satellite-based *after-the-fact* fire detection (e.g. NASA
  FIRMS alerts once a fire is already burning). A predictive, *before-the-
  fact* risk layer is a natural complement to this.
- Machine learning is well suited to this problem because wildfire risk is
  driven by many interacting, non-linear factors (dryness, heat, wind,
  fuel load, terrain, historical activity) that are difficult to combine
  with simple fixed rules.
- A grid-based, explainable approach keeps the system interpretable for
  non-ML stakeholders such as forest officers, which matters for trust and
  adoption in a disaster-management context.

## Honesty Note

This document describes the *problem* the prototype is designed to
address. The current implementation is a **software prototype validated on
synthetic demonstration data**; it does not yet solve this problem with
real-world validated accuracy. See `SCOPE_AND_LIMITATIONS.md`.
