# Results Guide

## Run Command

```bash
export MPLBACKEND=Agg && python -u main.py | tee outputs/run_server.log
```

## Key Settings

- Task: forecast_anom
- Event date: 2025-02-18
- Last observed InSAR epoch: 2025-02-18 (D20250218)
- Near-event window: last H=5 epochs before last observation
- Near-event epoch range: D20241220 to D20250206
- FAR target for tau calibration: 0.01

## Outputs

- Rolling risk maps (CSV): risk_maps_csv
- Rolling risk maps (PNG lat/lon): risk_maps_png_latlon
- Rolling risk maps (PNG UTM): risk_maps_png_utm
- Mean risk over time: mean_risk_over_time_by_class.csv and .png
- AUC over time (core vs near-field): auc_over_time_core_vs_near.csv and .png
- Trend summary: trend_summary.txt
- Lead-time summary: lead_time_summary.json

## How To Interpret

- Risk is standardized as risk_pk across tasks to enable shared evaluation/plots.
- A good early-warning signal should show rising core mean risk toward the last observed epochs.
- AUC(k) core vs near-field should improve near the end if the objective is working.
- The lead-time summary reports the first sustained exceedance of tau by the core mean risk.

## Notes

- Last observed epoch date in the data: 2025-02-18
- Landslide date is later than the last observation, so lead time is measured relative to the event date.