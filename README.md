# Six-hour river level forecasting at Kalady, Periyar basin

Code for the paper *"Horizon-Resolved Explainable Long Short-Term Memory for Six-Hour River Level Forecasting under Partial Rainfall Forecasts: A Deployed Case Study at Kalady, Periyar Basin, India"* (International Journal of Engineering and Manufacturing, under review).

## What it does
- Builds 39 hourly predictors, including unit-hydrograph-routed rainfall and travel-time-shifted upstream rainfall and discharge (`scripts/pipe.py`)
- Trains an encoder-decoder LSTM that predicts the six-hour change in water level, with a masked future-rainfall input (`scripts/train.py`)
- Compares it with persistence, ARIMA and XGBoost baselines (`scripts/arima.py`, `scripts/xgb.py`)
- Runs ablations, a kernel-length sensitivity test and a withheld August 2018 flood test (`scripts/train.py`, `scripts/ev_base.py`, `scripts/xgb_event.py`)
- Computes horizon-resolved LIME attributions on 500 validation forecasts (`scripts/lime_run.py`)
- Collects every reported number into `results/results.json` (`scripts/report.py`)

## Data
The historical hourly records are from the India Water Resources Information System (India-WRIS); level observations from 2026 onward come from the radar sensor installed at Kalady. The data file is not included in this repository. Place it at `data/lime.xlsx`. The code freezes the data at 2 November 2026, 08:00.

## Run
```
pip install -r requirements.txt
python run_all.py
```
On a Tesla T4 GPU (Google Colab) the full run takes about 2 hours. Interrupted runs can be restarted; finished runs are skipped.

## Results
`results/results.json` and `results/results.txt` contain the results reported in the paper; `results/hardware.json` records the hardware used.
