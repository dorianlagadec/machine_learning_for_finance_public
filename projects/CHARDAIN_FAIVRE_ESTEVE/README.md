# Satellite Parking Lot Analysis — Retail Consumption Proxy

## Authors

- Victor Chardain
- Benoit Faivre
- Amandine Esteve

## Track

Alternative Data

## Problem

Can parking lot occupancy detected from labeled parking lot imagery predict
retail consumption indicators (Walmart stock direction, consumer confidence index)?

## Data

- **Parking lot annotations**: PKLot dataset from Kaggle (COCO format).
  The merged annotation file `data/annotations.coco.json` is provided directly
  in the repository — no download required.
- **Walmart stock prices**: fetched automatically via Yahoo Finance (`yfinance`)
- **Consumer Confidence Index**: University of Michigan Consumer Sentiment (UMCSENT),
  downloaded from FRED. The file `data/consumer_confidence.csv` is provided
  directly in the repository.
- **Google Trends**: fetched automatically via `pytrends` for keywords
  "Walmart", "grocery store", "discount", "inflation"

## Structure

- `code/` → Python scripts (data collection, feature extraction, merging, modeling, backtesting, analysis)
- `data/` → Input data files (annotations.coco.json, consumer_confidence.csv) and generated outputs
- `oral_presentation/` → Slides for the 7-minute presentation
- `written_presentation/` → HTML export of Jupyter notebook

## How to run
```bash
pip install -r requirements.txt
python code/main.py
```

The script will automatically fetch Walmart stock prices and Google Trends data,
extract occupancy features from the COCO annotations, merge all signals into a
daily dataset, train and evaluate models, run a backtest, and perform a Granger
causality test.

## Notes

- Python 3.12 is recommended
- The `data/` folder is created automatically if it does not exist
- `data/annotations.coco.json` and `data/consumer_confidence.csv` must be present
  before running — both are included in the repository