"""
preprocess.py
─────────────
Preprocessing pipeline for the QRT dataset.

Loads the raw data, applies feature engineering (momentum, volatility, volume,
cross-sectional features), and saves the result to data/processed/qrt_ready.csv.

Usage:
    python scripts/preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "sample.csv"
OUTPUT = ROOT / "data" / "processed" / "qrt_ready.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
print(f"Loaded: {df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
RET_features  = [f'RET_{i}'  for i in range(1, 21)]
SVOL_features = [f'SIGNED_VOLUME_{i}' for i in range(2, 21)]  # SIGNED_VOLUME_1 dropped

# Benchmark features
for i in [3, 5, 10, 15, 20]:
    df[f'AVERAGE_PERF_{i}']             = df[RET_features[:i]].mean(axis=1)
    df[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = df.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')

for i in [20]:
    df[f'STD_PERF_{i}']             = df[RET_features[:i]].std(axis=1)
    df[f'ALLOCATIONS_STD_PERF_{i}'] = df.groupby('TS')[f'STD_PERF_{i}'].transform('mean')

# Momentum features
df['MOM_3']       = df[['RET_1', 'RET_2', 'RET_3']].sum(axis=1)
df['MOM_5']       = df[[f'RET_{i}' for i in range(1, 6)]].sum(axis=1)
df['MOM_vs_MEAN'] = df['RET_1'] - df[[f'RET_{i}' for i in range(2, 6)]].mean(axis=1)

ret_5 = df[[f'RET_{i}' for i in range(1, 6)]]
df['SHARPE_5'] = ret_5.mean(axis=1) / (ret_5.std(axis=1) + 1e-8)

# Volatility features
df['VOL_5']     = df[[f'RET_{i}' for i in range(1, 6)]].std(axis=1)
df['VOL_20']    = df[RET_features].std(axis=1)
df['VOL_RATIO'] = df['VOL_5'] / (df['VOL_20'] + 1e-8)

# Volume features
df['SVOL_SHORT'] = df[[f'SIGNED_VOLUME_{i}' for i in range(2, 5)]].sum(axis=1)
df['CONVICTION'] = df['RET_1'] * df['SIGNED_VOLUME_2']

# Cross-sectional features
df['CS_RANK_RET1'] = df.groupby('TS')['RET_1'].rank(pct=True)
df['CS_RANK_VOL5'] = df.groupby('TS')['VOL_5'].rank(pct=True)

df['CS_GROUP_MEAN_RET1'] = df.groupby(['TS', 'GROUP'])['RET_1'].transform('mean')
df['CS_GROUP_STD_RET1']  = df.groupby(['TS', 'GROUP'])['RET_1'].transform('std')
df['CS_GROUP_RANK_RET1'] = df.groupby(['TS', 'GROUP'])['RET_1'].rank(pct=True)
df['CS_GROUP_RANK_VOL5'] = df.groupby(['TS', 'GROUP'])['VOL_5'].rank(pct=True)

# ── 3. Target binaire ─────────────────────────────────────────────────────────
df['target_clf'] = (df['target'] > 0).astype(int)

# ── 4. Colonne date numérique (pour le split temporel) ────────────────────────
df['date'] = pd.to_datetime(
    df['TS'].str.extract(r'(\d+)')[0].astype(int),
    unit='D', origin='2020-01-01'
)

# ── 5. Liste finale des features (pour info) ──────────────────────────────────
features = (
    RET_features
    + SVOL_features
    + ['MEDIAN_DAILY_TURNOVER', 'GROUP']
    + [f'AVERAGE_PERF_{i}'             for i in [3, 5, 10, 15, 20]]
    + [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
    + [f'STD_PERF_{i}'                 for i in [20]]
    + [f'ALLOCATIONS_STD_PERF_{i}'     for i in [20]]
    + ['MOM_3', 'MOM_5', 'MOM_vs_MEAN', 'SHARPE_5']
    + ['VOL_5', 'VOL_20', 'VOL_RATIO']
    + ['SVOL_SHORT', 'CONVICTION']
    + ['CS_RANK_RET1', 'CS_RANK_VOL5']
    + ['CS_GROUP_MEAN_RET1', 'CS_GROUP_STD_RET1']
    + ['CS_GROUP_RANK_RET1', 'CS_GROUP_RANK_VOL5']
)

# ── 6. Save (features + target_clf + date) ────────────────────────────────────
cols_to_save = features + ['target_clf', 'date']
df[cols_to_save].to_csv(OUTPUT, index=False)

print(f"Saved : {OUTPUT}")
print(f"Shape : {df[cols_to_save].shape}")
print(f"Features : {len(features)}")
