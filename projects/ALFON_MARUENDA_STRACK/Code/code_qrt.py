import os
import sys
import time
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

def _import_deps():
    """Import heavy dependencies. Called once at the start of main()."""
    global pd, np, plt, sns, ks_2samp, KFold, accuracy_score, lgb, xgb, cb
    import pandas as _pd; pd = _pd
    import numpy as _np; np = _np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt; plt = _plt
    import seaborn as _sns; sns = _sns
    sns.set_style('whitegrid')
    from scipy.stats import ks_2samp as _ks; ks_2samp = _ks
    from sklearn.model_selection import KFold as _KF; KFold = _KF
    from sklearn.metrics import accuracy_score as _acc; accuracy_score = _acc
    import lightgbm as _lgb; lgb = _lgb
    import xgboost as _xgb; xgb = _xgb
    import catboost as _cb; cb = _cb

eps      = 1e-8
RET_COLS = [f'RET_{i}' for i in range(1, 21)]
SV_COLS  = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
N_SPLITS = 8

LGBM_PARAMS = {
    'objective':         'binary',
    'metric':            'binary_error',
    'verbosity':         -1,
    'boosting_type':     'dart',
    'n_estimators':      2000,
    'learning_rate':     5e-3,
    'num_leaves':        63,
    'max_depth':         6,
    'min_child_samples': 50,
    'feature_fraction':  0.7,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'drop_rate':         0.05,
    'seed':              42,
}

XGB_PARAMS = {
    'objective':             'binary:logistic',
    'eval_metric':           'error',
    'verbosity':             0,
    'n_estimators':          2000,
    'learning_rate':         5e-3,
    'max_depth':             5,
    'min_child_weight':      50,
    'subsample':             0.8,
    'colsample_bytree':      0.7,
    'reg_alpha':             0.1,
    'reg_lambda':            1.0,
    'tree_method':           'hist',
    'early_stopping_rounds': 100,
    'seed':                  42,
}

CAT_PARAMS = {
    'loss_function':    'Logloss',
    'iterations':       2000,
    'learning_rate':    5e-3,
    'depth':            6,
    'l2_leaf_reg':      3.0,
    'min_data_in_leaf': 50,
    'rsm':              0.7,
    'subsample':        0.8,
    'bootstrap_type':   'MVS',
    'random_seed':      42,
    'verbose':          False,
    'task_type':        'CPU',
    'thread_count':     4,
}

LGBM_FAST = {
    'objective': 'binary', 'metric': 'binary_error', 'verbosity': -1,
    'boosting_type': 'gbdt', 'n_estimators': 500, 'learning_rate': 1e-2,
    'max_depth': 4, 'num_leaves': 31, 'feature_fraction': 0.7,
    'min_child_samples': 50, 'seed': 42,
}

EARLY_STOP = 100
DENSITY_THRESHOLD = 250


# ──────────────────────────────────────────────────────────────────────────────
# VECTORIZED HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def autocorr_lag1_vec(ret):
    x  = ret[:, :-1]
    y  = ret[:, 1:]
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    cov   = (xc * yc).mean(axis=1)
    denom = xc.std(axis=1) * yc.std(axis=1)
    return np.where(denom < eps, 0.0, cov / denom)


def skew_vec(ret):
    x = ret - ret.mean(axis=1, keepdims=True)
    return (x**3).mean(axis=1) / (ret.std(axis=1)**3 + eps)


def kurt_vec(ret):
    x = ret - ret.mean(axis=1, keepdims=True)
    return (x**4).mean(axis=1) / (ret.std(axis=1)**4 + eps) - 3


def pearson_vec(a, b):
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    cov   = (ac * bc).mean(axis=1)
    denom = ac.std(axis=1) * bc.std(axis=1)
    return np.where(denom < eps, 0.0, cov / denom)


def spearman_vec(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    cov   = (ra * rb).mean(axis=1)
    denom = ra.std(axis=1) * rb.std(axis=1)
    return np.where(denom < eps, 0.0, cov / denom)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    X_train = pd.read_csv('X_train.csv', index_col='ROW_ID')
    X_test  = pd.read_csv('X_test.csv',  index_col='ROW_ID')
    y_train = pd.read_csv('y_train.csv', index_col='ROW_ID')
    sample_submission = pd.read_csv('sample_submission.csv', index_col='ROW_ID')
    y_bin = (y_train['target'] > 0).astype(int)
    print(f'X_train : {X_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_train : {y_train.shape}')
    return X_train, X_test, y_train, sample_submission, y_bin


# ──────────────────────────────────────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────────────────────────────────────

def run_eda(X_train, X_test, y_train, y_bin):

    print(f'X_train : {X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')

    missing     = X_train.isnull().sum()
    missing_pct = (missing / len(X_train) * 100).round(2)
    dtype_df    = pd.DataFrame({
        'dtype':     X_train.dtypes,
        'missing':   missing,
        'missing_%': missing_pct
    })
    print(dtype_df[dtype_df['missing'] > 0])

    print(f'Positive returns (label 1) : {y_bin.sum()} ({y_bin.mean()*100:.2f}%)')
    print(f'Negative returns (label 0) : {(1-y_bin).sum()} ({(1-y_bin.mean())*100:.2f}%)')

    print('\n' + '=' * 60)
    print('PANEL STRUCTURE')
    print('=' * 60)
    print(f'Unique allocations : {X_train["ALLOCATION"].nunique()}')
    print(f'Unique timestamps  : {X_train["TS"].nunique()}')
    print(f'ROW_IDs aligned    : {(X_train.index == y_train.index).all()}')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('EDA - Asset Allocation Challenge', fontsize=16)
    axes[0, 0].hist(y_train['target'], bins=100, color='steelblue', edgecolor='white')
    axes[0, 0].set_title('Target distribution (raw return)')
    axes[0, 0].set_xlabel('Return')
    y_bin.value_counts().plot(kind='bar', ax=axes[0, 1], color=['salmon', 'steelblue'])
    axes[0, 1].set_title('Class balance (sign of target)')
    axes[0, 1].set_xticklabels(['Negative (0)', 'Positive (1)'], rotation=0)
    axes[0, 2].hist(X_train['RET_1'], bins=100, color='green', edgecolor='white')
    axes[0, 2].set_title('RET_1 distribution (yesterday return)')
    axes[1, 0].hist(X_train['SIGNED_VOLUME_1'].dropna(), bins=100, color='orange', edgecolor='white')
    axes[1, 0].set_title('SIGNED_VOLUME_1 distribution (non-missing only)')
    axes[1, 1].hist(X_train['MEDIAN_DAILY_TURNOVER'], bins=100, color='purple', edgecolor='white')
    axes[1, 1].set_title('MEDIAN_DAILY_TURNOVER distribution')
    X_train_tmp = X_train.copy()
    X_train_tmp['target_sign'] = y_bin.values
    X_train_tmp.groupby('GROUP')['target_sign'].mean().plot(kind='bar', ax=axes[1, 2], color='teal')
    axes[1, 2].set_title('Positive return rate by GROUP')
    axes[1, 2].set_ylabel('% positive returns')
    axes[1, 2].axhline(0.5, color='red', linestyle='--', label='50%')
    axes[1, 2].legend()
    axes[1, 2].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig('plots/eda/distributions.png', dpi=120)
    plt.close()

    print('GROUP distribution - Train')
    print(X_train.groupby('GROUP').size().pipe(lambda s: pd.DataFrame({
        'count': s, 'proportion': (s / len(X_train)).round(4)
    })))
    print('\nGROUP distribution - Test')
    print(X_test.groupby('GROUP').size().pipe(lambda s: pd.DataFrame({
        'count': s, 'proportion': (s / len(X_test)).round(4)
    })))

    df_eda = X_train.copy()
    df_eda['target_bin'] = y_bin.values
    df_eda['REVERSAL_1'] = X_train['RET_1'] - X_train[[f'RET_{i}' for i in range(2, 6)]].mean(1)

    print('=== Signal accuracy by GROUP ===')
    for g in [1, 2, 3, 4]:
        sub = df_eda[df_eda['GROUP'] == g]
        mom_acc = ((sub['RET_1'] > 0) == sub['target_bin']).mean()
        rev_acc = ((sub['REVERSAL_1'] < 0) == sub['target_bin']).mean()
        corr    = sub['RET_1'].corr(y_train.loc[sub.index, 'target'])
        print(f'  GROUP {g}: momentum={mom_acc:.4f}  reversal={rev_acc:.4f}  corr(RET1, target)={corr:.4f}')

    ic_by_lag = [(lag, X_train[f'RET_{lag}'].corr(y_train['target'])) for lag in range(1, 21)]
    lags, ics = zip(*ic_by_lag)
    colors = ['tomato' if ic > 0 else 'steelblue' for ic in ics]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lags, ics, color=colors, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Return lag')
    ax.set_ylabel('Pearson correlation with target')
    ax.set_title('Information Coefficient by lag\n(positive = momentum, negative = reversal)')
    ax.set_xticks(range(1, 21))
    plt.tight_layout()
    plt.savefig('plots/eda/ic_by_lag.png', dpi=120)
    plt.close()
    print(f'Strongest lag: RET_{max(ic_by_lag, key=lambda x: abs(x[1]))[0]}')

    X_train['SV1_missing'] = X_train['SIGNED_VOLUME_1'].isna().astype(int)
    missing_rate = X_train['SV1_missing'].mean()
    print(f'Global SV1 missing rate: {missing_rate:.2%}')
    sv1_by_alloc = X_train.groupby('ALLOCATION')['SV1_missing'].mean().sort_values(ascending=False)
    print(f'Allocs with 100% missing : {(sv1_by_alloc == 1.0).sum()}')
    print(f'Allocs with 0% missing   : {(sv1_by_alloc == 0.0).sum()}')
    print(f'Allocs with partial miss : {((sv1_by_alloc > 0) & (sv1_by_alloc < 1)).sum()}')
    sv1_by_ts = X_train.groupby('TS')['SV1_missing'].mean()
    print(f'\nTimestamps 100% missing  : {(sv1_by_ts == 1.0).sum()} / {len(sv1_by_ts)}')
    print(f'Timestamps 0% missing    : {(sv1_by_ts == 0.0).sum()} / {len(sv1_by_ts)}')
    special_allocs = sv1_by_alloc[sv1_by_alloc == 0.0].index.tolist()
    print(f'\n24 special allocations (SV1 always available): {sorted(special_allocs)}')

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    sv1_by_alloc.plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('SV1 missing rate per allocation')
    axes[0].axhline(missing_rate, color='red', linestyle='--', label=f'Mean ({missing_rate:.2%})')
    axes[0].legend()
    sv1_by_ts.plot(ax=axes[1], color='steelblue', alpha=0.6, linewidth=0.8)
    axes[1].set_title('SV1 missing rate over time')
    axes[1].axhline(missing_rate, color='red', linestyle='--', label=f'Mean ({missing_rate:.2%})')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('plots/eda/sv1_missing.png', dpi=120)
    plt.close()

    alloc_train = set(X_train['ALLOCATION'].unique())
    alloc_test  = set(X_test['ALLOCATION'].unique())
    print(f'Allocations train only : {len(alloc_train - alloc_test)}')
    print(f'Allocations test only  : {len(alloc_test - alloc_train)}')
    print(f'Allocations in common  : {len(alloc_train & alloc_test)}')
    ts_overlap = len(set(X_train['TS'].unique()) & set(X_test['TS'].unique()))
    print(f'\nTimestamps overlap: {ts_overlap} -> zero temporal overlap between train and test')

    alloc_per_ts_train = X_train.groupby('TS').size()
    alloc_per_ts_test  = X_test.groupby('TS').size()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alloc_per_ts_train.plot(kind='hist', bins=30, ax=axes[0], alpha=0.6, density=True, color='steelblue', label='Train', edgecolor='white')
    alloc_per_ts_test.plot(kind='hist', bins=20, ax=axes[0], alpha=0.6, density=True, color='tomato', label='Test', edgecolor='white')
    axes[0].set_title('Allocations per timestamp - Train vs Test')
    axes[0].set_xlabel('Number of allocations present')
    axes[0].axvline(250, color='black', linestyle='--', alpha=0.7, label='250 threshold')
    axes[0].legend()
    group_train = X_train['GROUP'].value_counts(normalize=True).sort_index()
    group_test  = X_test['GROUP'].value_counts(normalize=True).sort_index()
    pd.DataFrame({'Train': group_train, 'Test': group_test}).plot(kind='bar', ax=axes[1], color=['steelblue', 'tomato'])
    axes[1].set_title('GROUP distribution - Train vs Test')
    axes[1].axhline(0.25, color='black', linestyle='--', label='25% (test target)')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('plots/eda/train_test_structure.png', dpi=120)
    plt.close()

    dense_train = (alloc_per_ts_train >= 250).mean()
    dense_test  = (alloc_per_ts_test >= 250).mean()
    print(f'\nDense timestamps (>=250 allocs): train={dense_train:.1%}  test={dense_test:.1%}')

    if 'SV1_missing' in X_train.columns:
        X_train.drop(columns=['SV1_missing'], inplace=True)

    print('\nEDA plots saved to plots/eda/')


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING V1
# ──────────────────────────────────────────────────────────────────────────────

def build_structural_profiles(X_df):
    grouped = X_df.groupby('ALLOCATION')
    profiles = pd.DataFrame({
        'ALLOC_VOL':          grouped['RET_1'].std(),
        'ALLOC_SHARPE_HIST':  grouped['RET_1'].apply(lambda x: x.mean() / (x.std() + eps)),
        'ALLOC_AUTOCORR':     grouped['RET_1'].apply(lambda x: x.autocorr(lag=1)),
        'ALLOC_TURNOVER_MEAN':grouped['MEDIAN_DAILY_TURNOVER'].mean(),
    })
    return profiles


def build_target_encoded_profiles(X_df, y_series, n_splits=8):
    df = X_df[['TS', 'ALLOCATION', 'GROUP', 'REVERSAL_1']].copy()
    df['target_bin'] = (y_series.values > 0).astype(float)
    df['ALLOC_HIT_RATE']             = np.nan
    df['ALLOC_REVERSAL_SENSITIVITY'] = np.nan
    df['GROUP_HIT_RATE']             = np.nan
    train_dates = df['TS'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr_idx, val_idx in kf.split(train_dates):
        tr_dates  = train_dates[tr_idx]
        val_dates = train_dates[val_idx]
        tr_mask   = df['TS'].isin(tr_dates)
        val_mask  = df['TS'].isin(val_dates)
        df_tr     = df[tr_mask]
        global_hit = df_tr['target_bin'].mean()
        alloc_hit = df_tr.groupby('ALLOCATION')['target_bin'].mean()
        df.loc[val_mask, 'ALLOC_HIT_RATE'] = df.loc[val_mask, 'ALLOCATION'].map(alloc_hit).fillna(global_hit)
        alloc_rev = df_tr.groupby('ALLOCATION').apply(lambda g: g['REVERSAL_1'].corr(g['target_bin']))
        df.loc[val_mask, 'ALLOC_REVERSAL_SENSITIVITY'] = df.loc[val_mask, 'ALLOCATION'].map(alloc_rev).fillna(0.0)
        group_hit = df_tr.groupby('GROUP')['target_bin'].mean()
        df.loc[val_mask, 'GROUP_HIT_RATE'] = df.loc[val_mask, 'GROUP'].map(group_hit).fillna(global_hit)
    test_profiles = {
        'ALLOC_HIT_RATE': df.groupby('ALLOCATION')['target_bin'].mean(),
        'ALLOC_REVERSAL_SENSITIVITY': df.groupby('ALLOCATION').apply(lambda g: g['REVERSAL_1'].corr(g['target_bin'])),
        'GROUP_HIT_RATE': df.groupby('GROUP')['target_bin'].mean(),
        'global_hit': df['target_bin'].mean(),
    }
    return df[['ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY', 'GROUP_HIT_RATE']], test_profiles


def apply_target_profiles_test(X_df, test_profiles):
    d   = X_df[['ALLOCATION', 'GROUP']].copy()
    gh  = test_profiles['global_hit']
    d['ALLOC_HIT_RATE'] = d['ALLOCATION'].map(test_profiles['ALLOC_HIT_RATE']).fillna(gh)
    d['ALLOC_REVERSAL_SENSITIVITY'] = d['ALLOCATION'].map(test_profiles['ALLOC_REVERSAL_SENSITIVITY']).fillna(0.0)
    d['GROUP_HIT_RATE'] = d['GROUP'].map(test_profiles['GROUP_HIT_RATE']).fillna(gh)
    return d[['ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY', 'GROUP_HIT_RATE']]


def build_features(df, fit_stats=None):
    df  = df.copy()
    ret = df[RET_COLS].values.astype(float)
    sv  = np.nan_to_num(df[SV_COLS].values.astype(float), nan=0.0)

    df['ACCEL_3_10']  = ret[:, :3].mean(1)  - ret[:, 3:10].mean(1)
    df['ACCEL_5_20']  = ret[:, :5].mean(1)  - ret[:, 5:20].mean(1)
    df['REVERSAL_1']  = ret[:, 0] - ret[:, 1:5].mean(1)
    df['HIT_RATE_5']  = (ret[:, :5]  > 0).mean(1)
    df['HIT_RATE_20'] = (ret[:, :20] > 0).mean(1)
    early             = ret[:, 13:20].mean(1)
    mid               = ret[:, 6:13].mean(1)
    late              = ret[:, 0:6].mean(1)
    df['CONVEXITY']   = early - 2 * mid + late

    df['SHARPE_20']  = ret.mean(1)        / (ret.std(1)        + eps)
    df['SHARPE_5']   = ret[:, :5].mean(1) / (ret[:, :5].std(1) + eps)
    df['VOL_RATIO']  = ret[:, :5].std(1)  / (ret[:, 5:].std(1) + eps)
    df['SKEW_20']    = skew_vec(ret)
    df['KURT_20']    = kurt_vec(ret)
    df['AUTOCORR_1'] = autocorr_lag1_vec(ret)

    sv_2_10  = sv[:, 1:10]
    ret_2_10 = ret[:, 1:10]
    sv_abs   = np.abs(sv_2_10)
    sv_w     = sv_abs / (sv_abs.sum(1, keepdims=True) + eps)
    df['VW_MOM']     = (ret_2_10 * sv_w).sum(1)
    df['ACCUM_DIST'] = spearman_vec(ret[:, 1:], sv[:, 1:])

    for i in [3, 5, 10, 15, 20]:
        df[f'AVERAGE_PERF_{i}'] = ret[:, :i].mean(1)
    df['STD_PERF_20'] = ret.std(1)

    for i in [3, 5, 10, 15, 20]:
        df[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = df.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    df['AVG_ALLOC_VOL_20'] = df.groupby('TS')['STD_PERF_20'].transform('mean')

    cs_feats = ['AVERAGE_PERF_5', 'AVERAGE_PERF_20', 'STD_PERF_20', 'SHARPE_20', 'AUTOCORR_1']
    for feat in cs_feats:
        grp_m = df.groupby(['TS', 'GROUP'])[feat].transform('mean')
        grp_s = df.groupby(['TS', 'GROUP'])[feat].transform('std').fillna(0)
        df[f'{feat}_ZSCORE_GROUP'] = (df[feat] - grp_m) / (grp_s + eps)

    rank_feats = ['AVERAGE_PERF_5', 'AVERAGE_PERF_20', 'SHARPE_20']
    for feat in rank_feats:
        df[f'{feat}_RANK_GROUP']  = df.groupby(['TS', 'GROUP'])[feat].rank(pct=True)
        df[f'{feat}_RANK_MARKET'] = df.groupby('TS')[feat].rank(pct=True)
        med = df.groupby('TS')[feat].transform('median')
        df[f'{feat}_DEV_MARKET']  = df[feat] - med

    df['DISP_GROUP_5']   = df.groupby(['TS', 'GROUP'])['AVERAGE_PERF_5'].transform('std')
    df['DISP_GROUP_20']  = df.groupby(['TS', 'GROUP'])['AVERAGE_PERF_20'].transform('std')
    df['MARKET_DISP']    = df.groupby('TS')['RET_1'].transform('std')
    df['MARKET_BREADTH'] = df.groupby('TS')['RET_1'].transform(lambda x: (x > 0).mean())

    df['REGIME_INTERACTION_5']  = df['AVERAGE_PERF_5']  * df['ALLOCATIONS_AVERAGE_PERF_5']
    df['REGIME_INTERACTION_20'] = df['AVERAGE_PERF_20'] * df['ALLOCATIONS_AVERAGE_PERF_20']

    med_cols  = [f'MED_{c}' for c in RET_COLS]
    group_med = df.groupby(['TS', 'GROUP'])[RET_COLS].median().rename(columns={c: f'MED_{c}' for c in RET_COLS})
    df = df.join(group_med, on=['TS', 'GROUP'])
    df['CROWDING'] = pearson_vec(df[RET_COLS].values.astype(float), df[med_cols].values.astype(float))
    df.drop(columns=med_cols, inplace=True)

    to_mean = df.groupby('TS')['MEDIAN_DAILY_TURNOVER'].transform('mean')
    to_std  = df.groupby('TS')['MEDIAN_DAILY_TURNOVER'].transform('std').fillna(1)
    df['TURNOVER_ZSCORE'] = (df['MEDIAN_DAILY_TURNOVER'] - to_mean) / (to_std + eps)

    if fit_stats is None:
        sv1_avail      = df.groupby('ALLOCATION')['SIGNED_VOLUME_1'].apply(lambda x: x.notna().mean())
        special_allocs = sv1_avail[sv1_avail > 0.99].index.tolist()
        fit_stats      = {'special_allocs': special_allocs}
    else:
        special_allocs = fit_stats['special_allocs']

    df['IS_SPECIAL_ALLOC']    = df['ALLOCATION'].isin(special_allocs).astype(int)
    df['SPECIAL_x_SHARPE20']  = df['IS_SPECIAL_ALLOC'] * df['SHARPE_20']
    df['SPECIAL_x_AUTOCORR']  = df['IS_SPECIAL_ALLOC'] * df['AUTOCORR_1']
    df['SPECIAL_x_MOM20']     = df['IS_SPECIAL_ALLOC'] * df['AVERAGE_PERF_20']

    df.fillna(0.0, inplace=True)
    return df, fit_stats


def get_features_v1():
    return (
        [f'AVERAGE_PERF_{i}'             for i in [3, 5, 10, 15, 20]] +
        [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]] +
        ['STD_PERF_20', 'AVG_ALLOC_VOL_20',
         'ACCEL_3_10', 'ACCEL_5_20', 'REVERSAL_1',
         'HIT_RATE_5', 'HIT_RATE_20', 'CONVEXITY',
         'SHARPE_20', 'SHARPE_5', 'VOL_RATIO',
         'SKEW_20', 'KURT_20', 'AUTOCORR_1',
         'VW_MOM', 'ACCUM_DIST',
         'AVERAGE_PERF_5_ZSCORE_GROUP',  'AVERAGE_PERF_20_ZSCORE_GROUP',
         'STD_PERF_20_ZSCORE_GROUP',     'SHARPE_20_ZSCORE_GROUP',
         'AUTOCORR_1_ZSCORE_GROUP',
         'AVERAGE_PERF_5_RANK_GROUP',    'AVERAGE_PERF_20_RANK_GROUP',
         'SHARPE_20_RANK_GROUP',
         'AVERAGE_PERF_5_RANK_MARKET',   'AVERAGE_PERF_20_RANK_MARKET',
         'AVERAGE_PERF_5_DEV_MARKET',    'AVERAGE_PERF_20_DEV_MARKET',
         'DISP_GROUP_5', 'DISP_GROUP_20',
         'MARKET_DISP', 'MARKET_BREADTH',
         'REGIME_INTERACTION_5', 'REGIME_INTERACTION_20',
         'CROWDING',
         'TURNOVER_ZSCORE', 'IS_SPECIAL_ALLOC',
         'SPECIAL_x_SHARPE20', 'SPECIAL_x_AUTOCORR', 'SPECIAL_x_MOM20',
         'ALLOC_VOL', 'ALLOC_SHARPE_HIST', 'ALLOC_AUTOCORR', 'ALLOC_TURNOVER_MEAN',
         'ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY', 'GROUP_HIT_RATE',
         'REVERSAL_x_SENSITIVITY', 'MOM_x_SHARPE_HIST', 'HIT_RATE_x_AUTOCORR']
    )


def prepare_v1_features(X_train, X_test, y_train):
    print('Building train features v1...')
    t0 = time.time()
    X_train_fe, fit_stats = build_features(X_train)
    print(f'Train: {time.time()-t0:.1f}s - {X_train_fe.shape}')

    print('Building test features v1...')
    t0 = time.time()
    X_test_fe, _ = build_features(X_test, fit_stats=fit_stats)
    print(f'Test:  {time.time()-t0:.1f}s - {X_test_fe.shape}')

    print('Structural profiles...')
    struct_profiles = build_structural_profiles(X_train)
    X_train_fe = X_train_fe.join(struct_profiles, on='ALLOCATION')
    X_test_fe  = X_test_fe.join(struct_profiles,  on='ALLOCATION')
    for col in struct_profiles.columns:
        X_train_fe[col] = X_train_fe[col].fillna(0.0)
        X_test_fe[col]  = X_test_fe[col].fillna(0.0)

    print('OOF target encoding...')
    t0 = time.time()
    oof_target, test_profiles = build_target_encoded_profiles(X_train_fe, y_train['target'])
    X_train_fe = pd.concat([X_train_fe, oof_target], axis=1)
    test_target = apply_target_profiles_test(X_test_fe, test_profiles)
    X_test_fe   = pd.concat([X_test_fe, test_target], axis=1)
    print(f'OOF done in {time.time()-t0:.1f}s')

    for df_ in [X_train_fe, X_test_fe]:
        df_['REVERSAL_x_SENSITIVITY'] = df_['REVERSAL_1'] * df_['ALLOC_REVERSAL_SENSITIVITY']
        df_['MOM_x_SHARPE_HIST']      = df_['AVERAGE_PERF_5'] * df_['ALLOC_SHARPE_HIST']
        df_['HIT_RATE_x_AUTOCORR']    = df_['HIT_RATE_20'] * df_['ALLOC_AUTOCORR']

    X_train_fe.fillna(0.0, inplace=True)
    X_test_fe.fillna(0.0, inplace=True)
    print(f'Final shape - Train: {X_train_fe.shape} | Test: {X_test_fe.shape}')
    return X_train_fe, X_test_fe


# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE CV
# ──────────────────────────────────────────────────────────────────────────────

def train_ensemble_cv(X_tr, y_tr, X_te, features, n_splits=N_SPLITS):
    train_dates = X_tr['TS'].unique()
    splits = KFold(n_splits=n_splits, shuffle=True, random_state=0).split(train_dates)

    oof_lgbm  = np.zeros(len(X_tr))
    oof_xgb   = np.zeros(len(X_tr))
    oof_cat   = np.zeros(len(X_tr))
    test_lgbm = np.zeros(len(X_te))
    test_xgb  = np.zeros(len(X_te))
    test_cat  = np.zeros(len(X_te))
    scores_lgbm, scores_xgb, scores_cat = [], [], []

    X_tr_feat = X_tr[features].fillna(0)
    X_te_feat = X_te[features].fillna(0)
    y_bin     = (y_tr.values > 0).astype(int)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        tr_dates  = train_dates[tr_idx]
        val_dates = train_dates[val_idx]
        tr_mask   = X_tr['TS'].isin(tr_dates).values
        val_mask  = X_tr['TS'].isin(val_dates).values

        X_f, y_f = X_tr_feat.values[tr_mask],  y_bin[tr_mask]
        X_v, y_v = X_tr_feat.values[val_mask], y_bin[val_mask]
        X_t      = X_te_feat.values

        print(f'\n--- Fold {fold+1}/{n_splits} ---')

        print('  LightGBM DART...')
        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(X_f, y_f, eval_set=[(X_v, y_v)],
                       callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(500)])
        p_lgbm_val  = lgbm_model.predict_proba(X_v)[:, 1]
        p_lgbm_test = lgbm_model.predict_proba(X_t)[:, 1]
        acc_l = ((p_lgbm_val > 0.5) == y_v).mean()
        scores_lgbm.append(acc_l)
        oof_lgbm[val_mask] = p_lgbm_val
        test_lgbm += p_lgbm_test / n_splits
        print(f'  LGBM val acc: {acc_l*100:.3f}%')

        print('  XGBoost...')
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(X_f, y_f, eval_set=[(X_v, y_v)], verbose=False)
        p_xgb_val  = xgb_model.predict_proba(X_v)[:, 1]
        p_xgb_test = xgb_model.predict_proba(X_t)[:, 1]
        acc_x = ((p_xgb_val > 0.5) == y_v).mean()
        scores_xgb.append(acc_x)
        oof_xgb[val_mask] = p_xgb_val
        test_xgb += p_xgb_test / n_splits
        print(f'  XGB  val acc: {acc_x*100:.3f}%')

        print('  CatBoost...')
        cat_model = cb.CatBoostClassifier(**CAT_PARAMS)
        cat_model.fit(X_f, y_f, eval_set=(X_v, y_v), early_stopping_rounds=EARLY_STOP)
        p_cat_val  = cat_model.predict_proba(X_v)[:, 1]
        p_cat_test = cat_model.predict_proba(X_t)[:, 1]
        acc_c = ((p_cat_val > 0.5) == y_v).mean()
        scores_cat.append(acc_c)
        oof_cat[val_mask] = p_cat_val
        test_cat += p_cat_test / n_splits
        print(f'  CAT  val acc: {acc_c*100:.3f}%')

        blend_val = (p_lgbm_val + p_xgb_val + p_cat_val) / 3
        print(f'  BLEND acc:   {((blend_val > 0.5) == y_v).mean()*100:.3f}%')

    print(f"\n{'='*55}")
    print(f'[LGBM]  {np.mean(scores_lgbm)*100:.3f}% +/- {np.std(scores_lgbm)*100:.3f}%')
    print(f'[XGB]   {np.mean(scores_xgb)*100:.3f}% +/- {np.std(scores_xgb)*100:.3f}%')
    print(f'[CAT]   {np.mean(scores_cat)*100:.3f}% +/- {np.std(scores_cat)*100:.3f}%')

    oof_blend = (oof_lgbm + oof_xgb + oof_cat) / 3
    y_bin_arr = (y_tr.values > 0).astype(int)
    print(f'[BLEND] OOF global: {((oof_blend > 0.5) == y_bin_arr).mean()*100:.3f}%')

    print('\nOOF accuracy by GROUP:')
    groups = X_tr['GROUP'].values
    for g in [1, 2, 3, 4]:
        mask_g = groups == g
        if mask_g.sum() == 0:
            continue
        acc_g = ((oof_blend[mask_g] > 0.5) == y_bin_arr[mask_g]).mean()
        print(f'  GROUP {g}: {acc_g*100:.3f}%  (n={mask_g.sum()})')

    print('\n--- Optimal blend weights ---')
    best_w, best_acc = (1/3, 1/3, 1/3), 0.
    for wl in np.arange(0.2, 0.6, 0.05):
        for wx in np.arange(0.2, 0.6, 0.05):
            wc = 1 - wl - wx
            if wc < 0.1:
                continue
            blend = wl * oof_lgbm + wx * oof_xgb + wc * oof_cat
            acc   = ((blend > 0.5) == y_bin_arr).mean()
            if acc > best_acc:
                best_acc, best_w = acc, (wl, wx, wc)
    wl, wx, wc = best_w

    return {
        'test_lgbm': test_lgbm, 'test_xgb': test_xgb, 'test_cat': test_cat,
        'test_blend_equal': (test_lgbm + test_xgb + test_cat) / 3,
        'test_blend_optim': wl*test_lgbm + wx*test_xgb + wc*test_cat,
        'oof_lgbm': oof_lgbm, 'oof_xgb': oof_xgb, 'oof_cat': oof_cat,
        'oof_blend': oof_blend, 'best_weights': best_w, 'yb': y_bin_arr,
        'scores': (scores_lgbm, scores_xgb, scores_cat),
    }


# ──────────────────────────────────────────────────────────────────────────────
# GBDT BASELINE V1
# ──────────────────────────────────────────────────────────────────────────────

def run_gbdt_v1(X_train, X_test, y_train, sample_submission):
    X_train_fe, X_test_fe = prepare_v1_features(X_train, X_test, y_train)
    features_v1 = get_features_v1()

    missing = [f for f in features_v1 if f not in X_train_fe.columns]
    print(f'Features v1: {len(features_v1)}')
    if missing:
        print(f'Missing: {missing}')


    t0 = time.time()
    results_v1 = train_ensemble_cv(X_train_fe, y_train['target'], X_test_fe, features_v1)

    with open('models/ensemble_results_v1.pkl', 'wb') as f:
        pickle.dump(results_v1, f)

    train_dates = X_train_fe['TS'].unique()
    kf          = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    importances = []
    for tr_idx, _ in kf.split(train_dates):
        tr_dates = train_dates[tr_idx]
        tr_mask  = X_train_fe['TS'].isin(tr_dates)
        X_f = X_train_fe.loc[tr_mask, features_v1].fillna(0)
        y_f = (y_train.loc[tr_mask, 'target'] > 0).astype(int)
        m   = lgb.LGBMClassifier(**LGBM_FAST)
        m.fit(X_f, y_f)
        importances.append(m.feature_importances_)

    imp_mean = pd.DataFrame(importances, columns=features_v1).mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 10))
    imp_mean.head(25).plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Top 25 feature importance - GBDT baseline')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/modeling/feature_importance_v1.png', dpi=120)
    plt.close()
    print('Top 10 features:')
    print(imp_mean.head(10).to_string())

    idx = sample_submission.index
    for name, preds in [
        ('v1_blend_equal', results_v1['test_blend_equal']),
        ('v1_blend_optim', results_v1['test_blend_optim']),
        ('v1_lgbm_solo',   results_v1['test_lgbm']),
    ]:
        s    = pd.DataFrame({'target': (preds > 0.5).astype(int)}, index=idx)
        taux = s['target'].mean()
        flag = ' WARNING: biased' if taux > 0.55 or taux < 0.45 else ''
        s.to_csv(f'submissions/submission_{name}.csv')
        print(f'  {name:<25s} rate_1={taux:.4f}{flag}')

    with open('models/gbdt_oof_v1.pkl', 'wb') as f:
        pickle.dump(results_v1, f)

    return X_train_fe, X_test_fe, results_v1


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMER
# ──────────────────────────────────────────────────────────────────────────────

def run_transformer(X_train, X_test, y_train, y_bin, X_train_fe, X_test_fe, sample_submission):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    device = torch.device('cpu')
    print(f'Device: {device}')

    SCALAR_FEATURES = [
        'AVERAGE_PERF_5', 'AVERAGE_PERF_20', 'SHARPE_20', 'VOL_RATIO',
        'REVERSAL_1', 'CROWDING', 'MARKET_BREADTH', 'MARKET_DISP',
        'ALLOC_SHARPE_HIST', 'ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY',
        'REVERSAL_x_SENSITIVITY', 'TURNOVER_ZSCORE',
        'DISP_GROUP_5', 'DISP_GROUP_20', 'AVG_ALLOC_VOL_20',
    ]

    scaler = StandardScaler()
    X_sca_train = scaler.fit_transform(X_train_fe[SCALAR_FEATURES].fillna(0).values)
    X_sca_test  = scaler.transform(X_test_fe[SCALAR_FEATURES].fillna(0).values)

    ret_arr    = X_train[RET_COLS].values.astype(np.float32)
    sv_arr     = np.nan_to_num(X_train[SV_COLS].values.astype(np.float32), nan=0.0)
    ret_arr_te = X_test[RET_COLS].values.astype(np.float32)
    sv_arr_te  = np.nan_to_num(X_test[SV_COLS].values.astype(np.float32), nan=0.0)

    ret_arr    = ret_arr    / (ret_arr.std(axis=1, keepdims=True) + 1e-6)
    ret_arr_te = ret_arr_te / (ret_arr_te.std(axis=1, keepdims=True) + 1e-6)
    sv_arr     = sv_arr     / (np.abs(sv_arr).max(axis=1, keepdims=True) + 1e-6)
    sv_arr_te  = sv_arr_te  / (np.abs(sv_arr_te).max(axis=1, keepdims=True) + 1e-6)

    seq_train = np.stack([ret_arr, sv_arr], axis=2)
    seq_test  = np.stack([ret_arr_te, sv_arr_te], axis=2)

    T_seq_train = torch.FloatTensor(seq_train)
    T_seq_test  = torch.FloatTensor(seq_test)
    T_sca_train = torch.FloatTensor(X_sca_train)
    T_sca_test  = torch.FloatTensor(X_sca_test)
    T_grp_train = torch.LongTensor(X_train['GROUP'].values - 1)
    T_grp_test  = torch.LongTensor(X_test['GROUP'].values - 1)
    T_y         = torch.FloatTensor(y_bin.values)

    class AllocationDataset(Dataset):
        def __init__(self, seq, scalar, group, target=None):
            self.seq, self.scalar, self.group, self.target = seq, scalar, group, target
        def __len__(self):
            return len(self.seq)
        def __getitem__(self, idx):
            out = {'seq': self.seq[idx], 'scalar': self.scalar[idx], 'group': self.group[idx]}
            if self.target is not None:
                out['target'] = self.target[idx].unsqueeze(0)
            return out

    class DualBranchTransformer(nn.Module):
        def __init__(self, n_scalar, seq_len=20, in_features=2, d_model=32,
                     n_heads=4, n_layers=2, ffn_dim=64, n_groups=4,
                     group_emb_dim=8, scalar_hidden=64, fusion_hidden=64, dropout=0.3):
            super().__init__()
            self.input_proj = nn.Linear(in_features, d_model)
            self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
            self.seq_drop = nn.Dropout(dropout)
            self.group_emb  = nn.Embedding(n_groups, group_emb_dim)
            scalar_in = n_scalar + group_emb_dim
            self.scalar_mlp = nn.Sequential(
                nn.Linear(scalar_in, scalar_hidden), nn.LayerNorm(scalar_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(scalar_hidden, scalar_hidden // 2), nn.LayerNorm(scalar_hidden // 2), nn.GELU(), nn.Dropout(dropout))
            fusion_in = d_model + scalar_hidden // 2
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden), nn.LayerNorm(fusion_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(fusion_hidden, 1))

        def forward(self, seq, scalar, group):
            B = seq.size(0)
            x   = self.input_proj(seq)
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1) + self.pos_embed
            x   = self.transformer(self.seq_drop(x))
            cls_out = x[:, 0, :]
            grp_emb    = self.group_emb(group)
            scalar_out = self.scalar_mlp(torch.cat([scalar, grp_emb], dim=1))
            return self.fusion(torch.cat([cls_out, scalar_out], dim=1)).squeeze(1)

    N_SPLITS_NN = 8
    BATCH_SIZE  = 2048
    LR, WEIGHT_DECAY, PATIENCE, EPOCHS = 3e-4, 1e-2, 8, 40
    ARCH = dict(n_scalar=len(SCALAR_FEATURES), d_model=32, n_heads=4, n_layers=2,
                ffn_dim=64, scalar_hidden=64, fusion_hidden=64, dropout=0.3)

    def run_fold(fold, tr_mask, val_mask):
        ds_tr  = AllocationDataset(T_seq_train[tr_mask], T_sca_train[tr_mask], T_grp_train[tr_mask], T_y[tr_mask])
        ds_val = AllocationDataset(T_seq_train[val_mask], T_sca_train[val_mask], T_grp_train[val_mask], T_y[val_mask])
        ds_te  = AllocationDataset(T_seq_test, T_sca_test, T_grp_test)
        dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=0)
        dl_te  = DataLoader(ds_te, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=0)
        model     = DualBranchTransformer(**ARCH).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.BCEWithLogitsLoss()
        best_acc, best_state, patience_ctr = 0., None, 0
        for epoch in range(EPOCHS):
            model.train()
            for b in dl_tr:
                optimizer.zero_grad()
                loss = criterion(model(b['seq'].to(device), b['scalar'].to(device), b['group'].to(device)), b['target'].to(device).squeeze(1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            model.eval()
            val_preds, val_tgts = [], []
            with torch.no_grad():
                for b in dl_val:
                    logits = model(b['seq'].to(device), b['scalar'].to(device), b['group'].to(device))
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_tgts.extend(b['target'].squeeze(1).numpy())
            val_acc = ((np.array(val_preds) > 0.5) == np.array(val_tgts)).mean()
            if val_acc > best_acc:
                best_acc, best_state, patience_ctr = val_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        test_preds = []
        with torch.no_grad():
            for b in dl_te:
                test_preds.extend(torch.sigmoid(model(b['seq'].to(device), b['scalar'].to(device), b['group'].to(device))).cpu().numpy())
        print(f'  Fold {fold+1} best val acc: {best_acc*100:.3f}%')
        return best_acc, np.array(val_preds), np.array(test_preds)

    train_dates_nn = X_train['TS'].unique()
    kf_nn = KFold(n_splits=N_SPLITS_NN, shuffle=True, random_state=0)
    oof_transformer  = np.zeros(len(X_train))
    test_transformer = np.zeros(len(X_test))
    nn_scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf_nn.split(train_dates_nn)):
        tr_mask = X_train['TS'].isin(train_dates_nn[tr_idx]).values
        val_mask = X_train['TS'].isin(train_dates_nn[val_idx]).values
        best_acc, oof_probs, test_probs = run_fold(fold, tr_mask, val_mask)
        oof_transformer[val_mask] = oof_probs
        test_transformer += test_probs / N_SPLITS_NN
        nn_scores.append(best_acc)

    print(f'\nTransformer CV: {np.mean(nn_scores)*100:.3f}% +/- {np.std(nn_scores)*100:.3f}%')
    print(f'OOF global: {((oof_transformer > 0.5) == y_bin.values).mean()*100:.3f}%')

    sub = pd.DataFrame({'target': (test_transformer > 0.5).astype(int)}, index=sample_submission.index)
    sub.to_csv('submissions/submission_transformer.csv')
    print(f'Transformer submission rate_1: {sub["target"].mean():.4f}')
    with open('models/transformer_oof.pkl', 'wb') as f:
        pickle.dump({'oof': oof_transformer, 'test': test_transformer}, f)


# ──────────────────────────────────────────────────────────────────────────────
# AUTOENCODER
# ──────────────────────────────────────────────────────────────────────────────

def run_autoencoder(X_train, X_test, y_train, y_bin, X_train_fe, X_test_fe, sample_submission):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    device = torch.device('cpu')
    features_v1 = get_features_v1()

    scaler_ae = StandardScaler()
    X_ae_train = scaler_ae.fit_transform(X_train_fe[features_v1].fillna(0).values).astype(np.float32)
    X_ae_test  = scaler_ae.transform(X_test_fe[features_v1].fillna(0).values).astype(np.float32)
    print(f'AE input dim: {X_ae_train.shape[1]}')

    class AEDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            out = {'x': self.X[idx]}
            if self.y is not None:
                out['target'] = self.y[idx]
            return out

    class SupervisedAutoencoder(nn.Module):
        def __init__(self, n_input=60, latent_dim=8, encoder_dims=[64, 32],
                     decoder_dims=[32, 64], mlp_dims=[128, 64], noise_std=0.03,
                     dropout_enc=0.1, dropout_mlp=0.3):
            super().__init__()
            self.noise_std = noise_std
            enc_layers, in_dim = [], n_input
            for out_dim in encoder_dims:
                enc_layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.SiLU(), nn.Dropout(dropout_enc)]
                in_dim = out_dim
            enc_layers.append(nn.Linear(in_dim, latent_dim))
            self.encoder = nn.Sequential(*enc_layers)
            dec_layers, in_dim = [], latent_dim
            for out_dim in decoder_dims:
                dec_layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.SiLU()]
                in_dim = out_dim
            dec_layers.append(nn.Linear(in_dim, n_input))
            self.decoder = nn.Sequential(*dec_layers)
            mlp_in = n_input + latent_dim
            mlp_layers, in_dim = [], mlp_in
            for out_dim in mlp_dims:
                mlp_layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.SiLU(), nn.Dropout(dropout_mlp)]
                in_dim = out_dim
            mlp_layers.append(nn.Linear(in_dim, 1))
            self.mlp = nn.Sequential(*mlp_layers)

        def forward(self, x):
            x_noisy = x + torch.randn_like(x) * self.noise_std if self.training else x
            z       = self.encoder(x_noisy)
            x_recon = self.decoder(z)
            logit   = self.mlp(torch.cat([x, z], dim=1))
            return x_recon, logit.squeeze(1)

    AE_ARCH = dict(n_input=X_ae_train.shape[1], latent_dim=8, encoder_dims=[64, 32],
                   decoder_dims=[32, 64], mlp_dims=[128, 64], noise_std=0.03, dropout_enc=0.1, dropout_mlp=0.3)
    AE_LR, AE_WD, AE_EPOCHS, AE_PATIENCE, AE_BS = 1e-3, 1e-4, 100, 10, 4096
    ALPHA_MSE, BETA_BCE = 0.1, 1.0
    N_SPLITS_NN = 8

    def train_ae_fold(fold, tr_mask, val_mask):
        ds_tr  = AEDataset(X_ae_train[tr_mask], y_bin.values[tr_mask])
        ds_val = AEDataset(X_ae_train[val_mask], y_bin.values[val_mask])
        ds_te  = AEDataset(X_ae_test)
        dl_tr  = DataLoader(ds_tr, batch_size=AE_BS, shuffle=True, num_workers=0)
        dl_val = DataLoader(ds_val, batch_size=AE_BS*4, shuffle=False, num_workers=0)
        dl_te  = DataLoader(ds_te, batch_size=AE_BS*4, shuffle=False, num_workers=0)
        model = SupervisedAutoencoder(**AE_ARCH).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WD)
        mse_loss, bce_loss = nn.MSELoss(), nn.BCEWithLogitsLoss()
        best_val_bce, best_state, patience_ctr = float('inf'), None, 0
        for epoch in range(AE_EPOCHS):
            model.train()
            for b in dl_tr:
                x, tgt = b['x'].to(device), b['target'].to(device)
                optimizer.zero_grad()
                x_recon, logit = model(x)
                loss = ALPHA_MSE * mse_loss(x_recon, x) + BETA_BCE * bce_loss(logit, tgt.squeeze(1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            model.eval()
            val_bce, val_preds, val_tgts = 0., [], []
            with torch.no_grad():
                for b in dl_val:
                    x, tgt = b['x'].to(device), b['target'].to(device)
                    _, logit = model(x)
                    val_bce += bce_loss(logit, tgt.squeeze(1)).item() * len(tgt)
                    val_preds.extend(torch.sigmoid(logit).cpu().numpy())
                    val_tgts.extend(tgt.squeeze(1).cpu().numpy())
            val_bce /= len(ds_val)
            if val_bce < best_val_bce:
                best_val_bce, best_state, patience_ctr = val_bce, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_ctr += 1
                if patience_ctr >= AE_PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        test_preds = []
        with torch.no_grad():
            for b in dl_te:
                _, logit = model(b['x'].to(device))
                test_preds.extend(torch.sigmoid(logit).cpu().numpy())
        val_acc = ((np.array(val_preds) > 0.5) == np.array(val_tgts)).mean()
        print(f'  Fold {fold+1} val acc: {val_acc*100:.3f}%')
        return val_acc, np.array(val_preds), np.array(test_preds)

    train_dates_ae = X_train['TS'].unique()
    kf_ae = KFold(n_splits=N_SPLITS_NN, shuffle=True, random_state=0)
    oof_ae, test_ae, ae_scores = np.zeros(len(X_train)), np.zeros(len(X_test)), []
    for fold, (tr_idx, val_idx) in enumerate(kf_ae.split(train_dates_ae)):
        tr_mask = X_train['TS'].isin(train_dates_ae[tr_idx]).values
        val_mask = X_train['TS'].isin(train_dates_ae[val_idx]).values
        val_acc, oof_probs, test_probs = train_ae_fold(fold, tr_mask, val_mask)
        oof_ae[val_mask] = oof_probs
        test_ae += test_probs / N_SPLITS_NN
        ae_scores.append(val_acc)

    print(f'\nAutoencoder CV: {np.mean(ae_scores)*100:.3f}% +/- {np.std(ae_scores)*100:.3f}%')
    print(f'OOF global: {((oof_ae > 0.5) == y_bin.values).mean()*100:.3f}%')

    sub = pd.DataFrame({'target': (test_ae > 0.5).astype(int)}, index=sample_submission.index)
    sub.to_csv('submissions/submission_autoencoder.csv')
    print(f'Autoencoder submission rate_1: {sub["target"].mean():.4f}')
    with open('models/autoencoder_oof.pkl', 'wb') as f:
        pickle.dump({'oof': oof_ae, 'test': test_ae}, f)


# ──────────────────────────────────────────────────────────────────────────────
# STABILITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def run_stability_analysis(X_train, X_test, y_train, X_train_fe, X_test_fe):
    features_v1 = get_features_v1()
    ts_density_train = X_train.groupby('TS')['ALLOCATION'].count()
    ts_density_test  = X_test.groupby('TS')['ALLOCATION'].count()

    X_train_fe['ts_density'] = X_train_fe['TS'].map(ts_density_train)
    dense_mask  = X_train_fe['ts_density'] >= DENSITY_THRESHOLD
    sparse_mask = X_train_fe['ts_density'] <  DENSITY_THRESHOLD

    print(f'Dense timestamps (>= {DENSITY_THRESHOLD} allocs)  : {dense_mask.sum():,} obs ({dense_mask.mean()*100:.1f}%)')
    print(f'Sparse timestamps (< {DENSITY_THRESHOLD} allocs)  : {sparse_mask.sum():,} obs ({sparse_mask.mean()*100:.1f}%)')
    print(f'Test timestamps dense: {(ts_density_test >= DENSITY_THRESHOLD).mean()*100:.1f}%')

    print('Computing KS statistics train vs test...')
    ks_results = {}
    for feat in features_v1:
        if feat not in X_test_fe.columns:
            continue
        tr_vals = X_train_fe[feat].dropna().values
        te_vals = X_test_fe[feat].dropna().values
        stat, _ = ks_2samp(tr_vals, te_vals)
        ks_results[feat] = stat
    ks_tt = pd.Series(ks_results).sort_values(ascending=False)
    print('\nTop 15 features by KS train/test:')
    print(ks_tt.head(15).round(4).to_string())

    print('\nComputing KS statistics dense vs sparse...')
    ks_ds_results = {}
    for feat in features_v1:
        dns = X_train_fe.loc[dense_mask, feat].dropna().values
        sps = X_train_fe.loc[sparse_mask, feat].dropna().values
        if len(dns) < 100 or len(sps) < 100:
            continue
        stat, _ = ks_2samp(dns, sps)
        ks_ds_results[feat] = stat
    ks_ds = pd.Series(ks_ds_results).sort_values(ascending=False)
    print('\nTop 15 features by KS dense/sparse:')
    print(ks_ds.head(15).round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, feat, title in zip(axes,
        ['MARKET_BREADTH', 'AVERAGE_PERF_20'],
        ['MARKET_BREADTH (cross-sectional - density sensitive)', 'AVERAGE_PERF_20 (individual - stable)']):
        lo = np.percentile(X_train_fe[feat].dropna(), 2)
        hi = np.percentile(X_train_fe[feat].dropna(), 98)
        bins = np.linspace(lo, hi, 40)
        ax.hist(X_train_fe.loc[sparse_mask, feat].clip(lo, hi), bins=bins, alpha=0.5, density=True, color='steelblue', label='train sparse', edgecolor='none')
        ax.hist(X_train_fe.loc[dense_mask, feat].clip(lo, hi), bins=bins, alpha=0.5, density=True, color='tomato', label='train dense', edgecolor='none')
        ax.hist(X_test_fe[feat].clip(lo, hi), bins=bins, alpha=0.4, density=True, color='green', label='test', edgecolor='none')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    plt.suptitle('Distribution shift: train sparse vs dense vs test', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/stability/density_effect.png', dpi=120)
    plt.close()

    print(f'Computing feature importance variance over {N_SPLITS} folds...')
    train_dates_imp = X_train_fe['TS'].unique()
    kf_imp = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    importances_all = []
    for tr_idx, _ in kf_imp.split(train_dates_imp):
        tr_dates = train_dates_imp[tr_idx]
        tr_mask  = X_train_fe['TS'].isin(tr_dates)
        X_f = X_train_fe.loc[tr_mask, features_v1].fillna(0)
        y_f = (y_train.loc[tr_mask, 'target'] > 0).astype(int)
        m = lgb.LGBMClassifier(**LGBM_FAST)
        m.fit(X_f, y_f)
        importances_all.append(m.feature_importances_)

    imp_df   = pd.DataFrame(importances_all, columns=features_v1)
    imp_mean = imp_df.mean()
    imp_cv   = imp_df.std() / (imp_mean + eps)

    stability_df = pd.DataFrame({
        'importance_mean': imp_mean, 'importance_cv': imp_cv,
        'ks_train_test': ks_tt.reindex(features_v1).fillna(0),
        'ks_dense_sparse': ks_ds.reindex(features_v1).fillna(0),
    }).sort_values('importance_mean', ascending=False)
    print('\nTop 20 by importance mean:')
    print(stability_df.head(20).round(3).to_string())

    imp_norm = imp_mean / (imp_mean.max() + eps)
    stability_df['importance_norm'] = imp_norm
    stability_df['score_composite'] = stability_df['importance_norm'] * (1 - stability_df['ks_train_test']) * (1 - stability_df['ks_dense_sparse'])
    stability_df_sorted = stability_df.sort_values('score_composite', ascending=False)
    print('=== Features ranked by composite score ===')
    print(stability_df_sorted[['importance_norm', 'importance_cv', 'ks_train_test', 'ks_dense_sparse', 'score_composite']].round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(stability_df['ks_train_test'], stability_df['importance_norm'], c='steelblue', alpha=0.7, s=60)
    flagged = stability_df[(stability_df['ks_train_test'] > 0.15) | (stability_df['ks_dense_sparse'] > 0.15)]
    for feat, row in flagged.iterrows():
        axes[0].annotate(feat, (row['ks_train_test'], row['importance_norm']), fontsize=7, color='red')
    axes[0].axvline(0.10, color='orange', linestyle='--', alpha=0.6, label='KS=0.10')
    axes[0].axvline(0.20, color='red', linestyle='--', alpha=0.6, label='KS=0.20')
    axes[0].set_xlabel('KS stat train/test')
    axes[0].set_ylabel('Importance (normalized)')
    axes[0].set_title('Importance vs distribution shift\n(ideal: top-left)')
    axes[0].legend()
    top20 = stability_df_sorted.head(20)
    top20['score_composite'].plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='white')
    axes[1].set_title('Composite robustness score - top 20')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('importance * (1 - KS_tt) * (1 - KS_ds)')
    plt.tight_layout()
    plt.savefig('plots/stability/composite_score.png', dpi=120)
    plt.close()

    def cv_dense_sparse(features, label):
        train_dates = X_train_fe['TS'].unique()
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
        global_sc, dense_sc, sparse_sc = [], [], []
        for tr_idx, val_idx in kf.split(train_dates):
            tr_dates_f = train_dates[tr_idx]
            val_dates  = train_dates[val_idx]
            tr_mask  = X_train_fe['TS'].isin(tr_dates_f)
            val_mask = X_train_fe['TS'].isin(val_dates)
            X_f = X_train_fe.loc[tr_mask, features].fillna(0)
            y_f = (y_train.loc[tr_mask, 'target'] > 0).astype(int)
            X_v = X_train_fe.loc[val_mask, features].fillna(0)
            y_v = (y_train.loc[val_mask, 'target'] > 0).astype(int)
            m = lgb.LGBMClassifier(**LGBM_FAST)
            m.fit(X_f, y_f, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = m.predict_proba(X_v)[:, 1]
            val_density = X_train_fe.loc[val_mask, 'ts_density']
            d_mask = (val_density >= DENSITY_THRESHOLD).values
            global_sc.append(((preds > 0.5) == y_v.values).mean())
            if d_mask.sum() > 0:
                dense_sc.append(((preds[d_mask] > 0.5) == y_v.values[d_mask]).mean())
            if (~d_mask).sum() > 0:
                sparse_sc.append(((preds[~d_mask] > 0.5) == y_v.values[~d_mask]).mean())
        g, d, s = np.mean(global_sc)*100, np.mean(dense_sc)*100, np.mean(sparse_sc)*100
        print(f'[{label:35s}] global={g:.3f}%  dense={d:.3f}%  sparse={s:.3f}%  gap={d-s:+.3f}%')
        return g, d, s

    print('\nCV with dense vs sparse evaluation...')
    cv_dense_sparse(features_v1, 'V1 baseline')
    print('\nStability plots saved to plots/stability/')


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING V2B
# ──────────────────────────────────────────────────────────────────────────────

def build_structural_profiles_v2b(X_df):
    grouped = X_df.groupby('ALLOCATION')
    market_ret_ts = X_df.groupby('TS')['RET_1'].mean()
    betas, alphas = {}, {}
    for alloc, sub in X_df.groupby('ALLOCATION'):
        ts_sub  = sub['TS'].values
        alloc_r = sub['RET_1'].values
        mkt_r   = market_ret_ts.reindex(ts_sub).values
        valid   = ~np.isnan(mkt_r)
        if valid.sum() < 50:
            betas[alloc] = 1.0; alphas[alloc] = 0.0
            continue
        ar, mr = alloc_r[valid], mkt_r[valid]
        cov = np.cov(ar, mr)
        b = cov[0, 1] / (cov[1, 1] + eps)
        betas[alloc]  = b
        alphas[alloc] = ar.mean() - b * mr.mean()
    return pd.DataFrame({
        'ALLOC_VOL':          grouped['RET_1'].std(),
        'ALLOC_SHARPE_HIST':  grouped['RET_1'].apply(lambda x: x.mean() / (x.std() + eps)),
        'ALLOC_TURNOVER_MEAN':grouped['MEDIAN_DAILY_TURNOVER'].mean(),
        'ALLOC_BETA_HIST':    pd.Series(betas),
        'ALLOC_ALPHA_HIST':   pd.Series(alphas),
    })


def build_target_profiles_oof_v2b(X_df, y_series, n_splits=8):
    df = X_df[['TS', 'ALLOCATION', 'REVERSAL_1']].copy()
    df['target_bin'] = (y_series.values > 0).astype(float)
    df['ALLOC_HIT_RATE']             = np.nan
    df['ALLOC_REVERSAL_SENSITIVITY'] = np.nan
    train_dates = df['TS'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr_idx, val_idx in kf.split(train_dates):
        tr_dates  = train_dates[tr_idx]
        val_dates = train_dates[val_idx]
        tr_mask   = df['TS'].isin(tr_dates)
        val_mask  = df['TS'].isin(val_dates)
        df_tr     = df[tr_mask]
        global_hit = df_tr['target_bin'].mean()
        alloc_hit = df_tr.groupby('ALLOCATION')['target_bin'].mean()
        df.loc[val_mask, 'ALLOC_HIT_RATE'] = df.loc[val_mask, 'ALLOCATION'].map(alloc_hit).fillna(global_hit)
        alloc_rev = df_tr.groupby('ALLOCATION').apply(lambda g: g['REVERSAL_1'].corr(g['target_bin']), include_groups=False)
        df.loc[val_mask, 'ALLOC_REVERSAL_SENSITIVITY'] = df.loc[val_mask, 'ALLOCATION'].map(alloc_rev).fillna(0.0)
    test_profiles = {
        'ALLOC_HIT_RATE': df.groupby('ALLOCATION')['target_bin'].mean(),
        'ALLOC_REVERSAL_SENSITIVITY': df.groupby('ALLOCATION').apply(lambda g: g['REVERSAL_1'].corr(g['target_bin']), include_groups=False),
        'global_hit': df['target_bin'].mean(),
    }
    return df[['ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY']], test_profiles


def build_features_v2b(df, fit_stats=None):
    df  = df.copy()
    ret = df[RET_COLS].values.astype(float)
    sv  = np.nan_to_num(df[SV_COLS].values.astype(float), nan=0.0)

    df['ACCEL_3_10']  = ret[:, :3].mean(1) - ret[:, 3:10].mean(1)
    df['ACCEL_5_20']  = ret[:, :5].mean(1) - ret[:, 5:20].mean(1)
    df['REVERSAL_1']  = ret[:, 0] - ret[:, 1:5].mean(1)
    df['HIT_RATE_20'] = (ret > 0).mean(1)
    early = ret[:, 13:20].mean(1)
    mid   = ret[:, 6:13].mean(1)
    late  = ret[:, 0:6].mean(1)
    df['CONVEXITY'] = early - 2 * mid + late

    df['SHARPE_20'] = ret.mean(1) / (ret.std(1) + eps)
    df['SHARPE_5']  = ret[:, :5].mean(1) / (ret[:, :5].std(1) + eps)
    df['VOL_RATIO'] = ret[:, :5].std(1) / (ret[:, 5:].std(1) + eps)
    df['SKEW_20']   = skew_vec(ret)
    df['KURT_20']   = kurt_vec(ret)

    sv_2_10  = sv[:, 1:10]
    ret_2_10 = ret[:, 1:10]
    sv_abs   = np.abs(sv_2_10)
    sv_w     = sv_abs / (sv_abs.sum(1, keepdims=True) + eps)
    df['VW_MOM']     = (ret_2_10 * sv_w).sum(1)
    df['ACCUM_DIST'] = spearman_vec(ret[:, 1:], sv[:, 1:])

    for i in [3, 5, 10, 15, 20]:
        df[f'AVERAGE_PERF_{i}'] = ret[:, :i].mean(1)
    df['STD_PERF_20'] = ret.std(1)

    for i in [3, 5, 10, 15, 20]:
        df[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = df.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
    df['AVG_ALLOC_VOL_20'] = df.groupby('TS')['STD_PERF_20'].transform('mean')

    cs_feats = ['AVERAGE_PERF_5', 'AVERAGE_PERF_20', 'STD_PERF_20', 'SHARPE_20']
    for feat in cs_feats:
        grp_m = df.groupby(['TS', 'GROUP'])[feat].transform('mean')
        grp_s = df.groupby(['TS', 'GROUP'])[feat].transform('std').fillna(0)
        df[f'{feat}_ZSCORE_GROUP'] = (df[feat] - grp_m) / (grp_s + eps)

    rank_feats = ['AVERAGE_PERF_5', 'AVERAGE_PERF_20', 'SHARPE_20']
    for feat in rank_feats:
        df[f'{feat}_RANK_MARKET'] = df.groupby('TS')[feat].rank(pct=True)
        med = df.groupby('TS')[feat].transform('median')
        df[f'{feat}_DEV_MARKET']  = df[feat] - med
    df['AVERAGE_PERF_20_RANK_GROUP'] = df.groupby(['TS', 'GROUP'])['AVERAGE_PERF_20'].rank(pct=True)
    df['SHARPE_20_RANK_GROUP']       = df.groupby(['TS', 'GROUP'])['SHARPE_20'].rank(pct=True)

    df['DISP_GROUP_5']   = df.groupby(['TS', 'GROUP'])['AVERAGE_PERF_5'].transform('std')
    df['DISP_GROUP_20']  = df.groupby(['TS', 'GROUP'])['AVERAGE_PERF_20'].transform('std')
    df['MARKET_DISP']    = df.groupby('TS')['RET_1'].transform('std')
    df['MARKET_BREADTH'] = df.groupby('TS')['RET_1'].transform(lambda x: (x > 0).mean())

    med_cols  = [f'MED_{c}' for c in RET_COLS]
    group_med = df.groupby(['TS', 'GROUP'])[RET_COLS].median().rename(columns={c: f'MED_{c}' for c in RET_COLS})
    df = df.join(group_med, on=['TS', 'GROUP'])
    df['CROWDING'] = pearson_vec(df[RET_COLS].values.astype(float), df[med_cols].values.astype(float))
    df.drop(columns=med_cols, inplace=True)

    group_med_ll = df.groupby(['TS', 'GROUP'])[RET_COLS[:19]].median().rename(columns={RET_COLS[i]: f'GRP_LL_{i}' for i in range(19)})
    df = df.join(group_med_ll, on=['TS', 'GROUP'])
    df['LEAD_LAG'] = pearson_vec(ret[:, 1:], df[[f'GRP_LL_{i}' for i in range(19)]].values.astype(float))
    df.drop(columns=[f'GRP_LL_{i}' for i in range(19)], inplace=True)

    df['MARKET_BREADTH'] = df.groupby('TS')['RET_1'].transform(lambda x: (x > 0).mean())

    to_mean = df.groupby('TS')['MEDIAN_DAILY_TURNOVER'].transform('mean')
    to_std  = df.groupby('TS')['MEDIAN_DAILY_TURNOVER'].transform('std').fillna(1)
    df['TURNOVER_ZSCORE'] = (df['MEDIAN_DAILY_TURNOVER'] - to_mean) / (to_std + eps)

    if fit_stats is None:
        sv1_avail = df.groupby('ALLOCATION')['SIGNED_VOLUME_1'].apply(lambda x: x.notna().mean())
        special_allocs = sv1_avail[sv1_avail > 0.99].index.tolist()
        fit_stats = {'special_allocs': special_allocs}
    else:
        special_allocs = fit_stats['special_allocs']

    is_special = df['ALLOCATION'].isin(special_allocs).astype(float)
    df['SPECIAL_x_SHARPE20'] = is_special * df['SHARPE_20']
    df['SPECIAL_x_MOM20']    = is_special * df['AVERAGE_PERF_20']

    df.fillna(0.0, inplace=True)
    return df, fit_stats


def get_features_v2b():
    return (
        [f'AVERAGE_PERF_{i}'             for i in [3, 5, 10, 15, 20]] +
        [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]] +
        ['STD_PERF_20', 'AVG_ALLOC_VOL_20',
         'ACCEL_3_10', 'ACCEL_5_20', 'REVERSAL_1', 'HIT_RATE_20', 'CONVEXITY',
         'SHARPE_20', 'SHARPE_5', 'VOL_RATIO', 'SKEW_20', 'KURT_20',
         'VW_MOM', 'ACCUM_DIST',
         'AVERAGE_PERF_5_ZSCORE_GROUP',  'AVERAGE_PERF_20_ZSCORE_GROUP',
         'STD_PERF_20_ZSCORE_GROUP',     'SHARPE_20_ZSCORE_GROUP',
         'AVERAGE_PERF_20_RANK_GROUP',   'SHARPE_20_RANK_GROUP',
         'AVERAGE_PERF_5_RANK_MARKET',   'AVERAGE_PERF_20_RANK_MARKET',
         'AVERAGE_PERF_5_DEV_MARKET',    'AVERAGE_PERF_20_DEV_MARKET',
         'DISP_GROUP_5', 'DISP_GROUP_20', 'MARKET_DISP', 'MARKET_BREADTH',
         'CROWDING', 'LEAD_LAG',
         'TURNOVER_ZSCORE',
         'SPECIAL_x_SHARPE20', 'SPECIAL_x_MOM20',
         'ALLOC_VOL', 'ALLOC_SHARPE_HIST', 'ALLOC_TURNOVER_MEAN',
         'ALLOC_BETA_HIST', 'ALLOC_ALPHA_HIST',
         'ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY',
         'REVERSAL_x_SENSITIVITY', 'MOM_x_SHARPE_HIST']
    )


# ──────────────────────────────────────────────────────────────────────────────
# GBDT V2B (FINAL)
# ──────────────────────────────────────────────────────────────────────────────

def run_gbdt_v2b(X_train, X_test, y_train, sample_submission):
    print('Building v2b features...')
    t0 = time.time()
    X_train_v2b, fit_stats_v2b = build_features_v2b(X_train)
    X_test_v2b,  _             = build_features_v2b(X_test, fit_stats=fit_stats_v2b)
    print(f'Features built in {time.time()-t0:.1f}s')

    print('Structural profiles v2b...')
    struct_v2b = build_structural_profiles_v2b(X_train)
    X_train_v2b = X_train_v2b.join(struct_v2b, on='ALLOCATION')
    X_test_v2b  = X_test_v2b.join(struct_v2b, on='ALLOCATION')
    for col in struct_v2b.columns:
        X_train_v2b[col] = X_train_v2b[col].fillna(0.0)
        X_test_v2b[col]  = X_test_v2b[col].fillna(0.0)

    print('OOF target encoding v2b...')
    oof_tgt_v2b, test_prof_v2b = build_target_profiles_oof_v2b(X_train_v2b, y_train['target'])
    X_train_v2b = pd.concat([X_train_v2b, oof_tgt_v2b], axis=1)
    X_test_v2b  = pd.concat([
        X_test_v2b,
        X_test_v2b[['ALLOCATION']].assign(
            ALLOC_HIT_RATE=X_test_v2b['ALLOCATION'].map(test_prof_v2b['ALLOC_HIT_RATE']).fillna(test_prof_v2b['global_hit']),
            ALLOC_REVERSAL_SENSITIVITY=X_test_v2b['ALLOCATION'].map(test_prof_v2b['ALLOC_REVERSAL_SENSITIVITY']).fillna(0.0)
        )[['ALLOC_HIT_RATE', 'ALLOC_REVERSAL_SENSITIVITY']]
    ], axis=1)

    for df_ in [X_train_v2b, X_test_v2b]:
        df_['REVERSAL_x_SENSITIVITY'] = df_['REVERSAL_1'] * df_['ALLOC_REVERSAL_SENSITIVITY']
        df_['MOM_x_SHARPE_HIST']      = df_['AVERAGE_PERF_5'] * df_['ALLOC_SHARPE_HIST']

    X_train_v2b.fillna(0.0, inplace=True)
    X_test_v2b.fillna(0.0, inplace=True)
    print(f'v2b shapes - Train: {X_train_v2b.shape} | Test: {X_test_v2b.shape}')

    features_v2b = get_features_v2b()
    missing = [f for f in features_v2b if f not in X_train_v2b.columns]
    print(f'Features v2b: {len(features_v2b)}')
    if missing:
        print(f'Missing: {missing}')

    t0 = time.time()
    results_v2b = train_ensemble_cv(X_train_v2b, y_train['target'], X_test_v2b, features_v2b)
    print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')

    with open('models/ensemble_results_v2b.pkl', 'wb') as f:
        pickle.dump(results_v2b, f)

    idx = sample_submission.index
    print('=== Final Submissions ===')
    for name, preds in [
        ('v2b_blend_equal', results_v2b['test_blend_equal']),
        ('v2b_blend_optim', results_v2b['test_blend_optim']),
        ('v2b_lgbm_solo',   results_v2b['test_lgbm']),
        ('v2b_xgb_solo',    results_v2b['test_xgb']),
        ('v2b_cat_solo',    results_v2b['test_cat']),
    ]:
        s    = pd.DataFrame({'target': (preds > 0.5).astype(int)}, index=idx)
        taux = s['target'].mean()
        flag = ' WARNING: biased' if taux > 0.55 or taux < 0.45 else ''
        s.to_csv(f'submissions/submission_{name}.csv')
        print(f'  {name:<25s} rate_1={taux:.4f}{flag}')

    sl, sx, sc = results_v2b['scores']
    oof_blend  = results_v2b['oof_blend']
    yb         = results_v2b['yb']
    print(f'\nCV results v2b:')
    print(f'  LGBM  : {np.mean(sl)*100:.3f}% +/- {np.std(sl)*100:.3f}%')
    print(f'  XGB   : {np.mean(sx)*100:.3f}% +/- {np.std(sx)*100:.3f}%')
    print(f'  CAT   : {np.mean(sc)*100:.3f}% +/- {np.std(sc)*100:.3f}%')
    print(f'  BLEND : {((oof_blend > 0.5) == yb).mean()*100:.3f}% (OOF)')


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='QRT Asset Allocation Challenge')
    parser.add_argument('--eda', action='store_true', help='Run EDA and save plots')
    parser.add_argument('--gbdt-v1', action='store_true', help='Train GBDT baseline v1 ensemble')
    parser.add_argument('--transformer', action='store_true', help='Train DualBranch Transformer')
    parser.add_argument('--autoencoder', action='store_true', help='Train Supervised Autoencoder')
    parser.add_argument('--stability', action='store_true', help='Run stability / distribution shift analysis')
    parser.add_argument('--gbdt-v2b', action='store_true', help='Train final GBDT v2b ensemble')
    parser.add_argument('--all', action='store_true', help='Run everything')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print('\nExamples:')
        print('  python code_qrt.py --eda')
        print('  python code_qrt.py --gbdt-v1')
        print('  python code_qrt.py --gbdt-v2b')
        print('  python code_qrt.py --transformer --autoencoder')
        print('  python code_qrt.py --all')
        return

    _import_deps()

    for d in ['plots/eda', 'plots/modeling', 'plots/stability', 'models', 'submissions']:
        os.makedirs(d, exist_ok=True)

    X_train, X_test, y_train, sample_submission, y_bin = load_data()

    if args.eda or args.all:
        run_eda(X_train, X_test, y_train, y_bin)

    X_train_fe, X_test_fe = None, None

    if args.gbdt_v1 or args.all:
 
        X_train_fe, X_test_fe, _ = run_gbdt_v1(X_train, X_test, y_train, sample_submission)

    if args.transformer or args.all:
        if X_train_fe is None:
            X_train_fe, X_test_fe = prepare_v1_features(X_train, X_test, y_train)
        run_transformer(X_train, X_test, y_train, y_bin, X_train_fe, X_test_fe, sample_submission)

    if args.autoencoder or args.all:

        if X_train_fe is None:
            X_train_fe, X_test_fe = prepare_v1_features(X_train, X_test, y_train)
        run_autoencoder(X_train, X_test, y_train, y_bin, X_train_fe, X_test_fe, sample_submission)

    if args.stability or args.all:

        if X_train_fe is None:
            X_train_fe, X_test_fe = prepare_v1_features(X_train, X_test, y_train)
        run_stability_analysis(X_train, X_test, y_train, X_train_fe, X_test_fe)

    if args.gbdt_v2b or args.all:
        run_gbdt_v2b(X_train, X_test, y_train, sample_submission)



if __name__ == '__main__':
    main()
