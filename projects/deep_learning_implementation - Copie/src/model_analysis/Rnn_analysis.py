import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── helper ─────────────────────────────────────────────────────────────────────

def _abs_returns(df: pd.DataFrame) -> pd.Series:
    s = df.iloc[:, 0].copy()
    s.index = pd.to_datetime(s.index)
    return s.abs()


# ── 1. full-series overview ────────────────────────────────────────────────────

def plot_abs_returns(df: pd.DataFrame, ax=None) -> None:
    """Plot the full absolute-return series."""
    abs_ret = _abs_returns(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(abs_ret.index, abs_ret.values, linewidth=0.6, color="steelblue")
    ax.set_ylabel("|return|")
    ax.set_title("SP500 — Daily Absolute Log Returns", fontsize=13, fontweight="bold")

    if standalone:
        plt.tight_layout()
        plt.show()


# ── 2. spike highlight on a zoom window ───────────────────────────────────────

def plot_abs_return_spikes(
    df: pd.DataFrame,
    start: str = None,
    end: str = None,
    spike_quantile: float = 0.95,
    ax=None,
) -> None:
    """
    Plot absolute returns for a sub-period and flag days above `spike_quantile`.

    Parameters
    ----------
    start, end     : str  e.g. "2008-01-01"  (None = full series)
    spike_quantile : float — days at or above this quantile are marked as spikes
    """
    abs_ret = _abs_returns(df)
    if start:
        abs_ret = abs_ret[abs_ret.index >= start]
    if end:
        abs_ret = abs_ret[abs_ret.index <= end]

    threshold = abs_ret.quantile(spike_quantile)
    spikes = abs_ret[abs_ret >= threshold]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    title = "Absolute Return Spikes"
    if start or end:
        title += f"  [{start or '…'} → {end or '…'}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.plot(abs_ret.index, abs_ret.values, linewidth=0.7, color="steelblue", label="|return|")
    ax.vlines(spikes.index, 0, spikes.values, color="crimson", linewidth=0.8, alpha=0.7,
              label=f"spikes (≥ {spike_quantile:.0%} quantile)")
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("|return|")
    ax.legend(fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── 3. regime isolation ───────────────────────────────────────────────────────

# Reference regimes used for visualisation (pre-2017, inside training data)
REGIMES: Dict[str, Tuple[str, str, str]] = {
    "calm":       ("2004-01-01", "2006-12-31", "mediumseagreen"),
    "clustering": ("2011-07-01", "2012-06-30", "goldenrod"),
    "spikes":     ("2008-09-01", "2009-03-31", "crimson"),
}

# Post-2017 regimes — held-out test set, never seen during training
TEST_REGIMES: Dict[str, Tuple[str, str, str]] = {
    "calm":       ("2017-01-01", "2017-12-31", "mediumseagreen"),
    "clustering": ("2022-01-01", "2022-12-31", "goldenrod"),
    "spikes":     ("2020-03-01", "2020-04-30", "crimson"),
}


def plot_regimes(df: pd.DataFrame, ax=None) -> None:
    """Full-series plot with the three reference regimes shaded."""
    abs_ret = _abs_returns(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(abs_ret.index, abs_ret.values, linewidth=0.6, color="steelblue", zorder=2)

    patches = []
    for label, (start, end, color) in REGIMES.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.25, color=color, zorder=1)
        patches.append(mpatches.Patch(color=color, alpha=0.5, label=label))

    ax.legend(handles=patches, fontsize=9)
    ax.set_title("SP500 Absolute Returns — Reference Regimes", fontsize=13, fontweight="bold")
    ax.set_ylabel("|return|")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_data_overview(
    df: pd.DataFrame,
    spike_start: str = "2007-01-01",
    spike_end: str = "2010-12-31",
    spike_quantile: float = 0.95,
) -> None:
    """
    Three-row combined overview:
      row 1 — full absolute-return series
      row 2 — spike zoom window
      row 3 — full series with regime shading
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    plot_abs_returns(df, ax=axes[0])
    plot_abs_return_spikes(df, start=spike_start, end=spike_end,
                           spike_quantile=spike_quantile, ax=axes[1])
    plot_regimes(df, ax=axes[2])

    fig.tight_layout()
    plt.show()


def extract_regime_samples(
    df: pd.DataFrame,
    sequence_length: int = 20,
    regimes: Dict[str, Tuple[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Slice the dataframe into sub-DataFrames for each regime.

    Returns { regime_name: sub_df } ready to pass into
    build_dataset_abs_returns_sequential for per-regime RNN evaluation.
    """
    if regimes is None:
        regimes = REGIMES

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    samples = {}
    for label, bounds in regimes.items():
        start, end = bounds[0], bounds[1]
        sub = df[(df.index >= start) & (df.index <= end)]
        if len(sub) < sequence_length + 1:
            print(f"[warning] regime '{label}' has only {len(sub)} rows — too short for seq_len={sequence_length}")
        samples[label] = sub

    return samples


def plot_abs_return_distribution(df: pd.DataFrame, bins: int = 100) -> None:
    """
    Distribution of absolute returns: histogram + log-scale y-axis to
    show the fat tail clearly.
    """
    abs_ret = _abs_returns(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Distribution of Absolute Returns", fontsize=13, fontweight="bold")

    # left — linear scale
    axes[0].hist(abs_ret.values, bins=bins, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("|return|")
    axes[0].set_ylabel("count")
    axes[0].set_title("Linear scale")

    # right — log y-axis to see the fat tail
    axes[1].hist(abs_ret.values, bins=bins, color="steelblue", edgecolor="none")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("|return|")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("Log scale — fat tail")

    for q, ls in [(0.95, "--"), (0.99, ":")]:
        val = abs_ret.quantile(q)
        for ax in axes:
            ax.axvline(val, color="crimson", linestyle=ls, linewidth=1,
                       label=f"q{int(q*100)}={val:.4f}")

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def build_test_df(
    df: pd.DataFrame,
    sequence_length: int = 20,
) -> pd.DataFrame:
    """
    Concatenate the three post-2017 held-out regime periods into a single
    DataFrame to use as a regime-aware test set.

    Periods (TEST_REGIMES):
      calm       — 2017  (low, stable moves)
      clustering — 2022  (persistently elevated, rate-hike cycle)
      spikes     — Mar-Apr 2020  (COVID crash)

    Returns a DataFrame with the original log-return column, sorted by date.
    Each row also has a 'regime' column for later error analysis per regime.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    parts = []
    for label, (start, end, _) in TEST_REGIMES.items():
        sub = df[(df.index >= start) & (df.index <= end)].copy()
        if len(sub) < sequence_length + 1:
            print(f"[warning] test regime '{label}' has only {len(sub)} rows — too short for seq_len={sequence_length}")
        sub["regime"] = label
        parts.append(sub)

    test_df = pd.concat(parts).sort_index()
    print(f"test_df: {len(test_df)} rows across {len(parts)} regimes")
    for label, (start, end, _) in TEST_REGIMES.items():
        n = (test_df["regime"] == label).sum()
        print(f"  {label:12s} {start} → {end}  ({n} rows)")

    return test_df


def plot_test_regimes(df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Plot the full absolute-return series with the three test regimes
    highlighted in their respective colors.
    """
    abs_ret = _abs_returns(df)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(abs_ret.index, abs_ret.values, linewidth=0.6, color="lightgrey", zorder=1)

    test_df = test_df.copy()
    test_df.index = pd.to_datetime(test_df.index)

    patches = []
    for label, (start, end, color) in TEST_REGIMES.items():
        regime_data = _abs_returns(test_df[test_df["regime"] == label].drop(columns="regime"))
        ax.plot(regime_data.index, regime_data.values, linewidth=0.9, color=color, zorder=2)
        patches.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=patches, fontsize=9)
    ax.set_title("Test Regimes — Held-Out Periods", fontsize=13, fontweight="bold")
    ax.set_ylabel("|return|")
    fig.tight_layout()
    plt.show()


def prepare_datasets(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_length: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Build all datasets for training and regime-aware evaluation.

    Steps
    -----
    1. Remove test_df rows from df → train_df
    2. Build sequences from train_df → train/val split
    3. Fit ONE scaler on X_train, ONE on y_train
    4. Build sequences for each regime in test_df, scale with the same scalers

    Returns
    -------
    X_train, y_train, X_val, y_val  — for model training
    regime_sets                      — dict { "calm": (X, y), "clustering": (X, y), "spikes": (X, y) }
    scaler_X, scaler_y               — fitted scalers (needed for inverse transform later)
    """
    from data_utils import build_dataset_abs_returns_sequential

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    test_df = test_df.copy()
    test_df.index = pd.to_datetime(test_df.index)

    # 1 — remove test rows from original df
    train_df = df[~df.index.isin(test_df.index)]
    if verbose:
        print(f"train_df: {len(train_df)} rows  (removed {len(df) - len(train_df)} test rows)")

    # 2 — build sequences from train_df
    X, y = build_dataset_abs_returns_sequential(train_df, sequence_length)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

    # 3 — fit scalers on training data only
    n, seq_len, n_features = X_train.shape
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train.reshape(-1, n_features)).reshape(n, seq_len, n_features)
    y_train = scaler_y.fit_transform(y_train)

    n_val = X_val.shape[0]
    X_val = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(n_val, seq_len, n_features)
    y_val = scaler_y.transform(y_val)

    if verbose:
        print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

    # 4 — build and scale each regime test set
    regime_sets = {}
    for label in TEST_REGIMES:
        sub = test_df[test_df["regime"] == label].drop(columns="regime")
        X_r, y_r = build_dataset_abs_returns_sequential(sub, sequence_length)
        n_r = X_r.shape[0]
        X_r = scaler_X.transform(X_r.reshape(-1, n_features)).reshape(n_r, seq_len, n_features)
        y_r = scaler_y.transform(y_r)
        regime_sets[label] = (X_r, y_r)
        if verbose:
            print(f"  {label:12s}  X: {X_r.shape}")

    return X_train, y_train, X_val, y_val, regime_sets, scaler_X, scaler_y


def summarise_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Stats table per regime: n_days, mean |return|, std |return|, max |return|."""
    abs_ret = _abs_returns(df)
    abs_ret_df = abs_ret.to_frame(name="abs_ret")
    abs_ret_df.index = pd.to_datetime(abs_ret_df.index)

    rows = []
    for label, (start, end, _) in REGIMES.items():
        sub = abs_ret_df[(abs_ret_df.index >= start) & (abs_ret_df.index <= end)]["abs_ret"]
        rows.append({
            "regime":        label,
            "period":        f"{start} → {end}",
            "n_days":        len(sub),
            "mean |return|": round(sub.mean(), 5),
            "std  |return|": round(sub.std(), 5),
            "max  |return|": round(sub.max(), 5),
        })

    summary = pd.DataFrame(rows).set_index("regime")
    print(summary.to_string())
    return summary


# ── step 1 — understand what the model learned ────────────────────────────────

def plot_weight_distributions(model, title: str = "") -> None:
    """
    Step 1.1 — Weight distributions.

    Plots histograms of three weight groups:
      - W_ih  : input → hidden  (how the model reads new data)
      - W_hh  : hidden → hidden (the "memory" matrix — key for vanishing gradient)
      - head  : hidden → output (how prediction is formed)

    What to look for
    ----------------
    Healthy  : roughly bell-shaped, centred near 0, spread ~0.1-0.3
    Dead     : almost all weights near 0 → that part of the network does nothing
    Exploding: very large values → training instability
    """
    W_ih = model.rnn.weight_ih_l0.detach().cpu().numpy().flatten()
    W_hh = model.rnn.weight_hh_l0.detach().cpu().numpy().flatten()
    head_w = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.head.parameters()])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    suptitle = "Step 1.1 — Weight Distributions"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ax, weights, label, color in zip(
        axes,
        [W_ih, W_hh, head_w],
        ["W_ih  (input → hidden)", "W_hh  (hidden → hidden)", "Head  (hidden → output)"],
        ["steelblue", "darkorange", "mediumseagreen"],
    ):
        ax.hist(weights, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("weight value")
        ax.set_ylabel("count")
        ax.set_title(f"{label}\nstd={weights.std():.3f}", fontsize=10)

    fig.tight_layout()
    plt.show()


def plot_hidden_state_activity(model, X: np.ndarray, title: str = "") -> None:
    """
    Step 1.2 — Hidden state activity over time.

    For each position in the sequence (0 = oldest input, T = most recent),
    computes the average absolute activation across all hidden units and all
    samples. Tells you: "how active is the network's memory at each step?"

    What to look for
    ----------------
    Good: activity grows as the model reads through the sequence — it is
          accumulating information and the memory is being used.
    Bad : flat line — the hidden state barely changes as new inputs arrive,
          meaning the model ignores the sequential structure.
    """
    import torch
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        rnn_out, _ = model.rnn(X_t)                              # (batch, seq_len, hidden)
        mean_activity = rnn_out.abs().mean(dim=(0, 2)).cpu().numpy()  # (seq_len,)

    seq_len = len(mean_activity)
    fig, ax = plt.subplots(figsize=(10, 4))
    suptitle = "Step 1.2 — Hidden State Activity"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    ax.plot(range(seq_len), mean_activity, color="darkorange", linewidth=2)
    ax.set_xlabel("position in sequence  (0 = oldest input,  T = most recent)")
    ax.set_ylabel("mean |h_t|")
    ax.set_xticks([0, seq_len - 1])
    ax.set_xticklabels(["oldest\n(lag N)", "most recent\n(lag 1)"])

    fig.tight_layout()
    plt.show()


def plot_input_sensitivity(model, X: np.ndarray, title: str = "") -> None:
    """
    Step 1.3 — Input sensitivity by lag.

    Computes the gradient of the model output w.r.t. each input position,
    then averages over samples and features. The result tells you how much
    each past day actually influences today's prediction.

    Sequence layout:
      X[:, 0, :] = lag N (oldest)   X[:, -1, :] = lag 1 (most recent)

    What to look for
    ----------------
    Good             : bar at lag 1 is tallest, decreases smoothly toward lag N
    Vanishing gradient: bars collapse to near zero after ~5-10 lags — the model
                        effectively ignores everything beyond that, regardless of
                        how long you make the sequence
    FNN-like         : all bars roughly equal — model treats all lags identically,
                        not exploiting temporal order
    """
    import torch
    model.eval()
    X_t = torch.tensor(X[:64], dtype=torch.float32, requires_grad=True)
    model(X_t).sum().backward()

    # grads[i] = sensitivity at sequence position i
    # position 0 = oldest = lag N,  position seq_len-1 = most recent = lag 1
    grads = X_t.grad.abs().mean(dim=(0, 2)).detach().cpu().numpy()
    seq_len = len(grads)

    # x-axis: lag 1 on the left (most recent), lag N on the right
    lags = np.arange(1, seq_len + 1)       # [1, 2, ..., seq_len]
    sensitivity = grads[::-1]              # flip: index 0 → lag 1 (most recent)

    fig, ax = plt.subplots(figsize=(12, 4))
    suptitle = "Step 1.3 — Input Sensitivity by Lag"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    ax.bar(lags, sensitivity, color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_xlabel("lag  (1 = yesterday,  N = N days ago)")
    ax.set_ylabel("mean |gradient|")
    ax.set_title("Higher bar = model pays more attention to that day", fontsize=10)

    fig.tight_layout()
    plt.show()


# ── model evaluation ──────────────────────────────────────────────────────────

def plot_pred_vs_true_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 80,
    title: str = "Predicted vs True Distribution",
) -> None:
    """
    Overlay histogram of true vs predicted absolute returns.
    Both arrays must already be in original (inverse-scaled) space.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y_true, bins=bins, alpha=0.6, color="steelblue", label="true |return|", density=True)
    ax.hist(y_pred, bins=bins, alpha=0.6, color="darkorange", label="predicted |return|", density=True)
    ax.set_xlabel("|return|")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


def evaluate_regimes(
    trainer,
    regime_sets: dict,
    scaler_y,
) -> pd.DataFrame:
    """
    For each regime: compute MAE + plot predicted vs true series side by side.

    Parameters
    ----------
    trainer     : fitted Trainer (from Rnn.py)
    regime_sets : dict { label: (X, y) } as returned by prepare_datasets — scaled
    scaler_y    : fitted scaler for inverse transform

    Returns
    -------
    DataFrame with MAE per regime
    """
    from sklearn.metrics import mean_absolute_error

    colors = {label: c for label, (_, _, c) in TEST_REGIMES.items()}
    results = []

    fig, axes = plt.subplots(len(regime_sets), 2, figsize=(14, 4 * len(regime_sets)))
    if len(regime_sets) == 1:
        axes = [axes]

    for i, (label, (X, y)) in enumerate(regime_sets.items()):
        y_pred = scaler_y.inverse_transform(trainer.predict(X).reshape(-1, 1))
        y_true = scaler_y.inverse_transform(y.reshape(-1, 1))

        mae = mean_absolute_error(y_true, y_pred)
        results.append({"regime": label, "n_days": len(y_true), "MAE": round(mae, 6)})

        color = colors.get(label, "steelblue")

        # left — time series: true vs predicted
        ax = axes[i][0]
        ax.plot(y_true, linewidth=0.8, color="lightgrey", label="true")
        ax.plot(y_pred, linewidth=1.0, color=color, alpha=0.85, label="predicted")
        ax.set_title(f"{label}  —  series  (MAE={mae:.5f})", fontsize=11, fontweight="bold")
        ax.set_ylabel("|return|")
        ax.legend(fontsize=8)

        # right — distribution: true vs predicted
        ax = axes[i][1]
        ax.hist(y_true, bins=40, alpha=0.6, color="lightgrey", label="true", density=True)
        ax.hist(y_pred, bins=40, alpha=0.7, color=color, label="predicted", density=True)
        ax.set_title(f"{label}  —  distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("|return|")
        ax.legend(fontsize=8)

    fig.suptitle("Regime Evaluation — Predicted vs True", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()

    summary = pd.DataFrame(results).set_index("regime")
    print(summary.to_string())
    return summary


# ── grid search ───────────────────────────────────────────────────────────────
from tqdm import tqdm

def grid_search_rnn(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    param_grid: dict = None,
    n_epochs: int = 80,
    early_stopping: int = 15,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Grid search over RNN hyperparameters (~108 combinations by default).

    Head is always Dense(hidden_size → 1, activation='none') — linear output
    for regression. Only the recurrent part and training dynamics are searched.

    Default param_grid  (3 × 3 × 2 × 3 × 2 = 108 combinations)
    ------------------
    sequence_length : [10, 20, 30]   — memory horizon
    hidden_size     : [32, 64, 128]  — capacity of hidden state
    num_layers      : [1, 2]         — recurrent depth
    lr              : [0.001, 0.0005, 0.0001]  — learning rate
    grad_clip       : [0.5, 1.0]     — gradient clipping threshold
    """
    import itertools
    from dl_utils.Rnn import Dense, RNN, Trainer
    from sklearn.metrics import mean_absolute_error

    if param_grid is None:
        param_grid = {
            'sequence_length': [10, 20, 30],
            'hidden_size':     [32, 64],
            'num_layers':      [1, 2],
            'lr':              [0.001, 0.0005],
            'grad_clip':       [0.5, 1.0],
        }

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total  = len(combos)
    print(f"grid search: {total} combinations  |  {n_epochs} max epochs  |  early stopping={early_stopping}")

    # cache datasets per sequence_length — avoid rebuilding for the same seq_len
    dataset_cache = {}

    results = []
    for values in tqdm(combos, total=total):
        cfg = dict(zip(keys, values))

        seq_len    = cfg['sequence_length']
        hidden_size = cfg['hidden_size']
        num_layers  = cfg['num_layers']
        lr          = cfg['lr']
        grad_clip   = cfg['grad_clip']

        if seq_len not in dataset_cache:
            dataset_cache[seq_len] = prepare_datasets(
                df, test_df, sequence_length=seq_len, test_size=0.2, random_state=42,
                verbose=False,
            )
        X_train, y_train, X_val, y_val, regime_sets, scaler_X, scaler_y = dataset_cache[seq_len]

        head = Dense(hidden_size, y_train.shape[-1],
                     activation='none', dropout=0.0,
                     weight_init='xavier_uniform', bias_init='zeros')

        model = RNN(
            input_size=X_train.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            rnn_dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=False,
            head=[head],
        )

        trainer = Trainer(
            model,
            loss_fn='mse',
            optimizer='adam',
            lr=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            shuffle=True,
            clip_grad_norm=grad_clip,
            early_stopping=early_stopping,
            verbose=0,
        )

        trainer.fit(X_train, y_train, X_val, y_val)

        y_pred_val = scaler_y.inverse_transform(trainer.predict(X_val).reshape(-1, 1))
        y_val_inv  = scaler_y.inverse_transform(y_val.reshape(-1, 1))
        val_mae    = mean_absolute_error(y_val_inv, y_pred_val)

        row = {**cfg, 'val_MAE': round(val_mae, 6)}
        for label, (X_r, y_r) in regime_sets.items():
            y_pred_r = scaler_y.inverse_transform(trainer.predict(X_r).reshape(-1, 1))
            y_r_inv  = scaler_y.inverse_transform(y_r.reshape(-1, 1))
            row[f'MAE_{label}'] = round(mean_absolute_error(y_r_inv, y_pred_r), 6)

        results.append(row)

    results_df = pd.DataFrame(results).sort_values('val_MAE').reset_index(drop=True)
    results_df.insert(0, 'rank', results_df.index + 1)

    print("\n── Top 10 ────────────────────────────────────────────────────")
    print(results_df.head(10).to_string(index=False))
    return results_df



# ── step 2 — error analysis ───────────────────────────────────────────────────

def _get_regime_predictions(trainer, regime_sets, scaler_y, test_df, sequence_length):
    """
    Helper — returns { label: (dates, y_true, y_pred) } all in original space.
    """
    results = {}
    for label, (X_scaled, y_scaled) in regime_sets.items():
        y_pred = scaler_y.inverse_transform(
            trainer.predict(X_scaled).reshape(-1, 1)
        ).flatten()
        y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        sub = test_df[test_df["regime"] == label]
        dates = sub.index[sequence_length:]
        n = min(len(dates), len(y_true))
        results[label] = (dates[:n], y_true[:n], y_pred[:n])
    return results


def plot_residuals_over_time(trainer, regime_sets, scaler_y, test_df, sequence_length, title=""):
    """
    Step 2.1 — One panel per regime showing true |return| (grey) and
    absolute error filled in regime colour. Errors that spike with the
    grey line = mean-reversion bias.
    """
    colors = {label: c for label, (_, _, c) in TEST_REGIMES.items()}
    regime_preds = _get_regime_predictions(trainer, regime_sets, scaler_y, test_df, sequence_length)

    fig, axes = plt.subplots(len(regime_preds), 1, figsize=(14, 4 * len(regime_preds)))
    if len(regime_preds) == 1:
        axes = [axes]
    suptitle = "Step 2.1 — Residuals Over Time"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ax, (label, (dates, y_true, y_pred)) in zip(axes, regime_preds.items()):
        color = colors.get(label, "steelblue")
        errors = np.abs(y_true - y_pred)
        ax.plot(dates, y_true, linewidth=0.7, color="lightgrey", label="true |return|")
        ax.fill_between(dates, errors, alpha=0.6, color=color, label="|error|")
        ax.set_title(
            f"{label}  —  mean |error|={errors.mean():.5f}   max |error|={errors.max():.5f}",
            fontsize=10, fontweight="bold",
        )
        ax.set_ylabel("|return|  /  |error|")
        ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_error_vs_magnitude(trainer, regime_sets, scaler_y, test_df, sequence_length, title=""):
    """
    Step 2.2 — Left: scatter true vs |error| coloured by regime + overall
    linear fit. Right: signed error histograms per regime overlaid.
    Positive slope + histograms shifted left = mean-reversion bias.
    """
    colors = {label: c for label, (_, _, c) in TEST_REGIMES.items()}
    regime_preds = _get_regime_predictions(trainer, regime_sets, scaler_y, test_df, sequence_length)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    suptitle = "Step 2.2 — Error vs Magnitude"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    all_true, all_abs_err = [], []

    for label, (dates, y_true, y_pred) in regime_preds.items():
        color = colors.get(label, "steelblue")
        abs_error = np.abs(y_true - y_pred)
        signed_error = y_pred - y_true
        all_true.append(y_true)
        all_abs_err.append(abs_error)

        axes[0].scatter(y_true, abs_error, s=10, alpha=0.5, color=color,
                        edgecolors="none", label=label)
        axes[1].hist(signed_error, bins=40, alpha=0.5, color=color, density=True,
                     label=f"{label}  (mean={signed_error.mean():.5f})")

    all_true_arr = np.concatenate(all_true)
    all_abs_err_arr = np.concatenate(all_abs_err)
    coeffs = np.polyfit(all_true_arr, all_abs_err_arr, 1)
    x_line = np.linspace(all_true_arr.min(), all_true_arr.max(), 200)
    axes[0].plot(x_line, np.polyval(coeffs, x_line), color="blue", linewidth=1.5,
                 linestyle="--", label=f"linear fit  (slope={coeffs[0]:.2f})")
    axes[0].set_xlabel("true |return|")
    axes[0].set_ylabel("|error|")
    axes[0].set_title("Absolute error vs true magnitude", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("signed error  (predicted − true)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Signed error distribution per regime", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_worst_predictions(trainer, regime_sets, scaler_y, scaler_X, test_df, sequence_length, top_pct=0.05):
    """
    Step 2.3 — For each regime, finds the top_pct worst predictions and
    plots the input look-back window that preceded each failure.
    Date shown in each subplot title.
    """
    colors = {label: c for label, (_, _, c) in TEST_REGIMES.items()}
    regime_preds = _get_regime_predictions(trainer, regime_sets, scaler_y, test_df, sequence_length)

    for label, (X_scaled, _) in regime_sets.items():
        dates, y_true, y_pred = regime_preds[label]
        abs_error = np.abs(y_true - y_pred)
        threshold = np.quantile(abs_error, 1 - top_pct)
        worst_idx = np.where(abs_error >= threshold)[0]

        n_features = X_scaled.shape[2]
        seq_len = X_scaled.shape[1]
        X_inv = scaler_X.inverse_transform(
            X_scaled.reshape(-1, n_features)
        ).reshape(-1, seq_len, n_features)

        color = colors.get(label, "steelblue")
        n_worst = len(worst_idx)
        ncols = min(5, n_worst)
        nrows = int(np.ceil(n_worst / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
        fig.suptitle(
            f"Step 2.3 — Worst {top_pct:.0%} predictions  |  regime: {label}  "
            f"({n_worst} samples,  error ≥ {threshold:.5f})",
            fontsize=11, fontweight="bold",
        )

        for plot_i, idx in enumerate(worst_idx):
            ax = axes[plot_i // ncols][plot_i % ncols]
            seq = X_inv[idx, :, 0]
            ax.plot(range(seq_len), seq, linewidth=1.0, color=color)
            ax.axhline(y_true[idx], color="blue", linewidth=0.8, linestyle="--",
                       label=f"true={y_true[idx]:.4f}")
            ax.axhline(y_pred[idx], color="purple", linewidth=0.8, linestyle="-",
                       label=f"pred={y_pred[idx]:.4f}")
            date_str = str(dates[idx])[:10] if idx < len(dates) else ""
            ax.set_title(f"{date_str}\nerr={abs_error[idx]:.4f}", fontsize=7)
            ax.set_xticks([0, seq_len - 1])
            ax.set_xticklabels(["lag N", "lag 1"], fontsize=6)
            ax.tick_params(labelsize=6)
            if plot_i == 0:
                ax.legend(fontsize=6)

        for plot_i in range(n_worst, nrows * ncols):
            axes[plot_i // ncols][plot_i % ncols].axis("off")

        fig.tight_layout()
        plt.show()




# ── step 3 — vanishing gradient diagnosis ─────────────────────────────────────

def plot_gradient_norms(model, X: np.ndarray, title: str = "") -> None:
    """
    Step 3.1 — Gradient norm per time step (BPTT).

    Runs a forward pass, then back-propagates from the output and records
    ||d(loss)/d(h_t)|| for each step t = 0 … T.

    If norms collapse to near zero after ~10 steps the model has no usable
    memory beyond that point — extending SEQUENCE_LENGTH further is pointless.
    """
    import torch
    model.eval()

    batch_size = min(64, X.shape[0])
    X_t = torch.tensor(X[:batch_size], dtype=torch.float32)

    rnn_out, _ = model.rnn(X_t)          # (batch, seq_len, hidden)
    rnn_out.retain_grad()                 # keep grad on non-leaf tensor

    x = rnn_out[:, -1, :]
    for layer in model.head:
        x = layer(x)
    y_hat = x
    y_hat.sum().backward()

    # rnn_out.grad : (batch, seq_len, hidden)
    grad_norms = rnn_out.grad.norm(dim=-1).mean(dim=0).detach().cpu().numpy()  # (seq_len,)

    seq_len = len(grad_norms)
    # x-axis: step T (most recent) on left → step 0 (oldest) on right
    steps_back = np.arange(seq_len)    # 0 = most recent … seq_len-1 = oldest

    fig, ax = plt.subplots(figsize=(12, 4))
    suptitle = "Step 3.1 — Gradient Norm per Time Step (BPTT)"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    ax.plot(steps_back, grad_norms[::-1], color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("steps back from prediction  (0 = most recent input,  T-1 = oldest)")
    ax.set_ylabel("||d loss / d h_t||")
    ax.set_title(
        "Gradient vanishes → curve collapses toward 0 after a few steps\n"
        "Flat curve → gradient flows uniformly (good memory)",
        fontsize=9,
    )

    # annotate half-life: first step where norm < 50% of the max
    max_norm = grad_norms.max()
    if max_norm > 0:
        half_idx = np.argmax(grad_norms[::-1] < 0.5 * max_norm)
        if half_idx > 0:
            ax.axvline(half_idx, color="crimson", linestyle="--", linewidth=1,
                       label=f"50%-decay at step {half_idx}")
            ax.legend(fontsize=9)

    fig.tight_layout()
    plt.show()


def plot_effective_memory(model, X: np.ndarray, title: str = "") -> None:
    """
    Step 3.2 — Effective memory length via h_T / x_t correlation.

    For each lag t (0 = oldest … T = most recent), computes the mean
    absolute Pearson correlation between the final hidden state h_T and
    the input x_t across all samples.

    High correlation at lag 1 and near-zero at lag 20 = model only
    remembers the last few steps regardless of sequence length.
    """
    import torch
    model.eval()

    n = min(256, X.shape[0])
    X_t = torch.tensor(X[:n], dtype=torch.float32)

    with torch.no_grad():
        _, h_T = model.rnn(X_t)              # (num_layers, batch, hidden)
        h_T = h_T[-1].cpu().numpy()          # last layer: (batch, hidden)

    X_np = X[:n]                             # (batch, seq_len, features)
    seq_len = X_np.shape[1]

    # Precompute centred / normalised h_T
    h_c = h_T - h_T.mean(axis=0)            # (batch, hidden)
    h_std = h_T.std(axis=0) + 1e-8          # (hidden,)

    corr_per_lag = []
    for t in range(seq_len):
        x_t = X_np[:, t, :]                  # (batch, features)
        x_c = x_t - x_t.mean(axis=0)
        x_std = x_t.std(axis=0) + 1e-8

        cov = (h_c.T @ x_c) / n             # (hidden, features)
        corr_mat = cov / (h_std[:, None] * x_std[None, :])
        corr_per_lag.append(np.abs(corr_mat).mean())

    # x-axis: lag 1 on the left (most recent), lag N on the right
    lags = np.arange(1, seq_len + 1)
    corr_arr = np.array(corr_per_lag)[::-1]   # flip: index 0 → lag 1

    fig, ax = plt.subplots(figsize=(12, 4))
    suptitle = "Step 3.2 — Effective Memory Length  (h_T vs x_t correlation)"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    ax.bar(lags, corr_arr, color="darkorange", alpha=0.8, edgecolor="none")
    ax.set_xlabel("lag  (1 = yesterday,  N = N days ago)")
    ax.set_ylabel("mean |Pearson r|  (h_T vs x_t)")
    ax.set_title(
        "High bar at lag 1 + rapid decay → model only uses recent inputs\n"
        "Slow decay → hidden state genuinely accumulates older context",
        fontsize=9,
    )

    fig.tight_layout()
    plt.show()


def plot_whh_spectrum(model, title: str = "") -> None:
    """
    Step 3.3 — W_hh singular value spectrum.

    The spectral radius (largest singular value σ₁) determines the
    long-run fate of gradients flowing through W_hh^T:
      σ₁ < 1 → gradients shrink (vanishing)   — typical for vanilla RNN
      σ₁ > 1 → gradients grow  (exploding)    — training instability
      σ₁ ≈ 1 → gradients are preserved         — "edge of chaos", rare without LSTM

    The histogram shows whether W_hh is uniformly near zero (dead memory),
    spread around 0 (healthy), or has heavy tails (risk of instability).
    """
    W_hh = model.rnn.weight_hh_l0.detach().cpu().numpy()   # (hidden, hidden)
    sv = np.linalg.svd(W_hh, compute_uv=False)              # sorted descending
    spectral_radius = sv[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    suptitle = "Step 3.3 — W_hh Singular Value Spectrum"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # left — ranked singular values
    axes[0].plot(sv, "o-", color="steelblue", markersize=4, linewidth=1.2)
    axes[0].axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="σ = 1  (boundary)")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("singular value")
    axes[0].set_title(
        f"Ranked singular values\nspectral radius = {spectral_radius:.3f}  "
        f"({'vanishing' if spectral_radius < 1 else 'exploding'})",
        fontsize=10,
    )
    axes[0].legend(fontsize=9)

    # right — distribution
    axes[1].hist(sv, bins=30, color="steelblue", alpha=0.8, edgecolor="none")
    axes[1].axvline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="σ = 1")
    axes[1].set_xlabel("singular value")
    axes[1].set_ylabel("count")
    axes[1].set_title("Distribution of singular values", fontsize=10)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    plt.show()


def plot_gradient_norms_by_regime(model, regime_sets: dict, title: str = "") -> None:
    """
    Step 3.1 (regime) — Gradient norm per time step, one panel per regime.

    Same computation as plot_gradient_norms but run separately on each
    held-out regime set so that calm / clustering / spikes can be compared.
    regime_sets : dict { label: (X, y) } as returned by prepare_datasets.
    """
    import torch
    model.eval()

    n_regimes = len(regime_sets)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 4), sharey=True)
    if n_regimes == 1:
        axes = [axes]

    suptitle = "Step 3.1 — Gradient Norm per Time Step by Regime (BPTT)"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ax, (label, (X, _)) in zip(axes, regime_sets.items()):
        model.zero_grad()
        batch_size = min(64, X.shape[0])
        X_t = torch.tensor(X[:batch_size], dtype=torch.float32)

        rnn_out, _ = model.rnn(X_t)
        rnn_out.retain_grad()

        x = rnn_out[:, -1, :]
        for layer in model.head:
            x = layer(x)
        x.sum().backward()

        grad_norms = rnn_out.grad.norm(dim=-1).mean(dim=0).detach().cpu().numpy()
        seq_len = len(grad_norms)
        steps_back = np.arange(seq_len)

        ax.plot(steps_back, grad_norms[::-1], color="steelblue", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("steps back from prediction")
        ax.set_title(label, fontsize=11)

        max_norm = grad_norms.max()
        if max_norm > 0:
            half_idx = np.argmax(grad_norms[::-1] < 0.5 * max_norm)
            if half_idx > 0:
                ax.axvline(half_idx, color="crimson", linestyle="--", linewidth=1,
                           label=f"50%-decay at step {half_idx}")
                ax.legend(fontsize=8)

    axes[0].set_ylabel("||d loss / d h_t||")
    fig.tight_layout()
    plt.show()


def plot_effective_memory_by_regime(model, regime_sets: dict, title: str = "") -> None:
    """
    Step 3.2 (regime) — Effective memory length, one panel per regime.

    Same computation as plot_effective_memory but run separately on each
    held-out regime set so that calm / clustering / spikes can be compared.
    regime_sets : dict { label: (X, y) } as returned by prepare_datasets.
    """
    import torch
    model.eval()

    n_regimes = len(regime_sets)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 4), sharey=True)
    if n_regimes == 1:
        axes = [axes]

    suptitle = "Step 3.2 — Effective Memory Length by Regime  (h_T vs x_t correlation)"
    if title:
        suptitle += f"  [{title}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ax, (label, (X, _)) in zip(axes, regime_sets.items()):
        n = min(256, X.shape[0])
        X_t = torch.tensor(X[:n], dtype=torch.float32)

        with torch.no_grad():
            _, h_T = model.rnn(X_t)
            h_T = h_T[-1].cpu().numpy()

        X_np = X[:n]
        seq_len = X_np.shape[1]

        h_c = h_T - h_T.mean(axis=0)
        h_std = h_T.std(axis=0) + 1e-8

        corr_per_lag = []
        for t in range(seq_len):
            x_t = X_np[:, t, :]
            x_c = x_t - x_t.mean(axis=0)
            x_std = x_t.std(axis=0) + 1e-8
            cov = (h_c.T @ x_c) / n
            corr_mat = cov / (h_std[:, None] * x_std[None, :])
            corr_per_lag.append(np.abs(corr_mat).mean())

        lags = np.arange(1, seq_len + 1)
        corr_arr = np.array(corr_per_lag)[::-1]

        ax.bar(lags, corr_arr, color="darkorange", alpha=0.8, edgecolor="none")
        ax.set_xlabel("lag  (1 = yesterday,  N = N days ago)")
        ax.set_title(label, fontsize=11)

    axes[0].set_ylabel("mean |Pearson r|  (h_T vs x_t)")
    fig.tight_layout()
    plt.show()
