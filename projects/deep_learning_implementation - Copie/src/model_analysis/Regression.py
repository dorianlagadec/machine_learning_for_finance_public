import sys
sys.path.append('./src')
from dl_utils import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def get_layer_signals(net : NeuralNetwork, X : np.ndarray):
    """
    Run a forward pass and collect pre-activation (input to layer)
    and post-activation (output of layer) for every hidden layer.
    """
    net.forward(X)
    signals = []
    for i, layer in enumerate(net.layers):
        signals.append({
            'pre_activation':  layer.input_.flatten(),
            'post_activation': layer.output.flatten()
        })
    return signals

def get_weights(net):
    """Return all weight matrices (not biases) as flat arrays."""
    return [layer.params[0].flatten() for layer in net.layers]


def plot_init_analysis(weight_init, activation_name, net, X_test):
    """
    For a given weight_init + activation combo:
    - Plot weight distributions before and after training
    - Plot pre/post activation distributions for each layer before and after training
    """
    

    # --- Before training ---
    signals_before = get_layer_signals(net, X_test)
    weights_before = get_weights(net)

    # --- After training ---
    signals_after = get_layer_signals(net, X_test)
    weights_after = get_weights(net)

    n_layers = len(net.layers)
    fig, axes = plt.subplots(3, n_layers, figsize=(5 * n_layers, 12))
    fig.suptitle(f'Init: {weight_init}  |  Activation: {activation_name}', fontsize=14, fontweight='bold', y=1.01)

    layer_labels = [f'Layer {i+1}' if i < n_layers - 1 else 'Output' for i in range(n_layers)]

    for i in range(n_layers):
        # Row 0: weight distributions
        ax = axes[0, i]
        ax.hist(weights_before[i], bins=60, alpha=0.6, label='Before', color='steelblue')
        ax.hist(weights_after[i],  bins=60, alpha=0.6, label='After',  color='tomato')
        ax.set_title(f'{layer_labels[i]} — Weights')
        ax.legend(fontsize=8)
        ax.set_xlabel('Weight value')

        # Row 1: pre-activation distributions
        ax = axes[1, i]
        ax.hist(signals_before[i]['pre_activation'], bins=60, alpha=0.6, label='Before', color='steelblue')
        ax.hist(signals_after[i]['pre_activation'],  bins=60, alpha=0.6, label='After',  color='tomato')
        ax.set_title(f'{layer_labels[i]} — Pre-activation')
        ax.legend(fontsize=8)
        ax.set_xlabel('Activation value')

        # Row 2: post-activation distributions
        ax = axes[2, i]
        ax.hist(signals_before[i]['post_activation'], bins=60, alpha=0.6, label='Before', color='steelblue')
        ax.hist(signals_after[i]['post_activation'],  bins=60, alpha=0.6, label='After',  color='tomato')
        ax.set_title(f'{layer_labels[i]} — Post-activation')
        ax.legend(fontsize=8)
        ax.set_xlabel('Activation value')

    plt.tight_layout()
    plt.show()





ACCENT = '#5BC8AF'
ACCENT2 = '#E8593C'
GRAY = '#888888'
GRID_COL = '#333333'


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, MSE, RMSE and QLIKE for a regression model."""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    eps = 1e-8
    qlike = np.mean(np.log(np.abs(y_pred) + eps) + np.abs(y_true) / (np.abs(y_pred) + eps))
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'QLIKE': qlike}

def print_metrics(metrics: dict, model_name: str = 'Model') -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'─'*40}")
    print(f"  {model_name}")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:.6f}")
    print(f"{'─'*40}\n")

def plot_coefficients(coef: np.ndarray, feature_names: list) -> None:
    """Bar chart of regression coefficients — |coef| = relative importance."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    colors = [ACCENT2 if c < 0 else ACCENT for c in coef]
    ax.bar(feature_names, coef, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
    ax.set_title('Linear regression — coefficients per lag\n(scaled features → |coef| = relative importance)', fontsize=12, pad=10)
    ax.set_xlabel('Feature (lag)')
    ax.set_ylabel('Coefficient value')
    ax.grid(axis='y', color=GRID_COL, linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = 'Model') -> None:
    """Scatter predicted vs actual + residuals over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor('#0E1117')
    for ax in axes:
        ax.set_facecolor('#0E1117')
    axes[0].scatter(y_true, y_pred, alpha=0.15, s=5, color=ACCENT)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, color=ACCENT2, linewidth=1.2, linestyle='--', label='Perfect prediction')
    axes[0].axhline(0, color='white', linewidth=0.6, linestyle=':', alpha=0.4, label='Predicted = 0')
    axes[0].set_xlabel('Actual volatility')
    axes[0].set_ylabel('Predicted volatility')
    axes[0].set_title(f'{model_name} — Predicted vs Actual')
    axes[0].legend(fontsize=9)
    axes[0].grid(color=GRID_COL, linewidth=0.4)
    residuals = y_true.flatten() - y_pred.flatten()
    axes[1].plot(residuals, color=ACCENT, linewidth=0.5, alpha=0.7)
    axes[1].axhline(0, color=ACCENT2, linewidth=0.8, linestyle='--', label='Zero residual (perfect)')
    axes[1].axhline(residuals.mean(), color='white', linewidth=0.8, linestyle=':', alpha=0.6, label=f'Mean residual = {residuals.mean():.2f}')
    axes[1].set_title(f'{model_name} — Residuals over time')
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Residual (actual − predicted)')
    axes[1].legend(fontsize=9)
    axes[1].grid(color=GRID_COL, linewidth=0.4)
    plt.tight_layout()
    plt.show()

def plot_performance_by_regime(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = 'Model', crisis_quantile: float = 0.90) -> dict:
    """Metrics and plots split by calm vs crisis regime."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    threshold = np.quantile(y_true, crisis_quantile)
    crisis_mask = y_true > threshold
    calm_mask = ~crisis_mask
    metrics_calm = compute_metrics(y_true[calm_mask], y_pred[calm_mask])
    metrics_crisis = compute_metrics(y_true[crisis_mask], y_pred[crisis_mask])
    print(f"\n  Regime threshold (p{int(crisis_quantile*100)}) = {threshold:.5f}")
    print(f"  Calm   samples : {calm_mask.sum()} ({calm_mask.mean()*100:.1f}%)")
    print(f"  Crisis samples : {crisis_mask.sum()} ({crisis_mask.mean()*100:.1f}%)")
    print_metrics(metrics_calm, f'{model_name} — Calm regime')
    print_metrics(metrics_crisis, f'{model_name} — Crisis regime')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor('#0E1117')
    for ax in axes:
        ax.set_facecolor('#0E1117')
    metric_keys = ['MAE', 'RMSE', 'QLIKE']
    x = np.arange(len(metric_keys))
    w = 0.3
    bars_calm = axes[0].bar(x - w/2, [metrics_calm[k] for k in metric_keys], width=w, label='Calm', color=ACCENT, alpha=0.85)
    bars_crisis = axes[0].bar(x + w/2, [metrics_crisis[k] for k in metric_keys], width=w, label='Crisis', color=ACCENT2, alpha=0.85)
    for bar in bars_calm:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=ACCENT)
    for bar in bars_crisis:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=ACCENT2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_keys)
    axes[0].set_title(f'{model_name} — Metrics by regime')
    axes[0].legend()
    axes[0].grid(axis='y', color=GRID_COL, linewidth=0.5)
    residuals = y_true - y_pred
    axes[1].bar(np.where(calm_mask)[0], residuals[calm_mask], color=ACCENT, alpha=0.6, width=1, label='Calm residuals')
    axes[1].bar(np.where(crisis_mask)[0], residuals[crisis_mask], color=ACCENT2, alpha=0.9, width=1, label='Crisis residuals')
    axes[1].axhline(0, color='white', linewidth=0.6, linestyle='--', alpha=0.4)
    axes[1].set_title(f'{model_name} — Residuals by regime')
    axes[1].set_xlabel('Sample index (test set)')
    axes[1].set_ylabel('Residual (actual − predicted)')
    axes[1].legend(fontsize=9)
    axes[1].grid(color=GRID_COL, linewidth=0.4)
    plt.tight_layout()
    plt.show()
    return {'calm': metrics_calm, 'crisis': metrics_crisis}