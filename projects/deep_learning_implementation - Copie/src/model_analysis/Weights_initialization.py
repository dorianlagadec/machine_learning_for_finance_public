import sys
sys.path.append('./src')
from dl_utils import NeuralNetwork, Trainer
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


def plot_init_analysis(net, X_train, y_train, X_test, y_test, optimizer ):
    """
    For a given weight_init + activation combo:
    - Plot weight distributions before and after training
    - Plot pre/post activation distributions for each layer before and after training
    """
    

    # --- Before training ---
    signals_before = get_layer_signals(net, X_test)
    weights_before = get_weights(net)

    trainer = Trainer(net, optimizer)
    trainer.fit(X_train, y_train, X_test, y_test,                                                                                                                                                             
                epochs=300,                                                                                                                                                                                           
                eval_every=10,  
                seed = 201906501,                                                                                                                                                                                     
                patience=5)


    # --- After training ---
    signals_after = get_layer_signals(net, X_test)
    weights_after = get_weights(net)

    n_layers = len(net.layers)
    fig, axes = plt.subplots(3, n_layers, figsize=(5 * n_layers, 12))
    fig.suptitle(f'Weights and Features distribution', fontsize=14, fontweight='bold', y=1.01)

    layer_labels = [f'Layer {i+1}' if i < n_layers - 1 else 'Output' for i in range(n_layers)]

    for i, layer in enumerate(net.layers):
        # Row 0: weight distributions
        ax = axes[0, i]
        ax.hist(weights_before[i], bins=60, alpha=0.6, label='Before', color='steelblue')
        ax.hist(weights_after[i],  bins=60, alpha=0.6, label='After',  color='tomato')
        ax.set_title(f' Weights, {layer_labels[i]}, Init : {layer._get_weight_init_method()} Activation : {layer._get_activation_function_name()} ')
        ax.legend(fontsize=8)
        ax.set_xlabel('Weight value')

        # Row 1: pre-activation distributions
        ax = axes[1, i]
        ax.hist(signals_before[i]['pre_activation'], bins=60, alpha=0.6, label='Before', color='steelblue')
        ax.hist(signals_after[i]['pre_activation'],  bins=60, alpha=0.6, label='After',  color='tomato')
        ax.set_title(f'Signals, {layer_labels[i]}  Pre-activation')
        ax.legend(fontsize=8)
        ax.set_xlabel('Activation value')

        # Row 2: post-activation distributions
        ax = axes[2, i]
        ax.hist(signals_before[i]['post_activation'], bins=60, alpha=0.6, label='Before', color='steelblue')
        ax.hist(signals_after[i]['post_activation'],  bins=60, alpha=0.6, label='After',  color='tomato')
        ax.set_title(f'Signals, {layer_labels[i]}  Post-activation')
        ax.legend(fontsize=8)
        ax.set_xlabel('Activation value')

    plt.tight_layout()
    plt.show()