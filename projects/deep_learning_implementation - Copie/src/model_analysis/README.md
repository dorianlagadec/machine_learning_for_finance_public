# model_analysis

Visualization utilities for inspecting the internal state of neural networks — weight distributions and signal flow across layers. Used primarily in `5_weight_init.ipynb`.

---

## Modules

### `Weights_initialization.py`

Tools for visualizing how weight initialization and training affect the distribution of weights and activations across layers.

| Function | Description |
|----------|-------------|
| `get_layer_signals(net, X)` | Runs a forward pass and returns the pre-activation and post-activation values for every layer |
| `get_weights(net)` | Returns the weight matrices of all layers as flat arrays |
| `plot_init_analysis(net, X_train, y_train, X_test, y_test, optimizer)` | Trains the network then plots a 3-row grid: weight distributions, pre-activation distributions, post-activation distributions — before and after training |

**What `plot_init_analysis` shows:**

| Row | What it shows | What to look for |
|-----|---------------|-----------------|
| Weights | Distribution of weight values per layer | Do weights move significantly after training? Are they spread or collapsed? |
| Pre-activation | Values entering the activation function | Should be roughly centered around 0. If very large → saturation with Sigmoid/Tanh |
| Post-activation | Values leaving the activation function | Should stay spread across layers. If all values are the same → information is lost |

---

### `Regression.py`

An earlier version of the weight/activation visualization, used during initial exploration. Superseded by `Weights_initialization.py` which takes a fully trained network as input rather than training inside the plotting function.

---

## Usage

```python
from model_analysis.Weights_initialization import plot_init_analysis
from dl_utils import *

model = NeuralNetwork(
    layers=[
        Dense(neurons=64, activation=ReLU(), weight_init='he_normal'),
        Dense(neurons=64, activation=ReLU(), weight_init='he_normal'),
        Dense(neurons=1,  activation=Linear())
    ],
    loss=MeanSquaredError()
)

plot_init_analysis(model, X_train, y_train, X_test, y_test,
                   optimizer=SGD_MOMENTUM(lr=0.001))
```
