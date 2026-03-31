# dl_utils — Neural Network from Scratch

A minimal, fully custom implementation of a feedforward neural network in pure NumPy. Built to understand the internals of deep learning: forward pass, backpropagation, parameter updates, and training loops — without relying on any deep learning framework.

---

## Architecture Overview

The library is organized in layers of abstraction, from low-level math to high-level training:

```
Operation       → atomic math operations (matmul, bias add, activation)
    ↓
Layer           → a stack of operations forming one layer of neurons
    ↓
NeuralNetwork   → a stack of layers + a loss function
    ↓
Trainer         → training loop with batching and early stopping
```

`Optimizer` is a separate component that plugs into `Trainer` and handles the parameter update rule.

---

## Modules

### `Operation.py`

The lowest level of the library. Every computation in the network — weight multiplication, bias addition, activation functions — is implemented as an `Operation`. Each operation knows how to compute its output (forward pass) and how to propagate gradients backward (backward pass).

---

#### `Operation` (base class)

The base class for all stateless operations (no learnable parameters).

**Methods:**
- `forward(input_)` — stores the input, computes and returns the output
- `backward(output_grad)` — receives the gradient from the next layer, computes and returns the gradient with respect to the input. Validates that shapes match before and after.

**Must implement:**
- `_output()` — defines the forward computation
- `_input_grad(output_grad)` — defines the gradient computation

---

#### `ParamOperation(Operation)` (base class)

Extends `Operation` for operations that have learnable parameters (weights, biases).

**Additional behavior:**
- Stores a `param` array at initialization
- `backward()` also computes `self.param_grad` — the gradient with respect to the parameter, which the optimizer will use to update it

**Must implement:**
- `_param_grad(output_grad)` — gradient with respect to the parameter

---

#### `WeightMultiply(ParamOperation)`

Matrix multiplication of the input by a weight matrix: `output = input @ W`

| Parameter | Description |
|-----------|-------------|
| `W` | Weight matrix of shape `(n_inputs, n_neurons)` |

- **Forward**: `input @ W`
- **Input grad**: `output_grad @ W.T`
- **Param grad**: `input.T @ output_grad`

---

#### `BiasAdd(ParamOperation)`

Adds a bias vector to the input: `output = input + b`

| Parameter | Description |
|-----------|-------------|
| `B` | Bias vector of shape `(1, n_neurons)` — must have `shape[0] == 1` |

- **Forward**: `input + b` (broadcasts across the batch)
- **Input grad**: `output_grad` (identity, bias addition doesn't affect input gradient)
- **Param grad**: sum of `output_grad` across the batch, reshaped to `(1, n_neurons)`

---

#### `Dropout(Operation)`

Randomly zeros out activations during training to prevent overfitting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keep_prob` | `float` | `0.8` | Fraction of neurons to keep. `1.0` = no dropout. |

- **Training** (`inference=False`): generates a binary mask where each value is 1 with probability `keep_prob`, multiplies the input by it
- **Inference** (`inference=True`): multiplies the input by `keep_prob` to match the expected magnitude seen during training
- The mask is stored and reused in `backward()` — the same neurons that were zeroed in the forward pass have zero gradient in the backward pass
- Not a `ParamOperation` — has no learnable parameters

Dropout is added automatically inside `Dense` when `dropout < 1.0`, placed after the activation function. Use `dropout=1.0` (the default) to disable it entirely.

---

#### `Sigmoid(Operation)`

Sigmoid activation function: `output = 1 / (1 + exp(-input))`

- **Forward**: element-wise sigmoid
- **Input grad**: `sigmoid(x) * (1 - sigmoid(x)) * output_grad`
- **Effect**: squashes all values to `(0, 1)`. Can cause **vanishing gradients** in deep networks because the gradient is always < 0.25 and gets multiplied across layers.

---

#### `ReLU(Operation)`

Rectified Linear Unit: `output = max(0, input)`

- **Forward**: element-wise `max(0, x)`
- **Input grad**: `output_grad` where `input > 0`, else `0`
- **Effect**: faster training than sigmoid, no vanishing gradient for positive inputs. Recommended for hidden layers. Use with **He initialization**.

---

#### `Linear(Operation)`

Identity activation — passes input through unchanged: `output = input`

- **Forward**: `input`
- **Input grad**: `output_grad`
- **Effect**: used for the output layer of regression networks so predictions are unbounded.

---

### `Layer.py`

A layer groups a sequence of operations into a single unit. It handles lazy initialization (weights are created on the first forward pass when input shape is known) and coordinates the forward/backward pass through its operations.

---

#### `Layer` (base class)

| Attribute | Description |
|-----------|-------------|
| `neurons` | Number of output neurons |
| `first` | If `True`, `_setup_layer()` will be called on the next forward pass to initialize weights |
| `params` | List of parameter arrays (weights, biases) |
| `param_grads` | List of gradient arrays, populated after each backward pass |
| `operations` | Ordered list of `Operation` objects |

**Methods:**
- `forward(input_)` — runs input through all operations in order. On the first call, initializes weights via `_setup_layer()`.
- `backward(output_grad)` — runs gradient backward through operations in reverse order, then calls `_param_grads()` to collect gradients.
- `_param_grads()` — extracts `param_grad` from each `ParamOperation` in the layer and stores them in `self.param_grads`.
- `_params()` — extracts `param` from each `ParamOperation`. Note: `self.params` is already set correctly in `_setup_layer`, so this is a utility method.

---

#### `Dense(Layer)`

A fully connected layer: applies `WeightMultiply → BiasAdd → Activation`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neurons` | `int` | required | Number of output neurons in this layer |
| `activation` | `Operation` | `Sigmoid()` | Activation function applied after the linear transformation |
| `weight_init` | `str` | `'glorot_normal'` | Weight initialization strategy — see table below |
| `dropout` | `float` | `1.0` | Dropout keep probability. `1.0` = no dropout. Applied after the activation. |

**Weight initialization strategies:**

| `weight_init` | Formula | When to use |
|---------------|---------|-------------|
| `'standard'` | `randn(n_in, n_out)` | Sigmoid, Linear activations |
| `'he'` | `randn(n_in, n_out) * sqrt(2 / n_in)` | ReLU activations |

He initialization scales the weights to compensate for the fact that ReLU kills half the neurons (outputs 0 for negative inputs). Without it, the variance of activations shrinks across layers, slowing learning.

Biases are always initialized to zeros.

**Example:**
```python
Dense(neurons=32, activation=ReLU(), weight_init='he')
Dense(neurons=1,  activation=Linear())
```

---

### `Loss.py`

Loss functions compute the scalar training objective and its gradient with respect to the network's predictions.

---

#### `Loss` (base class)

**Methods:**
- `forward(prediction, target)` — stores prediction and target, returns the scalar loss value
- `backward()` — returns the gradient of the loss with respect to `prediction`

---

#### `MeanSquaredError(Loss)`

`loss = sum((prediction - target)^2) / n`

- **Gradient**: `2 * (prediction - target) / n`
- **Properties**: smooth gradient everywhere, penalizes large errors heavily due to the square. Good default for regression.

---

#### `MeanAbsoluteError(Loss)`

`loss = sum(|prediction - target|) / n`

- **Gradient**: `sign(prediction - target) / n`
- **Properties**: gradient is constant in magnitude (always ±1/n), more robust to outliers than MSE but harder to optimize because the gradient doesn't shrink as predictions improve.

---

### `NeuralNetwork.py`

Orchestrates a list of layers and a loss function into a complete model.

---

#### `NeuralNetwork`

| Parameter | Type | Description |
|-----------|------|-------------|
| `layers` | `List[Layer]` | Ordered list of layers, applied left to right |
| `loss` | `Loss` | Loss function used during training |
| `seed` | `int` | Random seed propagated to all layers for reproducibility |

**Methods:**
- `forward(x_batch)` — passes input through all layers and returns predictions
- `backward(loss_grad)` — passes the loss gradient backward through all layers
- `train_batch(x_batch, y_batch)` — runs forward pass, computes loss, runs backward pass. Returns the loss value.
- `params()` — generator that yields all parameter arrays across all layers
- `param_grads()` — generator that yields all gradient arrays across all layers (must be called after a backward pass)

**Example:**
```python
model = NeuralNetwork(
    layers=[
        Dense(neurons=32, activation=ReLU(), weight_init='he'),
        Dense(neurons=16, activation=ReLU(), weight_init='he'),
        Dense(neurons=1,  activation=Linear())
    ],
    loss=MeanSquaredError(),
    seed=42
)
```

---

### `Optimizer.py`

Optimizers update the network's parameters after each batch using the gradients computed by backpropagation.

---

#### `Optimizer` (base class)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `0.01` | Learning rate — controls the size of each parameter update |

The optimizer is linked to a network by `Trainer` via `setattr(optim, 'net', net)`.

---

#### `SGD(Optimizer)`

Stochastic Gradient Descent. Updates each parameter by stepping in the direction opposite to its gradient:

`param -= lr * param_grad`

The update is applied **in-place** (`param[:] -= lr * param_grad`) so the original arrays stored in the network are actually modified.

| `lr` value | Effect |
|------------|--------|
| Too high (e.g. `0.1`) | Overshoots, loss oscillates or diverges |
| Too low (e.g. `0.00001`) | Learns very slowly |
| Typical range | `0.001` – `0.01` for this library |

---

### `Trainer.py`

Handles the full training loop: batching, shuffling, evaluation, and early stopping.

---

#### `permute_data(X, y)`

Shuffles `X` and `y` together along axis 0, preserving the correspondence between samples and labels.

---

#### `Trainer`

| Parameter | Type | Description |
|-----------|------|-------------|
| `net` | `NeuralNetwork` | The network to train |
| `optim` | `Optimizer` | The optimizer to use for parameter updates |

---

#### `Trainer.fit()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | `ndarray` | required | Training features |
| `y_train` | `ndarray` | required | Training targets |
| `X_test` | `ndarray` | required | Validation features |
| `y_test` | `ndarray` | required | Validation targets |
| `epochs` | `int` | `100` | Number of full passes through the training data |
| `eval_every` | `int` | `10` | Evaluate on validation set every N epochs |
| `batch_size` | `int` | `32` | Number of samples per mini-batch |
| `seed` | `int` | `1` | Random seed for data shuffling |
| `restart` | `bool` | `True` | If `True`, resets layer weights and best loss before training |
| `patience` | `int` | `3` | Number of consecutive evaluation checks with no improvement before stopping |

**Training loop:**
1. At each epoch, shuffle the training data
2. Split into mini-batches of size `batch_size`
3. For each batch: forward pass → compute loss → backward pass → optimizer step
4. Every `eval_every` epochs: compute validation loss
   - If improved: save as best, reset patience counter
   - If not improved: increment patience counter. If counter reaches `patience`, restore the best model and stop.

**Effect of key parameters:**

| Parameter | Too low | Too high |
|-----------|---------|----------|
| `epochs` | Network hasn't converged | Wastes compute (early stopping handles this) |
| `eval_every` | Evaluates too often, early stopping fires on noise | Misses the optimal stopping point |
| `batch_size` | Noisy gradients, slow per-epoch | Smoother gradients but less regularization effect |
| `patience` | Stops too early on temporary loss spikes | Trains too long after the network has stopped improving |

---

### `Metrics.py`

Utility functions for evaluating regression models.

#### `mae(y_true, y_pred)`
Returns mean absolute error: `mean(|y_true - y_pred|)`

#### `rmse(y_true, y_pred)`
Returns root mean squared error: `sqrt(mean((y_true - y_pred)^2))`

#### `eval_regression_model(model, X_test, y_test)`
Runs a forward pass and prints both MAE and RMSE. Convenience wrapper around the two functions above.

---

## Full Usage Example

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dl_utils import *

# --- Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test  = scaler_x.transform(X_test)

# --- Model ---
model = NeuralNetwork(
    layers=[
        Dense(neurons=32, activation=ReLU(), weight_init='he'),
        Dense(neurons=16, activation=ReLU(), weight_init='he'),
        Dense(neurons=1,  activation=Linear())
    ],
    loss=MeanSquaredError(),
    seed=42
)

# --- Training ---
trainer = Trainer(model, SGD(lr=0.001))

trainer.fit(X_train, y_train, X_test, y_test,
    epochs=300,
    eval_every=10,
    batch_size=32,
    patience=5)

# --- Evaluation ---
preds = model.forward(X_test)
print(f"MAE: {np.mean(np.abs(preds - y_test)):.4f}")
```

---

## Design Notes

- **Lazy initialization**: layer weights are created on the first forward pass, so you never need to specify input dimensions explicitly. The layer infers them from the data.
- **In-place parameter updates**: the optimizer uses `param[:] -= lr * grad` instead of `param -= lr * grad`. The latter rebinds the local variable and leaves the original array untouched — a subtle Python/NumPy gotcha.
- **Early stopping restores the best model**: when patience runs out, the trainer resets `self.net` to the saved checkpoint and re-links the optimizer to it.
