# FNN Workflow

This document traces a full training step through the custom NumPy framework, from raw input to parameter update. Each section maps to a source file in `src/dl_utils/`.

---

## High-Level Data Flow

```
X_batch ──► Dense ──► Dense ──► ... ──► Dense ──► Loss ──► scalar
                                                      │
               ◄──────────────────────────────────────┘  backward()
               param_grads collected layer by layer
                                                      │
                                               Optimizer.step()
                                               param[:] -= lr * grad
```

---

## 1. Operations — `Operation.py`

The `Operation` is the atomic unit of computation. Every arithmetic step (matrix multiply, bias add, activation) is its own `Operation` subclass.

### Base classes

| Class | Role |
|-------|------|
| `Operation` | Base: `forward()` stores input, calls `_output()`; `backward()` calls `_input_grad()`, asserts shapes |
| `ParamOperation(Operation)` | Adds a learnable `param` array; `backward()` also calls `_param_grad()` and stores it in `self.param_grad` |

### Concrete operations used in a Dense layer

| Class | `_output()` | `_input_grad(g)` | `_param_grad(g)` |
|-------|-------------|------------------|------------------|
| `WeightMultiply` | `X @ W` | `g @ W.T` | `X.T @ g` |
| `BiasAdd` | `X + B` | `ones_like(X) * g` | `sum(g, axis=0)` |
| `Sigmoid` | `1 / (1 + exp(-x))` | `σ(x) * (1 - σ(x)) * g` | — |
| `ReLU` | `max(0, x)` | `g * (x > 0)` | — |
| `Tanh` | `tanh(x)` | `(1 - tanh²(x)) * g` | — |
| `LeakyReLU(α)` | `x if x>0 else αx` | `g * (1 if x>0 else α)` | — |
| `ELU(α)` | `x if x>0 else α(eˣ-1)` | `g * (1 if x>0 else out+α)` | — |
| `Swish` | `x * σ(x)` | `g * (σ + x·σ·(1-σ))` | — |
| `Linear` | `x` (identity) | `g` | — |
| `Dropout(p)` | training: `x * mask`; inference: `x * p` | `g * mask` | — |

**Shape invariant**: `backward()` asserts `output_grad.shape == output.shape` and `input_grad.shape == input_.shape` before returning.

---

## 2. Layer — `Layer.py`

A `Layer` owns an ordered list of `Operations` and orchestrates the sequential forward/backward pass through them.

### `Layer.forward(input_, inference=False)`

1. On the **first call only**: calls `_setup_layer(input_)` to lazily initialize weights using the actual input shape.
2. Loops over `self.operations` in order, passing the output of one as the input to the next.
3. Returns the final output.

### `Layer.backward(output_grad)`

1. Loops over `self.operations` **in reverse**, threading the gradient backward.
2. Calls `self._param_grads()` to collect all `ParamOperation.param_grad` values into `self.param_grads`.
3. Returns `input_grad` (gradient to be passed to the previous layer).

### `Dense._setup_layer(input_)`

Sets up the operation chain for a fully connected layer:

```
WeightMultiply(W)  →  BiasAdd(B)  →  activation  [→  Dropout]
```

Weight shapes: `W: (n_in, n_out)`, `B: (1, n_out)`.

**Lazy initialization**: weights are created on the first forward pass, so you never need to specify input size when constructing the network.

### `Dense.__init__` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `neurons` | `int` | Output width of this layer (`n_out`) |
| `activation` | `Operation` | Activation function instance (default: `Sigmoid()`) |
| `weight_init` | `str` | Init strategy (see table below) |
| `dropout` | `float` | Keep probability in (0, 1]; `1.0` = no dropout |

### Weight initialization strategies

| `weight_init` | Formula | Recommended for |
|---------------|---------|-----------------|
| `'standard'` | N(0, 1) | Baseline only |
| `'glorot_normal'` | N(0, √(2/(n_in+n_out))) | Sigmoid, Tanh, Linear |
| `'glorot_uniform'` | U(±√(6/(n_in+n_out))) | Sigmoid, Tanh |
| `'he_normal'` | N(0, √(2/n_in)) | ReLU |
| `'he_uniform'` | U(±√(6/n_in)) | ReLU (PyTorch default) |
| `'lecun_normal'` | N(0, √(1/n_in)) | SELU |

---

## 3. NeuralNetwork — `NeuralNetwork.py`

`NeuralNetwork` composes a list of `Layer` objects and a `Loss` function.

### `NeuralNetwork.forward(x_batch, inference=False)`

Calls `layer.forward(x)` for each layer in sequence. Returns the final prediction array.

### `NeuralNetwork.backward(loss_grad)`

Calls `layer.backward(grad)` for each layer **in reverse**. Gradients flow back through the whole network.

### `NeuralNetwork.train_batch(x_batch, y_batch) → float`

One complete forward + backward step:

```python
predictions = self.forward(x_batch)          # 1. forward pass
loss        = self.loss.forward(predictions, y_batch)  # 2. compute loss
loss_grad   = self.loss.backward()           # 3. loss gradient
self.backward(loss_grad)                     # 4. backprop
return loss
```

### `NeuralNetwork.params()` / `.param_grads()`

Aggregate `layer.params` / `layer.param_grads` across all layers into flat lists. Used by the optimizer to iterate over all parameters.

---

## 4. Loss — `Loss.py`

| Method | Description |
|--------|-------------|
| `Loss.forward(prediction, target)` | Stores both arrays, returns the scalar loss value |
| `Loss.backward()` | Returns `∂loss/∂prediction` — the gradient passed into `NeuralNetwork.backward()` |

### `MeanSquaredError`

```
_output()      = mean((prediction - target)²)
_input_grad()  = 2 * (prediction - target) / N
```

---

## 5. Optimizer — `Optimizer.py`

### `SGD.step()`

Iterates over `(param, grad)` pairs from `net.params()` / `net.param_grads()` and applies:

```python
param[:] -= lr * grad   # in-place mutation — required to update the shared array
```

The `[:]` slice notation is critical: rebinding the local variable (`param = ...`) would not update the `WeightMultiply.param` array that the layer holds.

### `SGD_MOMENTUM.step()`

Uses a velocity accumulator:

```
v = momentum * v - lr * grad
param[:] += v
```

### Learning rate decay

Both optimizers support optional decay over training:
- `'linear'`: interpolates from `lr` to `final_lr` linearly across epochs.
- `'exponential'`: exponential schedule from `lr` to `final_lr`.

---

## 6. Trainer — `Trainer.py`

`Trainer` owns a `NeuralNetwork` and an `Optimizer` and runs the full training loop.

### `Trainer.fit(X_train, y_train, X_test, y_test, epochs, eval_every, batch_size, patience)`

```
for epoch in range(epochs):
    shuffle X_train, y_train
    for X_batch, y_batch in generate_batches(...):
        loss = net.train_batch(X_batch, y_batch)   # forward + backward
        optimizer.step()                            # update params
    if epoch % eval_every == 0:
        val_loss = evaluate on X_test
        if val_loss < best_loss:
            checkpoint model weights
        else if no improvement for `patience` checks:
            restore best weights and stop early
```

### `Trainer.generate_batches(X, y, size)`

Yields `(X_batch, y_batch)` pairs of the given batch size. The last batch may be smaller if the dataset size is not divisible by `size`.

### Key flags

| Flag | Effect |
|------|--------|
| `inference=True` in `forward()` | Dropout multiplies by `keep_prob` instead of masking — use for evaluation |
| `restart=True` in `fit()` | Re-initializes weights before training (calls `_setup_layer` again on next forward pass) |
