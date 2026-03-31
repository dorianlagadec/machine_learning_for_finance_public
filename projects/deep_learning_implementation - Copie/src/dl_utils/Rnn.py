import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, List, Literal
from typing import get_args, Literal
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
Activation = Literal['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'none']
Initializer = Literal['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'normal', 'uniform', 'zeros']
RNNNonlinearity = Literal['tanh', 'relu']

ACTIVATIONS = {
    'relu':       nn.ReLU(),
    'tanh':       nn.Tanh(),
    'sigmoid':    nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'elu':        nn.ELU(),
    'none':       nn.Identity(),
}


class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation = 'relu',
        dropout: float = 0.0,
        bias: bool = True,
        weight_init: Initializer = 'xavier_uniform',
        bias_init: Initializer = 'zeros',
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = ACTIVATIONS[activation]
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()

        self._init_weights(self.linear.weight, weight_init)
        if bias:
            self._init_weights(self.linear.bias, bias_init)

    def _init_weights(self, tensor: torch.Tensor, method: Initializer) -> None:
        with torch.no_grad():
            if method == 'xavier_uniform':
                if tensor.dim() >= 2: init.xavier_uniform_(tensor)
                else: init.zeros_(tensor)
            elif method == 'xavier_normal':
                if tensor.dim() >= 2: init.xavier_normal_(tensor)
                else: init.zeros_(tensor)
            elif method == 'kaiming_uniform':
                if tensor.dim() >= 2: init.kaiming_uniform_(tensor)
                else: init.zeros_(tensor)
            elif method == 'kaiming_normal':
                if tensor.dim() >= 2: init.kaiming_normal_(tensor)
                else: init.zeros_(tensor)
            elif method == 'normal':  init.normal_(tensor, mean=0.0, std=0.01)
            elif method == 'uniform': init.uniform_(tensor, -0.1, 0.1)
            elif method == 'zeros':   init.zeros_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: RNNNonlinearity = 'tanh',
        rnn_dropout: float = 0.0,
        bidirectional: bool = False,
        head: Optional[List[Dense]] = None,   # ← list of already built Dense layers
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )


        self.head = nn.ModuleList(head) if head is not None else nn.ModuleList([])

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None, ) -> torch.Tensor:

        out, _ = self.rnn(x, h0)
        last = out[:, -1, :]   # (batch, hidden_size)

        for layer in self.head:
            last = layer(last)

        return last


class RNNMeanPool(nn.Module):
    """
    RNN that feeds the temporal mean of all hidden states to the head,
    instead of only h_T.

    The vanilla RNN fails because the output Jacobian collapses at step 1:
    h_T is linearly dominated by the most recent input, so the head never
    learns to extract older information even though h_T nonlinearly encodes it.
    Mean-pooling across all T hidden states gives the head a direct, equal-weight
    view of every time step — bypassing the Jacobian collapse entirely.
    """
    def __init__(
        self,
        input_size:   int,
        hidden_size:  int,
        num_layers:   int = 1,
        nonlinearity: RNNNonlinearity = 'tanh',
        rnn_dropout:  float = 0.0,
        head: Optional[List[Dense]] = None,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.ModuleList(head) if head is not None else nn.ModuleList([])

    def forward(self, x: torch.Tensor, h0=None) -> torch.Tensor:
        out, _ = self.rnn(x, h0)      # (batch, seq_len, hidden)
        pooled = out.mean(dim=1)       # (batch, hidden) — equal weight to all steps
        for layer in self.head:
            pooled = layer(pooled)
        return pooled


class LSTM(nn.Module):
    """
    LSTM wrapper that mirrors the RNN class interface.

    The key difference from vanilla RNN: the cell state c_t acts as a
    gradient highway during BPTT. Forget/input/output gates control what
    information is written, retained, and exposed at each step, allowing
    gradients to flow back 50-100+ steps without exponential decay.

    The output gate specifically controls what the head reads from the cell
    state — this lets the model learn to expose older context to the
    prediction layer, which vanilla RNN cannot do (its Jacobian collapses
    at step 1).
    """
    def __init__(
        self,
        input_size:   int,
        hidden_size:  int,
        num_layers:   int = 1,
        lstm_dropout: float = 0.0,
        bidirectional: bool = False,
        head: Optional[List[Dense]] = None,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.head = nn.ModuleList(head) if head is not None else nn.ModuleList([])

    def forward(self, x: torch.Tensor, h0=None) -> torch.Tensor:
        out, _ = self.lstm(x, h0)
        last = out[:, -1, :]          # (batch, hidden_size)
        for layer in self.head:
            last = layer(last)
        return last


LossFunction = Literal['mse', 'mae', 'huber']
Optimizer    = Literal['adam', 'sgd', 'rmsprop']

LOSSES = {
    'mse':   nn.MSELoss(),
    'mae':   nn.L1Loss(),
    'huber': nn.HuberLoss(),
}


class Trainer:
    def __init__(
        self,
        model:          nn.Module,
        loss_fn:        LossFunction = 'mse',
        optimizer:      Optimizer = 'adam',
        lr:             float = 0.001,
        batch_size:     int = 32,
        n_epochs:       int = 100,
        shuffle:        bool = True,
        clip_grad_norm: Optional[float] = None,   # gradient clipping — useful for RNN
        early_stopping: Optional[int] = None,     # stop if test loss doesn't improve for N epochs
        verbose:        int = 10,                 # print every N epochs, 0 = silent
    ) -> None:

        if loss_fn not in get_args(LossFunction):
            raise ValueError(f"loss_fn must be one of {get_args(LossFunction)}, got '{loss_fn}'")
        if optimizer not in get_args(Optimizer):
            raise ValueError(f"optimizer must be one of {get_args(Optimizer)}, got '{optimizer}'")

        self.model      = model
        self.loss_fn    = LOSSES[loss_fn]
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.shuffle    = shuffle
        self.clip_grad_norm = clip_grad_norm
        self.early_stopping = early_stopping
        self.verbose    = verbose

        # build optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        # history
        self.train_losses: List[float] = []
        self.test_losses:  List[float] = []
        self._best_weights = None
        self.best_epoch: int = 0


    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(x, dtype=torch.float32)

    def fit(
        self,
        X_train,
        y_train,
        X_test  = None,
        y_test  = None,
    ) -> None:

        X_train = self._to_tensor(X_train)
        y_train = self._to_tensor(y_train)
        if X_test is not None:
            X_test = self._to_tensor(X_test)
        if y_test is not None:
            y_test = self._to_tensor(y_test)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        best_test_loss  = float('inf')
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(self.n_epochs):

            # ── train ──────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                #print(X_batch)
                pred = self.model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                loss.backward()

                # gradient clipping — prevents exploding gradients in RNN
                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(train_loss)

            # ── evaluate ───────────────────────────────────────
            if X_test is not None and y_test is not None:
                self.model.eval()
                with torch.no_grad():
                    test_pred = self.model(X_test)
                    test_loss = self.loss_fn(test_pred, y_test).item()
                    self.test_losses.append(test_loss)

                # always track the best model
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch
                    self.best_epoch = best_epoch
                    epochs_no_improve = 0
                    self._best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_no_improve += 1

                # early stopping (only if enabled)
                if self.early_stopping is not None and epochs_no_improve >= self.early_stopping:
                    if self.verbose > 0:
                        print(f"early stopping at epoch {epoch} — best test loss: {best_test_loss:.6f}")
                    self.model.load_state_dict(self._best_weights)
                    return

            # ── logging ────────────────────────────────────────
            if self.verbose > 0 and epoch % self.verbose == 0:
                log = f"epoch {epoch:4d}  train: {train_loss:.6f}"
                if X_test is not None:
                    log += f"  test: {test_loss:.6f}"
                print(log)

        # restore best model at end of full training
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)
            if self.verbose > 0:
                print(f"restoring best model from epoch {best_epoch} — test loss: {best_test_loss:.6f}")


    def restore_best_model(self) -> None:
        if self._best_weights is None:
            raise RuntimeError("No best weights saved — train with test data first.")
        self.model.load_state_dict(self._best_weights)

    def predict(
        self,
        X,
        scaler_y:   Optional[object] = None,   # pass scaler to inverse transform
    ) -> np.ndarray:

        X = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X).numpy()

        if scaler_y is not None:
            pred = scaler_y.inverse_transform(pred)

        return pred


    def plot_training_curve(self, title: str = "") -> None:
        if not self.train_losses:
            print("No training history — call fit() first.")
            return

        fig, ax = plt.subplots(figsize=(9, 4))
        epochs = range(len(self.train_losses))

        ax.plot(epochs, self.train_losses, label="train loss", linewidth=1.5)
        if self.test_losses:
            ax.plot(epochs, self.test_losses, label="val loss", linewidth=1.5)
            ax.axvline(self.best_epoch, color="gray", linestyle="--", linewidth=1,
                       label=f"best epoch ({self.best_epoch})")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title if title else "Training curve")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def summary(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"total parameters:     {total}")
        print(f"trainable parameters: {trainable}")
        print(f"best train loss:      {min(self.train_losses):.6f}" if self.train_losses else "not trained yet")
        print(f"best test loss:       {min(self.test_losses):.6f}"  if self.test_losses  else "no test data")

