from copy import deepcopy
from typing import Tuple, Generator
from .NeuralNetwork import *
from .Optimizer import *
import numpy as np

def permute_data(X: np.ndarray, y: np.ndarray):
    '''
    Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


class Trainer(object):
    ''':
    Trains a neural network
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        '''
        Requires a neural network and an optimizer in order for training to occur. 
        Assign the neural network as an instance variable to the optimizer.
        '''
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)
        
    def generate_batches(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         size: int = 32) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
                         
        '''
        Generates batches for training 
        '''
        assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch

            
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            restart: bool = True,
            patience: int = 3,
            verbose = True)-> None:
        '''
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        Stops early if validation loss does not improve for "patience" consecutive checks.
        '''

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        self.optim.max_epochs = epochs
        self.optim._setup_decay()

        no_improvement = 0

        for e in range(epochs):

            if (e+1) % eval_every == 0:

                # for early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    if verbose:
                        print(f"Validation loss after {e+1} epochs is {loss:.6f}")
                    self.best_loss = loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if verbose:
                        print(f"No improvement after epoch {e+1} ({no_improvement}/{patience}), best loss: {self.best_loss:.6f}")
                    if no_improvement >= patience:
                        if verbose:
                            print(f"Early stopping triggered. Restoring best model.")
                        self.net = last_model
                        setattr(self.optim, 'net', self.net)
                        break
            if self.optim.final_lr is not None:
                self.optim._decay_lr()
