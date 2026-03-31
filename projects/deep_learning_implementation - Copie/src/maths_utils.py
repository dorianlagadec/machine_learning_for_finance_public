from typing import Callable, List, Dict, Tuple
import numpy as np

def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray, 
          delta: float = 0.001) -> np.ndarray:

    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)



#defintion of a function taking  array as argument and returning array
Array_Function = Callable[[np.ndarray], np.ndarray]
#Chain of function, inex 0 of the Chain is the most nested function, first one to be called
Chain = List[Array_Function]


def chain_length_2(chain: Chain,
    a: np.ndarray) -> np.ndarray:
    assert len(chain) == 2
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(a))


def sigmoid(x : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def square(x : np.ndarray) -> np.ndarray:
    return x**2

def leaky_relu(x: np.ndarray) -> np.ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray
    '''
    return np.maximum(0.2 * x, x)

def chain_deriv_2(chain: Chain,
                   input_range: np.ndarray) -> np.ndarray:

    assert len(chain) == 2

    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1(input_range))
    return df1dx * df2du


def plot_chain(ax,
               chain: Chain, 
               input_range: np.ndarray) -> None:

    
    assert input_range.ndim == 1
    output_range = chain_length_2(chain, input_range)
    ax.plot(input_range, output_range)


def plot_chain_deriv(ax,
                     chain: Chain,
                     input_range: np.ndarray) -> np.ndarray:

    output_range = chain_deriv_2(chain, input_range)
    ax.plot(input_range, output_range)


def chain_deriv_3(chain: Chain,
           input_range: np.ndarray) -> np.ndarray:
    assert len(chain) == 3
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    f1_of_x = f1(input_range)
    f2_of_x = f2(f1_of_x)

    df3du = deriv(f3, f2_of_x)

    df2du = deriv(f2, f1_of_x)
    df1dx = deriv(f1, input_range)

    return df1dx * df2du * df3du


def multiple_inputs_add(x: np.ndarray,
                        y: np.ndarray,
                        sigma: Array_Function) -> float:

    assert x.shape == y.shape
    a = x + y
    return sigma(a)

def multiple_inputs_add_backward(x: np.ndarray,
                                y: np.ndarray,
                                sigma: Array_Function) -> float:
    a = x + y
    dsda = deriv(sigma, a)
    dadx, dady = 1, 1
    return dsda * dadx, dsda * dady

def matmul_forward(X: np.ndarray,
                    W: np.ndarray) -> np.ndarray:

        assert X.shape[1] == W.shape[0]


        N = X @ W
        return N

def matmul_backward_first(X: np.ndarray,
                            W: np.ndarray) -> np.ndarray:

                    
            dNdX = W.T
            return dNdX


def matrix_forward_extra(X: np.ndarray,
                        W: np.ndarray,
                        sigma: Array_Function) -> np.ndarray:

        assert X.shape[1] == W.shape[0]
        return sigma(X @ W)


def matrix_function_backward_1(X: np.ndarray,
                                W: np.ndarray,
                                sigma: Array_Function) -> np.ndarray:

    assert X.shape[1] == W.shape[0]

    N = X @ W
    S = sigma(N)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)
    dLdN = dLdS * dSdN
    dNdX = np.transpose(W, (1, 0))

    return np.dot(dLdN, dNdX)


def matrix_function_forward_sum(X: np.ndarray,
                                W: np.ndarray,
                                sigma: Array_Function) -> float:

    assert X.shape[1] == W.shape[0]
    N = X @ W
    S = sigma(N)
    L = np.sum(S)
    return L




def matrix_function_backward_sum_1(X, W, sigma):
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)
    dLdN = dLdS * dSdN        # elementwise
    dNdX = np.transpose(W, (1, 0))
    dLdX = np.dot(dLdN, dNdX) # matrix multiply

    return dLdX



def forward_linear_regression(X_batch: np.ndarray,
                              y_batch: np.ndarray,
                              weights: Dict[str, np.ndarray]
                              )-> Tuple[float, Dict[str, np.ndarray]]:

    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights['W'].shape[0]
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    N = np.dot(X_batch, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return loss, forward_info


def loss_gradients(forward_info: Dict[str, np.ndarray],
                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  
    batch_size = forward_info['X'].shape[0]

    dLdP = -2 * (forward_info['y'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1, 0))
      
    dLdW = np.dot(dNdW, dLdN)
 
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients: Dict[str, np.ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB

    return loss_gradients


Batch = Tuple[np.ndarray, np.ndarray]

def generate_batch(X: np.ndarray, 
                   y: np.ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    '''
    Generate batch from X and y, given a start position
    '''
    assert X.ndim == y.ndim == 2, \
    "X and Y must be 2 dimensional"

    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0] - start
    
    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]
    
    return X_batch, y_batch

def forward_loss(X: np.ndarray,
                 y: np.ndarray,
                 weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    '''
    Generate predictions and calculate loss for a step-by-step linear regression
    (used mostly during inference).
    '''
    N = np.dot(X, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y - P, 2))

    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y

    return forward_info, loss


def init_weights(n_in: int) -> Dict[str, np.ndarray]:
    
    weights: Dict[str, np.ndarray] = {}
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)
    
    weights['W'] = W
    weights['B'] = B

    return weights

def permute_data(X: np.ndarray, y: np.ndarray):
    '''
    Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def train(X: np.ndarray, 
          y: np.ndarray, 
          n_iter: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = 100,
          return_losses: bool = False, 
          return_weights: bool = False, 
          seed: int = 1) -> None:
    '''
    Train model for a certain number of epochs.
    '''
    if seed:
        np.random.seed(seed)
    start = 0

    # Initialize weights
    weights = init_weights(X.shape[1])

    # Permute data
    X, y = permute_data(X, y)
    
    if return_losses:
        losses = []

    for i in range(n_iter):

        # Generate batch
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0
        
        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size
    
        # Train net using generated batch
        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights
    
    return None

def mae(preds: np.ndarray, actuals: np.ndarray):
    '''
    Compute mean absolute error.
    '''
    return np.mean(np.abs(preds - actuals))

def rmse(preds: np.ndarray, actuals: np.ndarray):
    '''
    Compute root mean squared error.
    '''
    return np.sqrt(np.mean(np.power(preds - actuals, 2)))


def predict(X: np.ndarray,
            weights: Dict[str, np.ndarray]):
    '''
    Generate predictions from the step-by-step linear regression model.
    '''

    N = np.dot(X, weights['W'])

    return N + weights['B']