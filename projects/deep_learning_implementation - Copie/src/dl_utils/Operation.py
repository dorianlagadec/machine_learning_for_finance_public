import numpy as np
import matplotlib.pyplot as plt

def assert_same_shape(array: np.ndarray,
                      array_grad: np.ndarray):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None


class Operation(object):
    '''
    Base class for an "operation" in a neural network.
    '''
    def __init__(self):
        pass

    def forward(self, input_: np.ndarray, inference: bool = False):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function.
        '''
        self.input_ = input_
        self.inference = inference

        self.output = self._output()

        return self.output


    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad


    def _output(self) -> np.ndarray:
        '''
        The _output method must be defined for each Operation
        '''
        raise NotImplementedError()


    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        The _input_grad method must be defined for each Operation
        '''
        raise NotImplementedError()
    

class ParamOperation(Operation):
    '''
    An Operation with parameters.
    '''

    def __init__(self, param: np.ndarray) -> np.ndarray:
        '''
        The ParamOperation method
        '''
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        '''

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Every subclass of ParamOperation must implement _param_grad.
        '''
        raise NotImplementedError()
    

class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for a neural network.
    '''

    def __init__(self, W: np.ndarray):
        '''
        Initialize Operation with self.param = W.
        '''
        super().__init__(W)

    def _output(self) -> np.ndarray:
        '''
        Compute output.
        '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Compute input gradient.
        '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: np.ndarray)  -> np.ndarray:
        '''
        Compute parameter gradient.
        '''        
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)
    


class BiasAdd(ParamOperation):
    '''
    Compute bias addition.
    '''

    def __init__(self,
                 B: np.ndarray):
        '''
        Initialize Operation with self.param = B.
        Check appropriate shape.
        '''
        assert B.shape[0] == 1
        
        super().__init__(B)

    def _output(self) -> np.ndarray:
        '''
        Compute output.
        '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Compute input gradient.
        '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Compute parameter gradient.
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
    


class Sigmoid(Operation):
    '''
    Sigmoid activation function.
    '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> np.ndarray:
        '''
        Compute output.
        '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Compute input gradient.
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
    
class ReLU(Operation):
      def __init__(self) -> None:
          super().__init__()

      def _output(self) -> np.ndarray:
          return np.maximum(0, self.input_)

      def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
          return output_grad * (self.input_ > 0)
      
class Linear(Operation):
    '''
    "Identity" activation function
    '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> np.ndarray:
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''Pass through'''
        return output_grad


class Tanh(Operation):
    '''
    Hyperbolic tangent activation function.
    output = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    '''

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * (1.0 - self.output ** 2)


class LeakyReLU(Operation):
    '''
    Leaky ReLU activation function.
    output = x if x > 0 else alpha * x
    Fixes the dying ReLU problem by allowing a small gradient for negative inputs.
    '''

    def __init__(self, alpha: float = 0.01) -> None:
        '''
        alpha: slope for negative inputs. Default 0.01.
        '''
        super().__init__()
        self.alpha = alpha

    def _output(self) -> np.ndarray:
        return np.where(self.input_ > 0, self.input_, self.alpha * self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * np.where(self.input_ > 0, 1.0, self.alpha)


class ELU(Operation):
    '''
    Exponential Linear Unit activation function.
    output = x if x > 0 else alpha * (exp(x) - 1)
    Smooth for negative inputs, which helps gradient flow.
    '''

    def __init__(self, alpha: float = 1.0) -> None:
        '''
        alpha: scale for negative inputs. Default 1.0.
        '''
        super().__init__()
        self.alpha = alpha

    def _output(self) -> np.ndarray:
        return np.where(self.input_ > 0, self.input_, self.alpha * (np.exp(self.input_) - 1))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * np.where(self.input_ > 0, 1.0, self.output + self.alpha)


class Swish(Operation):
    '''
    Swish activation function (also known as SiLU).
    output = x * sigmoid(x)
    Smooth, non-monotonic, empirically outperforms ReLU on deeper networks.
    '''

    def __init__(self) -> None:
        super().__init__()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _output(self) -> np.ndarray:
        self.sigmoid_ = self._sigmoid(self.input_)
        return self.input_ * self.sigmoid_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * (self.sigmoid_ + self.input_ * self.sigmoid_ * (1.0 - self.sigmoid_))



class Dropout(Operation):
    '''
    Dropout operation helps prevent the network from overfitting the training set.
    In training mode it randomly sets activations to zero with probability (1 - keep_prob).
    In inference mode it multiplies by keep_prob to match the expected magnitude seen during training.
    '''

    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self) -> np.ndarray:
        if self.inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask

def plot_activation(activation, name, x_range=(-5, 5), ax=None):
    """
    Plot an activation function and its derivative on the same axes.
    Uses the Operation's own forward/backward to compute both.
    """
    x = np.linspace(x_range[0], x_range[1], 500).reshape(-1, 1)

    # forward pass
    y = activation.forward(x)

    # backward pass: passing ones gives us the raw derivative
    dy = activation.backward(np.ones_like(y))

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y,  label=f'{name}',            linewidth=2)
    ax.plot(x, dy, label=f"{name}'",  linewidth=2, linestyle='--')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)


