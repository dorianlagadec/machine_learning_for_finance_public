import numpy as np
from typing import List
from .Operation import *

class Layer(object):
    '''
    A "layer" of neurons in a neural network.
    '''

    def __init__(self,
                 neurons: int):
        '''
        The number of "neurons" roughly corresponds to the "breadth" of the layer
        '''
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []


    def _setup_layer(self, num_in: int) -> None:
        '''
        The _setup_layer function must be implemented for each layer
        '''
        raise NotImplementedError()

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        '''
        Passes input forward through a series of operations
        '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_, inference)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Passes output_grad backward through a series of operations
        Checks appropriate shapes
        '''

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        
        self._param_grads()

        return input_grad

    def _param_grads(self) -> np.ndarray:
        '''
        Extracts the _param_grads from a layer's operations
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> np.ndarray:
        '''
        Extracts the _params from a layer's operations
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

    def _get_activation_function_name(self) -> str:
        return self.operations[-1].__class__.__name__
    


class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid(),
                 weight_init: str = 'standard',
                 dropout: float = 1.0):
        '''
        Requires an activation function upon initialization.
        weight_init options:
            'standard'      : N(0, 1) — no scaling, baseline
            'glorot_normal' : N(0, sqrt(2 / (n_in + n_out))) — for Sigmoid, Tanh, Linear
            'glorot_uniform': U(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))) — for Sigmoid, Tanh
            'he_normal'     : N(0, sqrt(2 / n_in)) — for ReLU
            'he_uniform'    : U(-sqrt(6 / n_in), sqrt(6 / n_in)) — for ReLU
            'lecun_normal'  : N(0, sqrt(1 / n_in)) — for SELU
        dropout: keep_prob in (0, 1). 1.0 = no dropout (default).
        '''
        super().__init__(neurons)
        self.activation = activation
        self.weight_init = weight_init
        self.dropout = dropout

    def _setup_layer(self, input_: np.ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        n_in  = input_.shape[1]
        n_out = self.neurons

        if self.weight_init == 'standard':
            W = np.random.randn(n_in, n_out)

        elif self.weight_init == 'glorot_normal':
            std = np.sqrt(2 / (n_in + n_out))
            W = np.random.randn(n_in, n_out) * std

        elif self.weight_init == 'glorot_uniform':
            limit = np.sqrt(6 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, size=(n_in, n_out))

        elif self.weight_init == 'he_normal':
            std = np.sqrt(2 / n_in)
            W = np.random.randn(n_in, n_out) * std

        elif self.weight_init == 'he_uniform':
            limit = np.sqrt(6 / n_in)
            W = np.random.uniform(-limit, limit, size=(n_in, n_out))

        elif self.weight_init == 'lecun_normal':
            std = np.sqrt(1 / n_in)
            W = np.random.randn(n_in, n_out) * std

        else:
            raise ValueError(f"Unknown weight_init '{self.weight_init}'. "
                             f"Choose from: standard, glorot_normal, glorot_uniform, "
                             f"he_normal, he_uniform, lecun_normal.")

        self.params.append(W)
        self.params.append(np.zeros((1, n_out)))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        if self.dropout < 1.0:
            self.operations.append(Dropout(self.dropout))

        return None
    
    def _get_weight_init_method(self) -> str:
        return self.weight_init