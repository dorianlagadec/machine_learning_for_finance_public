import numpy as np

class Optimizer(object):
    def __init__(self, lr: float = 0.01, final_lr: float = None, decay_type: str = 'linear') -> None:
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:
        if self.final_lr is None:
            return
        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1.0 / (self.max_epochs - 1))
        elif self.decay_type == "linear":
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:
        if self.final_lr is None:
            return

        if self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch
        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

    def step(self) -> None:

        for param, param_grad in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()




#one update per sample
class SGD(Optimizer):
    '''
    Stochasitc gradient descent optimizer.
    '''    
    def __init__(self, lr: float = 0.01, final_lr: float = None, decay_type: str = 'linear') -> None:
        '''Pass'''
        super().__init__(lr)

    def step(self):
        '''
        For each parameter, adjust in the appropriate direction, with the magnitude of the adjustment 
        based on the learning rate.
        '''
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            param[:] -= self.lr * param_grad



class SGD_MOMENTUM(Optimizer):

    '''
    Stochastic gradient descent with momentum
    '''

    def __init__(self, lr: float = 0.01,final_lr: float = None, momentum = 0.9, decay_type: str = 'linear') ->None:
        super().__init__(lr)
        self.momentum = momentum
        self.first = True

    def step(self):
        '''
        If first iteration: intialize "velocities" for each param.
        Otherwise, simply apply _update_rule.
        '''
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False
        
        for param, param_grad, velocity in zip(self.net.params(), self.net.param_grads(), self.velocities):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

        
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += self.lr * kwargs["grad"]

        
        kwargs["param"] -= kwargs["velocity"]

