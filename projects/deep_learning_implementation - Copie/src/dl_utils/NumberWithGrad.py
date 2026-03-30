from typing import List, Union

Numberable = Union[float, int]

def ensure_number(num: Numberable) -> 'NumberWithGrad':
    if isinstance(num, NumberWithGrad):
        return num
    else:
        return NumberWithGrad(num)

"""
This class  is used for automatic 
differentiation but 
can be replaced by torch tensor
a = torch.tensor(3.0, requires_grad=True)
b = a * 4
c = b + 3
c.backward()
print(a.grad)
"""
class NumberWithGrad:

    def __init__(self,
                 num: Numberable,
                 depends_on: List[Numberable] = None,
                 creation_op: str = ''):
        self.num = num

        #the grad argument will actually contain
        #the gradient of the final output (end of the chain dependency)
        #w.r.t to this number
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op

    def __add__(self, other: Numberable) -> 'NumberWithGrad':
        return NumberWithGrad(self.num + ensure_number(other).num,
                              depends_on=[self, ensure_number(other)],
                              creation_op='add')

    def __mul__(self, other: Numberable = None) -> 'NumberWithGrad':
        return NumberWithGrad(self.num * ensure_number(other).num,
                              depends_on=[self, ensure_number(other)],
                              creation_op='mul')

    def backward(self, backward_grad: Numberable = None) -> None:

        # Computes the gradient of the final output with respect to this number
    # and propagates it back to all numbers that were used to create this one.
    # Each number accumulates its gradient so that reused variables
    # receive contributions from all paths they appear in.
        if backward_grad is None:
            self.grad = 1
        else:
            if self.grad is None:
                self.grad = backward_grad
            else:
                self.grad += backward_grad

        if self.creation_op == 'add':
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)

        if self.creation_op == 'mul':
            new = self.depends_on[1].num * self.grad
            self.depends_on[0].backward(new)
            new = self.depends_on[0].num * self.grad
            self.depends_on[1].backward(new)

