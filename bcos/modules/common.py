from torch import nn
import torch
from torch import Tensor
from bcos.common import BcosUtilMixin
from typing import Tuple, Optional

__all__ = ["DynamicMultiplication"]
class _DynamicMultiplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: "Tensor", input: "Tensor", state: "dict") -> "Tensor":
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.state = state
        ctx.save_for_backward(weight, input)
        return weight * input

    @staticmethod
    def backward(ctx, grad_output: "Tensor") -> "Tuple[Optional[Tensor], Tensor, None]":
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        weight, input = ctx.saved_tensors
        if ctx.state["fixed_weights"]:
            return None, grad_output * weight, None
        return grad_output * input, grad_output * weight, None


class DynamicMultiplication(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = {"fixed_weights": False}

    def set_explanation_mode(self, on: bool = True):
        self.state["fixed_weights"] = on

    @property
    def is_in_explanation_mode(self):
        # just for testing
        return self.state["fixed_weights"]

    def forward(self, *, weight: "Tensor", input: "Tensor") -> "Tensor":
        return _DynamicMultiplication.apply(weight, input, self.state)



class BcosSequential(BcosUtilMixin, nn.Sequential):
    """
    Wrapper for models which are nn.Sequential at the "root" module level.
    This only adds helper functionality from `BcosMixIn`.
    """

    def __init__(self, *args):
        super().__init__(*args)
