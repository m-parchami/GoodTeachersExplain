"""
This module contains utilities related to B-cos models.
None of this is "essential" to training or doing inference with the models.
(Most of the stuff can be done quickly and easily in a few lines of code.)
However, they are useful for e.g. visualizing the explanations etc.
So essentially it's a collection of convenience/helper functions/classes.
"""
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    # this isn't supposed to be a hard dependency
    import matplotlib
    import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if torch.__version__ < "2.0":
    from torch.autograd.grad_mode import _DecoratorContextManager  # noqa
else:
    from torch.utils._contextlib import _DecoratorContextManager  # noqa

__all__ = [
    "BcosUtilMixin",
    "explanation_mode",
    "gradient_to_image",
]


TensorLike = Union[Tensor, np.ndarray]


class BcosUtilMixin:
    #TODO DOCUMENT THIS
    """
    This mixin defines useful helpers for dealing with explanations.
    This is just a convenience to attach useful B-cos specific functionality.

    The parameters to ``__init__`` are just passed to the actual base class (e.g. ``torch.nn.Module``).

    Notes
    -----
    Since this is a mixin, if you want to use this, you need to inherit from this first
    and then from the actual base class (e.g. `torch.nn.Module`).


    Examples
    --------
    >>> from bcos.modules import BcosConv2d
    >>> class MyModel(BcosUtilMixin, torch.nn.Module):
    ...     def __init__(self, in_chan: int, out_chan: int):
    ...         super().__init__()
    ...         self.linear = BcosConv2d(in_chan, out_chan, 3)
    ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...         return self.linear(x)
    >>> model = MyModel(6, 16)
    >>> model.explain(torch.rand(1, 6, 32, 32))  # get explain method
    >>> with model.explanation_mode():  # explanation mode ctx (assuming we have detachable modules)
    ...     ...  # do something with explanation mode activated

    Parameters
    ----------
    args: Any
        Positional arguments to pass to the parent class.
    kwargs: Any
        Keyword arguments to pass to the parent class.
    """

    to_probabilities = torch.sigmoid
    """ Function to convert model outputs to probabilties. """

    def __init__(self, *args: Any, **kwargs: Any):
        self.__explanation_mode_ctx = explanation_mode(self)  # type: ignore
        super().__init__(*args, **kwargs)

    def explanation_mode(self) -> "explanation_mode":
        """
        Returns a context manager which puts model in explanation mode
        and when exiting puts it in normal mode back again.

        Returns
        -------
        explanation_mode
            The context manager which puts model in and out to/from explanation mode.
        """
        return self.__explanation_mode_ctx


class explanation_mode(_DecoratorContextManager):
    #TODO DOCUMENT THIS
    """
    A context manager which activates and puts model in to explanation
    mode and deactivates it afterwards.
    Can also be used as a decorator.

    Parameters
    ----------
    model : nn.Module
        The model to put in explanation mode.
    """

    def __init__(self, model: "nn.Module"):
        self.model = model
        self.expl_modules = None

    def find_expl_modules(self) -> None:
        """Finds all modules which have a `set_explanation_mode` method."""
        self.expl_modules = [
            m for m in self.model.modules() if hasattr(m, "set_explanation_mode")
        ]

    def __enter__(self):
        """
        Put model in explanation mode.
        """
        if self.expl_modules is None:
            self.find_expl_modules()

        for m in self.expl_modules:
            m.set_explanation_mode(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Turn off explanation mode for model.
        """
        for m in self.expl_modules:
            m.set_explanation_mode(False)


def gradient_to_image(image, linear_mapping, smooth=0, alpha_percentile=99.5):
    #TODO DOCUMENT THIS
    """
    From https://github.com/moboehle/B-cos/blob/0023500ce/interpretability/utils.py#L41.
    Computing color image from dynamic linear mapping of B-cos models.

    Parameters
    ----------
    image: Tensor
        Original input image (encoded with 6 color channels)
        Shape: [C, H, W] with C=6
    linear_mapping: Tensor
        Linear mapping W_{1\rightarrow l} of the B-cos model
        Shape: [C, H, W] same as image
    smooth: int
        Kernel size for smoothing the alpha values
    alpha_percentile: float
        Cut-off percentile for the alpha value. In range [0, 100].

    Returns
    -------
    np.ndarray
        image explanation of the B-cos model.
        Shape: [H, W, C] (C=4 ie RGBA)
    """
    # shape of img and linmap is [C, H, W], summing over first dimension gives the contribution map per location
    contribs = (image * linear_mapping).sum(0, keepdim=True)  # [H, W]
    # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
    rgb_grad = linear_mapping / (
        linear_mapping.abs().max(0, keepdim=True).values + 1e-12
    )
    # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
    rgb_grad = rgb_grad.clamp(min=0)
    # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
    rgb_grad = rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12)  # [3, H, W]

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    # Only show positive contributions
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    alpha = (alpha / torch.quantile(alpha, q=alpha_percentile / 100)).clip(0, 1)

    rgb_grad = torch.concatenate([rgb_grad, alpha], dim=0)  # [4, H, W]
    # Reshaping to [H, W, C]
    grad_image = rgb_grad.permute(1, 2, 0)
    return grad_image.detach().cpu().numpy()

# ==============================================================================
# Other misc. functions
# ==============================================================================

def param_to_buffer(module):
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in list(module.named_parameters(recurse=False)):
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param)
    for module in modules:
        param_to_buffer(module)
