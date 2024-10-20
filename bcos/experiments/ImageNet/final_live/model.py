from functools import partial

import bcos.models.densenet as densenet
import bcos.models.resnet as resnet
import bcos.models.vit as vit

from bcos.modules.bcosconv2d import BcosConv2d
from bcos.modules.bcoslinear import BcosLinear
from bcos.modules.common import BcosSequential
from bcos.modules.logitlayer import LogitLayer

import bcos.models.standard_nets as standard_nets


__all__ = ["get_model"]


def get_arch_builder(arch_name: str): # ONLY FOR B-COS MODELS
    arch_builder = None

    if arch_name.startswith("resne"):
        arch_builder = getattr(resnet, arch_name)
    elif arch_name.startswith("densenet"):
        arch_builder = getattr(densenet, arch_name)
    elif arch_name.startswith(('vit', 'simple_vit')):
        arch_builder = getattr(vit, arch_name)

    assert arch_builder is not None
    return arch_builder


def get_model(model_config):
    is_bcos = model_config["is_bcos"]
    assert is_bcos or model_config["name"].startswith('std_'), 'name and config don\'t match!'
    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]

    if is_bcos:
        bcos_args = model_config["bcos_args"]
        assert "norm_layer" in args, "norm_layer is required!"
        
        if 'vit' in arch_name:
            # B-Cos ViTs
            if "linear_layer" not in args or "conv2d_layer" not in args:
                args["linear_layer"] = partial(BcosLinear, **bcos_args)
                args["conv2d_layer"] = partial(BcosConv2d, **bcos_args)

            model = get_arch_builder(arch_name)(**args)
            logit_bias = model_config["logit_bias"]
            model = BcosSequential(model, LogitLayer(logit_bias=logit_bias)) # Logit Bias manually added
        else:
            # B-Cos CNNs
            if "conv_layer" not in args:
                args["conv_layer"] = partial(BcosConv2d, **bcos_args)

            model = get_arch_builder(arch_name)(**args) # Logit Bias embedded in the model
    else:
        # Standard CNNS
        if arch_name.startswith('std_resne'):
            model = getattr(standard_nets, arch_name)(**args)
        elif arch_name.startswith('bit_M_std_resne'):
            model = getattr(bit_resnets, arch_name)(**args)
        else:
            raise NotImplementedError
    return model
