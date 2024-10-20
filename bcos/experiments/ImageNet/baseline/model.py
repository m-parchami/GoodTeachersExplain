# from torchvision import models
import bcos.models.standard_nets as standard_nets

__all__ = ["get_model"]


def get_model(model_config):
    assert not model_config.get("is_bcos", False), "Should be false!"

    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]

    # create model
    model = getattr(standard_nets, arch_name)(**args)

    return model
