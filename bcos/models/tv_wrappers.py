

import torch
from torch import nn, Tensor
from torchvision.models import MobileNetV2, EfficientNet, ShuffleNetV2, ConvNeXt
from torchvision.models.efficientnet import _efficientnet_conf
from torchvision.models.convnext import CNBlockConfig

from functools import partial

__all__ = ["MyMobileNetV2", "std_mobilenetv2", 'MyEfficientNet', 'std_efficientnetv2s']

class MyMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        super(MyMobileNetV2, self).__init__(*args, **kwargs)

        self.also_return_features = False
        self.baseline_mode = False
        self.catkd_mode = False

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        features = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.also_return_features:
            return dict(output=x, features=features)
        else:
            return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def std_mobilenetv2(*args, **kwargs):
    return MyMobileNetV2(*args, **kwargs)



class MyEfficientNet(EfficientNet):
    def __init__(self, *args, **kwargs):
        super(MyEfficientNet, self).__init__(*args, **kwargs)

        self.also_return_features = False
        self.baseline_mode = False
        self.catkd_mode = False

    def _forward_impl(self, x: Tensor) -> Tensor:
        features = self.features(x)

        x = self.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.also_return_features:
            return dict(output=x, features=features)
        else:
            return x

def std_efficientnetv2s(**kwargs):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return MyEfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=kwargs.pop("dropout", 0.2),
        last_channel=last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )

class MyShuffleNetV2(ShuffleNetV2):
    def __init__(self, *args, **kwargs):
        super(MyShuffleNetV2, self).__init__(*args, **kwargs)

        self.also_return_features = False
        self.baseline_mode = False
        self.catkd_mode = False

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        features = self.conv5(x)
        x = features.mean([2, 3])  # globalpool
        x = self.fc(x)

        if self.also_return_features:
            return dict(output=x, features=features)
        else:
            return x

def std_shufflenetv2_x0_5(**kwargs):
    return MyShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)



class MyConvNeXt(ConvNeXt):
    def __init__(self, *args, **kwargs):
        super(MyConvNeXt, self).__init__(*args, **kwargs)

        self.also_return_features = False
        self.baseline_mode = False
        self.catkd_mode = False

    def _forward_impl(self, x: Tensor) -> Tensor:
        features = self.features(x)
        x = self.avgpool(features)
        x = self.classifier(x)

        if self.also_return_features:
            return dict(output=x, features=features)
        else:
            return x

def std_convnext_tiny(**kwargs):
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    return MyConvNeXt(block_setting, stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.1), **kwargs)