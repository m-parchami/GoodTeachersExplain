from typing import Any, Callable, List, Optional, Type, Union, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch import Tensor
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.densenet import DenseNet

class MyBasicBlock(BasicBlock): 
    """
    everything the same, except that it also returns pre_relu activations and self.relu is no longer inplace
    """
    def __init__(self, *args, **kwargs):
        super(MyBasicBlock, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False) # was True in torchvision
        self.baseline_mode = False
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        pre_relu = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        pre_relu += identity
        out = self.relu(pre_relu)
        if self.baseline_mode:
            return out, pre_relu
        else:
            return out

class MyBottleneck(Bottleneck):
    """
    everything the same, except that it also returns pre_relu activations and self.relu is no longer inplace
    """
    def __init__(self, *args, **kwargs) -> None:
        super(MyBottleneck, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False) # was True in torchvision
        self.baseline_mode = False

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)
        pre_relu = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        pre_relu += identity
        out = self.relu(pre_relu)

        if self.baseline_mode:
            return out, pre_relu
        else:
            return out

class MyStandardResnet(ResNet):
    def __init__(self, *args, **kwargs,) -> None:
        super(MyStandardResnet, self).__init__(*args, **kwargs)
        self.also_return_features = False
        self.baseline_mode = False
        self.catkd_mode = False
        self.no_grad_features = False

    def convert_to_catkd(self):
        conv_fc = nn.Conv2d(self.fc.in_features, self.fc.out_features, kernel_size=1, bias=self.fc.bias is not None)
        conv_fc.weight = torch.nn.Parameter(self.fc.weight.view(self.fc.out_features,self.fc.in_features, 1, 1))
        conv_fc.bias = torch.nn.Parameter(self.fc.bias)
        self.fc = conv_fc
        self.catkd_mode = True

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.also_return_features or not (self.baseline_mode or self.catkd_mode):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            features = self.layer4(x) # Main Diff from PyTorch


            if self.catkd_mode:
                cams = self.fc(features)
                avg = self.avgpool(cams)
                x = torch.flatten(avg, 1)
            else:
                x = self.avgpool(features)
                x = torch.flatten(x, 1)
                x = self.fc(x)

            # Main Diff from Pytorch
            return x if not self.also_return_features else dict(output=x, features=features)
        else:
            with torch.no_grad() if self.no_grad_features else nullcontext():
                preact_feats = []
                self.layer1[-1].baseline_mode=True; self.layer2[-1].baseline_mode=True
                self.layer3[-1].baseline_mode=True; self.layer4[-1].baseline_mode=True

                x = self.conv1(x)
                x = self.bn1(x)
                preact_feats += [x] #Stem
                x = self.relu(x)
                x = self.maxpool(x)

                x, pre_relu = self.layer1(x)
                preact_feats += [pre_relu]
                x, pre_relu = self.layer2(x)
                preact_feats += [pre_relu]
                x, pre_relu = self.layer3(x)
                preact_feats += [pre_relu]
                features, pre_relu = self.layer4(x) # Main Diff from PyTorch
                preact_feats += [pre_relu]

                self.layer1[-1].baseline_mode=False; self.layer2[-1].baseline_mode=False
                self.layer3[-1].baseline_mode=False; self.layer4[-1].baseline_mode=False
            
            if self.no_grad_features:
                features.requires_grad = True

            if self.catkd_mode:
                cams = self.fc(features)
                avg = self.avgpool(cams)
                out = torch.flatten(avg, 1)
                return dict(output=out, features=dict(cams=cams))
            else:
                x = self.avgpool(features)
                avg = torch.flatten(x, 1)
                out = self.fc(avg)

                return dict(
                    output=out,
                    features=dict(feats=[F.relu(f) for f in preact_feats],
                    preact_feats=preact_feats, pooled_feat=avg),
                    gcam_feats=features)

def _resnet(
    block,
    layers: List[int],
    **kwargs: Any,
) -> MyStandardResnet:

    # model = Resnet(block, layers, **kwargs) # Main Difference from PyTorch
    model = MyStandardResnet(block, layers, **kwargs)

    return model


# Different names than PyTorch (to avoid overlap with B-cos models)
def std_resnet18(*, weights= None, progress=True, **kwargs: Any) -> MyStandardResnet:
    return _resnet(MyBasicBlock, [2, 2, 2, 2], **kwargs)

def std_resnet34(*, weights=None, progress=True, **kwargs: Any) -> MyStandardResnet:
    return _resnet(MyBasicBlock, [3, 4, 6, 3], **kwargs)

def std_resnet50(*, weights=None, progress=True, **kwargs: Any) -> MyStandardResnet:
    return _resnet(MyBottleneck, [3, 4, 6, 3], **kwargs)


class MyStandardDenseNet(DenseNet):
    def __init__(self, *args, **kwargs) -> None:
        super(MyStandardDenseNet, self).__init__(*args, **kwargs)
        self.also_return_features = False
        self.baseline_mode = False
        self.catkd_mode = False

    def convert_to_catkd(self):
        conv_classifier = nn.Conv2d(self.classifier.in_features, self.classifier.out_features, kernel_size=1, bias=self.classifier.bias is not None)
        conv_classifier.weight = torch.nn.Parameter(self.classifier.weight.view(self.classifier.out_features,self.classifier.in_features, 1, 1))
        conv_classifier.bias = torch.nn.Parameter(self.classifier.bias)
        self.classifier = conv_classifier
        self.catkd_mode = True

    def forward(self, x: Tensor) -> Tensor:
        if self.also_return_features or not (self.baseline_mode or self.catkd_mode):
            features = self.features(x)

            # Main Diff from PyTorch
            # out = F.relu(features, inplace=True)
            # out = F.adaptive_avg_pool2d(out, (1, 1))

            features = F.relu(features)
            if self.catkd_mode:
                cams = self.classifier(features)
                out = F.adaptive_avg_pool2d(cams, (1, 1))
                out = torch.flatten(out, 1)
            else:
                out = F.adaptive_avg_pool2d(features, (1, 1))
                out = torch.flatten(out, 1)
                out = self.classifier(out)
            return out if not self.also_return_features else dict(output=out, features=features)
        else:
            preact_feats = [] # [stem and after every denseblock]
            for name, module in self.features.named_children():
                if name == 'pool0':
                    preact_feats += [x] # Stem
                x = module(x)
                if name.startswith('transition') or name == 'denseblock4':
                    preact_feats += [x]

            features = F.relu(x)

            if self.catkd_mode:
                cams = self.classifier(features)
                out = F.adaptive_avg_pool2d(cams, (1, 1))
                out = torch.flatten(out, 1)
                return dict(output=out, features=dict(cams=cams))
            else:
                out = F.adaptive_avg_pool2d(features, (1, 1))
                avg = torch.flatten(out, 1)
                out = self.classifier(avg)
                return dict(output=out, features=dict(feats=[F.relu(f) for f in preact_feats], preact_feats=preact_feats, pooled_feat=avg))
        
    def _transform_state_dict(self, state_dict) -> None:
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return state_dict

def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    **kwargs: Any,
) -> MyStandardDenseNet:

    # Main Diff from PyTorch
    model = MyStandardDenseNet(growth_rate, block_config, num_init_features, **kwargs)

    return model

def std_densenet121(*, weights=None, progress: bool = True, **kwargs: Any) -> MyStandardDenseNet:
    return _densenet(32, (6, 12, 24, 16), 64, **kwargs)

def std_densenet161(*, weights=None, progress: bool = True, **kwargs: Any) -> MyStandardDenseNet:
    return _densenet(48, (6, 12, 36, 24), 96, **kwargs)

def std_densenet169(*, weights=None, progress=True, **kwargs: Any) -> MyStandardDenseNet:
    return _densenet(32, (6, 12, 32, 32), 64, **kwargs)