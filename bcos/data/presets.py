import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

import bcos.data.transforms as custom_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        is_bcos=False,
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(
                        interpolation=interpolation, magnitude=ra_magnitude
                    )
                )
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            elif auto_augment_policy == "augmix":
                trans.append(
                    autoaugment.AugMix(
                        interpolation=interpolation, severity=augmix_severity
                    )
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if not is_bcos:
            trans.append(transforms.Normalize(mean=mean, std=std))
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        if is_bcos:
            trans.append(custom_transforms.AddInverse())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        # https://rich.readthedocs.io/en/stable/pretty.html
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result

class BiTPresetEval:
    def __init__(
        self,
        *,
        crop_size,
    ):

        trans = [
            transforms.ToTensor(),
            transforms.Resize((crop_size, crop_size)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ImageNetClassificationWithInfoPresetTrain(ImageNetClassificationPresetTrain):
    """
    Similar to presetTrain above just that the RandomResizedCrop is more careful now and the crop 
    will not be too small.
    """
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        min_scale=0.2,
        is_bcos=False,
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = []
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        if is_bcos:
            trans.append(custom_transforms.AddInverse())
        else:
            trans.append(transforms.Normalize(mean=mean, std=std))

        # These transforms were put at the end instead of the beginning, assuming that the above transforms
        # don't matter when they are applied.
        trans += [custom_transforms.RandomResizedCropWithInfo(crop_size, interpolation=interpolation)]

        if hflip_prob > 0:
            trans.append(custom_transforms.RandomHorizontalFlipIgnoreInfo(hflip_prob))

        
        self.transforms = transforms.Compose(trans)

class ImageNetClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        is_bcos=False,
    ):
        self.args = get_args_dict(ignore=["mean", "std"])
        trans = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            custom_transforms.AddInverse()
            if is_bcos
            else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    @property
    def resize(self):
        return self.transforms.transforms[0]

    @property
    def center_crop(self):
        return self.transforms.transforms[1]

    def no_scale(self, img):
        x = img
        for t in self.transforms.transforms[2:]:
            x = t(x)
        return x

    # this is intended for when using the pretrained models
    def transform_with_options(self, img, center_crop=True, resize=True):
        x = img
        if resize:
            x = self.resize(x)
        if center_crop:
            x = self.center_crop(x)
        x = self.no_scale(x)
        return x

    def with_args(self, **kwargs):
        args = self.args.copy()
        args.update(kwargs)
        return self.__class__(**args)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        result = dict(
            transform=repr(self),
            **self.args,
        )
        result["interpolation"] = str(result["interpolation"])
        return result



class VOCClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
            custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

class VOCClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),

            transforms.Resize(256), transforms.CenterCrop(crop_size), # For simple classification evaluation
            # transforms.Resize((224, 224)), # For when annotations are being returned as well.

            custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

class WaterbirdsClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
            ),
            custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

class WaterbirdsClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        is_bcos=False,
    ):
        trans = [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            custom_transforms.AddInverse() if is_bcos else transforms.Normalize(mean=mean, std=std),
        ]

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transforms)})"

    def __rich_repr__(self):
        yield "transforms", self.transforms.transforms

def get_args_dict(ignore: "tuple | list" = tuple()):
    """Helper for saving args easily."""
    import inspect

    frame = inspect.currentframe().f_back
    av = inspect.getargvalues(frame)
    ignore = tuple(ignore) + ("self", "cls")
    return {arg: av.locals[arg] for arg in av.args if arg not in ignore}

def break_preset(preset):
    """
    This separates geometric and non-geometric transformations, in a very bad and limited way.
    You have to manually define pre_trans instances that are relevant.
    """
    data_pre_transform_list = []
    main_transform_list = []
    pre_trans_instances = (
        custom_transforms.AddInverse, transforms.Normalize, 
        transforms.ToTensor, transforms.PILToTensor,
        transforms.ConvertImageDtype
    )
    if preset is not None:
        orig_transforms = preset.transforms.transforms
        for trans in orig_transforms:
            if isinstance(trans, pre_trans_instances):
                data_pre_transform_list.append(trans)
            else:
                main_transform_list.append(trans)
    return main_transform_list, data_pre_transform_list