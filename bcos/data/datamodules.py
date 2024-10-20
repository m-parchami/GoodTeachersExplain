import os
import pickle as pkl
import random
import time
from os.path import join as ospj
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_info
except ImportError:
    raise ImportError(
        "Please install pytorch-lightning for using data modules: "
        "`pip install pytorch-lightning`"
    )

from collections import defaultdict, deque, namedtuple
from copy import deepcopy
from functools import partial

import pandas as pd
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder

import bcos.settings as settings

from .categories import IMAGENET_CATEGORIES
from .imagenet_classnames import folder_label_map as imn_folder_label_map
from .presets import break_preset
from .sampler import RASampler
from .transforms import (MyToTensor, RandomCutmix, RandomMixup, SplitAndGrid)
from .datasets import VOCDataset, WaterBirdsDataset, PreComputedDataset
__all__ = ["ImageNetDataModule", "CIFAR10DataModule", "ClassificationDataModule", 'PreComputedDataModule']


def get_cut(dataset, nshots, nb_classes, heldout_pkl=None):
    all_indices = [*range(0, len(dataset))]
    if heldout_pkl is not None:
        heldout = heldout_pkl
    else:
        heldout = find_fewshot_indices(dataset, nshots, nb_classes, 42)
    rank_zero_info('removing the indices!!!')
    cnt = 0
    for idx in sorted(heldout):
        all_indices.pop(idx - cnt)
        cnt += 1

    return all_indices, heldout 

def get_random_cut(dataset, cut_ratio):
    all_indices = [*range(0, len(dataset))]
    cut_idx = int(cut_ratio* len(all_indices))
    state = random.getstate()
    random.seed(42)
    random.shuffle(all_indices) #Always shuffle the same way.
    random.setstate(state)
    return all_indices[:cut_idx], all_indices[cut_idx:]

class ClassificationDataModule(pl.LightningDataModule):
    """Base class for data modules for classification tasks."""

    NUM_CLASSES: int = None
    """Number of classes in the dataset."""
    NUM_TRAIN_EXAMPLES: int = None
    """Number of training examples in the dataset. Need not be defined."""
    NUM_EVAL_EXAMPLES: int = None
    """Number of evaluation examples in the dataset. Need not be defined."""
    CATEGORIES: List[str] = None
    """List of categories in the dataset. Need not be defined."""

    # ===================================== [ Registry stuff ] ======================================
    __data_module_registry = {}
    """Registry of data modules."""

    def __init_subclass__(cls, **kwargs):
        # check that the class attributes are defined
        super().__init_subclass__(**kwargs)
        # assert cls.NUM_CLASSES is not None
        # rest don't need to be defined

        # get name and remove DataModule suffix
        name = cls.__name__
        # check if name matches XXXDataModule
        if not name.endswith("DataModule"):
            raise ValueError(
                f"Data module class name '{name}' does not end with 'DataModule'"
            )
        name = name[: -len("DataModule")]
        # check if name is already registered
        if name in cls.__data_module_registry:
            raise ValueError(f"Data module {name} already registered")
        # register the class in the registry
        cls.__data_module_registry[name] = cls

    @classmethod
    def registry(cls):
        """Returns the registry of data modules."""
        return cls.__data_module_registry

    # ===================================== [ Normal stuff ] ======================================
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.batch_size = config["batch_size"]
        self.eval_batch_size = config.get('eval_batch_size', self.batch_size)
        self.num_workers = config["num_workers"]

        self.train_dataset = None
        self.eval_dataset = None

        mixup_alpha = config.get("mixup_alpha", 0.0)
        cutmix_alpha = config.get("cutmix_alpha", 0.0)
        p_gridified = config.get("p_gridified", 0.0)
        self.train_collate_fn = self.get_train_collate_fn(
            mixup_alpha, cutmix_alpha, p_gridified
        )

    def train_dataloader(self, shuffle=True):
        if not shuffle:
            train_sampler = None
            do_shuffle = False
        else:
            train_sampler = self.get_train_sampler()
            do_shuffle = None if train_sampler is not None else True
        return data.DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=do_shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

    def val_dataloader(self, shuffle=False, drop_last=False):
        return data.DataLoader(
            self.eval_dataset,
            self.eval_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=True,
        )

    def test_dataloader(self, shuffle=False):
        return data.DataLoader(
            self.eval_dataset,
            self.eval_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )

    @classmethod
    def get_train_collate_fn(
        cls,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        p_gridified: float = 0.0,
    ):
        assert not (p_gridified and mixup_alpha), "For now, do not use both."

        collate_fn = None
        if p_gridified:
            gridify = SplitAndGrid(p_gridified, num_classes=cls.NUM_CLASSES)

            def collate_fn(batch):
                return gridify(*data.default_collate(batch))

            rank_zero_info(f"Gridify active for training with {p_gridified=}")

        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(
                RandomMixup(cls.NUM_CLASSES, p=1.0, alpha=mixup_alpha)
            )
            rank_zero_info(f"Mixup active for training with {mixup_alpha=}")
        if cutmix_alpha > 0.0:
            mixup_transforms.append(
                RandomCutmix(cls.NUM_CLASSES, p=1.0, alpha=cutmix_alpha)
            )
            rank_zero_info(f"Cutmix active for training with {cutmix_alpha=}")
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):  # noqa: F811
                return mixupcutmix(*data.default_collate(batch))

        return collate_fn

    def get_train_sampler(self):
        train_sampler = None

        # see https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/pytorch_lightning/trainer/connectors/data_connector.py#L336
        # and https://github.com/Lightning-AI/lightning/blob/612d43e5bf38ba73b4f372d64594c2f9a32e6d6a/src/lightning_lite/utilities/seed.py#L54
        seed = int(os.getenv("PL_GLOBAL_SEED", 0))
        ra_reps = self.config.get("ra_repetitions", None)
        if ra_reps is not None:
            rank_zero_info(f"Activating RASampler with {ra_reps=}")
            train_sampler = RASampler(
                self.train_dataset,
                shuffle=True,
                seed=seed,
                repetitions=ra_reps,
            )

        return train_sampler

    def get_target_name(self, class_idx):
        raise NotImplementedError # Should be implemented per datamodule.

class ImageNetDataModule(ClassificationDataModule):
    # from https://image-net.org/download.php
    NUM_CLASSES: int = 1000

    NUM_TRAIN_EXAMPLES: int = 1_281_167
    NUM_EVAL_EXAMPLES: int = 50_000

    CATEGORIES: List[str] = IMAGENET_CATEGORIES

    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_per_node = self.config.get("cache_dataset", None) == "shm"

    def prepare_data(self) -> None:
        cache_dataset = self.config.get("cache_dataset", None)
        if cache_dataset != "shm":
            return

        # print because we also want global non-zero rank's
        start = time.perf_counter()
        print("Caching dataset into SHM!...")
        from .caching import cache_tar_files_to_shm

        cache_tar_files_to_shm()
        end = time.perf_counter()
        print(f"Caching successful! Time taken {end - start:.2f}s")

    def setup(self, stage: str) -> None:
        SHMTMPDIR = settings.SHMTMPDIR
        IMAGENET_PATH = settings.IMAGENET_PATH if self.config.get('overwrite_imn_path', None) is None else self.config['overwrite_imn_path']
        if stage == "fit":
            cache_dataset = self.config.get("cache_dataset", None)
            rank_zero_info("Setting up ImageNet train dataset...")
            train_root = os.path.join(
                SHMTMPDIR if cache_dataset == "shm" else IMAGENET_PATH,
                "train",
            )

            # See if we should cut a val set from training data
            if self.config.get('val_split_at', None) is not None:
                ############### Maybe load the whole Dataset object from a PKL ###############
                if self.config.get('image_folder_dump_path', None) is not None:
                    with open(self.config['image_folder_dump_path'], 'rb') as f:
                        entire_train_dataset = pkl.load(f)
                else:
                    # Distinct the CRD Dataset
                    if self.config.get('use_crd_folder', False):
                        entire_train_dataset = CRDImageFolder(
                            root=train_root,
                            transform=None, # Don't pass it here!
                            is_sample=True, k = self.config['nce_k']
                        )
                    else:
                        entire_train_dataset = ImageFolder(
                            root=train_root,
                            transform=None, # Don't pass it here!
                        )
                    
                ############### Maybe load Val Split indices from a PKL ###############
                if self.config.get('val_split_pkl', None) is not None:
                    with open(self.config['val_split_pkl'], 'rb') as f:
                        heldout_pkl = pkl.load(f)
                else:
                    heldout_pkl = None

                train_indices, eval_indices = get_cut(entire_train_dataset, nshots=self.config['val_split_at'], nb_classes=1000, heldout_pkl=heldout_pkl)
                self.train_dataset = MySubset(entire_train_dataset,
                            indices=train_indices,
                            transform=self.config["train_transform"] 
                            )
                self.eval_dataset = MySubset(deepcopy(entire_train_dataset),
                            indices=eval_indices,
                            transform=self.config["test_transform"] 
                            )
                self.train_idx_trans = lambda idx: train_indices[idx]
                self.eval_idx_trans = lambda idx: eval_indices[idx]
                rank_zero_info(f'[Fit and Eval Setup] {len(self.train_dataset), len(self.eval_dataset)} for train and eval.')
                rank_zero_info(f'Eval indices hash is: {hash(tuple(sorted(eval_indices)))}')

                if self.config.get('use_crd_folder', False):
                    rank_zero_info(f"Now creating train sample indices for CRD datasets")
                    self.train_dataset.dataset.create_samples(heldout_indices=tuple(eval_indices))
                    # with open('IMN-CRDtrain-samples.pkl', 'rb') as f:
                    #     self.train_dataset.dataset.cls_positive, self.train_dataset.dataset.cls_negative = pkl.load(f)

                    rank_zero_info(f"Setting eval CRD datasets to be normal")
                    self.eval_dataset.dataset.is_sample = False
                    rank_zero_info(f"{self.train_dataset.dataset.is_sample=}, {self.eval_dataset.dataset.is_sample=}")

                assert len(self.train_dataset) + len(self.eval_dataset) == self.NUM_TRAIN_EXAMPLES
            else:
                # No eval_split
                if self.config.get('use_crd_folder', False):
                    entire_train_dataset = CRDImageFolder(
                        root=train_root,
                        transform=self.config["train_transform"],
                        is_sample=True,
                        k=self.config['nce_k']
                    )
                    entire_train_dataset.create_samples(heldout_indices=[])
                else:
                    entire_train_dataset = ImageFolder(
                        root=train_root,
                        transform=self.config["train_transform"],
                    )

                self.train_dataset = entire_train_dataset
                # set the eval below.
                if self.config.get('overwrite_imn_path', None) is None:
                    assert len(self.train_dataset) == self.NUM_TRAIN_EXAMPLES

            if cache_dataset == "onthefly":
                rank_zero_info("Trying to setup Bagua's cached dataset!")
                from .caching import CachedImageFolder

                self.train_dataset = CachedImageFolder(self.train_dataset)
                rank_zero_info("Successfully setup cached dataset!")

        if stage in ['fit', 'val'] and self.config.get('val_split_at', None) is not None:
            rank_zero_info("Not Setting up anything as val split is part of fit and should be done in fit setup!")
            return

        eval_stage = stage

        if stage == "fit":
            eval_stage = 'val'

        rank_zero_info(f"Setting up ImageNet {eval_stage} dataset FROM {ospj(IMAGENET_PATH, eval_stage)}...")
        self.eval_dataset = ImageFolder(
            root=os.path.join(IMAGENET_PATH, eval_stage),
            transform=self.config["test_transform"],
        )
        print(f'[Separate Eval Setup] transformation is set to {self.config["test_transform"]}')

    def get_target_name(self, class_idx):
        return imn_folder_label_map[self.eval_dataset.classes[class_idx]]

class VOCDataModule(ClassificationDataModule):
    NUM_CLASSES: int = 20

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.VOC_PATH
        if stage == "fit":
            if self.config.get('train_split_portion', None) is not None:
                    entire_train_data = VOCDataset(
                        root=DATA_ROOT,
                        image_set='train',
                        download=False,
                        # transform=self.config["train_transform"], # Don't pass it here!
                        year='2012',
                        also_annotation=self.config['also_annotation'],
                    )
                    train_indices, eval_indices = get_random_cut(entire_train_data, self.config.get('train_split_portion'))
                    self.train_dataset = MySubset(entire_train_data,
                                indices=train_indices,
                                transform=self.config["train_transform"] 
                                )
                    self.eval_dataset = MySubset(entire_train_data,
                                indices=eval_indices,
                                transform=self.config["test_transform"] 
                                )
                    self.train_idx_trans = lambda idx: train_indices[idx]
                    self.eval_idx_trans = lambda idx: eval_indices[idx]
                    rank_zero_info(f'[Fit and Eval Setup] {len(self.train_dataset), len(self.eval_dataset)} for train and eval.')
                    rank_zero_info(f'Eval indices hash is: {hash(tuple(sorted(eval_indices)))}')
                    return
            else:
                self.train_dataset = VOCDataset(
                    root=DATA_ROOT,
                    image_set='train',
                    transform=self.config["train_transform"],
                    download=False,
                    year='2012',
                    also_annotation=self.config['also_annotation'],
                )

        if stage in ['fit', 'val'] and self.config.get('train_split_portion', None) is not None:
            rank_zero_info("Not Setting up anything as val split is part of fit and should be done in fit setup!")
            return

        eval_stage = stage
        if stage == 'fit':
            eval_stage = 'val'

        self.eval_dataset = VOCDataset(
            root=DATA_ROOT,
            image_set=eval_stage,
            transform=self.config["test_transform"],
            download=False,
            year='2012',
            also_annotation=self.config['also_annotation'],
        )

class SUN397DataModule(ClassificationDataModule):
    NUM_CLASSES: int = 397

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.SUN397_PATH
        entire_dataset = torchvision.datasets.SUN397(
            root=DATA_ROOT,
            download=False,
        )
        if stage == "fit":
            if self.config.get('val_split_at', None) is not None:
                    train_indices, eval_indices = get_cut(entire_dataset, self.config.get('val_split_at'), nb_classes=self.NUM_CLASSES)
                    self.train_dataset = MySubset(entire_dataset,
                                indices=train_indices,
                                transform=self.config["train_transform"] 
                                )
                    self.eval_dataset = MySubset(entire_dataset,
                                indices=eval_indices,
                                transform=self.config["test_transform"] 
                                )
                    self.train_idx_trans = lambda idx: train_indices[idx]
                    self.eval_idx_trans = lambda idx: eval_indices[idx]
                    rank_zero_info(f'[Fit and Eval Setup] {len(self.train_dataset), len(self.eval_dataset)} for train and eval.')
                    rank_zero_info(f'Eval indices hash is: {hash(tuple(sorted(eval_indices)))}')
                    return
            else:
                self.train_dataset = torchvision.datasets.SUN397(
                    root=DATA_ROOT,
                    download=False,
                    transform=self.config["train_transform"]
                )
                self.eval_dataset = torchvision.datasets.SUN397(
                    root=DATA_ROOT,
                    download=False,
                    transform=self.config["test_transform"]
                )
                return

        return

class WaterBirdsDataModule(ClassificationDataModule):
    NUM_CLASSES: int = 2
    def __init__(self, config):
        super(WaterBirdsDataModule, self).__init__(config)

    def setup(self, stage: str) -> None:
        DATA_ROOT = settings.WATERBIRDS_PATH
        if stage == "fit":
            self.train_dataset = WaterBirdsDataset(
                image_set='train',
                root=DATA_ROOT,
                transform=self.config['train_transform'],
                target_transform=self.config.get('target_transform', None),
                target_name=self.config['target_name'],
                confounder_names=self.config['confounder_names'],
                reverse_problem=self.config['reverse_problem'],
            )

        self.eval_dataset = WaterBirdsDataset(
            image_set='test' if stage=='test' else 'val',
            root=DATA_ROOT,
            transform=self.config['test_transform'],
            target_transform=self.config.get('target_transform', None),
            target_name=self.config['target_name'],
            confounder_names=self.config['confounder_names'],
            reverse_problem=self.config['reverse_problem'],
        )

class DataFreeDataModule(ClassificationDataModule):
    
    def __init__(self, config):
        super(DataFreeDataModule, self).__init__(config)
        _train_dm_config = deepcopy(config)
        _train_dm_config.update(num_classes=config['train_num_classes'])
        self.train_dm = config['train_dm'](_train_dm_config)

        _eval_dm_config = deepcopy(config)
        _eval_dm_config.update(
            **{key.removeprefix('eval_dm_'): val for key, val in config.items() if key.startswith('eval_dm_')}
        )
        self.eval_dm = config['eval_dm'](_eval_dm_config)

    def setup(self, stage):
        if stage == 'fit':
            rank_zero_info('--++-- Setting up self.eval_dm for fit --++--')
            self.eval_dm.setup('fit')
            rank_zero_info('--++-- Setting up self.train_dm for fit--++--')
            self.train_dm.setup('fit')
            self.train_dataset = self.train_dm.train_dataset
            self.eval_dataset = self.eval_dm.eval_dataset
            rank_zero_info(f'--++-- Train Size is {len(self.train_dataset)} and Eval Size is {len(self.eval_dataset)}')
            return
        rank_zero_info('--++-- Not Doing anything as both datasets are set up at fit --++--')

class PreComputedDataModule(ClassificationDataModule):
    def __init__(self, orig_data_module, method):
        # Use the config from orig_data_module; but later change things
        super(PreComputedDataModule, self).__init__(orig_data_module.config)

        # Setup the orig DataModule and then store essential stuff from it.
        self.orig_dm = orig_data_module
    
        self.method = method
        self.get_target_name = orig_data_module.get_target_name #simply delegate


    def setup(self, stage: str) -> None:        
        if stage == "fit":
            self.orig_dm.setup(stage='fit')
            self.orig_train_dataset = self.orig_dm.train_dataset
            self.orig_train_transform = self.orig_dm.train_dataset.transform
            train_idx_trans = getattr(self.orig_dm, 'train_idx_trans', None)

            # Break the train transforms into two
            train_main_transform_list, train_data_pre_transform_list = break_preset(self.orig_train_transform)
            rank_zero_info(f"[PRECOMPUTED] For Training: Main Transform will be {train_main_transform_list} and Pre-Transform will be {train_data_pre_transform_list}")
           
            # Create new train dataset
            self.train_dataset = PreComputedDataset(
                orig_dataset=self.orig_train_dataset,
                precomputed_dir=ospj(self.config['precomputed_dir'], stage),
                method=self.method,
                primary_transform=torchvision.transforms.Compose(train_main_transform_list),
                data_pre_transform=torchvision.transforms.Compose(train_data_pre_transform_list),
                rescale_maps_to_img=self.config['rescale_maps_to_img'],
                teacher_intact=self.config.get('teacher_intact', False),
                packed=self.config.get('packed', True),
                idx_trans = train_idx_trans,
                return_attr_portion=self.config.get('return_attr_portion', False),
            )
            rank_zero_info(f'[PRECOMPUTED] The new PreComputed wrapped train datasets have length of {len(self.train_dataset)}')

        if self.config.get('val_split_at', None) is not None:
            # Then eval_dataset is also set up already within self.orig_dm
            eval_stage = 'fit' # For paths below
        else:
            # Create new eval dataset
            eval_stage = 'val' if stage == 'fit' else stage
            self.orig_dm.setup(stage=eval_stage)

        self.orig_eval_dataset = self.orig_dm.eval_dataset
        self.orig_eval_transform = self.orig_dm.eval_dataset.transform
        eval_idx_trans = getattr(self.orig_dm, 'eval_idx_trans', None)
        
        # Break eval transforms into two
        eval_main_transform_list, eval_data_pre_transform_list = break_preset(self.orig_eval_transform)
        rank_zero_info(f"[PRECOMPUTED] For Eval: Main Transform will be {eval_main_transform_list} and Pre-Transform will be {eval_data_pre_transform_list}")

        self.eval_dataset = PreComputedDataset(
            orig_dataset=self.orig_eval_dataset,
            precomputed_dir=ospj(self.config['precomputed_dir'], eval_stage),
            method=self.method,
            primary_transform=torchvision.transforms.Compose(eval_main_transform_list),
            data_pre_transform=torchvision.transforms.Compose(eval_data_pre_transform_list),
            rescale_maps_to_img=self.config['rescale_maps_to_img'],
            teacher_intact=self.config.get('teacher_intact', False),
            packed=self.config.get('packed', True),
            idx_trans=eval_idx_trans,
            return_attr_portion=self.config.get('return_attr_portion', False)
        )
        rank_zero_info(f'[PRECOMPUTED] The new PreComputed wrapped eval datasets have length of {len(self.eval_dataset)}')

class FewshotDataModule(ClassificationDataModule):
    def __init__(self, datamodule, train_samples_per_class, nb_classes, shot_seed):
        super(FewshotDataModule, self).__init__(datamodule.config)
        rank_zero_info(f"[FEW SHOT] Using only {train_samples_per_class} samples per class")
        self.full_datamodule = datamodule
        self.nshots = train_samples_per_class
        self.nb_classes = nb_classes
        self.shot_seed = shot_seed
        self.get_target_name = self.full_datamodule.get_target_name #simply delegate

    def setup(self, stage: str) -> None:        
        rank_zero_info(f"[FEW SHOT] Setting up inner datamodule!")
        self.full_datamodule.prepare_data()
        self.full_datamodule.setup(stage)
        if stage == "fit":
            train_dataset = self.full_datamodule.train_dataset
            if self.config.get('fewshot_indices_pkl', None) is not None:
                with open(self.config['fewshot_indices_pkl'], 'rb') as f:
                    few_shot_indices = pkl.load(f)
            else:
                few_shot_indices = find_fewshot_indices(train_dataset, self.nshots, self.nb_classes, self.shot_seed)

            rank_zero_info(f"[FEW SHOT] After choosing the subset, there are overall {len(few_shot_indices)} samples")
            rank_zero_info(f'[FEW SHOT] hash of sorted indices is: {hash(tuple(sorted(few_shot_indices)))}')

            self.train_dataset = data.Subset(train_dataset, indices=few_shot_indices)

        self.eval_dataset = self.full_datamodule.eval_dataset

class AddIndexIter(torch.utils.data.dataloader._SingleProcessDataLoaderIter):
    """
    Imported from google-research/big_transfer/bit_pytorch/fewshot.py
    """
    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = torch.utils.data._utils.pin_memory.pin_memory(data)
        return index, data

def find_indices_loader(loader, n_shots, n_classes):
    """
    Imported from google-research/big_transfer/bit_pytorch/fewshot.py
    """
    per_label_indices = defaultdict(partial(deque, maxlen=n_shots))

    for ibatch, (indices, item) in enumerate(AddIndexIter(loader)):
        if isinstance(item, dict):
            labels = item.pop('target')
        else:
            _, labels = item
        for idx, lbl in zip(indices, labels):
            per_label_indices[lbl.item()].append(idx)
        
            findings = sum(map(len, per_label_indices.values()))
            if findings == n_shots * n_classes:
                return per_label_indices
    raise RuntimeError("Unable to find enough examples!")

def find_fewshot_indices(dataset, n_shots, nb_classes, shot_seed):
    """
    Imported from google-research/big_transfer/bit_pytorch/fewshot.py
    """
    curr_state = torch.random.get_rng_state()

    torch.manual_seed(shot_seed)

    orig_transform = dataset.transform
    dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(1),
        MyToTensor()
    ])

    rank_zero_info('Finding Fewshot indices with the loader!')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0, drop_last=False)
    per_label_indices = find_indices_loader(loader, n_shots, nb_classes)
    all_indices = [i for indices in per_label_indices.values() for i in indices]
    random.shuffle(all_indices)

    dataset.transform = orig_transform

    torch.random.set_rng_state(curr_state)
    return all_indices

class MySubset(data.Subset):
    """
    Subset dataset with a few more things:
    - supporting ransform
    - delegate rest of attra custom t./methods to internal dataset.

    Note: Mainly required for splitting and then using different transforms on
          created `Subset`s. (Otherwise, it's overwritten b/c internal is same.)
    Note: only for supervised data of form (x, y)
    """
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform
        if hasattr(dataset, "transform") and dataset.transform is not None:
            rank_zero_info(f"Internal dataset has transform will apply transform on top: {dataset}")

    def __getitem__(self, item):
        ret = super().__getitem__(item)
        x, y = ret[0], ret[1]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return (x, y) if len(ret) == 2 else (x, y, *ret[2:]) #just pass-through the rest.

    def __getattr__(self, item):
        if item in ['transform', 'target_transform']:
            return self.__dict__[item]
        # not found in attr so look in internal dataset
        return getattr(self.dataset, item)
