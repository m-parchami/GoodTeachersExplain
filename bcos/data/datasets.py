import torchvision
import torch
from torch.utils import data
import os
import pandas as pd
import numpy as np
try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import rank_zero_info
except ImportError:
    raise ImportError(
        "Please install pytorch-lightning for using data modules: "
        "`pip install pytorch-lightning`"
    )

from torchvision.transforms.functional import resize
from bcos.experiments.utils.config_utils import ExpMethod
from collections import namedtuple
from os.path import join as ospj

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from PIL import Image
class PreComputedDataset(data.Dataset):
    """
        A wrapper dataset class that wraps around an already implemented dataset, and loads
        pre-computed things (e.g. logits and explanations) together with the original dataset's items.
    """
    def __init__(self, 
            orig_dataset:data.Dataset, precomputed_dir: str, method: ExpMethod, 
            primary_transform, data_pre_transform,
            rescale_maps_to_img: bool=False, teacher_intact: bool=False, idx_trans=None,
            extra_precomputed_dirs=None):
        self.orig_dataset = orig_dataset
        self.orig_dataset.transform = data_pre_transform # Keep essential pre-transforms in the orig_dataset
        self.target_transform = None # Nothing other than what orig_dataset already does
        self.transform = primary_transform
        self.computed_path = precomputed_dir
        self.extra_computed_paths = extra_precomputed_dirs # For Ensemble
        self.computed_keys = ['output']

        self.extra_files_method = method
        if self.extra_files_method is not None:
            self.computed_keys += [self.extra_files_method.to_key()]

        self.rescale_maps_to_img = rescale_maps_to_img
        self.teacher_intact = teacher_intact
        self.data = None
        self.idx_trans = idx_trans

        self.preprocess = None
        self.postprocess = None

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        (image, target) = self.orig_dataset[idx]

        ## Transform the idx if necessary
        if self.idx_trans is not None:
            idx = self.idx_trans(idx)
        
        ## Load precomputed stuff
        computed_dict = torch.load(ospj(self.computed_path, f'{idx}.pt'))
        if 'weight' in computed_dict:
            computed_dict['weight'] = computed_dict['weight'].type(torch.float32).unsqueeze(0) #(B=1, C, H, W)


        if 'ouptut' in computed_dict: # When saving, I accidentally had saved them as "ouptut" ;)
            computed_dict['output'] = computed_dict.pop('ouptut').unsqueeze(0) #(B=1, Num_classes)

        # Apply the transform
        if self.transform:
            if self.extra_files_method is not None and not self.teacher_intact:
                method_key = self.extra_files_method.to_key()
                # Stack weight and image together and apply same transformation on both
                exp = computed_dict[method_key] # (B, C, H, W)
                b = exp.shape[0] # B for "Batch" here means multiple precomputed explanations (e.g. ensemble)
                if self.preprocess:
                    exp = self.preprocess(exp, image.shape[0])

                if self.rescale_maps_to_img:
                    exp = resize(exp, (image.shape[-2], image.shape[-1]))

                image = image.unsqueeze(0) #(1, C, H, W)
                stack = torch.vstack((exp, image)) # (B+1, C, H, W)
                if self.return_attr_portion:
                    # Don't use thee attr_map of this section for other purposes
                    # as it doesn't have resize/interpolation part!
                    transformed_stack, crop_info = self.transform(stack)
                    top, left, h, w = crop_info['top'], crop_info['left'], crop_info['h'], crop_info['w']
                    attr_map = (exp * image).sum(dim=-3) # Sum over C
                    attr_portion = attr_map[:, top:top+h, left:left+w].view(b, -1).sum(1) / attr_map.view(b, -1).sum(1) #(b, 1)
                else:
                    transformed_stack = self.transform(stack)

                exp = transformed_stack[:-1] # (B, C, H, W)
                image = transformed_stack[-1] # (C, H, W)

                if self.postprocess:
                    exp = self.postprocess(exp)
                computed_dict[method_key] = exp
            else:
                # Apply transformations on images only, as usual
                image = self.transform(image)
    
        # if self.target_transform:
        #     target = self.target_transform(target)

        ret_dict = dict(image=image, target=target, **computed_dict,)
        if self.return_attr_portion:
            ret_dict['attr_portion'] = attr_portion

        return ret_dict

class WaterBirdsDataset(torch.utils.data.Dataset):
    """
        Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
        and the Base ConfounderDataset class there.
        
        CUB dataset (already cropped and centered).
        Note: metadata_df is one-indexed.
    """

    def __init__(self, root,
            image_set='train', transform=None, 
            target_transform=None,
            target_name='waterbird_complete95',
            confounder_names=None,
            reverse_problem=False):
        super(WaterBirdsDataset, self).__init__()
        self.root_dir = root
        self.transform = transform
        self.target_name = target_name
        self.confounder_names = confounder_names or ['forest2water2']
        self.target_transform = target_transform
        self.reverse_problem=reverse_problem # Not looked into in this project :)

        self.data_dir = ospj(
            self.root_dir,
            'data',
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            ospj(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')
        self.ret_groups = False # can be enabled only manually afterwards.

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        assert image_set in ('train','val','test'), image_set+' is not a valid split'
        mask = self.split_array == self.split_dict[image_set]
        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        self.indices = indices
        self.also_return_groups = False # can be set from outside
        self.cub_dict, self.class_names = self.parse_cub()

    def parse_cub(self):
        cub_dir = 'CUB_200_2011' 
        images_path = ospj(self.root_dir, cub_dir, 'images.txt')
        bb_path = ospj(self.root_dir, cub_dir, 'bounding_boxes.txt')
        label_path = ospj(self.root_dir, cub_dir, 'image_class_labels.txt')
        cls_path = ospj(self.root_dir, cub_dir, 'classes.txt')
        with open(images_path, 'r') as img_file, open(bb_path, 'r') as bb_file,\
         open(cls_path, 'r') as cls_file, open(label_path, 'r') as label_file:
            # read the contents of each file into a list
            img_list = [line.strip() for line in img_file]
            bb_list = [line.strip() for line in bb_file]
            cls_list = [line.strip() for line in cls_file]
            label_list = [line.strip() for line in label_file]

        # create an empty dictionary to store the name map
        class_names = {}
        for cls_line in cls_list:
            label = int(cls_line.split(' ')[0])
            cls_name = cls_line.split(' ')[1] 
            assert label not in class_names
            class_names[label] = cls_name

        cub_dict = {}
        CubAnnotaiton = namedtuple("CubAnnotaiton", ['box', 'label'])

        for img_line, bb_line, label_line in zip(img_list, bb_list, label_list):
            img_name = img_line.split(' ')[1] # drop the image number
            label = int(label_line.split(' ')[1])

            bb_coords = bb_line.split(' ')[1:]
            xmin, ymin, width, height = [int(float(coord)) for coord in bb_coords]

            assert img_name not in cub_dict
            cub_dict[img_name] = CubAnnotaiton(
                box=[xmin, ymin, xmin+width, ymin+height],
                label=label) 

        return cub_dict, class_names

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = [self.indices[i] for i in idx]
        else:
            idx = self.indices[idx]
        
        # g [0 landbird/land, 1 landbid/water] -> y 0 land
        # g [2 waterbird/land, 3 waterbird/water] -> y 1 water
        y = int(self.y_array[idx])
        g = self.group_array[idx]

        if self.reverse_problem:
            # g [0 landbird/land, 2 waterbird/land]  -> y 0 land
            # g [1 landbid/water, 3 waterbird/water] -> y 1 water
            y = 0 if g in [0,2] else 1

        # Load the image
        img_filename = ospj(
            self.data_dir,
            self.filename_array[idx])
        sample = Image.open(img_filename).convert('RGB')

        # should be replaced by below:
        if self.transform is not None:
            sample = self.transform(sample)
     
        if self.target_transform is not None:
            y = self.target_transform(y)

        ret = sample, y

        if self.also_return_groups:
            ret = sample, y, g
        
        return  ret
    
    def __len__(self):
        return len(self.indices)

class VOCDataset(torchvision.datasets.VOCDetection):
    def __init__(self, *args, also_annotation=False, **kwargs):
        super(VOCDataset, self).__init__(*args, **kwargs)

        self.target_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,
                'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        self.reverse_target_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7:
                       'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 
                15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

        self.num_classes = 20
        self.also_annotation = also_annotation
        if self.also_annotation:
            rank_zero_info('Warning! you are asking VOC Dataset to return annotaitons, '+\
                    'but they are only valid if you only resize images to 224. No Crop/Flip etc. should be done!')
            
        self.transforms = None # Note that self.transform is still enabled.
        assert self.transforms is None, f'Not considered as of now!'

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        annotations = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        objects = annotations['annotation']['object']
        target = torch.zeros(self.num_classes)
        object_names = [item['name'] for item in objects]
        for name in object_names:
            target[self.target_dict[name]] = 1

        if self.transform is not None:
            img = self.transform(img)
        
        if self.also_annotation:
            # This only works for Resized images (no crop/flip etc.!!!)
            size = annotations['annotation']['size']
            width = int(size['width'])
            height = int(size['height'])
            wscale = 224 / width
            hscale = 224 / height

            object_bndboxes = [item['bndbox'] for item in objects]
            bbs = []
            for name, bndbox in zip(object_names, object_bndboxes):
                index = self.target_dict[name]
                xmin, xmax = int(bndbox['xmin']), int(bndbox['xmax'])
                ymin, ymax = int(bndbox['ymin']), int(bndbox['ymax'])

                new_xmin, new_xmax = int(min(max(xmin*wscale, 0), 223)), int(min(max(xmax*wscale, 0), 223))
                new_ymin, new_ymax = int(min(max(ymin*hscale, 0), 223)), int(min(max(ymax*hscale, 0), 223))

                bbs.append([index, new_xmin, new_ymin, new_xmax, new_ymax])
            return img, target, bbs
        else:
            return img, target

    

    
        # This applies transforms to both img and target (irrelevant for us!)
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

