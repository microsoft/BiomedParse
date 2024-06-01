# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import random

import numpy as np
import torch

from transformers import AutoTokenizer, LlamaForCausalLM

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.structures.boxes import pairwise_iou
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog
from pycocotools import mask as coco_mask

from utils import prompt_engineering
from modeling.language import build_tokenizer
from modeling.language.misc import text_noun_with_prompt_all
from modeling.utils import configurable

from ..visual_sampler.sampler import build_shape_sampler

__all__ = ["BioMedDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []
    
    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])
    
    return augmentation

def build_transform_gen_se(cfg, is_train):
    # min_scale = cfg['INPUT']['MIN_SIZE_TEST']
    # max_scale = cfg['INPUT']['MAX_SIZE_TEST']

    augmentation = []
    # augmentation.extend([
    #     T.ResizeShortestEdge(
    #         min_scale, max_size=max_scale
    #     ),
    # ])    
    return augmentation

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

# This is specifically designed for the COCO dataset.
class BioMedDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        caption_thres,
        grounding,
        lvis,
        lvis_thres,
        max_grounding_num,
        shape_sampler,
        retrieval,
        max_token_num,
        tokenizer,
        binary_classes: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.caption_thres = caption_thres
        self.grounding = grounding
        self.lvis = lvis
        self.lvis_thres = lvis_thres
        self.max_grounding_num = max_grounding_num

        self.shape_sampler = shape_sampler

        self.retrieval = retrieval
        self.tokenizer = tokenizer
        self.max_token_num = max_token_num

        self.binary_classes = binary_classes

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            tfm_gens = build_transform_gen(cfg, is_train)
        else:
            tfm_gens = build_transform_gen_se(cfg, is_train)
            
        shape_sampler = build_shape_sampler(cfg)

        retrieval = cfg['MODEL']['DECODER']['RETRIEVAL']['ENABLED']
        tokenizer, max_token_num = None, None
        if retrieval:
            lang_model = cfg['MODEL']['TEXT']['NAME']
            max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
            if 'llama' in lang_model:
                tokenizer = AutoTokenizer.from_pretrained(lang_model, padding_side='right')
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "caption_thres": cfg['MODEL']['DECODER']['CAPTION']['SIM_THRES'],
            "grounding": cfg['MODEL']['DECODER']['GROUNDING']['ENABLED'],
            "lvis": cfg['MODEL']['DECODER']['LVIS']['ENABLED'],
            "lvis_thres": cfg['MODEL']['DECODER']['LVIS']['THRES'],
            "max_grounding_num": cfg['MODEL']['DECODER']['GROUNDING']['MAX_LEN'],
            "shape_sampler": shape_sampler,
            "retrieval": retrieval,
            "max_token_num": max_token_num,
            "tokenizer": tokenizer,
            "binary_classes": cfg['MODEL']['ENCODER']['BINARY_CLASSES']
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        while True:
            try:
                image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
                break
            except:
                print('Image loading error:', dataset_dict["file_name"])

        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))


        grounding_anno = dataset_dict['grounding_info']
        if len(grounding_anno) == 0:
            print(dataset_dict['file_name'])
        assert len(grounding_anno) > 0
        masks_grd = []
        texts_grd = []
        boxes_grd = []
        hash_grd = []
        classes = []
        masks_orig = []
        for ann in grounding_anno:
            if 'segmentation' in ann:
                if len(ann['segmentation']) == 0:
                    print('Empty segmentation!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    continue
                rle = coco_mask.frPyObjects(
                    ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
                m = coco_mask.decode(rle)
                masks_orig.append(m)
                # sometimes there are multiple binary map (corresponding to multiple segs)
                m = np.sum(m, axis=2)
            else:
                # directly read from mask file
                while True:
                    try:
                        m = utils.read_image(ann["mask_file"], format=self.img_format)
                        break
                    except:
                        print('Image loading error:', ann["mask_file"])
                m = np.sum(m, axis=2)
                m = 1 * (m > 0)
            m = m.astype(np.uint8)  # convert to np.uint8
            m = transforms.apply_segmentation(255*m[:,:,None])[:,:,0]
            masks_grd += [m]
            rand_id = random.randint(0, len(ann['sentences'])-1)
            texts_grd.append(ann['sentences'][rand_id]['raw'].lower())
            hash_grd.append(hash(ann['sentences'][rand_id]['raw'].lower()))
            if self.binary_classes:
                ann["category_id"] = 1 * (ann["category_id"] > 0)
            classes.append(ann["category_id"])
        #masks_grd = torch.from_numpy(np.stack(masks_grd))
        boxes_grd = torch.tensor(boxes_grd)
        groundings = {'masks': masks_grd, 'texts': texts_grd, 'hash': hash_grd, 'mode': 'text'}
        dataset_dict["groundings"] = groundings

        masks_grd = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks_grd])

        instances = Instances(image_shape)

        instances.gt_masks = BitMasks(masks_grd)
        instances.gt_boxes = BitMasks(masks_grd).get_bounding_boxes()

        classes = np.array(classes)
        is_things = np.array([1 for _ in classes])
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.is_things = torch.tensor(is_things, dtype=torch.int64)

        dataset_dict["instances"] = instances

        
        spatial_query_utils = self.shape_sampler(instances)
        dataset_dict['spatial_query'] = spatial_query_utils

        if self.retrieval:
            captions = dataset_dict['captions']
            tokens = self.tokenizer(
                captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

        if self.grounding:
            grounding_anno = dataset_dict['grounding_info']
            grounding_len = random.randint(1, self.max_grounding_num-1)
            if len(grounding_anno) > 0:
                masks_grd = []
                texts_grd = []
                mode = 'text'
                random.shuffle(grounding_anno)
                for ann in grounding_anno:
                    if 'segmentation' in ann:
                        if len(ann['segmentation']) == 0:
                            print('Empty segmentation!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            continue
                        rle = coco_mask.frPyObjects(
                            ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
                        m = coco_mask.decode(rle)
                        # sometimes there are multiple binary map (corresponding to multiple segs)
                        m = np.sum(m, axis=2)
                    else:
                        # directly read from mask file
                        while True:
                            try:
                                m = utils.read_image(ann["mask_file"], format=self.img_format)
                                break
                            except:
                                print('Image loading error:', ann["mask_file"])
                        m = np.sum(m, axis=2)
                        m = 1 * (m > 0)

                    m = m.astype(np.uint8)  # convert to np.uint8
                    m = transforms.apply_segmentation(m[:,:,None])[:,:,0]
                    masks_grd += [m]
                    # random select a sentence of a single annotation.
                    rand_index = random.randint(0, len(ann['sentences'])-1)
                    texts_grd += [ann['sentences'][rand_index]['raw'].lower()]
                # max_len = min(grounding_len, len(texts_grd))
                max_len = len(masks_grd)
                indices = np.random.permutation(max_len)
                texts_grd = list(np.array(texts_grd)[indices])
                masks_grd = torch.tensor(np.stack(masks_grd)[indices])
                hash_grd = np.array([hash(txt) for txt in texts_grd])
            else:
                masks_grd = instances.gt_masks.tensor
                mode = 'class'
                if len(masks_grd) == 0:
                    masks_grd = torch.tensor([])
                    texts_grd = ['none']
                    hash_grd = np.array([hash(txt) for txt in texts_grd])
                else:
                    biomed_classes = ['liver', 'lung', 'kidney', 'pancreas', 'heart anatomies', 'brain anatomies', 
                                      'eye anatomies', 'vessel', 'other organ', 'tumor', 'infection', 'other lesion', 
                                      'fluid disturbance', 'other abnormality', 'histology structure', 'other']
                    if self.binary_classes:
                        biomed_classes = ['target']
                    texts_grd = np.array(biomed_classes)
                    hash_grd = np.array([hash(txt) for txt in texts_grd])
                    unique_hash_grd = np.unique(hash_grd)
                    np.random.shuffle(unique_hash_grd)
                    max_len = min(grounding_len, len(unique_hash_grd))
                    indices = np.random.permutation(max_len)                    
                    selected_unique_hash_grd = unique_hash_grd[indices]
                    selected_mask = np.in1d(hash_grd, selected_unique_hash_grd)
                    texts_grd = texts_grd[selected_mask]
                    hash_grd = hash_grd[selected_mask]
                    masks_grd = masks_grd[selected_mask]
                    texts_grd = [prompt_engineering(text.replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                                        for text in texts_grd]
            groundings = {'masks': masks_grd, 'texts': texts_grd, 'mode': mode, 'hash': hash_grd}
            dataset_dict["groundings"] = groundings
            assert len(masks_grd) == len(dataset_dict['grounding_info']), f"len(masks_grd)={len(masks_grd)}, len(dataset_dict['grounding_info'])={len(dataset_dict['grounding_info'])}, mask shape={masks_grd.shape}, max_len={max_len}, grounding_len={grounding_len}, len(texts_grd)={len(texts_grd)}, len(hash_grd)={len(hash_grd)}"
        # gt_masks_orisize = torch.stack([torch.from_numpy(m.squeeze(-1)) for m in masks_orig])
        # dataset_dict['gt_masks_orisize'] = gt_masks_orisize # (nm,h,w)

        return dataset_dict
