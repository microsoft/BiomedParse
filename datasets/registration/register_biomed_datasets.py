# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_BIOMED = {}

datasets = ['ACDC', 'AbdomenCT-1K', 'BreastCancerCellSegmentation', 'BreastUS', 'CAMUS', 'CDD-CESM', 'COVID-QU-Ex', 
            'CXR_Masks_and_Labels', 'FH-PS-AOP', 'G1020', 'GlaS', 'ISIC', 'LGG', 'LIDC-IDRI', 'MMs', 
            'MSD/Task01_BrainTumour', 'MSD/Task02_Heart', 'MSD/Task03_Liver', 'MSD/Task04_Hippocampus', 
            'MSD/Task05_Prostate', 'MSD/Task06_Lung', 'MSD/Task07_Pancreas', 'MSD/Task08_HepaticVessel', 
            'MSD/Task09_Spleen', 'MSD/Task10_Colon', 'NeoPolyp', 'OCT-CME', 'PROMISE12', 'PolypGen', 'QaTa-COV19', 
            'REFUGE', 'Radiography/COVID', 'Radiography/Lung_Opacity', 'Radiography/Normal', 'Radiography/Viral_Pneumonia', 
            'Totalsegmentator2/abdomen', 'Totalsegmentator2/chest', 'UWaterlooSkinCancer', 'amos22/CT', 'amos22/MRI', 
            'kits23', 'siim-acr-pneumothorax', 'PanNuke', 'LiverUS', 'BTCV-Cervix', 'COVID-19_CT', 'DRIVE']

for name in datasets:
    for split in ['train', 'test']:
        dataname = f'biomed_{name.replace("/", "-")}_{split}'
        image_root = f"BiomedSeg/{name}/{split}"
        ann_root = f"BiomedSeg/{name}/{split}.json"
        _PREDEFINED_SPLITS_BIOMED[dataname] = (image_root, ann_root)



# extra test data version
data_versions = {'illegal': ['ISIC', 'PolypGen', 'GlaS_cell', 'LGG', 'REFUGE', 'COVID-19_CT', 'Radiography/Lung_Opacity', 'BreastUS'],
                 'absent': ['amos22/CT', 'amos22/MRI', 'MSD/Task08_HepaticVessel', 'kits23', 'MSD/Task05_Prostate', 'ACDC', 
                            'MSD/Task01_BrainTumour', 'MSD/Task04_Hippocampus', 'NeoPolyp', 'PanNuke'],
                 }

for version in data_versions:
    for name in data_versions[version]:
        for split in ['test']:
            dataname = f'biomed_{name.replace("/", "-")}_{version}_{split}'
            image_root = f"BiomedSeg/{name}/{split}"
            ann_root = f"BiomedSeg/{name}/{split}_{version}.json"
            _PREDEFINED_SPLITS_BIOMED[dataname] = (image_root, ann_root)
            

def get_metadata():
    meta = {}
    return meta


def load_biomed_json(image_root, annot_json, metadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(annot_json) as f:
        json_info = json.load(f)
        
    # build dictionary for grounding
    grd_dict = collections.defaultdict(list)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    mask_root = image_root + '_mask'
    ret = []
    for image in json_info["images"]:
        image_id = int(image["id"])
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]
        for ann in grounding_anno:
            if 'mask_file' not in ann:
                ann['mask_file'] = image['file_name']
            ann['mask_file'] = os.path.join(mask_root, ann['mask_file'])
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "grounding_info": [ann],
                }
            )
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_biomed(
    name, metadata, image_root, annot_json):
    DatasetCatalog.register(
        name,
        lambda: load_biomed_json(image_root, annot_json, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_biomed(root):
    for (
        prefix,
        (image_root, annot_root),
    ) in _PREDEFINED_SPLITS_BIOMED.items():
        register_biomed(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, annot_root),
        )


_root = os.getenv("DATASET", "datasets")
register_all_biomed(_root)
