#%% import packages
import numpy as np
import os
join = os.path.join 
from skimage import io, transform
from tqdm import tqdm
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image     
import json
import nibabel as nib
import pandas as pd
import shutil


# convert 2D data to png files, including images and corresponding masks
data_name = 'amos22'
#modality = 'X-Ray' # e.g., 'Dermoscopy 
anatomy = 'abdomen'  # e.g., 'SkinCancer'

CT_WINDOWS = {'liver': [-25, 230],
              'abdomen': [-150, 250],
              'colon': [-68, 187],
              'pancreas': [-100, 200],
              'lung': [-1000, 1000],}

labels = {"1": "spleen", "2": "right kidney", "3": "left kidney", "4": "gallbladder", "5": "esophagus", "6": "liver", "7": "stomach", "8": "aorta", "9": "postcava", "10": "pancreas", "11": "right adrenal gland", "12": "left adrenal gland", "13": "duodenum", "14": "bladder", "15": "prostate/uterus"}

merge_classes = {(2,3): 'kidney', (11,12): 'adrenal gland'}

### Dataset specific settings
split_names = {'Tr': 'train', 'Va': 'test'}

# record image and mask statistics
splits = []
fnames = []
original_shapes = []
mask_areas = []
mask_bboxes = []
intersect_portion = []



image_size = 1024
# set label ids that are excluded
remove_label_ids = [15]     # remove prostate/uterus since the name is not specific

basepath = '/storage/data/MedSegmentation/'
datapath = os.path.join(basepath, data_name)
basetarget = '/storage/data/MedSegmentation/BiomedSeg/'
targetpath = os.path.join(basetarget, data_name)

def get_mask_stats(mask):
    masked_points = np.nonzero(mask)
    area = len(masked_points[0])
    min_x = np.min(masked_points[0])
    max_x = np.max(masked_points[0])
    min_y = np.min(masked_points[1])
    max_y = np.max(masked_points[1])
    bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
    return area, bbox


# preprocessing
for split in split_names.keys():
    
    files = os.listdir(os.path.join(datapath, f'images{split}'))
    label_files = os.listdir(os.path.join(datapath, f'labels{split}'))
    for file in tqdm(files):
        if '.nii.gz' not in file:
            continue
        
        if file not in label_files:
            continue
        
        target_split = split_names[split]
        extra_train = int(file[5:9]) <= 228 or (int(file[5:9]) > 500 and int(file[5:9]) <= 556)
        if target_split == 'test' and extra_train:
            target_split = 'train'
        
        ### Image INFO
        if int(file[5:9]) <= 500:
            modality = 'CT'
        else:
            modality = 'MRI'
            
        f = file.split('.')[0]
        
        # load image and preprocess
        image_data = nib.load(os.path.join(datapath, f'images{split}', file)).get_fdata()
        
        # nii preprocess start
        if modality == "CT":
            lower_bound = CT_WINDOWS['abdomen'][0]
            upper_bound = CT_WINDOWS['abdomen'][1]
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            # different processing for liver CT
            lower_bound = CT_WINDOWS['liver'][0]
            upper_bound = CT_WINDOWS['liver'][1]
            image_data_liver = np.clip(image_data, lower_bound, upper_bound)
            image_data_liver = (
                (image_data_liver - np.min(image_data_liver))
                / (np.max(image_data_liver) - np.min(image_data_liver))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0
        
        # process mask annotations
        mask = nib.load(os.path.join(datapath, f'labels{split}', file)).get_fdata()
        shape = list(mask.shape[:2])
        n_slices = mask.shape[2]
        
        # remove label ids
        for remove_label_id in remove_label_ids:
            mask[mask==remove_label_id] = 0
        
        # pad to square with equal padding on both sides
        if shape[0] > shape[1]:
            pad = (shape[0]-shape[1])//2
            pad_width = ((0,0), (pad, pad), (0,0))
        elif shape[0] < shape[1]:
            pad = (shape[1]-shape[0])//2
            pad_width = ((pad, pad), (0,0), (0,0))
        else:
            pad_width = None
        
        if pad_width is not None:
            image_data_pre = np.pad(image_data_pre, pad_width, 'constant', constant_values=0)
            if modality=='CT':
                image_data_liver = np.pad(image_data_liver, pad_width, 'constant', constant_values=0)
            mask = np.pad(mask, pad_width, 'constant', constant_values=0)
        # preprocessing ends
            
            
        # find largest slice area for each class
        max_area = {}
        for c in labels:
            mask_c = 1*(mask==int(c))
            max_area[c] = mask_c.sum(axis=(0,1)).max()
        for c in merge_classes:
            mask_c = 1*(mask==int(c[0]))
            for cc in c[1:]:
                mask_c += 1*(mask==int(cc))
            mask_c = 1*(mask_c>0)
            max_area[c] = mask_c.sum(axis=(0,1)).max()  
        
        
        # process by slice
        for i in range(n_slices):
            # remove masks of class that are too small
            for c in labels:
                if np.sum(mask[:,:,i]==int(c)) < max_area[c]/10:
                    mask[:,:,i][mask[:,:,i]==int(c)] = 0
            # skip slices without mask
            if int(mask[:,:,i].max()) == 0:
                continue
            
            # resize image to 1024x1024
            resize_image = transform.resize(image_data_pre[:,:,i], (image_size, image_size), order=3, mode='constant', preserve_range=True, anti_aliasing=True)
            
            # save image
            filename = f"{f}_{i}_{modality}_{anatomy}.png"
            plt.imsave(os.path.join(targetpath, modality, f"{target_split}/{filename}"), 
                       resize_image.astype(np.uint8), cmap='gray')
            
            
            # save extra image slice for liver CT
            if modality=='CT' and 6 in mask[:,:,i]:
                # resize image to 1024x1024
                resize_image = transform.resize(image_data_liver[:,:,i], (image_size, image_size), order=3, mode='constant', preserve_range=True, anti_aliasing=True)
                # save image
                filename = f"{f}_{i}_{modality}_liver.png"
                plt.imsave(os.path.join(targetpath, modality, f"{target_split}/{filename}"), 
                        resize_image.astype(np.uint8), cmap='gray')
        
        
            # Get masks for each class
            for c in labels:
                target = labels[c]
                target_name = target.replace(' ', '+')
                
                mask_c = 1*(mask[:,:,i]==int(c))
                # make sure the class exists
                if mask_c.max() == 0:
                    continue
                
                portion = int(10 * mask_c.sum() / max_area[c])
                
                # resize mask to 1024x1024
                resize_gt = transform.resize(mask_c, (image_size, image_size), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
                
                # stats of resized mask
                area, bbox = get_mask_stats(resize_gt)
            
                # colored mask
                colored_mask = 255*np.repeat(resize_gt[:,:,None], 3, axis=-1)
                # save output
                mask_filename = f"{f}_{i}_{modality}_{anatomy}_{target_name}.png"
                plt.imsave(os.path.join(targetpath, modality, f"{target_split}_mask/{mask_filename}"), 
                        colored_mask.astype(np.uint8))
                # record image and mask statistics
                splits.append(target_split)
                fnames.append(mask_filename)
                original_shapes.append(tuple(shape))
                mask_areas.append(int(area))
                mask_bboxes.append(tuple(bbox))
                intersect_portion.append(portion)
                
                # save mask for liver CT
                if modality=='CT' and target_name == 'liver':
                    # save output
                    mask_filename = f"{f}_{i}_{modality}_liver_{target_name}.png"
                    plt.imsave(os.path.join(targetpath, modality, f"{target_split}_mask/{mask_filename}"), 
                            colored_mask.astype(np.uint8))
                    # record image and mask statistics
                    splits.append(target_split)
                    fnames.append(mask_filename)
                    original_shapes.append(tuple(shape))
                    mask_areas.append(int(area))
                    mask_bboxes.append(tuple(bbox))
                    intersect_portion.append(portion)
                    
                
                
            # get mask for merged classes
            for c in merge_classes:
                target = merge_classes[c]
                target_name = target.replace(' ', '+')
                
                # make sure both classes exist
                if (mask[:,:,i]==int(c[0])).sum() == 0 or (mask[:,:,i]==int(c[1])).sum() == 0:
                    continue
                
                mask_c = 1*((mask[:,:,i]==int(c[0])) | (mask[:,:,i]==int(c[1])))
                
                portion = int(10 * mask_c.sum() / max_area[c])
                
                # resize mask to 1024x1024
                resize_gt = transform.resize(mask_c, (image_size, image_size), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
                
                # stats of resized mask
                area, bbox = get_mask_stats(resize_gt)
            
                mask_filename = f"{f}_{i}_{modality}_{anatomy}_{target_name}.png"
                
                # save colored mask
                colored_mask = 255*np.repeat(resize_gt[:,:,None], 3, axis=-1)
                plt.imsave(os.path.join(targetpath, modality, f"{target_split}_mask/{mask_filename}"), 
                        colored_mask.astype(np.uint8))

                # record image and mask statistics
                splits.append(target_split)
                fnames.append(mask_filename)
                original_shapes.append(tuple(shape))
                mask_areas.append(int(area))
                mask_bboxes.append(tuple(bbox))
                intersect_portion.append(portion)
        
            
                
# save statistics
df = pd.DataFrame({'split': splits, 'fname': fnames, 'original_shape': original_shapes, 
                   'mask_area': mask_areas, 'mask_bbox': mask_bboxes, 'slice_ratio': intersect_portion})
df.to_csv(os.path.join(targetpath, 'mask_stats.csv'), index=False)