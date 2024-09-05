# **BiomedParse**
:grapes: \[[Read Our arXiv Paper](https://arxiv.org/abs/2405.12971)\] &nbsp; :apple: \[[Check Our Demo](https://microsoft.github.io/BiomedParse/)\]

## Installation

### Conda Environment Setup
Create a new conda environment
```sh
conda create -n biomedparse python=3.9
conda activate biomedparse
```

Install Pytorch
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
In case there is issue with detectron2 installation in the following, make sure your pytorch version is compatible with CUDA version on your machine at https://pytorch.org/.

Install dependencies
```sh
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt
```

### Install Docker

In order to make sure the environment is set up correctly, we use run BiomedParse on a Docker image. Follow these commands to install Docker on Ubuntu:

```sh
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
```

## Prepare Docker Environment

Specify the project directories in `docker/README.md`.

Run the following commands to set up the Docker environment:

```sh
bash docker/docker_build.sh
bash docker/docker_run.sh
bash docker/setup_inside_docker.sh
source docker/data_env.sh 
```

## Dataset Description and Preparation

### Raw Image and Annotation
For each dataset, put the raw image and mask files in the following format
```
├── biomedparse_datasets
    ├── YOUR_DATASET_NAME
        ├── train
        ├── train_mask
        ├── test
        └── test_mask
```

Each folder should contain .png files. The mask files should be binary images where pixels != 0 indicates the foreground region.

### File Name Convention
Each file name follows certain convention as

[IMAGE-NAME]\_[MODALITY]\_[SITE].png

[IMAGE-NAME] is any string that is unique for one image. The format can be anything.
[MODALITY] is a string for the modality, such as "X-Ray"
[SITE] is the anatomic site for the image, such as "chest"

One image can be associated with multiple masks corresponding to multiple targets in the image. The mask file name convention is

[IMAGE-NAME]\_[MODALITY]\_[SITE]\_[TARGET].png

[IMAGE-NAME], [MODALITY], and [SITE] are the same with the image file name.
[TARGET] is the name of the target with spaces replaced by '+'. E.g. "tube" or "chest+tube". Make sure "_" doesn't appear in [TARGET].

### Get Final Data File with Text Prompts
In biomedparse_datasets/create-customer-datasets.py, specify YOUR_DATASET_NAME.
Once the create-custom-coco-dataset script is run, the dataset folder should be of the following format
```
├── dataset_name
        ├── train
        ├── train_mask
        ├── train.json
        ├── test
        ├── test_mask
        └── test.json
```

### Register Your Dataset for Training and Evaluation
In datasets/registration/register_biomed_datasets.py, simply add YOUR_DATASET_NAME to the datasets list. Registered datasets are ready to be added to the training and evaluation config file configs/biomed_seg_lang_v1.yaml. Your training dataset is registered as biomed_YOUR_DATASET_NAME_train, and your test dataset is biomed_YOUR_DATASET_NAME_test.


## Training

To train the model using the example BiomedParseData-Demo, run:

```sh
bash assets/scripts/train.sh
```

### Customizing Training Settings
See Training Parameters section for example.

## Evaluation

To evaluate the model on the example BiomedParseData-Demo, run:

```sh
bash assets/scripts/eval.sh
```

## Inference
Example inference code is provided in `example_prediction.py`. We provided example images in `examples` to load from. Model checkpoint is provided in `pretrained` to load from. Model configuration is provided in `configs/biomedparse_inference.yaml`.

### Model Setup
```sh
from PIL import Image
import torch
import argparse
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

# Build model config
def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/biomedparse_inference.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--model_path', default="pretrained/biomed_parse.pt", metavar="FILE", help='path to model file')
    cfg = parser.parse_args()
    return cfg

cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)

# Load model from pretrained weights
pretrained_pth = 'pretrained/biomed_parse.pt'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
```

### Segmentation On Example Images
```sh
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open('examples/Part_3_226_pathology_breast.png', formats=['png']) 
image = image.convert('RGB')
# text prompts querying objects in the image. Multiple ones can be provided.
prompts = ['neoplastic cells in breast pathology', 'inflammatory cells']

pred_mask, pred_text = interactive_infer_image(model, image, prompts)
```

<!-- 
Detection and recognition inference code are provided in `inference_utils/output_processing.py`.

- `check_mask_stats()`: Outputs p-value for model-predicted mask for detection.
- `combine_masks()`: Combines predictions for non-overlapping masks. -->



## Reproducing Results
To reproduce the exact results presented in the paper, use the following table of parameters and configurations:

| Configuration Parameter     | Description                              | Value                              |
|-----------------------------|------------------------------------------|------------------------------------|
| Data Directory              | Path to the dataset                      | `/path/to/data/`                   |
| Pre-trained Model Checkpoint| Path to the pre-trained model checkpoint | `pretrained/biomed_parse.pt`    |
| Training Script             | Script used for training                 | `assets/scripts/train.sh`          |
| Evaluation Script           | Script used for evaluation               | `assets/scripts/eval.sh`           |
| Inference Script            | Script for running inference             | `example_prediction.py` |
| Environment Variables       | Required environment variables           | See below                          |                     |
| Configuration File          | Configuration file for the model         | `configs/biomed_seg_lang_v1.yaml` |
| Training Parameters         | Additional training parameters           | See below                          |
| Output Directory            | Directory to save outputs                | `outputs/`                         |

### Environment Variables
```sh
export DETECTRON2_DATASETS=data/
export DATASET=data/
export DATASET2=data/
export VLDATASET=data/
export PATH=$PATH:data/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:data/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here
```

### Training Parameters
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 4 python entry.py train \
    --conf_files configs/biomedseg/biomed_seg_lang_v1.yaml \
    --overrides \
    FP16 True \
    RANDOM_SEED 2024 \
    BioMed.INPUT.IMAGE_SIZE 1024 \
    MODEL.DECODER.HIDDEN_DIM 512 \
    MODEL.ENCODER.CONVS_DIM 512 \
    MODEL.ENCODER.MASK_DIM 512 \
    TEST.BATCH_SIZE_TOTAL 4 \
    TRAIN.BATCH_SIZE_TOTAL 4 \
    TRAIN.BATCH_SIZE_PER_GPU 1 \
    SOLVER.MAX_NUM_EPOCHS 20 \
    SOLVER.BASE_LR 0.00001 \
    SOLVER.FIX_PARAM.backbone False \
    SOLVER.FIX_PARAM.lang_encoder False \
    SOLVER.FIX_PARAM.pixel_decoder False \
    MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 1.0 \
    MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 1.0 \
    MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 1.0 \
    MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
    MODEL.DECODER.SPATIAL.ENABLED True \
    MODEL.DECODER.GROUNDING.ENABLED True \
    LOADER.SAMPLE_PROB prop \
    FIND_UNUSED_PARAMETERS True \
    ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
    MODEL.DECODER.SPATIAL.MAX_ITER 0 \
    ATTENTION_ARCH.QUERY_NUMBER 3 \
    STROKE_SAMPLER.MAX_CANDIDATE 10 \
    WEIGHT True \
    RESUME_FROM pretrained/biomed_parse.pt
```




