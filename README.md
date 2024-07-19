# **BiomedParse**
:grapes: \[[Read Our arXiv Paper](https://arxiv.org/abs/2405.12971)\] &nbsp; :apple: \[[Check Our Demo](https://microsoft.github.io/BiomedParse/)\]

## Installation

### Install Docker

Follow these commands to install Docker on Ubuntu:

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

## Data Description and Preparation

### Data Description

@IceBubble217 Provide a detailed description of the dataset used. Include information such as the source, structure, and any preprocessing steps required.

### Data Preparation
Preprocessing scripts or commands that need to be run.


## Training

To train the model using the example BioParseData, run:

```sh
bash assets/scripts/train.sh
```

### Customizing Training Settings

**Placeholder:**
- Changing Parameters: [Describe how to change parameters, e.g., learning rate, batch size, etc.]
- Customizing Training Settings: [Provide examples of how to customize the training settings]

## Evaluation

To evaluate the model on the example BioParseData, run:

```sh
bash assets/scripts/eval.sh
```

## Inference

Detection and recognition inference code are provided in `inference_utils/output_processing.py`.

- `check_mask_stats()`: Outputs p-value for model-predicted mask for detection.
- `combine_masks()`: Combines predictions for non-overlapping masks.

@IceBubble217 details/example for usage.


## Reproducing Results
To reproduce the exact results presented in the paper, use the following table of parameters and configurations:

| Configuration Parameter     | Description                              | Value                              |
|-----------------------------|------------------------------------------|------------------------------------|
| Data Directory              | Path to the dataset                      | `/path/to/data/`                   |
| Pre-trained Model Checkpoint| Path to the pre-trained model checkpoint | `/path/to/checkpoint/model.pth`    |
| Training Script             | Script used for training                 | `assets/scripts/train.sh`          |
| Evaluation Script           | Script used for evaluation               | `assets/scripts/eval.sh`           |
| Inference Script            | Script for running inference             | `inference_utils/output_processing.py` |
| Environment Variables       | Required environment variables           | See below                          |                     |
| Configuration File          | Configuration file for the model         | `configs/biomedseg/biomed_seg_lang_v1.yaml` |
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
export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here
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
    SOLVER.MAX_NUM_EPOCHS 10 \
    SOLVER.BASE_LR 0.00005 \
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


## Additional Notes
- Refer to the Method section in our paper for more details on the algorithms and implementation.



