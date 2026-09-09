## Preparations
For native 2D grayscale/RGB images, use the [2D training guide](2D.md), which reuses this training stack and generates negative image-prompt pairs automatically. The instructions below cover the existing 3D-volume fine-tuning workflow.

You need to prepare the public model checkpoint and finetuning data under ```<YOUR MODEL AND DATA DIR>``` and put it in `finetune_biomedparse.yaml` as
```yaml
mounts:
  external: <YOUR MODEL AND DATA DIR>
```


## Model Weights
Download the pretrained checkpoint ```biomedparse_v2.ckpt``` and put it under ```<YOUR MODEL AND DATA DIR>```

### Option 1: Hugging Face Hub
You can download the pretrained model weights directly from the Hugging Face Hub.

First, install the required package:
```bash
pip install huggingface_hub
```

Then, download the checkpoint file using the Hugging Face Hub API:
```python
from huggingface_hub import hf_hub_download

# Download the checkpoint file
file_path = hf_hub_download(
    repo_id="microsoft/BiomedParse",
    filename="biomedparse_v2.ckpt"
)

print("Model weights downloaded to:", file_path)
```

### Option 2: Direct Download via Command Line
You can also download the file directly using `wget` or `curl`:
```bash
wget https://huggingface.co/microsoft/BiomedParse/resolve/main/biomedparse_v2.ckpt
```
or
```bash
curl -L -o biomedparse_v2.ckpt https://huggingface.co/microsoft/BiomedParse/resolve/main/biomedparse_v2.ckpt
```

> 💡 **Note:** If the repository is private, log in with your Hugging Face token using:
> ```bash
> huggingface-cli login
> ```
> before attempting to download.


Now you should have the model weights ready for use!

### 💾 Custom Checkpoints

Fine-tuning starts from the pretrained checkpoint specified in your config ```configs/olympus_checkpoint/biomedparse_checkpoint_loader.yaml```:

```yaml
checkpoint_path: ${mounts.external}/biomedparse_v2.ckpt
```

You can replace this path with your own checkpoint for continued training or domain adaptation.

## Data Preparation
Your finetuning data should be stored under ```<YOUR MODEL AND DATA DIR>/data```. We provided detailed instruction in [DATASET](DATASET.md).


## Fine-tuning BiomedParse V2

Once the model weights are downloaded and datasets are prepared, you can fine-tune **BiomedParse V2** using our modular YAML configuration system powered by [Hydra](https://hydra.cc/) and [AzureML Olympus](https://learn.microsoft.com/en-us/azure/machine-learning/).

---

### 🧩 How Hydra Works

Hydra enables **composable configuration management** — each logical part of training (model, dataset, trainer, optimizer, etc.) is defined in a separate YAML file and referenced in a master config via the `defaults:` list.

Example structure of `finetune_biomedparse.yaml`:

```yaml
defaults:
  - model: biomedparse
  - datamodule: biomedparse_finetune_datamodule
  - trainer: biomedparse_trainer
  - evaluator: biomedparse_evaluator
  - loss: biomedparse_loss
  - optimizer: adamw
  - olympus_checkpoint: biomedparse_checkpoint_loader
  - _self_
```

When you run a job, Hydra automatically merges these component configs into one runtime configuration.  

---

### ⚙️ Running the Fine-tuning Job

To launch a fine-tuning run with default parameters, execute:

```bash
python -m azureml.acft.image.components.olympus.app.main \
  --config-path <YOUR ABSOLUTE CONFIG DIRECTORY PATH> \
  --config-name finetune_biomedparse
```

This will:
1. Load all YAML config components via Hydra.  
2. Initialize the Olympus training pipeline.  
3. Start fine-tuning from the checkpoint defined in the configuration.

---

### 🧾 Baseline Configuration

The baseline configuration is located in the config directories. Start with finetune_biomedparse.yaml and follow the nested structure. 

---

### 📦 Outputs

Training logs, checkpoints, and metrics are saved to:

```
${mounts.external}/outputs
```

Monitor progress in AzureML or your chosen logging backend.

---

### ✅ Example Override Commands
You can override any field on the command line without editing YAML files.

Change the optimizer and batch size:

```bash
python -m azureml.acft.image.components.olympus.app.main \
  --config-path <YOUR ABSOLUTE CONFIG DIRECTORY PATH> \
  --config-name finetune_biomedparse \
  optimizer=adamw optimizer.lr=1e-4 datamodule.dataloaders.train.batch_size=16
```

---

### 🔍 Learn More

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [AzureML Components](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)


<!-- ## Dataset
BiomedParseData was created from preprocessing publicly available biomedical image segmentation datasets. Check a subset of our processed datasets on HuggingFace: https://huggingface.co/datasets/microsoft/BiomedParseData. For the source datasets, please check the details here: [BiomedParseData](assets/readmes/DATASET.md). As a quick start, we've samples a tiny demo dataset at biomedparse_datasets/BiomedParseData-Demo -->

<!-- ## Model Checkpoints
We host our model checkpoints on HuggingFace here: https://huggingface.co/microsoft/BiomedParse. See example code below on model loading.

Please expect future updates of the model as we are making it more robust and powerful based on feedbacks from the community. We recomment using the latest version of the model.

## Running Inference with BiomedParse

We’ve streamlined the process for running inference using BiomedParse. Below are details and resources to help you get started.

### How to Run Inference
To perform inference with BiomedParse, use the provided example code and resources:

- **Inference Code**: Use the example inference script in `example_prediction.py`.
- **Sample Images**: Load and test with the provided example images located in the `examples` directory.
- **Model Configuration**: The model settings are defined in `configs/biomedparse_inference.yaml`.

### Example Notebooks

We’ve included sample notebooks to guide you through running inference with BiomedParse:

- **RGB Inference Example**: Check out the `inference_examples_RGB.ipynb` notebook for example using normal RGB images, including Pathology, X-ray, Ultrasound, Endoscopy, Dermoscopy, OCT, Fundus.
- **DICOM Inference Example**: Check out the `inference_examples_DICOM.ipynb` notebook for example using DICOM images.
- **NIFTI Inference Example**: Check out the `inference_examples_NIFTI.ipynb` notebook for example using NIFTI image slices.
- You can also try a quick online demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/BiomedParse/blob/main/inference_colab_demo.ipynb)

### Model Setup
```sh
from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
import numpy as np

# Build model config
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
#pretrained_pth = 'pretrained/biomed_parse.pt'
pretrained_pth = 'hf_hub:microsoft/BiomedParse'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
```

### Segmentation On Example Images
```sh
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open('examples/Part_1_516_pathology_breast.png', formats=['png'])
image = image.convert('RGB')
# text prompts querying objects in the image. Multiple ones can be provided.
prompts = ['neoplastic cells', 'inflammatory cells']

# load ground truth mask
gt_masks = []
for prompt in prompts:
    gt_mask = Image.open(f"examples/Part_1_516_pathology_breast_{prompt.replace(' ', '+')}.png", formats=['png'])
    gt_mask = 1*(np.array(gt_mask.convert('RGB'))[:,:,0] > 0)
    gt_masks.append(gt_mask)

pred_mask = interactive_infer_image(model, image, prompts)

# prediction with ground truth mask
for i, pred in enumerate(pred_mask):
    gt = gt_masks[i]
    dice = (1*(pred>0.5) & gt).sum() * 2.0 / (1*(pred>0.5).sum() + gt.sum())
    print(f'Dice score for {prompts[i]}: {dice:.4f}')
    check_mask_stats(image, pred_mask[i]*255, 'X-Ray-Chest', text_prompt[i])
    print(f'p-value for {prompts[i]}: {p_value:.4f}')
```


Detection and recognition inference code are provided in `inference_utils/output_processing.py`.

- `check_mask_stats()`: Outputs p-value for model-predicted mask for detection. Check the `inference_examples_RGB.ipynb` notebook.
- `combine_masks()`: Combines predictions for non-overlapping masks.

## Finetune on Your Own Data
While BiomedParse can take in arbitrary image and text prompt, it can only reasonably segment the targets that it has learned during pretraining! If you have a specific segmentation task that the latest checkpint doesn't do well, here is the instruction on how to finetune it on your own data.

### Raw Image and Annotation
BiomedParse expects images and ground truth masks in 1024x1024 PNG format. For each dataset, put the raw image and mask files in the following format
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
In biomedparse_datasets/create-customer-datasets.py, specify YOUR_DATASET_NAME. Run the script with
```
cd biomedparse_datasets
python create-customer-datasets.py
```
After that, the dataset folder should be of the following format
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


## Train BiomedParse
To train the BiomedParse model, run:

```sh
bash assets/scripts/train.sh
```
This will continue train the model using the training datasets you specified in configs/biomed_seg_lang_v1.yaml

## Evaluate BiomedParse
To evaluate the model, run:
```sh
bash assets/scripts/eval.sh
```
This will continue evaluate the model on the test datasets you specified in configs/biomed_seg_lang_v1.yaml. We put BiomedParseData-Demo as the default. You can add any other datasets in the list. -->


## Citation

Please cite our paper if you use the code, model, or data.

```bibtex
@article{zhao2025foundation,
  title={A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities},
  author={Zhao, Theodore and Gu, Yu and Yang, Jianwei and Usuyama, Naoto and Lee, Ho Hin and Kiblawi, Sid and Naumann, Tristan and Gao, Jianfeng and Crabtree, Angela and Abel, Jacob and others},
  journal={Nature methods},
  volume={22},
  number={1},
  pages={166--176},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```

If you use the v2 code or model, please also cite the BoltzFormer paper:
```bibtex
@inproceedings{zhao2025boltzmann,
  title={Boltzmann Attention Sampling for Image Analysis with Small Objects},
  author={Zhao, Theodore and Kiblawi, Sid and Usuyama, Naoto and Lee, Ho Hin and Preston, Sam and Poon, Hoifung and Wei, Mu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25950--25959},
  year={2025}
}
```

## Usage and License Notices
The model described in this repository is provided for research and development use only. The model is not intended for use in clinical decision-making or for any other clinical use, and the performance of the model for clinical use has not been established. You bear sole responsibility for any use of this model, including incorporation into any product intended for clinical use.
