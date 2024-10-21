from PIL import Image
import torch
import argparse
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image

# Build model config
def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/biomedparse_inference.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--model_path', default="pretrained/biomedparse.pt", metavar="FILE", help='path to model file')
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


# Load image and run inference
# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open('examples/Part_3_226_pathology_breast.png', formats=['png']) 
image = image.convert('RGB')
# text prompts querying objects in the image. Multiple ones can be provided.
prompts = ['neoplastic cells in breast pathology', 'inflammatory cells']

pred_mask, pred_text = interactive_infer_image(model, image, prompts)

# show prediction stats
print(pred_mask.shape, pred_mask.sum(axis=(1,2)), pred_mask.min(axis=(1,2)), pred_mask.max(axis=(1,2)), pred_text)