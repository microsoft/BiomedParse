# FROM naotous/flash_attn:2.0.5-pytorch23.07
FROM wangkenpu/pytorch:1.8.0-py39-cuda11.1-cudnn8-ubuntu18.04

# RUN touch tensorboard_patcher.py && cp tensorboard_patcher.py $$USERSITE/usercustomize.py


# RUN pip install --upgrade pip

# RUN pip install -I torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install -I torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --user
# RUN pip install kornia
# RUN pip install timm==0.4.12
# RUN python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install git+https://github.com/openai/CLIP.git

# RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

COPY assets/requirements/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY assets/requirements/requirements_custom.txt /tmp/requirements_custom.txt
RUN pip install -r /tmp/requirements_custom.txt

#RUN pip install -U protobuf

# Set environment variables
ENV MKL_THREADING_LAYER=GNU
ENV NCCL_DEBUG=INFO

# Set the working directory HERE!
WORKDIR /path/to/BiomedParse