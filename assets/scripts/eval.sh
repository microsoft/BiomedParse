export DETECTRON2_DATASETS=data/
export DATASET=data/
export DATASET2=data/
export VLDATASET=data/
export PATH=$PATH:data/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:data/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 4 python entry.py evaluate \
            --conf_files configs/biomedseg/biomed_seg_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 4 \
            FP16 True \
            WEIGHT True \
            STANDARD_TEXT_FOR_EVAL False \
            RESUME_FROM pretrained/biomed_parse.pt \
            