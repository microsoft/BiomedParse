export DETECTRON2_DATASETS=data/
export DATASET=data/
export DATASET2=data/
export VLDATASET=data/
export PATH=$PATH:data/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:data/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here
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