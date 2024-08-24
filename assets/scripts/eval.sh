export DETECTRON2_DATASETS=biomedparse_datasets/
export DATASET=biomedparse_datasets/
export DATASET2=biomedparse_datasets/
export VLDATASET=biomedparse_datasets/
export PATH=$PATH:biomedparse_datasets/coco_caption/jre1.8.0_321/bin/
export PYTHONPATH=$PYTHONPATH:biomedparse_datasets/coco_caption/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 python entry.py evaluate \
            --conf_files configs/biomed_seg_lang_v1.yaml \
            --overrides \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 1 \
            FP16 True \
            WEIGHT True \
            STANDARD_TEXT_FOR_EVAL False \
            RESUME_FROM pretrained/biomed_parse.pt \
            