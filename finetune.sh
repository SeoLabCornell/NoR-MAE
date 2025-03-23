IMAGENET_DIR="/scratch/dataset/imagenet-100/"
FINETUNED_CHECKPOINT_PATH="output/imagenet100-asymmae-vit-tiny-pretrain-wfm-mr0.75-kmr0.25-dd12-ep1600-b4096-off-diag-lambda5e-5/checkpoint.pth"

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port 1238 main_finetune.py \
    --batch_size 256 \
    --model vit_tiny_patch16 \
    --finetune $FINETUNED_CHECKPOINT_PATH \
    --epoch 100 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path ${IMAGENET_DIR} \
    --output_dir output/imagenet100-asymmae-vit-tiny-pretrain-wfm-mr0.75-kmr0.25-dd12-ep1600-b4096-off-diag-lambda5e-5-finetune/ \
    --multi_epochs_dataloader \
    --nb_classes 100 \
    --accum_iter 1 \
    --project_name "nor-mae" \
    --exp_name "vit_tiny_patch16_ep1600_finetune_imagenet100"