IMAGENET_DIR="/share/seo/imagenet-100/"
# FINETUNED_CHECKPOINT_PATH="/home/jm2787/NMAE/output/imagenet100-mae-vit-tiny-pretrain-wfm-mr0.75-kmr0.25-dd12-ep200-b2048/checkpoint.pth"
FINETUNED_CHECKPOINT_PATH="/home/jm2787/NMAE/output/imagenet100-asymmae-vit-tiny-pretrain-wfm-mr0.75-kmr0.25-dd12-ep200-b2048-off-diag/checkpoint.pth"
# FINETUNED_CHECKPOINT_PATH="/home/jm2787/NMAE/output/imagenet100-asymmae-vit-base-pretrain-wfm-mr0.75-kmr0.25-dd12-ep200-b2048-off-diag/checkpoint.pth"

CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port 1235 encode_features.py \
    --batch_size 1 \
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
    --output_dir /home/jm2787/NMAE/output/imagenet100-asymmae-vit-base-pretrain-wfm-mr0.75-kmr0.25-dd12-ep200-b2048-off-diag-features/ \
    --multi_epochs_dataloader \
    --nb_classes 1000