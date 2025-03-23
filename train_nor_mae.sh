export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port 48008 main_pretrain.py \
    --batch_size 1024 \
    --model mae_vit_tiny_patch16 \
    --norm_pix_loss --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /scratch/dataset/imagenet-1k/ \
    --num_workers 2 \
    --multi_epochs_dataloader \
    --output_dir ./output/imagenet1k-normae-vit-tiny-pretrain-wfm-mr0.75-kmr0.25-dd12-ep200-b1024-off-diag-lambda5e-5-patchsize16 \
    --asym_mae \
    --decoder_depth 12 \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --lamda 5e-5 \
    --off_diag \
    --accum_iter 1 \
    --project_name "nor-mae" \
    --exp_name "vit_tiny_patch16_epoch200_imagenet1k" \
    
