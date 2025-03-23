CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port 48009 main_pretrain.py \
    --batch_size 1024 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /share/seo/imagenet-100/ \
    --num_workers 2 \
    --multi_epochs_dataloader \
    --output_dir ./output/imagenet100-crossmae-vit-base-pretrain-wfm-mr0.75-kmr0.25-dd12-ep200-b2048 \
    --cross_mae \
    --weight_fm \
    --decoder_depth 12 \
    --mask_ratio 0.75 \
    --kept_mask_ratio 0.25 \
    --epochs 200 \
    --warmup_epochs 40 \
    --use_input