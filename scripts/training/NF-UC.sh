#NCCL_SOCKET_IFNAME=eno2

let port=$RANDOM%5000+5000
let id=$RANDOM%5000+5000
gpu_num=1
WANDB__SERVICE_WAIT=300

CUDA_VISIBLE_DEVICES=0

torchrun --rdzv_id $id --rdzv_backend=c10d --nproc_per_node=$gpu_num --rdzv_endpoint=127.0.0.1:$port \
         main.py --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --output-dir checkpoints/NF \
         --dataset hicodet --zs --zs_type non_rare_first --num_classes 117 --num-workers 4 \
         --epochs 20 --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
         --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1. \
         --print-interval 100