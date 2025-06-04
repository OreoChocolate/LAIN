#NCCL_SOCKET_IFNAME=eno2

CUDA_VISIBLE_DEVICES=0 python main.py \
                              --pretrained checkpoints/pretrained_detr/detr-r50-hicodet.pth --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
                              --dataset hicodet --num-workers 6 --num_classes 117 --zs --zs_type unseen_object --output-dir checkpoints/debug \
                              --use_hotoken --use_prompt --N_CTX 36 --CSC --use_exp \
                              --use_insadapter --adapt_dim 32 --use_prior \
                              --resume /home/oreo/PycharmProjects/LAIN/checkpoints/UO.pt --eval --debug --port 11547