# CUDA_VISIBLE_DEVICES=7 python -m \
# torch.distributed.launch --nnodes=1 \
# --nproc_per_node=1 train.py --config-path \
# config/vidstg/config_weak_rcnn.json

CUDA_VISIBLE_DEVICES=3 python train.py --config-path \
config/vidstg/config_weak_c3d.json