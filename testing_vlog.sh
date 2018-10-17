#!/usr/bin/env bash

# Envs
# source activate pytorch-0.4.0

orn_ckpt_dir=/vision2/u/bingbin/ORN/ckpt
ckpt_name='epic_headsobject+context_gcnObj0_gcnCtxt0_bt4_lr1e-04_wd1e-05/ckpt_epoch7_best.pth'

# full dir
# VLOG masks
# VLOG=/vision2/u/cy3/data/VLOG
# EPIC masks
EPIC=/vision2/u/cy3/data/mask-AR

# testing dir
# VLOG_masks="/sailhome/bingbin/object_level_visual_reasoning/data/vlog/masks/preds_100x100_50"
# VLOG_videos="/sailhome/bingbin/object_level_visual_reasoning/data/vlog/videos"

# Pythonpath
PYTHONPATH=.

python main.py \
--resume $orn_ckpt_dir \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--depth 50 \
-t 4 \
-b 16 \
--cuda \
--dataset vlog \
--heads object+context \
-j 4 \
--nb-crops 8 \
--mask-confidence 0.75 \
-e
