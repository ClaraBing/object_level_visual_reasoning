#!/usr/bin/env bash

# Envs
# source activate pytorch-0.4.0

# Pythonpath
PYTHONPATH=.

CUDA_ID=0

# Settings
resume=/vision2/u/bingbin/ORN/
root=./data/epic/
# save_token='sampled1PerClip'
save_token='CEloss_tmp'
# mask_dir='masks/sampled_1perClip'
mask_dir='masks/preds_100x100_50'

use_obj_gcn=1
use_context_gcn=1
adj_type='prior'

# Train the object head only with f=MLP
epochs=10
heads=object
CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --resume $resume \
--root $root \
--save-token $save_token \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--use-obj-gcn=$use_obj_gcn \
--use-context-gcn=$use_context_gcn \
--adj-type=$adj_type \
--depth 50 \
-t 4 \
-b 8 \
--cuda \
--dataset epic \
--heads $heads \
--epochs $epochs \
--pooling avg \
--print-freq 100 \
--pooling avg \
--mask-dir $mask_dir \
--mask-confidence 0.5 \
-j 4

exit;

# no GCN:
# --ckpt-name 'epic_headsobject_gcnObj0_gcnCtxt0_bt8_lr1e-04_wd1e-05/ckpt_epoch10_best.pth' \
# both GCN:
# 'epic_headsobject_gcnObj1_gcnCtxt1_bt8_lr1e-04_wd1e-05_adjuniform_oEmb128_vEmb128_nLayer2_nTopObjs10/ckpt_epoch10_best.pth'
ckpt_name='epic_headsobject_gcnObj0_gcnCtxt0_bt8_lr1e-04_wd1e-05_CEloss/ckpt_epoch10_best.pth'
# ckpt_name='epic_headsobject_gcnObj1_gcnCtxt1_bt8_lr1e-04_wd1e-05_adjprior_oEmb128_vEmb128_nLayer2_nTopObjs10_CEloss/ckpt_epoch10_best.pth'

# Train the two heads with f=RNN and pooling is avg for context head
epochs=10
heads=object+context
CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --resume $resume \
--ckpt-name $ckpt_name \
--root $root \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--use-obj-gcn=$use_obj_gcn \
--use-context-gcn=$use_context_gcn \
--adj-type=$adj_type \
--depth 50 \
-t 4 \
-b 4 \
--cuda \
--dataset epic \
--heads $heads \
--epochs $epochs \
--print-freq 100 \
--pooling rnn \
--mask-confidence 0.5 \
-j 4

# no GCN:
# --ckpt-name 'epic_headsobject+context_gcnObj0_gcnCtxt0_bt4_lr1e-04_wd1e-05/ckpt_epoch10_best.pth'

# Finally validate on the test set
epochs=10
heads=object+context
CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --resume $resume \
--ckpt-name 'epic_headsobject+context_gcnObj1_gcnCtxt1_bt4_lr1e-04_wd1e-05_adjuniform_oEmb128_vEmb128_nLayer2_nTopObjs10/ckpt_epoch10_best.pth' \
--root $root \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--use-obj-gcn=$use_obj_gcn \
--use-context-gcn=$use_context_gcn \
--adj-type=$adj_type \
--depth 50 \
-t 4 \
-b 5 \
--cuda \
--dataset epic \
--heads $heads \
--epochs $epochs \
--print-freq 100 \
-j 4 \
--pooling rnn \
--nb-crops 8 \
--mask-confidence 0.5 \
-e
