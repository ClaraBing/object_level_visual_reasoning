#!/usr/bin/env bash

# Envs
# source activate pytorch-0.4.0

cd /sailhome/bingbin/object_level_visual_reasoning;

# Pythonpath
PYTHONPATH=.

machine='tibet5'
# machine='macondo2'
CUDA_ID=1
task_id=0

lr='1e-04'

# Settings
resume='/vision2/u/bingbin/ORN/ckpt/'
root='./data/epic/'
# mask_dir='masks/sampled_1perClip_GT'
mask_dir='masks/sampled_1perClip_reformat'
# mask_dir='masks/preds_100x100_50'


dataset="epic"
use_obj_rel=0
use_obj_logits=0
gcn_version='v2'
use_obj_gcn=0
use_context_gcn=0
two_layer_context=0
use_wv_weights=0
freeze_wv_weights=0
adj_type='prior'

heads=object
# ckpt_dir='/vision2/u/bingbin/ORN/ckpt/epic_headsobject_gcnObj0_gcnCtxt0_bt8_lr1e-04_wd1e-05_ORN_reproduce/'
# ckpt_dir='/vision2/u/bingbin/ORN/ckpt/epic_headsobject+context+vo_gcnObj0_gcnCtxt0_bt2_lr1e-04_wd1e-05_pretrainedContext_addVO_corrected'
ckpt_dir=$resume'epic_headsobject+context+vo_gcnObj0_gcnCtxt0_bt2_lr1e-04_wd1e-05_pretrainedContext_addVO_adjWV'
resume_file=$resume$machine'_gpu'$CUDA_ID'_task'$task_id'.exp'
result_file=$ckpt_dir'find_ckpt_results.txt'

echo "Begin testing.";
epochs=10
for nEoch in 1 2 3 4 5 6 7 8 9 10
do
  echo $nEoch >> $result_file

  ckpt_file=$ckpt_dir'ckpt_epoch'$nEoch'_best.pth'
  echo $ckpt_file
  echo $ckpt_file > $resume_file
  echo $ckpt_file >> $result_file

  CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --resume $resume \
  --machine $machine \
  --gpu-id $CUDA_ID \
  --task-id $task_id \
  --root $root \
  --blocks 2D_2D_2D_2.5D \
  --object-head 2D \
  --add-background \
  --train-set train+val \
  --arch orn_two_heads \
  --use-obj-rel $use_obj_rel \
  --two-layer-context $two_layer_context \
  --use-wv-weights $use_wv_weights \
  --freeze-wv-weights $freeze_wv_weights \
  --gcn-version $gcn_version \
  --use-obj-gcn=$use_obj_gcn \
  --use-context-gcn=$use_context_gcn \
  --adj-type=$adj_type \
  --depth 50 \
  -t 4 \
  -b 16 \
  --cuda \
  --dataset $dataset \
  --heads $heads \
  --epochs $epochs \
  --print-freq 100 \
  -j 4 \
  --pooling rnn \
  --nb-crops 8 \
  --mask-dir $mask_dir \
  --mask-confidence 0.75 \
  -e >> $result_file
done
