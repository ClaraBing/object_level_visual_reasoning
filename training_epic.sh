#!/usr/bin/env bash

# Envs
# source activate pytorch-0.4.0

cd /sailhome/bingbin/object_level_visual_reasoning;

# Pythonpath
PYTHONPATH=.

machine='tibet6'
# machine='macondo2'
CUDA_ID=0
task_id=0

stage1=true
stage2=false
perform_test=true

lr='1e-04'

# Settings
resume=/vision2/u/bingbin/ORN/ckpt
root=./data/epic/
save_token='pretrainedContext_twoLayer_WVInit'
# mask_dir='masks/sampled_1perClip_GT'
mask_dir='masks/sampled_1perClip_reformat'
# mask_dir='masks/preds_100x100_50'


dataset="epic"
use_obj_rel=0
use_obj_logits=0
gcn_version='v2'
use_obj_gcn=0
use_context_gcn=0
two_layer_context=1
use_wv_weights=1
freeze_wv_weights=0
adj_type='wv'

heads=context

if $stage1;
then
  # Train the object head only with f=MLP
  epochs=10
  # heads=object
  CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --resume $resume \
  --machine $machine \
  --gpu-id $CUDA_ID \
  --task-id $task_id \
  --root $root \
  --save-token $save_token \
  --blocks 2D_2D_2D_2.5D \
  --object-head 2D \
  --add-background \
  --train-set train+val \
  --arch orn_two_heads \
  --use-obj-rel $use_obj_rel \
  --use-obj-logits $use_obj_logits \
  --two-layer-context $two_layer_context \
  --use-wv-weights $use_wv_weights \
  --freeze-wv-weights $freeze_wv_weights \
  --gcn-version $gcn_version \
  --use-obj-gcn=$use_obj_gcn \
  --use-context-gcn=$use_context_gcn \
  --adj-type=$adj_type \
  --depth 50 \
  -t 4 \
  -b 1 \
  --cuda \
  --dataset $dataset \
  --heads $heads \
  --epochs $epochs \
  --pooling avg \
  --print-freq 100 \
  --pooling avg \
  --mask-dir $mask_dir \
  --mask-confidence 0.5 \
  -j 4
else
  echo "Skip training stage 1.";
fi



if $stage2;
then
  # ckpt_name='feats_epic_heads'$heads'_gcnObj'$use_obj_gcn'_gcnCtxt'$use_context_gcn'_bt8_lr'$lr'_wd1e-05_adjprior_oEmb32_vEmb32_nLayer2_nTopObjs10_'$save_token'/ckpt_epoch10_best.pth'

  # Train the two heads with f=RNN and pooling is avg for context head
  epochs=10
  heads=object+context
  CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --resume $resume \
  --machine $machine \
  --gpu-id $CUDA_ID \
  --task-id $task_id \
  --root $root \
  --save-token $save_token \
  --blocks 2D_2D_2D_2.5D \
  --object-head 2D \
  --add-background \
  --train-set train+val \
  --arch orn_two_heads \
  --use-obj-rel $use_obj_rel \
  --use-obj-logits $use_obj_logits \
  --two-layer-context $two_layer_context \
  --use-wv-weights $use_wv_weights \
  --freeze-wv-weights $freeze_wv_weights \
  --gcn-version $gcn_version \
  --use-obj-gcn=$use_obj_gcn \
  --use-context-gcn=$use_context_gcn \
  --adj-type=$adj_type \
  --depth 50 \
  -t 4 \
  -b 4 \
  --cuda \
  --dataset $dataset \
  --heads $heads \
  --epochs $epochs \
  --print-freq 100 \
  --pooling rnn \
  --mask-dir $mask_dir \
  --mask-confidence 0.5 \
  -j 4
else
  echo "Skip training stage 2.";
fi

if $perform_test;
then
  echo "Begin testing.";
  # ckpt_name='feats_epic_heads'$heads'_gcnObj'$use_obj_gcn'_gcnCtxt'$use_context_gcn'_bt4_lr'$lr'_wd1e-05_adjprior_oEmb32_vEmb32_nLayer2_nTopObjs10_'$save_token'/ckpt_epoch10_best.pth'
 # Finally validate on the test set
  epochs=10
  heads=$heads
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
  --use-obj-logits $use_obj_logits \
  --two-layer-context $two_layer_context \
  --use-wv-weights $use_wv_weights \
  --freeze-wv-weights $freeze_wv_weights \
  --gcn-version $gcn_version \
  --use-obj-gcn=$use_obj_gcn \
  --use-context-gcn=$use_context_gcn \
  --adj-type=$adj_type \
  --depth 50 \
  -t 4 \
  -b 4 \
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
  -e
else
  echo "Skip testing.";
fi

# NOTE: the only diff between testing here and the testing .sh is mask-confidence:
# 0.5 here and 0.75 there; copying over that setting to here.
