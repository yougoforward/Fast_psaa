#!/usr/bin/env bash

#train
python train.py --dataset ade20k \
    --model aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_seloss_jpu_res101_ade20k

#test [single-scale]
python test.py --dataset ade20k \
    --model aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/ade20k/aanet/aanet_seloss_jpu_res101_ade20k/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset ade20k \
    --model aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/ade20k/aanet/aanet_seloss_jpu_res101_ade20k/model_best.pth.tar \
    --split val --mode testval --ms