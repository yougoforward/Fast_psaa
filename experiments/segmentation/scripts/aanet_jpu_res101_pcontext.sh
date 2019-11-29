#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model aanet --aux --jpu --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_seloss_jpu_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aanet --aux --jpu --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet/aanet_seloss_jpu_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aanet --aux --jpu --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet/aanet_seloss_jpu_res101_pcontext/model_best.pth.tar \
    --split val --mode testval --ms
