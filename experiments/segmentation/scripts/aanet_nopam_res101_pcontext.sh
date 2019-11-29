#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model aanet_nopam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_nopam_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aanet_nopam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/aanet_nopam/aanet_nopam_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aanet_nopam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/aanet_nopam/aanet_nopam_res101_pcontext/model_best.pth.tar \
    --split val --mode testval --ms