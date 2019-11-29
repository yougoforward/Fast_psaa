#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model aspoc_gsecam --aux --dilated --base-size 520 --crop-size 480 \
    --backbone resnet101 --checkname aspoc_gsecam_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aspoc_gsecam --aux --dilated --base-size 520 --crop-size 480 \
    --backbone resnet101 --resume runs/pcontext/aspoc_gsecam/aspoc_gsecam_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aspoc_gsecam --aux --dilated --base-size 520 --crop-size 480 \
    --backbone resnet101 --resume runs/pcontext/aspoc_gsecam/aspoc_gsecam_res101_pcontext/model_best.pth.tar \
     --split val --mode testval --ms