#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model aspoc_gsecam --aux --dilated --base-size 520 --crop-size 480 \
    --backbone resnet50 --checkname aspoc_gsecam_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aspoc_gsecam --aux --dilated --base-size 520 --crop-size 480 \
    --backbone resnet50 --resume runs/pcontext/aspoc_gsecam/aspoc_gsecam_res50_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aspoc_gsecam --aux --dilated --base-size 520 --crop-size 480 \
    --backbone resnet50 --resume runs/pcontext/aspoc_gsecam/aspoc_gsecam_res50_pcontext/model_best.pth.tar \
     --split val --mode testval --ms