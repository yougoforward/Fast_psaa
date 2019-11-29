#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model aanet_simple --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_simple_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aanet_simple --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet_simple/aanet_simple_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aanet_simple --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet_simple/aanet_simple_res101_pcontext/model_best.pth.tar \
    --split val --mode testval --ms