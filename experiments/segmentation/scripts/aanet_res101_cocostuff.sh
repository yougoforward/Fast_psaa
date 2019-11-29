#!/usr/bin/env bash

#train
python train.py --dataset cocostuff \
    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_res101_cocostuff

#test [single-scale]
python test.py --dataset cocostuff \
    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/aanet/aanet_res101_cocostuff/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset cocostuff \
    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/aanet/aanet_res101_cocostuff/model_best.pth.tar \
    --split val --mode testval --ms