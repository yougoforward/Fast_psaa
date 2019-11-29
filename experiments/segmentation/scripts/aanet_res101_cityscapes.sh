#!/usr/bin/env bash

#train
python train.py --dataset cityscapes --batch-size 8 \
    --model aanet --aux --dilated --base-size 1024 --crop-size 768 \
    --backbone resnet101 --checkname aanet_res101_cityscapes

#test [single-scale]
python test.py --dataset cityscapes --batch-size 8 \
    --model aanet --aux --dilated --base-size 2048 --crop-size 768 \
    --backbone resnet101 --resume runs/cityscapes/aanet/aanet_res101_cityscapes/model_best.pth.tar \
    --split val --mode testval

#test [single-scale]
python test.py --dataset cityscapes --batch-size 8 \
    --model aanet --aux --dilated --base-size 2048 --crop-size 1024 \
    --backbone resnet101 --resume runs/cityscapes/aanet/aanet_res101_cityscapes/model_best.pth.tar \
    --split val --mode testval --ms