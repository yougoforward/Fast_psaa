#!/usr/bin/env bash

#train
python train.py --dataset cityscapes \
    --model aanet --aux --jpu --se-loss --batch-size 8 --base-size 1024 --crop-size 768 --lr 0.004 \
    --backbone resnet50 --checkname aanet_seloss_jpu_res50_cityscapes

#test [single-scale]
python test.py --dataset cityscapes \
    --model aanet --aux --jpu --se-loss --base-size 2048 --crop-size 768 \
    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_seloss_jpu_res50_cityscapes/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset cityscapes \
    --model aanet --aux --jpu --se-loss --base-size 2048 --crop-size 1024 \
    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_seloss_jpu_res50_cityscapes/model_best.pth.tar \
    --split val --mode testval --ms