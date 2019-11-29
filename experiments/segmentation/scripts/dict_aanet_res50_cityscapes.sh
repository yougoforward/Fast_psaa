#!/usr/bin/env bash

#train
python train.py --dataset cityscapes \
    --model dict_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname dict_aanet_res50_cityscapes

#test [single-scale]
python test.py --dataset cityscapes \
    --model dict_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/dict_aanet/dict_aanet_res50_cityscapes/model_best.pth.tar \
    --split val --mode testval