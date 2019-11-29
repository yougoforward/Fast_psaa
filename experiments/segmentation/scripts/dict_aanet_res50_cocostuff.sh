#!/usr/bin/env bash

#train
python train.py --dataset cocostuff \
    --model dict_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname dict_aanet_res50_cocostuff

#test [single-scale]
python test.py --dataset cocostuff \
    --model dict_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/cocostuff/dict_aanet/dict_aanet_res50_cocostuff/model_best.pth.tar \
    --split val --mode testval