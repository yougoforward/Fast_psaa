#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model psaa35 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname psaa35_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model psaa35 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psaa35/psaa35_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psaa35 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psaa35/psaa35_res50_pcontext/model_best.pth.tar --split val --mode testval --ms