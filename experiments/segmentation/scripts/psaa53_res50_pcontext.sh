#!/usr/bin/env bash
#train
python train_lovasz.py --dataset pcontext \
    --model psaa53 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname psaa53_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model psaa53 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psaa53/psaa53_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psaa53 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psaa53/psaa53_res50_pcontext/model_best.pth.tar --split val --mode testval --ms