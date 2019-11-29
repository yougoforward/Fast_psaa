#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model fcn --dilated --aux \
    --backbone resnet50 --checkname fcn_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model fcn --dilated --aux \
    --backbone resnet50 --resume runs/pcontext/fcn/fcn_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model fcn --dilated --aux \
    --backbone resnet50 --resume runs/pcontext/fcn/fcn_res50_pcontext/model_best.pth.tar --split val --mode testval --ms