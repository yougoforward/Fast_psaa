#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model fcn_du --aux \
    --backbone resnet50 --checkname fcn_du_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model fcn_du --aux \
    --backbone resnet50 --resume runs/pcontext/fcn_du/fcn_du_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model fcn_du --aux \
    --backbone resnet50 --resume runs/pcontext/fcn_du/fcn_du_res50_pcontext/model_best.pth.tar --split val --mode testval --ms