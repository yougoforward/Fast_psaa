#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model psaa7 --aux --dilated --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname psaa7_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model psaa7 --aux --dilated --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psaa7/psaa7_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psaa7 --aux --dilated --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psaa7/psaa7_res50_pcontext/model_best.pth.tar --split val --mode testval --ms