#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model pap7 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname pap7_res50_pcontext --no-val

#test [single-scale]
python test.py --dataset pcontext \
    --model pap7 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/pap7/pap7_res50_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model pap7 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/pap7/pap7_res50_pcontext/checkpoint.pth.tar --split val --mode testval --ms