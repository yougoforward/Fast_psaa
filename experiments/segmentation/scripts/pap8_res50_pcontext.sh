#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model pap8 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname pap8_res50_pcontext --no-val

#test [single-scale]
python test.py --dataset pcontext \
    --model pap8 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/pap8/pap8_res50_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model pap8 --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/pap8/pap8_res50_pcontext/checkpoint.pth.tar --split val --mode testval --ms