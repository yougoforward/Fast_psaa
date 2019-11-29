#!/usr/bin/env bash
#train
# python train.py --dataset pcontext \
#     --model psp --aux --dilated --base-size 608 --crop-size 576 \
#     --backbone resnet50 --checkname psp_res50_pcontext --no-val

#test [single-scale]
python test.py --dataset pcontext \
    --model psp --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psp/psp_res50_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psp --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/psp/psp_res50_pcontext/checkpoint.pth.tar --split val --mode testval --ms