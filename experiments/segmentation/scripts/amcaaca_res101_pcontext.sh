#!/usr/bin/env bash
#train
#python train.py --dataset pcontext \
#    --model amcaaca --aux --dilated --base-size 608 --crop-size 576 \
#    --backbone resnet101 --checkname amcaaca_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model amcaaca --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/amcaaca/amcaaca_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model amcaaca --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/amcaaca/amcaaca_res101_pcontext/model_best.pth.tar --split val --mode testval --ms