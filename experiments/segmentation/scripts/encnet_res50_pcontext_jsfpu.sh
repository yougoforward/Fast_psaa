#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --checkname encnet_res50_pcontext_jsfpu

#test [single-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext_jsfpu/model_best.pth.tar \
    --split val --mode testval