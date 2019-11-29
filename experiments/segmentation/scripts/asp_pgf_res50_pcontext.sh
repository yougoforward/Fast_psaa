#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model asp_pgfnet --jpu --aux --se-loss --base-size 520 --crop-size 480 \
    --backbone resnet50 --checkname asp_pgfnet_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model asp_pgfnet --jpu --aux --se-loss --base-size 520 --crop-size 480 \
    --backbone resnet50 --resume runs/pcontext/asp_pgfnet/asp_pgfnet_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model asp_pgfnet --jpu --aux --se-loss --base-size 520 --crop-size 480 \
    --backbone resnet50 --resume runs/pcontext/asp_pgfnet/asp_pgfnet_res50_pcontext/model_best.pth.tar --split val --mode testval --ms