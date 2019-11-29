#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --checkname encnet_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume runs/pcontext/encnet/encnet_res101_pcontext/model_best.pth.tar \
     --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume runs/pcontext/encnet/encnet_res101_pcontext/model_best.pth.tar \
     --split val --mode testval --ms

#predict [single-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume runs/pcontext/encnet/encnet_res101_pcontext/model_best.pth.tar \
     --split val --mode test

#predict [multi-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume runs/pcontext/encnet/encnet_res101_pcontext/model_best.pth.tar \
     --split val --mode test --ms

#fps
CUDA_VISIBLE_DEVICES=0 python test_fps_params.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101
