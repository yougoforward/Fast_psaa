#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --checkname encnet_res50_pcontext_jfpu

#test [single-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext_jfpu/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext_jfpu/model_best.pth.tar \
     --split val --mode testval --ms

#predict [single-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext_jfpu/model_best.pth.tar \
     --split val --mode test

#predict [multi-scale]
python test.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume runs/pcontext/encnet/encnet_res50_pcontext_jfpu/model_best.pth.tar \
     --split val --mode test --ms

#fps
CUDA_VISIBLE_DEVICES=0 python test_fps_params.py --dataset pcontext \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50
