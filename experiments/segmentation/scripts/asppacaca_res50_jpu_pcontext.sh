#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model asppacaca --jpu --aux \
    --backbone resnet50 --checkname asppacaca_res50_pcontext_jpu

#test [single-scale]
python test.py --dataset pcontext \
    --model asppacaca --jpu --aux \
    --backbone resnet50 --resume runs/pcontext/asppacaca/asppacaca_res50_pcontext_jpu/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model asppacaca --jpu --aux \
    --backbone resnet50 --resume runs/pcontext/asppacaca/asppacaca_res50_pcontext_jpu/model_best.pth.tar --split val --mode testval --ms