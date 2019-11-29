#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname topk20_aanet_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/topk_aanet/topk20_aanet_res50_pcontext/model_best.pth.tar \
    --split val --mode testval