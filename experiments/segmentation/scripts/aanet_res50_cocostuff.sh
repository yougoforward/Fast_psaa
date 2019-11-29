#!/usr/bin/env bash

#train
python train.py --dataset cocostuff \
    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname aanet_res50_cocostuff

#test [single-scale]
python test.py --dataset cocostuff \
    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/cocostuff/aanet/aanet_res50_cocostuff/model_best.pth.tar \
    --split val --mode testval

##test [multi-scale]
#python test.py --dataset cocostuff \
#    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
#    --backbone resnet50 --resume runs/cocostuff/aanet/aanet_res50_cocostuff/model_best.pth.tar \
#     --split val --mode testval --ms
#
##predict [single-scale]
#python test.py --dataset cocostuff \
#    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
#    --backbone resnet50 --resume runs/cocostuff/aanet/aanet_res50_cocostuff/model_best.pth.tar \
#     --split val --mode test
#
##predict [multi-scale]
#python test.py --dataset cocostuff \
#    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
#    --backbone resnet50 --resume runs/cocostuff/aanet/aanet_res50_cocostuff/model_best.pth.tarr \
#     --split val --mode test --ms
#
##fps
#CUDA_VISIBLE_DEVICES=0 python test_fps_params.py --dataset cocostuff \
#    --model aanet --aux --dilated --base-size 608 --crop-size 576 \
#    --backbone resnet50
