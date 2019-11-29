#!/usr/bin/env bash

#train
python train.py --dataset cityscapes --batch-size 8 \
    --model aanet --aux --dilated --base-size 1024 --crop-size 768 \
    --backbone resnet50 --checkname aanet_res50_cityscapes

#test [single-scale]
python test.py --dataset cityscapes --batch-size 8 \
    --model aanet --aux --dilated --base-size 2048 --crop-size 768 \
    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_res50_cityscapes/model_best.pth.tar \
    --split val --mode testval

#test [single-scale]
python test.py --dataset cityscapes --batch-size 8 \
    --model aanet --aux --dilated --base-size 2048 --crop-size 1024 \
    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_res50_cityscapes/model_best.pth.tar \
    --split val --mode testval --ms

##test [multi-scale]
#python test.py --dataset cityscapes --batch-size 8 \
#    --model aanet --aux --dilated --base-size 2048 --crop-size 1024 \
#    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_res50_cityscapes/model_best.pth.tar \
#     --split val --mode testval --ms
#
##predict [single-scale]
#python test.py --dataset cityscapes --batch-size 8 \
#    --model aanet --aux --dilated --base-size 2048 --crop-size 768 \
#    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_res50_cityscapes/model_best.pth.tar \
#     --split val --mode test
#
##predict [multi-scale]
#python test.py --dataset cityscapes --batch-size 8 \
#    --model aanet --aux --dilated --base-size 2048 --crop-size 1024 \
#    --backbone resnet50 --resume runs/cityscapes/aanet/aanet_res50_cityscapesmodel_best.pth.tarr \
#     --split val --mode test --ms
#
##fps
#CUDA_VISIBLE_DEVICES=0 python test_fps_params.py --dataset cityscapes \
#    --model aanet --aux --dilated --base-size 2048 --crop-size 768 \
#    --backbone resnet50
