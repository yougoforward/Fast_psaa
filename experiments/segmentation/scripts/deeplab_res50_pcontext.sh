#!/usr/bin/env bash

#train
#python train.py --dataset pcontext \
#    --model deeplab --jpu --aux \
#    --backbone resnet50 --checkname deeplab_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model deeplab --jpu --aux \
    --backbone resnet50 --resume runs/pcontext/deeplab/deeplab_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model deeplab --jpu --aux \
    --backbone resnet50 --resume runs/pcontext/deeplab/deeplab_res50_pcontext/model_best.pth.tar --split val --mode testval --ms

##predict [single-scale]
#python test.py --dataset pcontext \
#    --model deeplab --jpu --aux \
#    --backbone resnet50 --resume runs/pcontext/deeplab/deeplab_res50_pcontext/model_best.pth.tar --split val --mode test
#
##predict [multi-scale]
#python test.py --dataset pcontext \
#    --model deeplab --jpu --aux \
#    --backbone resnet50 --resume runs/pcontext/deeplab/deeplab_res50_pcontext/model_best.pth.tar --split val --mode test --ms

##fps
#python test_fps_params.py --dataset pcontext \
#    --model deeplab --jpu --aux \
#    --backbone resnet50
