#!/usr/bin/env bash

#train
python train_metric.py --dataset pcontext \
    --model aanet_metric --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_metric_seloss_jpu_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aanet_metric --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet_metric/aanet_metric_seloss_jpu_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aanet_metric --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet_metric/aanet_metric_seloss_jpu_res101_pcontext/model_best.pth.tar \
    --split val --mode testval --ms