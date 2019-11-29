#!/usr/bin/env bash
#train
#python train.py --dataset ade20k \
#    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
#    --backbone resnet101 --checkname topk10_aanet_res101_ade20k

#test [single-scale]
python test.py --dataset ade20k \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/ade20k/topk_aanet/topk10_aanet_res101_ade20k/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset ade20k \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/ade20k/topk_aanet/topk10_aanet_res101_ade20k/model_best.pth.tar \
    --split val --mode testval --ms