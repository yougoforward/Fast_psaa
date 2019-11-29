!/usr/bin/env bash
train
python train.py --dataset cocostuff \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname topk10_aanet_res101_cocostuff

#test [single-scale]
python test.py --dataset cocostuff \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/topk_aanet/topk10_aanet_res101_cocostuff/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset cocostuff \
    --model topk_aanet --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/topk_aanet/topk10_aanet_res101_cocostuff/model_best.pth.tar \
    --split val --mode testval --ms