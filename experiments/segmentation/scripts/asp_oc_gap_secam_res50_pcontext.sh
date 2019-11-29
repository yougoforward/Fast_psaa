#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model asp_oc_gap_secam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname asp_oc_gap_secam_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model asp_oc_gap_secam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/asp_oc_gap_secam/asp_oc_gap_secam_res50_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model asp_oc_gap_secam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/asp_oc_gap_secam/asp_oc_gap_secam_res50_pcontext/model_best.pth.tar \
     --split val --mode testval --ms#!/usr/bin/env bash