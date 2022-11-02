#!/bin/bash

declare -a arr=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut'
            'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
for i in "${arr[@]}";
do
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset $i  --niter 300 --isize 64 --datasetroot mvtec --repetition 5 --d_th 0.99 --training_mode AAT --batchsize 128 --p1 50 --p2 30
done
exit 0
