#! /bin/bash


python3 train.py --gpu-id $1 --loss $2 --model resnet50 --embedding-size 256  --batch-size 150 --lr 6e-4 --dataset SOP --warm 5 --bn-freeze 0 --lr-decay-step 10 --lr-decay-gamma 0.5  --epoch 60
