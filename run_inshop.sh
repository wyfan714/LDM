#! /bin/bash


python3 train.py --gpu-id $1 --loss $2 --model resnet50 --embedding-size 512  --batch-size 120 --lr 6e-4 --dataset Inshop --warm 5 --bn-freeze 0 --lr-decay-step 10 --lr-decay-gamma 0.5  --epoch 60
