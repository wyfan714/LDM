#! /bin/bash


python3 train.py --gpu-id $1 --loss $2 --model bn_inception --embedding-size 512  --batch-size 90 --lr 6e-4 --dataset $3 --warm 1 --bn-freeze 0 --lr-decay-step 20 --lr-decay-gamma 0.25  --epoch 60
