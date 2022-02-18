#! /bin/bash


python3 train.py --gpu-id 1 --loss MS --model bn_inception --embedding-size 512  --batch-size 90 --lr 1e-4 --dataset $1 --warm 1 --bn-freeze 1 --lr-decay-step 10 --IPC 0
