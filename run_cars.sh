#! /bin/bash


python3 train.py --gpu-id $1 --loss $2 --model resnet50 --embedding-size 512  --batch-size 90 --lr 1e-4 --dataset cars --warm 5 --bn-freeze 1 --lr-decay-step 10  --alphap $3 --alphan $3 --epoch 40 
