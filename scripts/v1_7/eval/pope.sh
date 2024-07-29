#!/bin/bash

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.7-7b \
    --question-file $VAST/eval/pope/llava_pope_test.jsonl \
    --image-folder $VAST/eval/pope/val2014 \
    --answers-file $VAST/eval/pope/answers/llava-v1.7-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

$SCRATCH/aa10460/pytorch-example/python llava/eval/eval_pope.py \
    --annotation-dir $VAST/eval/pope/coco \
    --question-file $VAST/eval/pope/llava_pope_test.jsonl \
    --result-file $VAST/eval/pope/answers/llava-v1.7-7b.jsonl
