#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/llava-v1.7-7b \
    --question-file $VAST/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file $VAST/eval/mmbench_cn/answers/$SPLIT/llava-v1.7-7b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $VAST/eval/mmbench_cn/answers_upload/$SPLIT

$SCRATCH/aa10460/pytorch-example/python scripts/convert_mmbench_for_submission.py \
    --annotation-file $VAST/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir $VAST/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir $VAST/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment llava-v1.7-7b
