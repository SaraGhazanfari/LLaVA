#!/bin/bash

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.7-7b \
    --question-file $VAST/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $VAST/eval/scienceqa/images/test \
    --answers-file $VAST/eval/scienceqa/answers/llava-v1.7-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

$SCRATCH/aa10460/pytorch-example/python llava/eval/eval_science_qa.py \
    --base-dir $VAST/eval/scienceqa \
    --result-file $VAST/eval/scienceqa/answers/llava-v1.7-7b.jsonl \
    --output-file $VAST/eval/scienceqa/answers/llava-v1.7-7b_output.jsonl \
    --output-result $VAST/eval/scienceqa/answers/llava-v1.7-7b_result.json
