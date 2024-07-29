#!/bin/bash

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.7-7b \
    --question-file $VAST/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $VAST/eval/mm-vet/images \
    --answers-file $VAST/eval/mm-vet/answers/llava-v1.7-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $VAST/eval/mm-vet/results

$SCRATCH/aa10460/pytorch-example/python scripts/convert_mmvet_for_eval.py \
    --src $VAST/eval/mm-vet/answers/llava-v1.7-7b.jsonl \
    --dst $VAST/eval/mm-vet/results/llava-v1.7-7b.json

