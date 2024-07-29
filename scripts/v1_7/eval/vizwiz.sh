#!/bin/bash

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.7-7b \
    --question-file $VAST/eval/vizwiz/llava_test.jsonl \
    --image-folder $VAST/eval/vizwiz/test \
    --answers-file $VAST/eval/vizwiz/answers/llava-v1.7-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

$SCRATCH/aa10460/pytorch-example/python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $VAST/eval/vizwiz/llava_test.jsonl \
    --result-file $VAST/eval/vizwiz/answers/llava-v1.7-7b.jsonl \
    --result-upload-file $VAST/eval/vizwiz/answers_upload/llava-v1.7-7b.json
