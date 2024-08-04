#!/bin/bash

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/var-v1.7-7b \
    --question-file $VAST/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $VAST/eval/textvqa/train_images \
    --answers-file $VAST/eval/textvqa/answers/var-v1.7-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.eval_textvqa \
    --annotation-file $VAST/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $VAST/eval/textvqa/answers/var-v1.7-7b.jsonl
