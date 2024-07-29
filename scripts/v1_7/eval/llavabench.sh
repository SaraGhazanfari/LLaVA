#!/bin/bash

$SCRATCH/aa10460/pytorch-example/python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-v1.7-7b \
    --question-file $VAST/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $VAST/eval/llava-bench-in-the-wild/images \
    --answers-file $VAST/eval/llava-bench-in-the-wild/answers/llava-v1.7-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $VAST/eval/llava-bench-in-the-wild/reviews

$SCRATCH/aa10460/pytorch-example/python llava/eval/eval_gpt_review_bench.py \
    --question $VAST/eval/llava-bench-in-the-wild/questions.jsonl \
    --context $VAST/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        $VAST/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $VAST/eval/llava-bench-in-the-wild/answers/llava-v1.7-7b.jsonl \
    --output \
        $VAST/eval/llava-bench-in-the-wild/reviews/llava-v1.7-7b.jsonl

$SCRATCH/aa10460/pytorch-example/python llava/eval/summarize_gpt_review.py -f $VAST/eval/llava-bench-in-the-wild/reviews/llava-v1.7-7b.jsonl
