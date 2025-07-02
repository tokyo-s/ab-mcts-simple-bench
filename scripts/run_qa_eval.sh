#!/bin/bash

# QA Evaluation Script
# This script processes QA experiment results and generates visualizations

EXP_ID="qa_abmcts_test"

echo "Processing QA experiment results for: $EXP_ID"

# Process results
python eval/proc_qa_results.py \
    --exp_name $EXP_ID \
    --n_jobs 4 \
    --dataset_path "simple_bench_public.json"

echo "QA evaluation completed!"
echo "Results saved in outputs/qa/${EXP_ID}/" 