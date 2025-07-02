#!/bin/bash

# QA Experiments Runner Script
# This script runs AB-MCTS experiments on the Simple Bench QA dataset

# Experiment configuration
EXP_ID="qa_abmcts_test"
MAX_NUM_NODES=20
ALGO_CLASS_NAME="ABMCTSA"
DIST_TYPE="beta"
N_JOBS=4
INDICES_FILE="experiments/qa/simple_bench_question_ids.txt"
DATASET_PATH="simple_bench_public.json"

# Create outputs directory
mkdir -p outputs/qa

# Get the list of question IDs
QUESTION_IDS=$(cat $INDICES_FILE)

echo "Starting QA experiments with the following configuration:"
echo "  Experiment ID: $EXP_ID"
echo "  Max nodes: $MAX_NUM_NODES"
echo "  Algorithm: $ALGO_CLASS_NAME"
echo "  Dataset: $DATASET_PATH"
echo "  Questions: $(echo $QUESTION_IDS | wc -w)"
echo "  Parallel jobs: $N_JOBS"

# Function to run a single experiment
run_experiment() {
    local question_id=$1
    echo "Starting experiment for question $question_id"
    
    cd experiments/qa
    python run.py \
        question_id=$question_id \
        dataset_path="../../$DATASET_PATH" \
        max_num_nodes=$MAX_NUM_NODES \
        algo.class_name=$ALGO_CLASS_NAME \
        hydra.run.dir="../../outputs/qa/${EXP_ID}/${question_id}"
    cd - > /dev/null
    
    echo "Completed experiment for question $question_id"
}

# Export function for parallel execution
export -f run_experiment
export EXP_ID
export MAX_NUM_NODES
export ALGO_CLASS_NAME
export DATASET_PATH

# Run experiments in parallel
echo $QUESTION_IDS | tr ' ' '\n' | head -n 10 | parallel -j $N_JOBS run_experiment {}

echo "All QA experiments completed!"
echo "Results saved in outputs/qa/${EXP_ID}/"
echo ""
echo "To process results, run:"
echo "  python eval/proc_qa_results.py --exp_name $EXP_ID" 