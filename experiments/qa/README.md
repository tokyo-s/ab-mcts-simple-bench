# QA Experiments with AB-MCTS

This directory contains the adaptation of the AB-MCTS algorithm for multiple choice question answering tasks.

## ⚠️ **Important: No Data Leakage Design**

**This implementation strictly avoids data leakage by not using ground truth during tree search.**

Unlike the original ARC implementation where models can test code against provided examples, QA benchmarks only provide the question and final answer. Using the correct answer to provide feedback during search would constitute data leakage.

**Our approach:**
- ✅ Encourage different reasoning perspectives without revealing correctness
- ✅ Use self-reflection prompts to explore alternative interpretations  
- ✅ Let multiple models explore different reasoning paths independently
- ❌ **Never** tell models if their answers are right or wrong during search
- ❌ **Never** reveal the ground truth answer until final evaluation

This makes the QA adaptation more challenging but ensures fair evaluation on benchmarks.

## Overview

The QA implementation adapts the original ARC-AGI-2 AB-MCTS algorithm for multiple choice questions. Instead of generating Python code to transform grids, the system:

1. **Analyzes questions**: LLMs read and understand multiple choice questions
2. **Reasons through options**: Uses structured reasoning to evaluate answer choices 
3. **Tree search**: AB-MCTS explores different reasoning approaches
4. **Multi-LLM collaboration**: Different models provide diverse perspectives
5. **Feedback refinement**: Incorrect answers trigger re-reasoning with feedback

## Key Differences from ARC

- **Task Type**: Text reasoning instead of visual pattern recognition
- **Output Format**: Letter choices (A-F) instead of Python code
- **Evaluation**: Direct answer matching instead of code execution
- **Feedback**: Reasoning analysis instead of execution results

## Files Structure

```
experiments/qa/
├── README.md           # This file
├── run.py             # Main experiment runner
├── prompt.py          # QA-specific prompts
├── utils.py           # QA utilities
├── configs/
│   └── config.yaml    # Experiment configuration
└── simple_bench_question_ids.txt  # Question IDs to run
```

## Setup

1. **Place your dataset**: Put `simple_bench_public.json` in the project root
2. **Configure models**: Edit `configs/config.yaml` to set your preferred models
3. **Set question IDs**: Modify `simple_bench_question_ids.txt` for your questions

## Running Experiments

### Single Question
```bash
cd experiments/qa
python run.py question_id=1 dataset_path="../../simple_bench_public.json"
```

### Multiple Questions (Parallel)
```bash
# From project root
./scripts/run_qa_experiments.sh
```

### Custom Configuration
```bash
cd experiments/qa
python run.py \
    question_id=5 \
    max_num_nodes=30 \
    dataset_path="../../simple_bench_public.json" \
    algo.class_name="ABMCTSA"
```

## Processing Results

```bash
# From project root
./scripts/run_qa_eval.sh

# Or manually:
python eval/proc_qa_results.py --exp_name qa_abmcts_test
```

## Configuration Options

### Key Parameters in `config.yaml`:

- `question_id`: Which question to solve (1-10 for simple_bench)
- `dataset_path`: Path to your QA dataset JSON file
- `max_num_nodes`: Maximum search tree nodes (20 recommended for QA)
- `models`: List of LLMs to use with temperatures
- `algo.class_name`: Search algorithm ("ABMCTSA" recommended)

### Model Configuration:
```yaml
models:
  - name: "o4-mini-2025-04-16"
    temperature: 0.7
  - name: "gemini-2.5-pro-preview-05-06" 
    temperature: 0.7
```

## Expected Dataset Format

Your JSON dataset should follow this structure:
```json
{
  "eval_data": [
    {
      "question_id": 1,
      "prompt": "Question text with options A. B. C. D. E. F.",
      "answer": "B"
    }
  ]
}
```

## How It Works

1. **Initial Reasoning**: LLM analyzes the question and provides structured reasoning
2. **Answer Extraction**: System extracts the chosen letter (A-F) from the response
3. **Tree Search**: AB-MCTS explores alternative reasoning paths if the answer is wrong
4. **Multi-LLM**: Different models contribute diverse reasoning approaches
5. **Feedback Loop**: Incorrect answers trigger re-analysis with feedback

## Benefits of AB-MCTS for QA

- **Robust Reasoning**: Multiple attempts at complex questions
- **Model Diversity**: Combines strengths of different LLMs
- **Self-Correction**: Learns from wrong answers to improve reasoning
- **Systematic Exploration**: Tree search prevents getting stuck in one approach

## Example Output

```
Question 1: Beth places four whole ice cubes...
Initial attempt: Model reasoning -> Answer: A (Incorrect)
Feedback attempt: Refined reasoning -> Answer: B (Correct)
Final result: 1/1 correct
```

## Tips for Best Results

1. **Temperature**: Use 0.7 for good reasoning diversity
2. **Max Nodes**: 20-30 nodes usually sufficient for QA
3. **Model Mix**: Combine reasoning-focused models (Claude, GPT-4) with others
4. **Question Selection**: Start with a few questions to test setup

## Troubleshooting

- **Import errors**: Ensure you're running from the correct directory
- **Dataset not found**: Check the `dataset_path` in your config
- **No answer extracted**: Check the answer extraction patterns in `task.py`
- **LLM API errors**: Verify your API keys and model names 