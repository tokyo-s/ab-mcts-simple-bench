import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm

from ab_mcts_arc2.tasks.qa.task import QAProblem

sys.path.append("./experiments/qa")
from utils import NodeState


def get_private_score(task: QAProblem, node_state: NodeState | None) -> float:
    """Calculate the private score for a QA node state.
    
    Args:
        task: The QA problem instance
        node_state: The node state to evaluate
        
    Returns:
        The private score (0.0 or 1.0 for QA tasks)
    """
    if node_state is not None:
        eval_results, _score = task.evaluate_on_test(
            llm_answer=node_state.generation_result
        )
        if len(eval_results) == 0:
            private_score = 0
        else:
            private_score = sum(
                [eval_result.get_score() for eval_result in eval_results]
            ) / len(eval_results)
    else:
        private_score = 0
    return private_score


def process_node(node, task: QAProblem):
    """Helper function to process a single node and calculate scores.
    
    Args:
        node: The tree node to process
        task: The QA problem instance
        
    Returns:
        Tuple of (node_idx, public_score, private_score) or None
    """
    if node.expand_idx < 0:
        return None

    node_idx = node.expand_idx
    public_score = node.score

    # calc private score
    node_state = node.state
    private_score = get_private_score(task, node_state)

    return node_idx, public_score, private_score


def main(
    exp_name: str,
    n_jobs: int = 4,
    path_to_ckpt: str = "./outputs/qa/{exp_name}/{question_id}/checkpoints/checkpoint_n_answers_32.pkl",
    save_path: str = "./outputs/qa/{exp_name}/",
    path_to_qa_questions: str = "./experiments/qa/simple_bench_question_ids.txt",
    dataset_path: str = "./simple_bench_public.json",
):
    """Process QA experiment results.
    
    Args:
        exp_name: Name of the experiment
        n_jobs: Number of parallel jobs
        path_to_ckpt: Path template for checkpoint files
        save_path: Path template for saving results
        path_to_qa_questions: Path to file containing question IDs
        dataset_path: Path to the QA dataset
    """
    with open(path_to_qa_questions, "r") as f:
        question_list = f.readlines()
    question_list = [int(q.strip()) for q in question_list]

    node_idx_dict = {}
    public_scores_dict = {}
    private_scores_dict = {}
    
    for question_id in question_list:
        task = QAProblem.load_from_dataset(dataset_path, question_id)

        # load state
        state_path = Path(path_to_ckpt.format(exp_name=exp_name, question_id=question_id))
        proc_ret_path = Path(state_path.as_posix().replace(".pkl", "_proc_result.json"))
        
        if not state_path.exists():
            print(f"State path {state_path} does not exist")
            continue
            
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        if proc_ret_path.exists():
            print(f"Processing result {proc_ret_path} already exists")
            with open(proc_ret_path, "r") as f:
                proc_ret = json.load(f)
            node_idx_list = proc_ret["node_idx_list"]
            public_scores = proc_ret["public_scores"]
            private_scores = proc_ret["private_scores"]
        else:
            # Filter nodes with expand_idx >= 0 before processing
            valid_nodes = [
                node for node in state.tree.get_nodes() if node.expand_idx >= 0
            ]

            # Parallel processing of nodes
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(process_node)(node, task)
                for node in tqdm(valid_nodes, desc=f"Processing question {question_id}")
            )

            # Filter out None results
            results = [r for r in results if r is not None]

            # Sort results by node_idx
            results.sort(key=lambda x: x[0])

            # Unpack sorted results
            node_idx_list = []
            public_scores = []
            private_scores = []
            for result in results:
                node_idx, public_score, private_score = result
                node_idx_list.append(node_idx)
                public_scores.append(public_score)
                private_scores.append(private_score)
                
            proc_ret = {
                "node_idx_list": node_idx_list,
                "public_scores": public_scores,
                "private_scores": private_scores,
            }
            with open(proc_ret_path, "w") as f:
                json.dump(proc_ret, f)

        node_idx_dict[question_id] = node_idx_list
        public_scores_dict[question_id] = public_scores
        private_scores_dict[question_id] = private_scores

    df_reward = pd.DataFrame(public_scores_dict)
    df_test = pd.DataFrame(private_scores_dict)

    # Save results
    save_dir = save_path.format(exp_name=exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    df_reward.to_csv(os.path.join(save_dir, "df_reward.csv"), index=False)
    df_test.to_csv(os.path.join(save_dir, "df_test.csv"), index=False)

    print(f"Saved to {save_dir}")
    
    # Calculate and print results
    correct_answers = df_test.max(0).sum()
    total_questions = df_test.shape[1]
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
    print(f"Results: {correct_answers} / {total_questions} correct")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Public scores sum: {df_reward.max(0).sum()}")


if __name__ == "__main__":
    Fire(main) 