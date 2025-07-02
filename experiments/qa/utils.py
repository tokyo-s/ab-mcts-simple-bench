import math
from dataclasses import dataclass

import treequest as tq

from ab_mcts_arc2.llm_generation_interface import GenerationResult
from ab_mcts_arc2.eval_result import EvalResult
from ab_mcts_arc2.tasks.qa.task import QAProblem


def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def get_top_k(state, algorithm, k: int):
    """Get top k nodes from the search state."""
    if hasattr(state, "tree"):
        # nodes in ascending order
        nodes = state.tree.get_nodes()
        nodes.sort(key=lambda x: x.score, reverse=True)
        top_k = [(node.state, node.score) for node in nodes[:k]]
        return top_k
    else:
        return tq.top_k(state, algorithm, k=k)


@dataclass
class NodeState:
    """State representation for nodes in the search tree."""
    generation_result: GenerationResult
    eval_results: EvalResult
    model_name: str 