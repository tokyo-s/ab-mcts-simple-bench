from typing import Dict, List, Literal, Tuple, TypeAlias, get_args

TaskType: TypeAlias = Literal["train", "test"]

Action: TypeAlias = Literal["transform", "question", "multi_questions", "answer"]
ACTIONS: Tuple[Action, ...] = get_args(Action)

GridType: TypeAlias = Literal["input", "output"]
Grid: TypeAlias = List[List[int]]

ARCProbData: TypeAlias = Dict[TaskType, List[Dict[GridType, Grid]]]

ScorerType: TypeAlias = Literal["default", "verifier"]

# New types for QA tasks
QAOption: TypeAlias = Literal["A", "B", "C", "D", "E", "F"]
QAProbData: TypeAlias = Dict[str, str | int]  # question_id, prompt, answer
QADataset: TypeAlias = List[QAProbData]
