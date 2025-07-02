import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ab_mcts_arc2.data_types import Action, QAProbData, QAOption
from ab_mcts_arc2.llm_generation_interface import GenerationResult
from ab_mcts_arc2.eval_result import EvalResult, EvalResultWithAns
from ab_mcts_arc2.tasks.base import Task


class QAProblem(Task):
    """Task class for multiple choice question answering."""
    
    def __init__(self, question_data: QAProbData) -> None:
        """Initialize QA problem with question data.
        
        Args:
            question_data: Dictionary containing question_id, prompt, and answer
        """
        self.question_id = question_data["question_id"]
        self.prompt = question_data["prompt"]
        self.correct_answer = question_data["answer"]
        
    @classmethod
    def load_from_dataset(cls, dataset_path: Path | str, question_id: int) -> "QAProblem":
        """Load a specific question from the dataset file.
        
        Args:
            dataset_path: Path to the JSON dataset file
            question_id: ID of the question to load
            
        Returns:
            QAProblem instance for the specified question
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise RuntimeError(f"QA dataset not found at {str(dataset_path)}")
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Find the question with the specified ID
        for question_data in dataset["eval_data"]:
            if question_data["question_id"] == question_id:
                return cls(question_data)
                
        raise RuntimeError(f"Question with ID {question_id} not found in dataset")
    
    def generate_eval_results(
        self, llm_answer: GenerationResult, kind: Action
    ) -> Optional[List[EvalResult]]:
        """Generate evaluation results for the LLM's answer.
        
        For QA tasks, this doesn't really apply since we don't have training examples
        to validate against. We'll return None to indicate no intermediate evaluation.
        """
        if kind == "answer":
            # Extract the answer choice from the LLM response
            predicted_answer = self._extract_answer_choice(llm_answer.generation)
            if predicted_answer is None:
                return None
                
            # Create a simple evaluation result
            eval_result = EvalResultWithAns(
                answer=predicted_answer,
                groundtruth=self.correct_answer
            )
            return [eval_result]
        else:
            # For other actions, return None (no intermediate evaluation)
            return None
    
    def evaluate_on_test(
        self, llm_answer: GenerationResult
    ) -> Tuple[List[EvalResult], float]:
        """Evaluate the LLM's final answer against the correct answer.
        
        Args:
            llm_answer: The LLM's response containing the predicted answer
            
        Returns:
            Tuple of (evaluation results, score)
        """
        predicted_answer = self._extract_answer_choice(llm_answer.generation)
        
        if predicted_answer is None:
            # Could not extract a valid answer
            return [], 0.0
            
        eval_result = EvalResultWithAns(
            answer=predicted_answer,
            groundtruth=self.correct_answer
        )
        
        score = 1.0 if predicted_answer == self.correct_answer else 0.0
        return [eval_result], score
    
    def _extract_answer_choice(self, llm_response: str) -> Optional[QAOption]:
        """Extract the answer choice (A, B, C, D, E, F) from LLM response.
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            The extracted answer choice or None if not found
        """
        # Look for patterns like "Answer: A" or "The answer is B" or just "A"
        patterns = [
            r"(?:answer\s*:?\s*|the\s+answer\s+is\s*:?\s*|i\s+choose\s*:?\s*)([A-F])",
            r"\b([A-F])\b(?:\s*(?:is|\.|\s*$))",  # Single letter followed by space, period, or end
            r"^([A-F])$",  # Just the letter alone
        ]
        
        # Convert to uppercase for case-insensitive matching
        response_upper = llm_response.upper()
        
        for pattern in patterns:
            matches = re.findall(pattern, response_upper, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Return the first valid match
                answer = matches[0].upper()
                if answer in ["A", "B", "C", "D", "E", "F"]:
                    return answer
        
        # If no pattern matches, look for the last occurrence of any answer choice
        # This handles cases where reasoning is followed by a final answer
        all_choices = re.findall(r'\b([A-F])\b', response_upper)
        if all_choices:
            return all_choices[-1]  # Return the last mentioned choice
            
        return None 