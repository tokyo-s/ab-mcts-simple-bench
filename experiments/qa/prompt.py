from typing import List, Optional

from ab_mcts_arc2.data_types import Action
from ab_mcts_arc2.llm_generation_interface import GenerationRequest, GenerationResult
from ab_mcts_arc2.eval_result import EvalResultWithAns
from ab_mcts_arc2.prompts.prompt_configs import PromptConfig
from ab_mcts_arc2.prompts.base import PromptTemplate
from ab_mcts_arc2.tasks.qa.task import QAProblem


class QAPrompt(PromptTemplate):
    """Prompt template for multiple choice question answering tasks."""
    
    version = "qa_baseline"

    def __init__(self, prompt_config: PromptConfig, problem: QAProblem) -> None:
        """Initialize the QA prompt template.
        
        Args:
            prompt_config: Configuration for the prompt
            problem: The QA problem instance
        """
        self.problem = problem

    def initial_prompt(self) -> str:
        """Generate the initial prompt for the QA task."""
        prompt = initial_qa_prompt()
        prompt += question_prompt(self.problem)
        return prompt

    def feedback_prompt(
        self,
        action: Action,
        eval_results: Optional[List[EvalResultWithAns]],
        generation_result: GenerationResult,
    ) -> str:
        """Generate feedback prompt based on evaluation results.
        
        Args:
            action: The action type
            eval_results: Results from evaluation (if any)
            generation_result: The LLM's previous response
            
        Returns:
            Feedback prompt string
        """
        match action:
            case "answer":
                return answer_feedback_prompt(
                    problem=self.problem,
                    eval_results=eval_results,
                    previous_response=generation_result.generation,
                )
            case _:
                raise NotImplementedError(
                    f"feedback_prompt not implemented for action {action}"
                )

    def add_next_action_instruction(
        self, action: Action, next_prompt: GenerationRequest
    ) -> GenerationRequest:
        """Add instruction for the next action.
        
        Args:
            action: The action to perform next
            next_prompt: The generation request to modify
            
        Returns:
            Modified generation request
        """
        last_user_msg = next_prompt.messages[-1]
        assert last_user_msg["role"] == "user"
        # Only use the last user message for QA tasks
        next_prompt.messages = next_prompt.messages[-1:]
        return next_prompt


def initial_qa_prompt() -> str:
    """Generate the initial instruction prompt for QA tasks."""
    task_explanation = """
You are an AI assistant tasked with answering multiple choice questions. You will be presented with a question followed by several answer options labeled A through F.

Your goal is to carefully read and understand the question, think through the problem step by step, and select the most appropriate answer from the given options.
"""

    reasoning_instruction = """
Please approach each question methodically:

1. Start by carefully analyzing the question and understanding what is being asked
2. Consider each answer option and evaluate its validity
3. Use logical reasoning to eliminate incorrect options
4. Provide your reasoning in <reasoning></reasoning> tags
5. After your reasoning, clearly state your final answer choice

Your response should follow this format:
<reasoning>
[Your step-by-step reasoning here]
</reasoning>

Answer: [Your chosen letter: A, B, C, D, E, or F]
"""

    quality_guidelines = """
Guidelines for high-quality reasoning:
- Break down complex problems into smaller parts
- Consider edge cases and potential ambiguities
- Use logical deduction and common sense
- Be explicit about your reasoning process
- Double-check your logic before selecting an answer
- If multiple answers seem plausible, explain why you chose one over the others
"""

    return task_explanation + reasoning_instruction + quality_guidelines


def question_prompt(problem: QAProblem) -> str:
    """Generate the prompt containing the specific question.
    
    Args:
        problem: The QA problem instance
        
    Returns:
        Formatted question prompt
    """
    prompt = f"""

## Question

{problem.prompt}

Please provide your reasoning and select the best answer.
"""
    return prompt


def answer_feedback_prompt(
    problem: QAProblem, 
    eval_results: Optional[List[EvalResultWithAns]], 
    previous_response: str
) -> str:
    """Generate feedback prompt based on the previous answer attempt.
    
    This feedback does NOT use knowledge of the correct answer to avoid data leakage.
    Instead, it encourages different reasoning approaches and self-reflection.
    
    Args:
        problem: The QA problem instance
        eval_results: Evaluation results from the previous attempt (not used for correctness)
        previous_response: The LLM's previous response
        
    Returns:
        Feedback prompt string
    """
    prompt = initial_qa_prompt()
    
    if eval_results is None or len(eval_results) == 0:
        prompt += f"\n### Previous Response Analysis\n"
        prompt += f"Your previous response did not contain a clear answer choice. "
        prompt += f"Please make sure to explicitly state your answer as A, B, C, D, E, or F.\n\n"
        prompt += f"Your previous response was:\n{previous_response}\n\n"
    else:
        # We have an extracted answer, but we don't reveal if it's correct or not
        eval_result = eval_results[0]
        predicted_answer = eval_result.answer
        
        prompt += f"\n### Previous Response Analysis\n"
        prompt += f"You previously selected answer '{predicted_answer}'. "
        prompt += f"Let's consider this question from a different angle to ensure "
        prompt += f"we've thoroughly explored all possibilities.\n\n"
        
        prompt += f"Consider these reflection questions:\n"
        prompt += f"- Did you consider all aspects of the question?\n"
        prompt += f"- Are there alternative interpretations you might have missed?\n"
        prompt += f"- Could any of the other answer choices be valid under different reasoning?\n"
        prompt += f"- What assumptions did you make in your previous analysis?\n\n"
    
    # Re-present the question
    prompt += "## Question (Revisited)\n\n"
    prompt += question_prompt(problem)
    
    prompt += "\nPlease reconsider this question with fresh perspective. "
    prompt += "You may reach the same conclusion or a different one - "
    prompt += "focus on providing the most thorough reasoning possible."
    
    return prompt 