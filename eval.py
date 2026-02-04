from __future__ import annotations
import os
from typing import Any
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from main import run 

EVALUATOR_MODEL = "google-gla:gemini-2.0-flash"

rag_dataset = Dataset(
    cases=[
        Case(
            name="check_plan",
            inputs="What is my plan?",
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric="Review should confirm if the answer correctly identifies the plan from the docs.",
                    model=EVALUATOR_MODEL,
                ),
            ),
        ),
        Case(
            name="out_of_scope",
            inputs="What is my age?",
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric="Since this is not in the docs, the answer should be 'Not found'.",
                    model=EVALUATOR_MODEL,
                ),
            ),
        ),
    ],
    # 全局准则
    evaluators=[
        LLMJudge(
            rubric="The response must be professional and include a source snippet.",
            model=EVALUATOR_MODEL,
        ),
    ],
)

def start_eval():
    print("🚀 starting evaluation...")
    
    report = rag_dataset.evaluate_sync(run)
    
    print("\n" + "="*30)
    print("📊 Done")
    print("="*30)
    report.print()

if __name__ == "__main__":
    start_eval()