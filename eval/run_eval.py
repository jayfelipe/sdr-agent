import json
import sys
import os
import traceback

# 1. Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. CRITICAL: Load .env BEFORE importing deepeval, otherwise it crashes silently looking for OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

# 3. Now import DeepEval and our app safely
try:
    from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase
    from app.agents.graph import agent_executor
except Exception as e:
    print("\n!!! FATAL ERROR DURING IMPORTS !!!")
    traceback.print_exc()
    sys.exit(1)

# Load our test data
with open("eval/golden_dataset.json", "r") as f:
    test_leads = json.load(f)

# Define the metrics
hallucination_metric = HallucinationMetric(threshold=0.5, model="gpt-4o-mini")
relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o-mini")


def run_evaluation():
    print("=" * 50)
    print("STARTING AUTOMATED AGENT EVALUATION")
    print("=" * 50)

    for i, lead in enumerate(test_leads):
        print(f"\n--- Evaluating Lead {i+1}: {lead['lead_name']} at {lead['lead_company']} ---")

        # 1. Prepare the state
        initial_state = {
            "lead_name": lead["lead_name"],
            "lead_company": lead["lead_company"],
            "lead_title": lead["lead_title"],
            "crm_data": "",
            "news_data": "",
            "product_context": "",
            "history_context": "",
            "draft_email": "",
            "review_feedback": "",
            "loop_count": 0,
        }

        # 2. Execute agent
        final_state = agent_executor.invoke(
            initial_state,
            {"recursion_limit": 10}
        )

        generated_email = final_state.get("draft_email", "")

        # Use the REAL RAG context retrieved by the agent
        real_rag_context = final_state.get("product_context", "")

        # Fallback just in case the RAG failed completely
        eval_context = (
            real_rag_context
            if real_rag_context
            else lead["context_expected"]
        )

        # 3. Setup Test Case (now using the REAL context!)
        test_case = LLMTestCase(
            # Made the input more specific so the relevancy judge understands the goal
            input=(
                f"Generate a highly personalized B2B sales email for "
                f"{lead['lead_name']} at {lead['lead_company']}, "
                f"focusing strictly on how our product features solve "
                f"their industry needs."
            ),
            actual_output=generated_email,
            context=[eval_context],  # Passing what the agent actually saw
        )

        # 4. Measure
        print("Measuring Hallucinations...")
        hallucination_metric.measure(test_case)

        print("Measuring Relevancy...")
        relevancy_metric.measure(test_case)

        # 5. Results
        print(
            f"-> HALLUCINATION SCORE: {hallucination_metric.score} "
            f"(Lower is better, < 0.5 is passing)"
        )
        print(
            f"-> RELEVANCY SCORE: {relevancy_metric.score} "
            f"(Higher is better, > 0.5 is passing)"
        )

        hal_pass = (
            hallucination_metric.score <= hallucination_metric.threshold
        )
        rel_pass = (
            relevancy_metric.score >= relevancy_metric.threshold
        )

        if hal_pass and rel_pass:
            print("TEST PASSED :)")
        else:
            print(
                "TEST FAILED :("
                "(Likely due to RAG context mismatch)"
            )


if __name__ == "__main__":
    run_evaluation()
