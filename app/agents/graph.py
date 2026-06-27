from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes import (
    retrieve_memory_node,
    enrich_lead_node,
    fetch_product_info_node,
    draft_email_node,
    review_email_node
)

def build_agent_graph():
    """Compiles the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("enrich_lead", enrich_lead_node)
    workflow.add_node("fetch_rag", fetch_product_info_node)
    workflow.add_node("draft_email", draft_email_node)
    workflow.add_node("review_email", review_email_node)

    # Entry point
    workflow.set_entry_point("retrieve_memory")

    # Linear flow
    workflow.add_edge("retrieve_memory", "enrich_lead")
    workflow.add_edge("enrich_lead", "fetch_rag")
    workflow.add_edge("fetch_rag", "draft_email")
    workflow.add_edge("draft_email", "review_email")

    # Decision logic after review
    def decide_next_step(state: AgentState):
        review = state.get("review_feedback")
        loop_count = state.get("loop_count", 0)

        # Stop conditions
        if review == "APPROVED" or loop_count >= 2:
            if loop_count >= 2 and review != "APPROVED":
                print("--- WARNING: Force stopping after 2 attempts ---")
            return END

        # Otherwise retry drafting
        return "draft_email"

    workflow.add_conditional_edges(
        "review_email",
        decide_next_step,
        {
            "draft_email": "draft_email",
            END: END
        }
    )

    return workflow.compile()


# Global executor
agent_executor = build_agent_graph()