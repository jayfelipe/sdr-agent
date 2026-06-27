from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Inputs from the API
    lead_name: str
    lead_company: str
    lead_title: Optional[str]
    
    # Data gathered by the tools and RAG
    crm_data: str
    news_data: str
    product_context: str
    history_context: str
    
    # Outputs
    draft_email: str
    review_feedback: str 
    
    # NEW: Safety net to prevent infinite loops caused by stubborn LLMs
    loop_count: int