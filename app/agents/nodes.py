from langchain_openai import ChatOpenAI
from app.config import settings
from app.agents.state import AgentState  # <--- AÑADE ESTA LÍNEA
from app.tools.crm_tool import fetch_lead_crm_data
from app.tools.web_scraper_tool import scrape_recent_news
from app.rag.retriever import get_relevant_context
from app.memory.manager import retrieve_lead_history

# Initialize models outside the nodes so we don't recreate them every run
# Using mini for logic/review to save cash, big boy for the actual writing
cheap_llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0.1)
writer_llm = ChatOpenAI(model=settings.openai_expensive_model, temperature=0.7) 

def retrieve_memory_node(state: AgentState):
    """Checks if we have talked to this guy before."""
    print("--- NODE: Checking Memory ---")
    history = retrieve_lead_history(state["lead_name"], state["lead_company"])
    return {"history_context": history}

def enrich_lead_node(state: AgentState):
    """Uses the mock tools to get company info and news."""
    print("--- NODE: Enriching Lead Data ---")
    crm = fetch_lead_crm_data.invoke({"company_name": state["lead_company"]})
    news = scrape_recent_news.invoke({"company_name": state["lead_company"]})
    return {"crm_data": crm, "news_data": news}

def fetch_product_info_node(state: AgentState):
    """Hits the RAG vector DB to find relevant product features."""
    print("--- NODE: Fetching Product RAG ---")
    # We build a query combining what we know so far to get better vector search results
    query = f"{state['lead_company']} {state['crm_data']} needs supply chain features"
    context = get_relevant_context(query, k=2)
    return {"product_context": context}

def draft_email_node(state: AgentState):
    """The main writer LLM puts it all together."""
    print("--- NODE: Drafting Email ---")
    
    # A bit of a long prompt, but structuring it clearly helps the LLM
    prompt = f"""
    You are an elite B2B Sales Development Representative for NexusAI.
    Write a highly personalized, short outreach email to {state['lead_name']} ({state.get('lead_title', 'Decision Maker')}) at {state['lead_company']}.

    CONTEXT TO USE:
    - Company Info: {state['crm_data']}
    - Recent News: {state['news_data']}
    - Our Product Features to highlight: {state['product_context']}
    
    {f"IMPORTANT - PAST INTERACTION: {state['history_context']} (If there is history, write a short follow-up, DO NOT introduce yourself again)." if state['history_context'] else "This is a cold first contact. Keep it punchy."}

    RULES:
    - Do not invent product features not listed in the Product Features.
    - Do not use cheesy sales jargon like "synergy" or "circle back".
    - Maximum 3 short paragraphs.
    - End with a low-friction call to action.
    """
    
    response = writer_llm.invoke(prompt)
    return {"draft_email": response.content}

def review_email_node(state: AgentState):
    """A secondary LLM acts as a strict manager to review the draft."""
    print("--- NODE: Reviewing Draft ---")
    
    review_prompt = f"""
    Review this sales email drafted by an AI. 
    Email:
    ---
    {state['draft_email']}
    ---
    
    Product Knowledge Base:
    ---
    {state['product_context']}
    ---

    Check for these failures:
    1. Did it invent a feature NOT in the knowledge base?
    2. Is it completely irrelevant to the company news?
    
    YOUR INSTRUCTIONS:
    You MUST output ONLY ONE SINGLE WORD. No punctuation.
    If it is acceptable, output exactly: APPROVED
    If it failed the checks above, output exactly: REWRITE
    """
    response = cheap_llm.invoke(review_prompt)
    
    # DEBUG: Let's see what the LLM is actually thinking
    print(f"[DEBUG] Reviewer said: {response.content.strip()}")
    
    # Using .upper() to catch "Approved", "APPROVED.", etc.
    feedback = "APPROVED" if "APPROVED" in response.content.upper() else "REWRITE"
    
    # Increment our safety loop counter
    current_count = state.get("loop_count", 0)
    
    return {"review_feedback": feedback, "loop_count": current_count + 1}