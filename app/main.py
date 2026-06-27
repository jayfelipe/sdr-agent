from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.config import settings
# Importamos el agente que acabamos de construir
from app.agents.graph import agent_executor
# Importamos la memoria para guardar el historial
from app.memory.manager import save_lead_interaction

app = FastAPI(
    title="AI SDR Agent API",
    description="Backend for an autonomous B2B sales agent.",
    version="0.1.0"
)

# Pydantic models
class OutreachRequest(BaseModel):
    lead_name: str
    lead_company: str
    lead_title: str | None = None 

class OutreachResponse(BaseModel):
    lead_name: str
    generated_email: str
    status: str

# endpoints
@app.get("/health")
def health_check():
    return {"status": "ok", "llm_in_use": settings.openai_chat_model}

@app.post("/api/v1/generate_outreach", response_model=OutreachResponse)
def generate_outreach(request: OutreachRequest):
    """
    Takes lead info, runs it through the LangGraph agent, returns email, and saves to memory.
    """
    try:
        # initialize the state dictionary that the graph expects
        initial_state = {
            "lead_name": request.lead_name,
            "lead_company": request.lead_company,
            "lead_title": request.lead_title,
            # These start empty, the nodes will fill them up
            "crm_data": "",
            "news_data": "",
            "product_context": "",
            "history_context": "",
            "draft_email": "",
            "review_feedback": "",
            "loop_count": 0 
        }

        # execute the graph. This triggers all the nodes sequentially
        print(f"\n[API] Starting agent workflow for {request.lead_name}...")
        final_state = agent_executor.invoke(initial_state, {"recursion_limit": 10})

        # extract the final email from the returned state
        email_text = final_state.get("draft_email", "Error: Agent failed to generate email.")

        # save this interaction to our Long-Term Memory (Vector DB)
        # this makes the system stateful across different API calls
        memory_summary = f"Sent an initial outreach email. Context used: {final_state.get('news_data', 'N/A')}"
        save_lead_interaction(request.lead_name, request.lead_company, memory_summary)

        return OutreachResponse(
            lead_name=request.lead_name,
            generated_email=email_text,
            status="success"
        )

    except Exception as e:
        # TODO: Add proper logging here, printing to console is bad for prod
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)