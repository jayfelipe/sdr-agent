import os
from langchain_openai import OpenAIEmbeddings
# Ignoring the deprecation warning for now to keep dependencies stable
from langchain_community.vectorstores import Chroma 
from langchain_community.docstore.document import Document
from app.config import settings

COLLECTION_NAME = "lead_history"

def _get_vectorstore() -> Chroma:
    """Helper to connect to the memory DB"""
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
    return Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def save_lead_interaction(lead_name: str, company: str, email_summary: str):
    """
    Saves a summary of an email sent to a lead so the agent remembers it later.
    """
    vectorstore = _get_vectorstore()
    
    # We format the text so it's searchable
    text_to_save = f"Lead: {lead_name} at {company}. Previous interaction: {email_summary}"
    doc = Document(page_content=text_to_save, metadata={"lead_name": lead_name, "company": company})
    
    vectorstore.add_documents([doc])
    print(f"--- MEMORY: Saved interaction for {lead_name} ---")

def retrieve_lead_history(lead_name: str, company: str) -> str:
    """
    Checks if we have ever contacted this lead before.
    Returns empty string if this is a cold first contact.
    """
    vectorstore = _get_vectorstore()
    
    # Search using the name and company to find past exact or similar matches
    query = f"Interactions with {lead_name} at {company}"
    results = vectorstore.similarity_search(query, k=1)
    
    if results:
        return results[0].page_content
    
    return "" # Empty string means no history found (Cold Outreach)