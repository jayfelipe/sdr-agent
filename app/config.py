import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Silenciar LangSmith para que no spamee la consola si no hay API Key
os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_dotenv()

class Settings(BaseSettings):
    # OpenAI configs - using mini for most things to save cash
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = "gpt-4o-mini"
    
    # We only use the big boy model for the final email draft to keep costs down
    openai_expensive_model: str = "gpt-4o" 
    
    openai_embedding_model: str = "text-embedding-3-small"

    # Langsmith configs for LLMOps tracing
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", False)
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "sdr-agent-local")

    # Local paths for our vector DB and logs
    chroma_db_path: str = "./data/db/chroma"
    sqlite_db_path: str = "./data/db/sdr_logs.db"

    class Config:
        # Ignore extra env vars that might be in the system
        extra = "ignore"

# Create a global settings instance to use anywhere
settings = Settings()