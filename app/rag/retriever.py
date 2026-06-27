from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings

def get_relevant_context(query: str, k: int = 2) -> str:
    """
    Searches the vector DB for chunks related to the query.
    Returns a single string with the combined context.
    """
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
    
    # Connect to the EXISTING DB (don't create a new one)
    vectorstore = Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
        collection_name="product_docs"
    )
    
    # Do the actual vector search
    results = vectorstore.similarity_search(query, k=k)
    
    # Join the page content of the results so the LLM can read it easily
    context_text = "\n\n".join([doc.page_content for doc in results])
    return context_text

# Quick local test to see if it works
if __name__ == "__main__":
    test_query = "How much does the enterprise plan cost?"
    print(f"Searching for: {test_query}\n")
    
    found_context = get_relevant_context(test_query)
    print("--- FOUND CONTEXT ---")
    print(found_context)