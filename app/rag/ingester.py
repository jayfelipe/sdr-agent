import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings

def ingest_documents(data_dir: str = "./data/pdfs"):
    print(f"Starting ingestion from {data_dir}...")
    
    # Make sure the DB folder exists so chroma doesn't crash
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    
    docs = []
    # Super basic loop to load all text files in the folder
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            # encoding="utf-8" is crucial to avoid windows errors
            loader = TextLoader(filepath, encoding="utf-8") 
            docs.extend(loader.load())
    
    if not docs:
        print("No .txt files found. Aborting.")
        return

    # Splitting the text. Kept it simple, 1000 chars with 200 overlap
    # TODO: maybe upgrade to RecursiveCharacterTextSplitter later if PDFs are needed
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Split into {len(splits)} chunks. Calling OpenAI for embeddings...")

    # Initialize the OpenAI embedding model we defined in config
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model)
    
    # Create or update the Chroma database
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=settings.chroma_db_path,
        collection_name="product_docs" # Hardcoding the collection name for now
    )
    print("Ingestion complete! Vector DB is ready.")

# This lets us run it directly: python -m app.rag.ingester
if __name__ == "__main__":
    ingest_documents()