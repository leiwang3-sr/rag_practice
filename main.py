from pathlib import Path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent
import inspect
import logfire
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

logfire.configure()
logfire.instrument_pydantic_ai()

class AnswerOutput(BaseModel):
    answer: str = Field(description="The answer to the question")
    source_snippet: str = Field(description="The source snippet from the document that supports the answer")


class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: dict


def load_and_chunk_documents(path: Path) -> List[DocumentChunk]:
    if not path.exists():
        print(f"⚠️ Path {path} does not exist.")
        return []

    chunks = []

    files = list(path.glob("*.md")) + list(path.glob("*.txt"))
    
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        parts = basic_splitter.split_text(content)
        
        for i, part in enumerate(parts):
            unique_id = f"{file_path.name}-{i}"
            chunks.append(DocumentChunk(
                id=unique_id,
                content=part,
                metadata={
                    "source": file_path.name, 
                    "chunk_index": i
                },
            ))
    return chunks

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

basic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

def init_data(kb_path):
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="knowledge_base", 
        embedding_function=chroma_ef
    )
    
    chunks = load_and_chunk_documents(kb_path)
    if chunks:
        collection.add(
            documents=[c.content for c in chunks],
            ids=[c.id for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
        print(f"✅ Indexed {len(chunks)} chunks.")
    return collection


def search_collection(collection, query_text: str, n_results: int = 3) -> List[DocumentChunk]:
    print(f"\n🔍 Searching for: {query_text}")
    
    results = collection.query(
        query_texts=[query_text], 
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    chunks = []
    if results["documents"]:
        for doc_id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
            chunks.append(DocumentChunk(
                id=doc_id,
                content=doc,
                metadata=meta
            ))
    
    print(f"🔍 Found {len(chunks)} relevant chunks")
    return chunks

def augment_prompt(query: str, chunks: List[DocumentChunk]) -> str:
    if not chunks:
        context_text = "NO RELEVANT DATA FOUND IN KNOWLEDGE BASE."
    else:
        context_text = "\n\n".join(
            [f"--- Document: {c.metadata['source']} ---\n{c.content}" for c in chunks]
        )
    
    full_prompt = inspect.cleandoc(f"""
        ### CONTEXT DATA
        {context_text}
        
        ### USER QUERY
        {query}
        
        ### INSTRUCTION
        Based on the CONTEXT DATA provided above, provide a structured answer. 
        If the data is insufficient, state 'Not found'.
    """)
    return full_prompt


def setup_agent() -> Agent:
    return Agent(
        model="google-gla:gemini-2.5-pro",
        system_prompt=(
            "You are a helpful knowledge assistant. "
            "Answer the query solely based on the provided context. "
            "If the context is empty or irrelevant, say 'Not found'."
        ),
        output_type=AnswerOutput,
    )

async def run_async(question: str) -> AnswerOutput:
    kb_path = Path(__file__).parent / "docs"
    collection = init_data(kb_path)
    search_results = search_collection(collection, question)
    augmented_prompt_str = augment_prompt(question, search_results)
    agent = setup_agent()
    result = await agent.run(augmented_prompt_str)
    return result.output

def run(question: str) -> AnswerOutput:
    kb_path = Path(__file__).parent / "docs"
    
    collection = init_data(kb_path)

    search_results = search_collection(collection, question)
    
    augmented_prompt_str = augment_prompt(question, search_results)
    
    agent = setup_agent()
    result = agent.run_sync(augmented_prompt_str)
    
    return result.output

def main():
    question = "What is my plan?"
    try:
        output = run(question)
        print("\n" + "="*30)
        print(f"answer: {output.answer}")
        print(f"source: {output.source_snippet}")
        print("="*30)
    except Exception as e:
        import traceback
        traceback.print_exc()
    sys.exit(0)

if __name__ == "__main__":
    main()
