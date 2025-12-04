import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import AzureOpenAI, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from utils import process_file, split_documents

load_dotenv()

app = FastAPI(title="QNA Tool API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the vector store
vector_store = None

class OpenAIEmbeddingsWrapper(Embeddings):
    """
    Wrapper for OpenAI/AzureOpenAI Embeddings using the official openai client
    to be compatible with LangChain's vectorstore.
    """
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure texts are not empty strings to avoid API errors
        valid_texts = [t if t else " " for t in texts]
        response = self.client.embeddings.create(
            input=valid_texts,
            model=self.model_name
        )
        # Sort by index to ensure order is preserved
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [data.embedding for data in sorted_data]

    def embed_query(self, text: str) -> List[float]:
        text = text if text else " "
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

def get_llm_client():
    use_azure = os.getenv("USE_AZURE_OPENAI", "True").lower() == "true"
    
    if use_azure:
        api_key = os.getenv("GENAIHUB_API_KEY")
        endpoint = os.getenv("OPENAI_SDK_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI configuration (GENAIHUB_API_KEY, OPENAI_SDK_ENDPOINT) is missing in .env")
            
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env for standard OpenAI usage")
            
        return OpenAI(api_key=api_key)

def get_embeddings_wrapper():
    client = get_llm_client()
    # For Azure, this is the deployment name. For standard OpenAI, this is the model name (e.g., text-embedding-3-small)
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not model_name:
        # Fallback for backward compatibility or Azure specific env var
        model_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    
    if not model_name:
        raise ValueError("EMBEDDING_MODEL_NAME (or AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME) is missing")
        
    return OpenAIEmbeddingsWrapper(client, model_name)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global vector_store
    
    all_docs = []
    
    for file in files:
        content = await file.read()
        docs = process_file(content, file.filename)
        all_docs.extend(docs)
    
    if not all_docs:
        return {"message": "No text extracted from files."}
    
    chunks = split_documents(all_docs)
    
    try:
        embeddings = get_embeddings_wrapper()
        if vector_store is None:
            vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            vector_store.add_documents(chunks)
            
        return {"message": f"Successfully processed {len(files)} files and added {len(chunks)} chunks to the knowledge base."}
    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload documents first.")
    
    try:
        # Detect if this is a comparison query
        comparison_keywords = ['compare', 'difference', 'differences', 'changed', 'changes', 'vs', 'versus',
                               'between', 'contrast', 'differ', 'what changed', 'what is different',
                               'version', 'v1', 'v2', 'old', 'new']
        is_comparison = any(keyword in request.question.lower() for keyword in comparison_keywords)

        # 1. Retrieve relevant documents with similarity scores
        # Increase retrieval for comparison queries to ensure we get diverse sources
        k_value = 20 if is_comparison else 10
        source_docs_with_scores = vector_store.similarity_search_with_score(request.question, k=k_value)

        # For comparison queries, use diversity-aware selection
        if is_comparison:
            # Group documents by source file
            docs_by_source = {}
            for doc, score in source_docs_with_scores:
                source = doc.metadata.get('source', 'Unknown')
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append((doc, score))

            # Check if this is a DBC file comparison (need ALL messages from both files)
            is_dbc_comparison = any('.dbc' in source.lower() for source in docs_by_source.keys())

            if is_dbc_comparison and len(docs_by_source) == 2:
                # For DBC comparisons, we need ALL messages from both files
                # Get all documents from the vector store that match these source files
                source_docs = []
                all_docs = vector_store.similarity_search("", k=1000)  # Get many docs

                dbc_sources = [s for s in docs_by_source.keys() if '.dbc' in s.lower()]
                for doc in all_docs:
                    if doc.metadata.get('source') in dbc_sources:
                        source_docs.append(doc)

                # Limit to avoid context overflow (take all unique messages)
                seen_messages = set()
                unique_docs = []
                for doc in source_docs:
                    msg_id = (doc.metadata.get('source'), doc.metadata.get('message_name'))
                    if msg_id not in seen_messages:
                        seen_messages.add(msg_id)
                        unique_docs.append(doc)

                source_docs = unique_docs[:30]  # Up to 30 messages total for comprehensive comparison
            else:
                # Standard diversity-aware selection for other comparisons
                source_docs = []
                RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))  # More lenient for comparisons

                # Get top 3-5 documents from each source file
                docs_per_source = max(3, 10 // len(docs_by_source)) if docs_by_source else 5

                for source, docs_scores in docs_by_source.items():
                    # Sort by score (lower is better)
                    docs_scores.sort(key=lambda x: x[1])
                    # Take top documents from this source that meet threshold
                    for doc, score in docs_scores[:docs_per_source]:
                        if score < RELEVANCE_THRESHOLD or len(source_docs) < 6:  # Ensure minimum documents
                            source_docs.append(doc)

                # Limit total to avoid context overflow
                source_docs = source_docs[:15]

        else:
            # Standard retrieval for non-comparison queries
            RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
            source_docs = [doc for doc, score in source_docs_with_scores if score < RELEVANCE_THRESHOLD]

            # If no documents pass the threshold, take the top 3 most relevant
            if not source_docs:
                source_docs = [doc for doc, score in source_docs_with_scores[:3]]

            # Limit to top 3 most relevant documents to reduce noise
            source_docs = source_docs[:3]
        
        # 2. Construct context with clear source identification
        context_text = ""
        for i, doc in enumerate(source_docs):
            source_file = doc.metadata.get('source', 'Unknown')
            page_or_row = doc.metadata.get('page', doc.metadata.get('row', 'N/A'))
            message_name = doc.metadata.get('message_name', '')

            # Build a clear source identifier
            if message_name:
                source_info = f"File: {source_file}, Message: {message_name}"
            else:
                source_info = f"File: {source_file}, Location: {page_or_row}"

            context_text += f"\n[Chunk {i+1} from {source_info}]:\n{doc.page_content}\n"

        # 3. Call OpenAI/Azure OpenAI Chat Completion
        client = get_llm_client()
        
        # Determine model/deployment name
        use_azure = os.getenv("USE_AZURE_OPENAI", "True").lower() == "true"
        if use_azure:
            model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        else:
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
            
        if not model_name:
            raise ValueError("Model name (AZURE_OPENAI_DEPLOYMENT_NAME or LLM_MODEL_NAME) is missing")

        system_prompt = """You are a helpful technical assistant. Use the provided context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS cite sources by referencing the FILENAME (e.g., "vehicle_v1.dbc", "vehicle_v2.dbc") not chunk numbers. When citing, use the actual filename from the context.

IMPORTANT FORMATTING RULES:
- Format your responses using Markdown syntax for better readability
- When generating code (Python, JavaScript, etc.), wrap it in code blocks with the language specified
  Example: ```python\ncode here\n```
- Use **bold** for important terms and concepts
- Use bullet points (-) or numbered lists (1.) for multiple items
- Use headings (## Heading) to organize longer responses
- For inline code or variable names, use `backticks`
- Make responses clear, well-structured, and easy to read

When the user asks you to generate code based on the documentation:
1. Read the relevant information from the provided context
2. Generate working, well-commented code
3. Wrap the code in proper markdown code blocks with language identifier
4. Explain what the code does before or after the code block

When the user asks to COMPARE documents or versions:
1. Carefully analyze ALL provided chunks from DIFFERENT source files
2. Identify what exists in one file but not the other (use FILENAMES not chunk numbers)
3. Identify what has changed between files (values, IDs, parameters, etc.)
4. Organize the comparison clearly with sections for: Added, Removed, Modified, and Unchanged (if relevant)
5. ALWAYS cite using the actual FILENAME (e.g., "In vehicle_v1.dbc..." or "In vehicle_v2.dbc...")
6. Do NOT use chunk numbers like "Document 3" - use filenames instead
7. Do NOT only focus on similarities - actively look for and highlight DIFFERENCES"""

        user_message = f"""Context:
{context_text}

Question: {request.question}
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # Extract unique sources for the UI
        sources = []
        seen_sources = set()
        for doc in source_docs:
            source_file = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page')
            row = doc.metadata.get('row')

            # Only include page/row info if it's actually available (not N/A)
            if page and page != 'N/A':
                source_str = f"{source_file} (Page: {page})"
            elif row and row != 'N/A':
                source_str = f"{source_file} (Row: {row})"
            else:
                source_str = source_file

            if source_str not in seen_sources:
                sources.append(source_str)
                seen_sources.add(source_str)
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"Error during ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
