from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import pypdf
import docx
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
import gc
import hashlib
import pickle
import os
import re
import requests
import time

# Import the main Pinecone class
from pinecone import Pinecone

# --- 2. Configuration (Reads from Environment Variables) ---
# This script will be configured by the Colab launcher cell.
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
API_TOKEN = os.getenv('API_TOKEN')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')

# Configure Gemini
import google.generativeai as genai
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured successfully from environment variable.")
    except Exception as e:
        print(f"❌ Could not configure Gemini API from env var. Error: {e}")
else:
    print("⚠️ Warning: GEMINI_API_KEY environment variable not found. LLM will not function.")

# --- 3. Pydantic Schemas (Matching Competition Specs) ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- 4. Global Configuration ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CACHE_DIR = Path("./temp_indexes")
CACHE_DIR.mkdir(exist_ok=True)
PROCESSED_DOCS_TRACKER = CACHE_DIR / "processed_docs.json"

# --- 5. Document Processing Class ---
class DocumentProcessor:
    @staticmethod
    def download_from_blob(url: str) -> bytes:
        response = requests.get(url); response.raise_for_status(); return response.content

    @classmethod
    def load_and_process_document(cls, document_url: str) -> List[Dict]:
        if document_url.startswith(('http://', 'https://')):
            print("downloading the file")
            content = cls.download_from_blob(document_url)
            print("download complete")
            source_name = document_url.split('/')[-1].split('?')[0]
        else:
            raise ValueError("Input must be a valid URL.")
        
        doc_type_map = {'.pdf': "pdf", '.docx': "docx", '.doc': "docx", '.eml': "email", '.msg': "email"}
        ext = Path(source_name).suffix.lower()
        document_type = doc_type_map.get(ext, "pdf")
        
        process_map = {"pdf": cls.process_pdf, "docx": cls.process_docx, "email": cls.process_email}
        if document_type not in process_map: raise ValueError(f"Unsupported type: {document_type}")
        
        docs = process_map[document_type](content)
        for doc in docs: doc["metadata"]["source"] = source_name
        return docs

    @staticmethod
    def process_pdf(content: bytes) -> List[Dict]:
        print("processing the pdf downloaded")
        reader = pypdf.PdfReader(BytesIO(content)); docs = []
        print("processing the ",reader.pages,"pages in pdf")
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip(): docs.append({"content": text, "metadata": {"page": i + 1}})
        print("processing complete")
        return docs

    @staticmethod
    def process_docx(content: bytes) -> List[Dict]:
        doc = docx.Document(BytesIO(content)); full_text = "\n".join([p.text for p in doc.paragraphs])
        return [{"content": full_text, "metadata": {"page": 1}}] if full_text.strip() else []

    @staticmethod
    def process_email(content: bytes) -> List[Dict]:
        msg = BytesParser(policy=default).parsebytes(content)
        header_text = f"From: {msg.get('from', 'N/A')}\nTo: {msg.get('to', 'N/A')}\nSubject: {msg.get('subject', 'N/A')}\n\n"; body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain": body += part.get_content(decode=True).decode('utf-8', 'ignore')
        else:
            body = msg.get_content(decode=True).decode('utf-8', 'ignore')
        full_text = header_text + body
        return [{"content": full_text, "metadata": {"page": 1}}] if full_text.strip() else []

# --- 6. The RAG Processor Class ---
class DynamicRAGProcessor:
    def __init__(self):
        print("Initializing Dynamic RAG Processor...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        if PINECONE_API_KEY:
            try:
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                self.pinecone_index = self.pc.Index(PINECONE_INDEX_NAME)
                print(f"✅ Pinecone index '{PINECONE_INDEX_NAME}' connected successfully.")
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Failed to connect to Pinecone. Error: {e}")
                self.pinecone_index = None
        else:
            self.pinecone_index = None
            print("⚠️ Warning: Pinecone credentials not found.")

        if GEMINI_API_KEY:
            self.llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print(f"✅ Gemini model '{GEMINI_MODEL_NAME}' initialized.")
        else: self.llm = None
        print("Processor Initialized.")

    def _track_processed_doc(self, doc_hash):
        processed = {}
        if PROCESSED_DOCS_TRACKER.exists():
            with open(PROCESSED_DOCS_TRACKER, 'r') as f: processed = json.load(f)
        processed[doc_hash] = True
        with open(PROCESSED_DOCS_TRACKER, 'w') as f: json.dump(processed, f)

    def _is_doc_processed(self, doc_hash):
        if not PROCESSED_DOCS_TRACKER.exists(): return False
        with open(PROCESSED_DOCS_TRACKER, 'r') as f: processed = json.load(f)
        return doc_hash in processed

    def _index_document_in_pinecone(self, doc_url: str):
        if not self.pinecone_index: raise ValueError("Pinecone index is not initialized.")
        doc_hash = hashlib.md5(doc_url.encode()).hexdigest()
        if self._is_doc_processed(doc_hash):
            print(f"CACHE HIT: Document '{doc_url}' already indexed in this session.")
            return

        print(f"CACHE MISS: Indexing document '{doc_url}' in Pinecone.")
        documents = DocumentProcessor.load_and_process_document(doc_url)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("starting to process the document")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunked_docs = []
        for doc in documents:
            for i, chunk in enumerate(text_splitter.split_text(doc['content'])):
                chunked_docs.append({"content": chunk, "metadata": {**doc['metadata'], "chunk_id": i}})
        if not chunked_docs: raise ValueError("Document is empty.")
        
        embeddings = self.embedding_model.encode([d['content'] for d in chunked_docs])
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunked_docs, embeddings)):
            chunk_id = f"{doc_hash}-{i}"
            clean_metadata = {"text": chunk['content'], "source": str(chunk['metadata'].get('source')), "page": int(chunk['metadata'].get('page', 0))}
            vectors_to_upsert.append((chunk_id, embedding.tolist(), clean_metadata))
        
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.pinecone_index.upsert(vectors=batch)
        
        self._track_processed_doc(doc_hash)
        print("Upsert to Pinecone complete.")

    def _llm_evaluate(self, query: str, context_clauses: List[Dict]) -> str:
        if not self.llm: return "Error: Gemini model not initialized."
        if not context_clauses: return "Information not available in the provided document."

        context_str = "\n".join([f"- {c['text']}" for c in context_clauses])
        
        prompt = f"""Based ONLY on the context provided below, answer the user's question.
Your answer must be in a single line, direct, and concise sentence. Do not add any introductory phrases. Do not explain your reasoning or mention the context.
If the information is not in the context, your answer must be "Information not available in the provided document." , but before seaying so double check that 
the information is not actually there in the file that was given.For every answer that u come up with make sure u test the answers accuracy by asking youre self the same question 
and trying to guess a new answer , compare the new answer and the old one and determine how right you were , do this trial and error tests at least 3 to 4 times before u return the answer

Also , make sure you give accurate answers and do not give partialy correct answers

[CONTEXT]:
{context_str}

[USER QUESTION]:
{query}

Answer:"""
        try:
            safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
            response = self.llm.generate_content(prompt, safety_settings=safety_settings)
            if not response.parts: return f"Response blocked by safety filters. Reason: {response.prompt_feedback.block_reason}"
            return response.text.strip()
        except Exception as e:
            return f"An error occurred with the Gemini API: {e}"

    def process_request(self, request: HackRxRequest) -> Dict:
        try:
            self._index_document_in_pinecone(request.documents)
            print("get the requested doc")
            all_answers = []
            for question in request.questions:
                query_embedding = self.embedding_model.encode([question]).tolist()
                query_response = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True)
                retrieved_clauses = [match['metadata'] for match in query_response['matches']]
                print("Answereing question",question,":passing to llm")
                answer = self._llm_evaluate(question, retrieved_clauses)
                all_answers.append(answer)
            return {"answers": all_answers}
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"answers": [f"An error occurred: {str(e)}"]}

# --- 7. Authorization Middleware ---
def verify_api_key(authorization: str = Header(...)):
    if not API_TOKEN: raise HTTPException(status_code=500, detail="API_TOKEN not configured on server.")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer' or token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid API key or authentication scheme")
        return token
    except:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

# --- 8. FastAPI Application ---
app = FastAPI(title="Competition RAG System")
processor = None
@app.on_event("startup")
async def startup_event():
    global processor
    processor = DynamicRAGProcessor()

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, api_key: str = Depends(verify_api_key)):
    result = processor.process_request(request)
    return HackRxResponse(**result)

@app.get("/")
def read_root():
    return {"status": "API is running"}
