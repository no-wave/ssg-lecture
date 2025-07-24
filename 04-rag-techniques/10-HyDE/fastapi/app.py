# app.py

import os
import io
import config
import base64
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- 1. 기본 설정 및 FastAPI 앱 초기화 ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# API 키 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
try:
    client = OpenAI(api_key=config.API_KEY)
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    raise ValueError("OpenAI API 키를 설정해주세요. .env 파일을 확인하거나 환경 변수 설정이 필요합니다.")

# 벡터 스토어를 저장할 전역 변수 (데모용)
vector_store = None

# --- 2. Pydantic 모델 정의 ---
# 요청 본문의 데이터 타입을 명시하고 검증합니다.
class ChatRequest(BaseModel):
    query: str
    mode: str = 'compare'
    k: int = 5

# --- 3. RAG 핵심 로직 함수들 (이전 코드와 거의 동일) ---
# SimpleVectorStore, 파일 처리, 임베딩, RAG 함수들...

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors: return []
        query_vector = np.array(query_embedding)
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [{'text': self.texts[idx], 'metadata': self.metadata[idx], 'similarity': float(score)} for idx, score in similarities[:min(k, len(similarities))]]
        return results

# FastAPI의 UploadFile 객체를 처리하기 위해 파일 경로 대신 bytes를 받도록 수정
def extract_text_from_pdf(pdf_bytes: bytes):
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if len(text.strip()) > 50:
                pages.append({"text": text, "metadata": {"page": page_num + 1}})
    return pages

def describe_image(image_bytes: bytes, filename: str):
    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "이 이미지는 문서의 일부입니다. 이미지의 모든 내용을 텍스트로 상세히 설명해주세요."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ], max_tokens=1024)
        return {"text": response.choices[0].message.content, "metadata": {"source_image": filename}}
    except Exception as e:
        print(f"Error describing image {filename}: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({"text": chunk_text, "metadata": {"start_pos": i}})
    return chunks

def create_embeddings(texts, model="text-embedding-3-small"):
    if not texts: return []
    try:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []

def generate_hypothetical_document(query, model="gpt-4o-mini"):
    system_prompt = "당신은 전문 문서 작성자입니다. 아래 질문에 대한 완벽한 답변이 될 가상의 문서를 작성하세요. 실제 문서처럼 사실, 예시, 개념 등을 자연스럽게 포함하여 작성하세요."
    user_prompt = f"질문: {query}\n\n이 질문에 대한 문서를 작성해 주세요:"
    response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.1)
    return response.choices[0].message.content

def generate_response(query, relevant_chunks, model="gpt-4o-mini"):
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다. 제공된 문맥을 기반으로 사용자의 질문에 상세하고 친절하게 답변하세요."},
            {"role": "user", "content": f"문맥:\n{context}\n\n질문: {query}"}
        ], temperature=0.5, max_tokens=1000)
    return response.choices[0].message.content


# --- 4. FastAPI 라우트 (API 엔드포인트) ---

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """메인 HTML 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_files(files: List[UploadFile] = File(...)):
    """파일을 업로드하고 RAG 지식 기반을 구축합니다."""
    global vector_store
    
    if not files:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")

    vector_store = None
    all_processed_texts = []

    for file in files:
        file_bytes = await file.read()
        if file.content_type == "application/pdf":
            pages = extract_text_from_pdf(file_bytes)
            for page in pages:
                page["metadata"]["source"] = file.filename
                all_processed_texts.append(page)
        elif file.content_type and file.content_type.startswith("image/"):
            desc_obj = describe_image(file_bytes, file.filename)
            if desc_obj:
                all_processed_texts.append(desc_obj)

    if not all_processed_texts:
        raise HTTPException(status_code=400, detail="처리할 수 있는 콘텐츠가 없습니다.")

    all_chunks_with_meta = []
    for item in all_processed_texts:
        page_chunks = chunk_text(item["text"])
        for chunk in page_chunks:
            chunk["metadata"].update(item["metadata"])
        all_chunks_with_meta.extend(page_chunks)

    chunk_texts_only = [chunk["text"] for chunk in all_chunks_with_meta]
    chunk_embeddings = create_embeddings(chunk_texts_only)

    if chunk_embeddings:
        temp_vs = SimpleVectorStore()
        for i, chunk in enumerate(all_chunks_with_meta):
            temp_vs.add_item(text=chunk["text"], embedding=chunk_embeddings[i], metadata=chunk["metadata"])
        vector_store = temp_vs
        return JSONResponse(content={'status': 'success', 'message': f'{len(files)}개 파일에서 총 {len(all_chunks_with_meta)}개의 정보 조각을 처리했습니다.'})
    else:
        raise HTTPException(status_code=500, detail="임베딩 생성에 실패했습니다.")


@app.post("/chat")
async def chat(req: ChatRequest):
    """사용자 질문에 답변합니다."""
    if vector_store is None:
        raise HTTPException(status_code=400, detail="먼저 파일을 처리해주세요.")

    results = {}
    try:
        # Standard RAG
        if req.mode in ['standard', 'compare']:
            query_embedding = create_embeddings([req.query])[0]
            std_chunks = vector_store.similarity_search(query_embedding, k=req.k)
            std_response = generate_response(req.query, std_chunks)
            results['standard'] = {'answer': std_response, 'chunks': std_chunks}

        # HyDE RAG
        if req.mode in ['hyde', 'compare']:
            hypo_doc = generate_hypothetical_document(req.query)
            hypo_embedding = create_embeddings([hypo_doc])[0]
            hyde_chunks = vector_store.similarity_search(hypo_embedding, k=req.k)
            hyde_response = generate_response(req.query, hyde_chunks)
            results['hyde'] = {'answer': hyde_response, 'chunks': hyde_chunks, 'hypo_doc': hypo_doc}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류: {str(e)}")
        
    return JSONResponse(content=results)

# uvicorn으로 실행하기 위한 설정
# 터미널에서 `uvicorn app:app --reload --port 8000` 명령어로 실행