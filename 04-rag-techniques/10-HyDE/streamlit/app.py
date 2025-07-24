import streamlit as st
import numpy as np
import fitz  # PyMuPDF
import os
import io
import config
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# --- 1. 기본 설정 및 API 키 로드 ---

st.set_page_config(
    page_title="HyDE RAG 문서 분석 챗봇",
    page_icon="🤖",
    layout="wide"
)

# .env 파일에서 환경 변수 로드
load_dotenv()
try:
    client = OpenAI(api_key=config.API_KEY)
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    st.error("OpenAI API 키를 설정해주세요. .env 파일을 확인하거나 환경 변수 설정이 필요합니다.")
    st.stop()


# --- 2. Jupyter Notebook의 핵심 로직 (클래스 및 함수) ---
# SimpleVectorStore, 문서 처리, 임베딩, RAG 함수들을 그대로 가져옵니다.

class SimpleVectorStore:
    """NumPy를 사용한 간단한 벡터 저장소 구현 클래스입니다."""
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding)
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        return results

def extract_text_from_pdf(pdf_bytes):
    """PDF 파일의 바이트 데이터에서 텍스트를 추출합니다."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if len(text.strip()) > 50:
                pages.append({
                    "text": text,
                    "metadata": {"page": page_num + 1}
                })
    return pages

def describe_image(image_bytes, filename):
    """OpenAI Vision API(gpt-4o)를 사용하여 이미지에 대한 설명을 생성합니다."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지는 문서의 일부입니다. 이미지의 모든 내용을 텍스트로 상세히 설명해주세요. 텍스트, 다이어그램, 차트, 주요 객체, 전체적인 맥락을 모두 포함해야 합니다."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_bytes}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        description = response.choices[0].message.content
        return {
            "text": description,
            "metadata": {"source_image": filename}
        }
    except Exception as e:
        st.error(f"이미지 설명 생성 실패: {filename}, 오류: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    """텍스트를 일정 길이의 중첩 청크로 분할합니다."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {"start_pos": i, "end_pos": i + len(chunk_text)}
            })
    return chunks

def create_embeddings(texts, model="text-embedding-3-small"):
    """주어진 텍스트 목록에 대해 임베딩 벡터를 생성합니다."""
    if not texts:
        return []
    try:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"임베딩 생성 중 오류 발생: {e}")
        return []

def generate_hypothetical_document(query, model="gpt-4o-mini"):
    """주어진 질문에 대한 가상의 문서를 생성합니다."""
    system_prompt = """당신은 전문 문서 작성자입니다. 
    아래 질문에 대한 완벽한 답변이 될 가상의 문서를 작성하세요.
    이 문서는 주제에 대해 깊이 있고 정보가 풍부한 설명을 포함해야 합니다.
    실제 문서처럼 사실, 예시, 개념 등을 자연스럽게 포함하여 작성하세요."""
    user_prompt = f"질문: {query}\n\n이 질문에 대한 문서를 작성해 주세요:"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def generate_response(query, relevant_chunks, model="gpt-4o-mini"):
    """질문과 관련된 문서 청크를 기반으로 최종 응답을 생성합니다."""
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다. 제공된 문맥을 기반으로 사용자의 질문에 상세하고 친절하게 답변하세요."},
            {"role": "user", "content": f"문맥:\n{context}\n\n질문: {query}"}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- 3. Streamlit UI 및 상태 관리 ---

# 세션 상태 초기화
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- 사이드바 UI ---
with st.sidebar:
    st.title("📄 HyDE RAG 챗봇 설정")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "PDF 또는 이미지 파일을 업로드하세요",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.subheader("RAG 설정")
    rag_mode = st.radio(
        "RAG 방식 선택",
        ("Standard RAG", "HyDE RAG", "Compare Both"),
        index=2, # 기본값 'Compare Both'
        help="Standard: 질문을 직접 사용. HyDE: 질문으로 가상 문서를 만들어 검색 정확도 향상. Compare: 두 방식 비교."
    )
    
    k_chunks = st.slider("검색할 청크 수 (k)", 1, 10, 5)

    if st.button("파일 처리 및 지식 기반 생성"):
        if not uploaded_files:
            st.warning("분석할 파일을 먼저 업로드해주세요.")
        else:
            with st.spinner("파일 처리 및 임베딩 중... 잠시만 기다려주세요."):
                all_processed_texts = []
                for uploaded_file in uploaded_files:
                    # 파일 타입에 따라 처리
                    if uploaded_file.type == "application/pdf":
                        pdf_bytes = uploaded_file.getvalue()
                        pages = extract_text_from_pdf(pdf_bytes)
                        for page in pages:
                            page["metadata"]["source"] = uploaded_file.name
                            all_processed_texts.append(page)
                    else: # 이미지 파일
                        image_bytes = uploaded_file.getvalue()
                        image_base64 = io.BytesIO(image_bytes).read()
                        import base64
                        encoded_image = base64.b64encode(image_base64).decode('utf-8')
                        desc_obj = describe_image(encoded_image, uploaded_file.name)
                        if desc_obj:
                            all_processed_texts.append(desc_obj)
                
                # 텍스트 청킹
                all_chunks_with_meta = []
                for item in all_processed_texts:
                    page_chunks = chunk_text(item["text"])
                    for chunk in page_chunks:
                        chunk["metadata"].update(item["metadata"])
                    all_chunks_with_meta.extend(page_chunks)

                # 임베딩 및 벡터 스토어 생성
                chunk_texts_only = [chunk["text"] for chunk in all_chunks_with_meta]
                chunk_embeddings = create_embeddings(chunk_texts_only)
                
                if chunk_embeddings:
                    vector_store = SimpleVectorStore()
                    for i, chunk in enumerate(all_chunks_with_meta):
                        vector_store.add_item(
                            text=chunk["text"],
                            embedding=chunk_embeddings[i],
                            metadata=chunk["metadata"]
                        )
                    st.session_state.vector_store = vector_store
                    st.session_state.processed = True
                    st.session_state.messages = [] # 새 파일 처리 시 채팅 기록 초기화
                    st.success(f"{len(uploaded_files)}개 파일에서 총 {len(all_chunks_with_meta)}개의 정보 조각을 처리했습니다.")
                else:
                    st.error("파일 처리 중 오류가 발생했습니다. 임베딩을 생성할 수 없습니다.")
    st.markdown("---")
    st.info("만든이: AI Assistant\n- Jupyter Notebook의 HyDE RAG 로직을 Streamlit 앱으로 구현했습니다.")

# --- 메인 채팅 화면 ---
st.title("HyDE RAG 챗봇")
st.markdown("사이드바에서 파일을 업로드하고 '파일 처리' 버튼을 누른 후, 아래 채팅창에 질문을 입력하세요.")

# 이전 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if query := st.chat_input("업로드한 내용에 대해 질문하세요..."):
    if not st.session_state.processed:
        st.warning("먼저 파일을 업로드하고 처리 버튼을 눌러주세요.")
    else:
        # 사용자 메시지 표시 및 저장
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # AI 응답 생성
        with st.chat_message("assistant"):
            # 로딩 스피너
            with st.spinner("답변 생성 중..."):
                
                def run_standard_rag(q, vs, k):
                    query_embedding = create_embeddings([q])[0]
                    chunks = vs.similarity_search(query_embedding, k=k)
                    response = generate_response(q, chunks)
                    return response, chunks

                def run_hyde_rag(q, vs, k):
                    hypo_doc = generate_hypothetical_document(q)
                    hypo_embedding = create_embeddings([hypo_doc])[0]
                    chunks = vs.similarity_search(hypo_embedding, k=k)
                    response = generate_response(q, chunks)
                    return response, chunks, hypo_doc

                # 선택된 모드에 따라 RAG 실행
                if rag_mode == "Standard RAG":
                    response, chunks = run_standard_rag(query, st.session_state.vector_store, k_chunks)
                    st.markdown(response)
                    with st.expander("참고한 정보 조각 (Standard RAG)"):
                        for chunk in chunks:
                            st.info(f"**유사도: {chunk['similarity']:.4f}** (출처: {chunk['metadata'].get('source', 'N/A')}, 페이지: {chunk['metadata'].get('page', 'N/A')})\n\n> {chunk['text']}")
                    st.session_state.messages.append({"role": "assistant", "content": response})

                elif rag_mode == "HyDE RAG":
                    response, chunks, hypo_doc = run_hyde_rag(query, st.session_state.vector_store, k_chunks)
                    st.markdown(response)
                    with st.expander("참고한 정보 조각 (HyDE RAG)"):
                        for chunk in chunks:
                            st.info(f"**유사도: {chunk['similarity']:.4f}** (출처: {chunk['metadata'].get('source', 'N/A')}, 페이지: {chunk['metadata'].get('page', 'N/A')})\n\n> {chunk['text']}")
                    with st.expander("HyDE가 생성한 가상 문서"):
                        st.warning(hypo_doc)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                else: # Compare Both
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Standard RAG 결과")
                        std_response, std_chunks = run_standard_rag(query, st.session_state.vector_store, k_chunks)
                        st.markdown(std_response)
                        with st.expander("참고한 정보 조각 (Standard)"):
                            for chunk in std_chunks:
                                st.info(f"**유사도: {chunk['similarity']:.4f}**\n\n> {chunk['text']}")
                    
                    with col2:
                        st.subheader("HyDE RAG 결과")
                        hyde_response, hyde_chunks, hypo_doc = run_hyde_rag(query, st.session_state.vector_store, k_chunks)
                        st.markdown(hyde_response)
                        with st.expander("참고한 정보 조각 (HyDE)"):
                            for chunk in hyde_chunks:
                                st.info(f"**유사도: {chunk['similarity']:.4f}**\n\n> {chunk['text']}")
                        with st.expander("HyDE 생성 가상 문서"):
                            st.warning(hypo_doc)
                    
                    # 비교 모드에서는 채팅 기록에 두 답변을 모두 추가
                    comparison_response = f"**Standard RAG 답변:**\n{std_response}\n\n---\n\n**HyDE RAG 답변:**\n{hyde_response}"
                    st.session_state.messages.append({"role": "assistant", "content": comparison_response})