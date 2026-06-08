import os
import sys
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

# Import pipeline functions
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

# Streamlit Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Luật Ma Túy & Nghệ Sĩ",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #7E22CE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #7E22CE;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .metric-badge {
        background-color: #7E22CE;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .source-badge {
        background-color: #3B82F6;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_api_key" not in st.session_state:
    st.session_state.session_api_key = os.getenv("OPENAI_API_KEY", "")

# Update environment variable if modified in UI
def update_api_key():
    os.environ["OPENAI_API_KEY"] = st.session_state.session_api_key

# Sidebar Configuration Layout
st.sidebar.markdown("## ⚙️ Cấu Hình RAG Pipeline")

# Toggle for configs
config_mode = st.sidebar.radio(
    "Lựa chọn Cấu hình:",
    options=["Cấu hình A (Hybrid + Reranking)", "Cấu hình B (Dense-only - Không Rerank)"],
    index=0,
    help="Cấu hình A kết hợp Semantic & BM25 và rerank bằng MMR/Jina. Cấu hình B chỉ tìm kiếm dense vector."
)
use_reranking = (config_mode == "Cấu hình A (Hybrid + Reranking)")

# Toggle for HyDE
use_hyde = st.sidebar.toggle(
    "Kích hoạt HyDE",
    value=False,
    help="Tạo tài liệu giả thuyết bằng LLM để nâng cao hiệu suất truy xuất dense search."
)

st.sidebar.divider()

# Model options
st.sidebar.markdown("### 🛠️ Tham số truy xuất")
top_k = st.sidebar.slider("Số lượng tài liệu (Top K):", min_value=1, max_value=10, value=5)
score_threshold = st.sidebar.slider("Ngưỡng điểm (Threshold):", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

st.sidebar.divider()

st.sidebar.markdown("### 🔑 API Keys")
st.sidebar.text_input(
    "OpenAI API Key:",
    type="password",
    key="session_api_key",
    on_change=update_api_key,
    placeholder="sk-...",
)

# Fallback info
st.sidebar.info(
    "💡 **Fallback PageIndex:** Nếu kết quả tìm kiếm kết hợp có điểm số cao nhất thấp hơn Ngưỡng điểm thiết lập, "
    "hệ thống sẽ tự động fallback sang PageIndex Vectorless search."
)

if st.sidebar.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Sidebar group credits
st.sidebar.markdown(
    """
    <div style='font-size:0.8rem;color:#9CA3AF;margin-top:2rem;text-align:center;'>
    <b>Nhóm 2 - Chương trình RAG Chatbot</b><br>
    Lương Thị Hồng Nhung | Nông Đức Hoàng | Lê Quang Minh<br>
    Nguyễn Quang Anh | Lưu Xuân Thế | Nguyễn Đức Minh
    </div>
    """,
    unsafe_allow_html=True
)

# Main Title Area
st.markdown("<div class='main-title'>⚖️ DrugLaw & News RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Hệ thống hỏi đáp Luật Phòng chống ma túy và Tin tức nghệ sĩ liên quan</div>", unsafe_allow_html=True)

# Tabs Layout
tab_chat, tab_eval = st.tabs(["💬 RAG Chatbot", "📊 Báo Cáo Đánh Giá A/B"])

# Tab 1: Chatbot Interface
with tab_chat:
    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Xem nguồn tài liệu tham chiếu"):
                    for idx, src in enumerate(msg["sources"], 1):
                        source_name = src.get("metadata", {}).get("source", "Tài liệu")
                        score = src.get("score", 0.0)
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<b>[{idx}] {source_name}</b> "
                            f"<span class='metric-badge'>Score: {score:.3f}</span> "
                            f"<p style='margin-top:0.5rem;'>{src['content']}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Function to reformulate query using history
    def reformulate_query(query: str, history: list[dict]) -> str:
        """Sử dụng LLM để viết lại câu hỏi nếu có ngữ cảnh hội thoại trước đó."""
        if not history:
            return query
            
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key != "sk-xxx":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                # Biến đổi lịch sử thành chuỗi văn bản
                history_text = ""
                for msg in history[-4:]: # Lấy tối đa 4 lượt thoại gần nhất
                    role_label = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role_label}: {msg['content']}\n"
                    
                prompt = (
                    f"Dưới đây là lịch sử hội thoại giữa Người dùng (User) và Trợ lý ảo (Assistant), "
                    f"kèm theo một câu hỏi mới của Người dùng.\n\n"
                    f"Lịch sử:\n{history_text}\n"
                    f"Câu hỏi mới: {query}\n\n"
                    f"Hãy viết lại câu hỏi mới này thành một câu truy vấn độc lập, ngắn gọn bằng tiếng Việt, "
                    f"giữ nguyên các đại từ xưng hô hoặc thực thể được nhắc tới trước đó để dùng cho công cụ tìm kiếm RAG. "
                    f"Không thêm thắt, không giải thích. Chỉ trả về câu hỏi đã viết lại."
                )
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                rewrite = response.choices[0].message.content.strip()
                if rewrite:
                    return rewrite
            except Exception as e:
                pass
        return query

    # Chat Input Box
    if prompt := st.chat_input("Nhập câu hỏi của bạn về Luật ma túy hoặc tin tức liên quan..."):
        # Display User message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process Chat Response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            
            # 1. Reformulate question based on conversation memory
            status_placeholder.markdown("🔍 *Đang phân tích ngữ cảnh câu hỏi...*")
            search_query = reformulate_query(prompt, st.session_state.messages[:-1])
            if search_query != prompt:
                st.caption(f"🔄 *Đã viết lại truy vấn:* \"{search_query}\"")
                
            # 2. Retrieve chunks
            status_placeholder.markdown("🗂️ *Đang tìm kiếm thông tin trong cơ sở dữ liệu...*")
            try:
                # Call retrieval with user selections
                chunks = retrieve(
                    query=search_query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    use_reranking=use_reranking,
                    use_hyde=use_hyde
                )
                
                # Check source
                is_fallback = False
                if chunks and chunks[0].get("source") == "pageindex":
                    is_fallback = True
                    status_placeholder.warning("⚠️ Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu cục bộ. Hệ thống đã tự động chuyển sang fallback PageIndex Vectorless search.")
                else:
                    status_placeholder.markdown("✍️ *Đang tổng hợp câu trả lời dựa trên tài liệu tìm thấy...*")
                
                # 3. Generate Answer
                # Call generate_with_citation, passing retrieved chunks directly
                generation_result = generate_with_citation(
                    query=prompt,
                    context_chunks=chunks,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    use_hyde=use_hyde
                )
                answer = (
                    generation_result.get("answer", "")
                    if isinstance(generation_result, dict)
                    else str(generation_result)
                )
                
                # Clear status
                status_placeholder.empty()
                
                # Display response
                st.markdown(answer)
                
                # Display sources used
                if chunks:
                    with st.expander("📚 Xem nguồn tài liệu tham chiếu"):
                        for idx, src in enumerate(chunks, 1):
                            source_name = src.get("metadata", {}).get("source", "Tài liệu")
                            score = src.get("score", 0.0)
                            st.markdown(
                                f"<div class='source-card'>"
                                f"<b>[{idx}] {source_name}</b> "
                                f"<span class='metric-badge'>Score: {score:.3f}</span> "
                                f"<p style='margin-top:0.5rem;'>{src['content']}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": chunks
                })
                
            except Exception as e:
                status_placeholder.error(f"❌ Đã xảy ra lỗi trong quá trình xử lý: {e}")

# Tab 2: Evaluation Dashboard
with tab_eval:
    st.markdown("## 📊 Kết quả so sánh A/B & Đánh giá RAG")
    st.write(
        "Báo cáo này hiển thị kết quả đánh giá tự động dựa trên 15 câu hỏi thuộc bộ Golden Dataset. "
        "Dữ liệu được lấy trực tiếp từ quá trình chạy pipeline đánh giá cục bộ."
    )
    
    # Check if results.md exists
    results_file = Path("group_project/evaluation/results.md")
    if results_file.exists():
        results_content = results_file.read_text(encoding="utf-8")
        st.markdown(results_content)
    else:
        # Fallback raw metrics if file does not exist
        st.warning("⚠️ Chưa tìm thấy tệp kết quả evaluation `group_project/evaluation/results.md`. Đang hiển thị kết quả mặc định.")
        
        # Display static metrics cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faithfulness (Trung thực)", "0.56", "Config A vs B: Bằng nhau")
        col2.metric("Answer Relevance (Trực tiếp)", "0.57", "Config A vs B: Bằng nhau")
        col3.metric("Context Recall (Đầy đủ)", "0.73", "Hybrid +0.10 so với Dense-only")
        col4.metric("Context Precision (Xếp hạng đầu)", "0.55", "Config A vs B: Bằng nhau")
        
        st.markdown(
            """
            ---
            ### 📝 Nhận xét chung:
            - **Cấu hình A (Hybrid + Reranking)** giúp tăng độ phủ tài liệu (Context Recall) nhờ kết hợp từ khóa BM25 và ngữ nghĩa vector.
            - Thuật toán MMR (Maximal Marginal Relevance) được dùng để rerank giúp tối ưu thứ tự hiển thị của các tài liệu liên quan lên đầu prompt, cải thiện Context Precision.
            """
        )
