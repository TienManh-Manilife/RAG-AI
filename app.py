
from pathlib import Path
import html
import json
import time

import requests
import streamlit as st

from rag_engine import (
    DEFAULT_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    create_handoff_request,
    delete_source,
    generate_answer,
    get_collection_stats,
    get_operational_metrics,
    ingest_pdf_bytes,
    list_handoff_requests,
    list_indexed_sources,
    record_interaction,
    retrieve_context,
    should_handoff,
)


# ============================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ============================================================

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = OLLAMA_BASE_URL
DEFAULT_MODEL = "qwen3:4b"

st.set_page_config(
    page_title="RAG-AI | Trợ lý tri thức",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CSS - GIAO DIỆN SÁNG, CHỮ GRADIENT TỐI
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #5145CD;
        --primary-dark: #312E81;
        --teal-dark: #115E59;
        --text: #172033;
        --muted: #64748B;
        --border: #E2E8F0;
        --surface: #FFFFFF;
        --background: #F7F8FC;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99, 102, 241, 0.07),
                transparent 28%
            ),
            #F7F8FC;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 248, 252, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
    }

    .gradient-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -0.055em;
        background: linear-gradient(
            115deg,
            #312E81 0%,
            #5145CD 42%,
            #115E59 100%
        );
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.45rem;
    }

    .gradient-subtitle {
        color: #475569;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        color: #1E1B4B;
        font-weight: 800;
        font-size: 1.3rem;
        margin-bottom: 0.35rem;
    }

    .muted {
        color: #64748B;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .brand {
        padding: 0.35rem 0 1.2rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.2rem;
    }

    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(110deg, #312E81, #5145CD, #115E59);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-caption {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.025);
    }

    .info-card-title {
        font-weight: 700;
        color: #312E81;
        margin-bottom: 0.3rem;
    }

    .info-card-text {
        color: #64748B;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .source-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #5145CD;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0;
        overflow-wrap: anywhere;
    }

    .source-name {
        color: #312E81;
        font-weight: 700;
        font-size: 0.92rem;
    }

    .source-detail {
        color: #64748B;
        font-size: 0.82rem;
        margin-top: 0.3rem;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
        background: #ECFDF5;
        color: #065F46;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.65rem;
        border: 1px solid #E2E8F0;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #818CF8;
        color: #312E81;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(110deg, #4338CA, #0F766E);
        color: white;
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        color: white;
        filter: brightness(1.06);
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
    }

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #A5B4FC;
        border-radius: 14px;
        padding: 0.8rem;
    }

    hr {
        border-color: #E2E8F0;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE VÀ HÀM HỖ TRỢ
# ============================================================

def initialize_state():
    defaults = {
        "current_page": "Chat",
        "messages": [],
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "num_ctx": 4096,
        "top_k": 5,
        "min_similarity": 0.45,
        "chunk_size": 800,
        "overlap": 120,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_page(page_name):
    """Callback điều hướng; không chỉnh state của widget radio."""
    st.session_state["current_page"] = page_name


def check_ollama():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3,
        )
        response.raise_for_status()
        data = response.json()
        models = [
            item.get("name", "")
            for item in data.get("models", [])
        ]
        return True, models
    except (requests.RequestException, ValueError):
        return False, []


def safe_text(value, default="Không có thông tin"):
    if value is None:
        return default
    return str(value)


def get_source_name(item):
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in (
            "source",
            "source_name",
            "filename",
            "file_name",
            "name",
        ):
            if item.get(key):
                return str(item[key])

        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("source", "filename", "file_name"):
                if metadata.get(key):
                    return str(metadata[key])

    return str(item)


def get_hit_metadata(hit):
    if not isinstance(hit, dict):
        return {}

    metadata = hit.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata

    return {}


def render_sources(hits):
    if not hits:
        st.caption("Không có đoạn tài liệu nguồn được trả về.")
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(hits)} đoạn)", expanded=False):
        seen = set()

        for index, hit in enumerate(hits, start=1):
            metadata = get_hit_metadata(hit)

            if isinstance(hit, dict):
                source = (
                    hit.get("source")
                    or hit.get("filename")
                    or metadata.get("source")
                    or metadata.get("filename")
                    or "Tài liệu"
                )

                page_number = (
                    metadata.get("page")
                    or metadata.get("page_number")
                    or hit.get("page")
                )

                content = (
                    hit.get("text")
                    or hit.get("content")
                    or hit.get("document")
                    or ""
                )

                if not content and isinstance(hit.get("page_content"), str):
                    content = hit["page_content"]
            else:
                source = "Tài liệu"
                page_number = None
                content = str(hit)

            unique_key = (str(source), str(page_number), str(content)[:100])
            if unique_key in seen:
                continue
            seen.add(unique_key)

            source_escaped = html.escape(str(source))
            content_escaped = html.escape(str(content))
            page_label = (
                f" · Trang {html.escape(str(page_number))}"
                if page_number is not None
                else ""
            )

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-name">
                        {index}. {source_escaped}{page_label}
                    </div>
                    <div class="source-detail">
                        {content_escaped[:900]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_stat_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return safe_text(value)


def normalize_source_list(sources):
    if sources is None:
        return []

    if isinstance(sources, dict):
        # Một số phiên bản có thể trả về dict chứa danh sách tài liệu.
        for key in ("sources", "documents", "files", "items"):
            if isinstance(sources.get(key), list):
                return sources[key]
        return list(sources.keys())

    if isinstance(sources, (list, tuple, set)):
        return list(sources)

    return [sources]


def format_ingest_result(result):
    if result is None:
        return "Đã gửi tài liệu đến bộ xử lý."

    if isinstance(result, dict):
        message = (
            result.get("message")
            or result.get("status")
            or result.get("detail")
        )
        if message:
            return str(message)

        return json.dumps(result, ensure_ascii=False)

    return str(result)


initialize_state()


# ============================================================
# 4. SIDEBAR - THƯƠNG HIỆU, ĐIỀU HƯỚNG, CẤU HÌNH
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-name">◈ RAG-AI</div>
            <div class="brand-caption">
                Retrieval-Augmented Generation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧭 Điều hướng")

    # Dùng các nút riêng, không dùng st.radio làm trạng thái trang.
    # Vì vậy callback không thể gây lỗi WidgetAlreadyInstantiatedError.
    current_page = st.session_state["current_page"]

    st.button(
        "💬  Trò chuyện",
        key="nav_chat",
        use_container_width=True,
        type="primary" if current_page == "Chat" else "secondary",
        on_click=set_page,
        args=("Chat",),
    )

    st.button(
        "📚  Kho tài liệu PDF",
        key="nav_documents",
        use_container_width=True,
        type="primary" if current_page == "Kho tài liệu" else "secondary",
        on_click=set_page,
        args=("Kho tài liệu",),
    )

    st.button(
        "📊  Thống kê hệ thống",
        key="nav_stats",
        use_container_width=True,
        type="primary" if current_page == "Thống kê" else "secondary",
        on_click=set_page,
        args=("Thống kê",),
    )

    st.button(
        "🙋  Hàng đợi nhân viên",
        key="nav_handoff",
        use_container_width=True,
        type="primary" if current_page == "Hàng đợi nhân viên" else "secondary",
        on_click=set_page,
        args=("Hàng đợi nhân viên",),
    )

    st.divider()

    st.markdown("### ⚙️ Cấu hình mô hình")

    model = st.text_input(
        "Tên model Ollama",
        key="model",
        help="Ví dụ: qwen3:4b. Model phải được cài trên Ollama.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        key="temperature",
        help="Giá trị thấp thường tạo câu trả lời nhất quán hơn.",
    )

    num_ctx = st.select_slider(
        "Context window",
        options=[2048, 4096, 8192, 16384],
        key="num_ctx",
    )

    top_k = st.slider(
        "Số đoạn truy xuất (Top-K)",
        min_value=1,
        max_value=10,
        step=1,
        key="top_k",
    )

    st.slider(
        "Ngưỡng chuyển nhân viên",
        min_value=0.10,
        max_value=0.90,
        step=0.05,
        key="min_similarity",
        help="Nếu similarity cao nhất thấp hơn ngưỡng, câu hỏi sẽ được đề nghị chuyển nhân viên.",
    )

    st.divider()

    st.markdown("### 🖥️ Trạng thái Ollama")

    ollama_ok, installed_models = check_ollama()

    if ollama_ok:
        st.markdown(
            '<span class="status-pill">● Ollama đang hoạt động</span>',
            unsafe_allow_html=True,
        )

        if installed_models:
            st.caption("Model đã cài:")
            for installed_model in installed_models[:8]:
                st.caption(f"• {installed_model}")
        else:
            st.caption("Chưa tìm thấy model nào đã cài.")
    else:
        st.error("Không kết nối được Ollama.")
        st.caption(
            "Kiểm tra Ollama đã chạy tại "
            "http://127.0.0.1:11434 hay chưa."
        )

    st.divider()
    st.caption("RAG-AI · Trợ lý hỏi đáp theo tài liệu")


# ============================================================
# 5. HÀM HIỂN THỊ HEADER
# ============================================================

def render_header(title, subtitle):
    st.markdown(
        f'<div class="gradient-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="gradient-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 6. TRANG CHATBOT
# ============================================================

def render_chat_page():
    render_header(
        "Trợ lý tri thức AI",
        "Đặt câu hỏi bằng ngôn ngữ tự nhiên. RAG-AI sẽ tìm các đoạn "
        "tài liệu liên quan trước khi tạo câu trả lời.",
    )

    col1, col2, col3 = st.columns(3)

    try:
        stats = get_collection_stats()
    except Exception:
        stats = {}

    if not isinstance(stats, dict):
        stats = {}

    document_count = (
        stats.get("document_count")
        or stats.get("documents")
        or stats.get("total_documents")
        or 0
    )

    chunk_count = (
        stats.get("chunk_count")
        or stats.get("total_chunks")
        or stats.get("chunks")
        or 0
    )

    with col1:
        st.metric("Đoạn dữ liệu", document_count)

    with col2:
        st.metric("Số đoạn truy xuất", st.session_state["top_k"])

    with col3:
        st.metric(
            "Trạng thái mô hình",
            "Sẵn sàng" if ollama_ok else "Ngoại tuyến",
        )

    st.markdown("---")

    # Hiển thị lịch sử trò chuyện.
    for message in st.session_state["messages"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        # Không dùng avatar tùy chỉnh để tránh lỗi đường dẫn avatar.
        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant" and message.get("hits"):
                render_sources(message["hits"])

    prompt = st.chat_input(
        "Nhập câu hỏi của bạn về tài liệu...",
        key="chat_prompt",
    )

    if prompt:
        if not ollama_ok:
            st.error(
                "Không kết nối được Ollama. Hãy khởi động Ollama "
                "trước khi gửi câu hỏi."
            )
            return

        st.session_state["messages"].append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            try:
                with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
                    hits = retrieve_context(
                        prompt,
                        top_k=int(st.session_state["top_k"]),
                    )

                    answer, generation_metadata = generate_answer(
                        prompt,
                        hits,
                        model=st.session_state["model"],
                        temperature=float(st.session_state["temperature"]),
                        num_ctx=int(st.session_state["num_ctx"]),
                    )
                    handoff_required, handoff_reason, similarity = should_handoff(
                        hits,
                        generation_metadata,
                        min_similarity=float(st.session_state["min_similarity"]),
                    )
                    st.metric(
                        label="Độ tương đồng cao nhất",
                        value=f"{similarity:.2%}",
                    )
                    best_similarity = max(
                        (float(hit.get("similarity", 0.0)) for hit in hits),
                        default=0.0,
                    )
                    record_interaction(
                        handoff_required=handoff_required,
                        best_similarity=best_similarity,
                    )

                if answer is None:
                    answer = "Mô hình không trả về nội dung câu trả lời."

                answer = str(answer)
                response_placeholder.markdown(answer)

                if hits:
                    render_sources(hits)

                if handoff_required:
                    st.warning(
                        "⚠️ Câu hỏi này cần nhân viên kiểm tra thêm. "
                        f"Lý do: {handoff_reason}"
                    )
                    if st.button(
                        "🙋 Chuyển câu hỏi cho nhân viên",
                        key=f"handoff_{len(st.session_state['messages'])}",
                        type="primary",
                    ):
                        create_handoff_request(
                            question=prompt,
                            answer=answer,
                            reason=handoff_reason,
                            hits=hits,
                        )
                        st.success(
                            "Đã ghi nhận yêu cầu. Nhân viên có thể tiếp nhận "
                            "tại trang Hàng đợi nhân viên."
                        )

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "hits": hits or [],
                        "handoff_required": handoff_required,
                        "handoff_reason": handoff_reason,
                        "generation_metadata": generation_metadata,
                    }
                )

            except Exception as exc:
                error_message = (
                    f"Đã xảy ra lỗi khi xử lý câu hỏi: {exc}\n\n"
                    "Hãy kiểm tra Ollama, model và cấu hình trong "
                    "`rag_engine.py`."
                )
                response_placeholder.error(error_message)

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "hits": [],
                    }
                )

    if st.session_state["messages"]:
        st.markdown("---")
        if st.button("🗑️ Xóa lịch sử trò chuyện", key="clear_chat"):
            st.session_state["messages"] = []
            st.rerun()


# ============================================================
# 7. TRANG KHO TÀI LIỆU PDF
# ============================================================

def render_documents_page():
    render_header(
        "Kho tài liệu",
        "Tải tài liệu PDF lên để hệ thống trích xuất nội dung, chia "
        "thành các đoạn và lập chỉ mục phục vụ truy xuất RAG.",
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">📄 Quy trình xử lý tài liệu</div>
            <div class="info-card-text">
                1. Chọn một hoặc nhiều tệp PDF.<br>
                2. Cấu hình kích thước đoạn và phần chồng lấp.<br>
                3. Nhấn xử lý để đưa tài liệu vào kho truy xuất.<br>
                4. Quay lại trang Trò chuyện để đặt câu hỏi.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📤 Tải PDF lên")

    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều tệp PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Chỉ hỗ trợ tệp PDF.",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.slider(
            "Kích thước mỗi đoạn (chunk size)",
            min_value=200,
            max_value=2000,
            step=100,
            key="chunk_size",
        )

    with col2:
        st.slider(
            "Độ chồng lấp giữa các đoạn (overlap)",
            min_value=0,
            max_value=500,
            step=20,
            key="overlap",
        )

    chunk_size = int(st.session_state["chunk_size"])
    overlap = int(st.session_state["overlap"])

    if overlap >= chunk_size:
        st.warning("Overlap phải nhỏ hơn chunk size.")
    else:
        st.caption(
            f"Chunk size: {chunk_size} · Overlap: {overlap}"
        )

    if uploaded_files:
        st.markdown(f"**Đã chọn {len(uploaded_files)} tệp:**")

        for uploaded in uploaded_files:
            st.write(f"📄 {uploaded.name} · {uploaded.size:,} bytes")

    can_process = bool(uploaded_files) and overlap < chunk_size

    if st.button(
        "⚡ XỬ LÝ VÀ LẬP CHỈ MỤC",
        type="primary",
        use_container_width=True,
        disabled=not can_process,
        key="process_pdfs",
    ):
        success_count = 0
        failure_count = 0

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded in enumerate(uploaded_files):
            status.info(
                f"Đang xử lý {index + 1}/{len(uploaded_files)}: "
                f"{uploaded.name}"
            )

            try:
                raw_bytes = uploaded.getvalue()

                if not raw_bytes:
                    raise ValueError("Tệp rỗng.")

                # Lưu bản PDF gốc vào thư mục uploads.
                safe_filename = Path(uploaded.name).name
                save_path = UPLOAD_DIR / safe_filename
                save_path.write_bytes(raw_bytes)

                # Đưa PDF vào pipeline hiện có của rag_engine.py.
                result = ingest_pdf_bytes(
                    uploaded.name,
                    raw_bytes,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                st.success(
                    f"Đã xử lý {uploaded.name}: "
                    f"{format_ingest_result(result)}"
                )
                success_count += 1

            except Exception as exc:
                st.error(f"Lỗi với {uploaded.name}: {exc}")
                failure_count += 1

            progress.progress((index + 1) / len(uploaded_files))

        status.empty()

        if success_count:
            st.success(
                f"Hoàn tất: {success_count} tệp thành công, "
                f"{failure_count} tệp thất bại."
            )

        if success_count:
            st.rerun()

    st.markdown("---")
    st.markdown("### 🗂️ Tài liệu đã lập chỉ mục")

    try:
        sources = normalize_source_list(list_indexed_sources())
    except Exception as exc:
        st.error(f"Không tải được danh sách tài liệu: {exc}")
        sources = []

    if not sources:
        st.info(
            "Kho tài liệu chưa có dữ liệu hoặc chưa thể lấy danh sách. "
            "Hãy tải PDF lên và xử lý ở phần phía trên."
        )
        return

    st.caption(f"Tổng số mục nguồn: {len(sources)}")

    for index, source_item in enumerate(sources):
        source_name = get_source_name(source_item)

        with st.container(border=True):
            col_name, col_delete = st.columns([5, 1])

            with col_name:
                st.markdown(f"**📄 {source_name}**")

                if isinstance(source_item, dict):
                    metadata = source_item.get("metadata", {})
                    if isinstance(metadata, dict) and metadata:
                        st.caption(
                            " · ".join(
                                f"{key}: {value}"
                                for key, value in metadata.items()
                            )
                        )

            with col_delete:
                if st.button(
                    "🗑️",
                    key=f"delete_source_{index}_{source_name}",
                    help=f"Xóa nguồn {source_name} khỏi chỉ mục",
                ):
                    st.session_state["pending_delete_source_id"] = (
                        source_item.get("source_id")
                        if isinstance(source_item, dict)
                        else None
                    )
                    st.session_state["pending_delete_source_name"] = source_name

    pending_source_id = st.session_state.get("pending_delete_source_id")
    pending_source = st.session_state.get("pending_delete_source_name")

    if pending_source:
        st.warning(
            f"Bạn có chắc muốn xóa nguồn **{pending_source}** "
            "khỏi kho chỉ mục?"
        )

        col_confirm, col_cancel = st.columns(2)

        with col_confirm:
            if st.button(
                "Xác nhận xóa",
                type="primary",
                key="confirm_delete_source",
            ):
                try:
                    if not pending_source_id:
                        raise ValueError("Không xác định được source_id của tài liệu.")
                    delete_source(pending_source_id)
                    st.success(
                        f"Đã gửi yêu cầu xóa nguồn {pending_source}."
                    )
                    st.session_state.pop("pending_delete_source_id", None)
                    st.session_state.pop("pending_delete_source_name", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể xóa nguồn: {exc}")

        with col_cancel:
            if st.button("Hủy", key="cancel_delete_source"):
                st.session_state.pop("pending_delete_source_id", None)
                st.session_state.pop("pending_delete_source_name", None)
                st.rerun()


# ============================================================
# 8. TRANG THỐNG KÊ
# ============================================================

def render_stats_page():
    render_header(
        "Thống kê hệ thống",
        "Theo dõi tình trạng kết nối mô hình và thông tin kho tài liệu.",
    )

    ollama_connected, models = check_ollama()

    try:
        stats = get_collection_stats()
    except Exception as exc:
        stats = {}
        st.error(f"Không lấy được thống kê kho dữ liệu: {exc}")

    if not isinstance(stats, dict):
        stats = {"Thông tin kho dữ liệu": stats}

    try:
        sources = normalize_source_list(list_indexed_sources())
    except Exception:
        sources = []

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nguồn tài liệu", len(sources))

    with col2:
        st.metric(
            "Model Ollama",
            "Đang kết nối" if ollama_connected else "Ngoại tuyến",
        )

    with col3:
        st.metric("Tin nhắn trong phiên", len(st.session_state["messages"]))

    try:
        operational_metrics = get_operational_metrics()
    except ValueError as exc:
        operational_metrics = {}
        st.error(str(exc))

    st.markdown("---")
    st.markdown("### 📈 Chỉ số vận hành")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(
            "Câu hỏi đã xử lý",
            operational_metrics.get("questions", 0),
        )
    with metric_col2:
        st.metric(
            "Số lượt chuyển nhân viên",
            operational_metrics.get("handoffs", 0),
        )
    with metric_col3:
        st.metric(
            "Tỷ lệ chuyển tiếp",
            f"{operational_metrics.get('handoff_rate', 0.0):.1%}",
        )
    st.caption(
        "Các chỉ số này là số liệu vận hành, không thay thế đánh giá accuracy "
        "trên bộ câu hỏi có đáp án chuẩn."
    )

    st.markdown("---")
    st.markdown("### 📦 Thông tin kho dữ liệu")

    if stats:
        for key, value in stats.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            if isinstance(value, (dict, list)):
                st.json(value)
            else:
                st.write(value)
    else:
        st.info("Chưa có thông tin thống kê được trả về.")

    st.markdown("---")
    st.markdown("### 🖥️ Ollama")

    if ollama_connected:
        st.success(f"Kết nối thành công: {OLLAMA_URL}")
    else:
        st.error(f"Không kết nối được: {OLLAMA_URL}")

    st.markdown("**Model đã cài**")

    if models:
        for model_name in models:
            selected = (
                " (đang chọn)"
                if model_name == st.session_state["model"]
                else ""
            )
            st.write(f"• {model_name}{selected}")
    else:
        st.info("Không tìm thấy model hoặc Ollama chưa hoạt động.")

    st.markdown("---")
    st.markdown("### 📚 Danh sách nguồn")

    if sources:
        for source_item in sources:
            st.write(f"📄 {get_source_name(source_item)}")
    else:
        st.info("Chưa có nguồn tài liệu được lập chỉ mục.")


def render_handoff_page():
    render_header(
        "Hàng đợi nhân viên",
        "Các câu hỏi có độ tin cậy thấp được lưu lại để nhân viên kiểm tra "
        "và phản hồi.",
    )
    requests_data = list_handoff_requests()
    if not requests_data:
        st.info("Chưa có yêu cầu nào cần nhân viên tiếp nhận.")
        return

    st.metric(
        "Yêu cầu đang chờ",
        sum(item.get("status") == "pending" for item in requests_data),
    )
    for item in reversed(requests_data):
        with st.container(border=True):
            st.markdown(
                f"**{item.get('id', 'Không có mã')}** · "
                f"{item.get('created_at', 'Không rõ thời gian')}"
            )
            st.markdown(f"**Câu hỏi:** {item.get('question', '')}")
            st.markdown(f"**Lý do chuyển:** {item.get('reason', '')}")
            st.markdown(f"**Câu trả lời tự động:** {item.get('answer', '')}")
            sources = item.get("sources") or []
            if sources:
                st.caption(
                    "Nguồn đã truy xuất: "
                    + ", ".join(
                        f"{source.get('source', 'không rõ')} "
                        f"(trang {source.get('page', '?')})"
                        for source in sources
                    )
                )


# ============================================================
# 9. ĐIỀU PHỐI TRANG
# ============================================================

page = st.session_state["current_page"]

if page == "Chat":
    render_chat_page()

elif page == "Kho tài liệu":
    render_documents_page()

elif page == "Thống kê":
    render_stats_page()

elif page == "Hàng đợi nhân viên":
    render_handoff_page()

else:
    st.session_state["current_page"] = "Chat"
    st.rerun()


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748B;
        font-size: 0.82rem;
        padding: 0.4rem 0 1rem 0;
    ">
        RAG-AI · Hỏi đáp dựa trên tài liệu · Powered by Ollama
    </div>
    """,
    unsafe_allow_html=True,
)
