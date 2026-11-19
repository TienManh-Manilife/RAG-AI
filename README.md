# ✦ RAG–AI Enterprise — Hỏi đáp tài liệu có hỗ trợ nhân viên

**Mục tiêu đồ án:** chatbot dùng dữ liệu quy chế đào tạo / chính sách bảo hành thực tế, chuyển nhân viên khi thiếu căn cứ; đo chất lượng tra cứu, tỷ lệ chuyển tiếp và phân tích Human-in-the-loop.

## 1. Tính năng

- Đăng ký, đăng nhập cổng người dùng/nhân viên; quản trị viên được bootstrap bằng secret khi **CSDL hoàn toàn trống**.
- Password PBKDF2-HMAC-SHA256 (600.000 vòng + salt), khóa tài khoản 15 phút sau 5 lần đăng nhập sai; role `user`, `staff`, `admin`.
- Nhân viên chỉ đăng ký bằng **mã mời bí mật** (tối thiểu 12 ký tự), hoặc do admin tạo. Người dùng không truy cập trang quản trị/kho chung/hàng đợi.
- RAG chế độ **Kho hệ thống** (ChromaDB, nhân viên quản lý) và **Tài liệu riêng** (PDF/DOCX/TXT; vector embedding trong session, không ghi vào Chroma chung).
- Chat có lịch sử riêng theo tài khoản, hỗ trợ truy vấn tiếp nối ngữ cảnh, phản hồi hữu ích/chưa đúng, nguồn tham khảo theo trang.
- Đề xuất chuyển tiếp dựa trên điểm bằng chứng, tình trạng thiếu căn cứ, lời từ chối và trích dẫn; người dùng có thể chuyển bất cứ câu trả lời nào.
- Nhân viên nhận xử lý, phản hồi, đóng phiếu; quản trị được mở lại.
- Dashboard hiển thị tổng số câu hỏi, số đề xuất chuyển, số chuyển thật, tỷ lệ, thời gian xử lý và phản hồi người dùng.
- Bộ đánh giá CSV: exact match, token F1, Retrieval Recall@5 (nếu có `expected_source`), citation rate, precision/recall chuyển tiếp (nếu có nhãn), gợi ý ngưỡng trên tập đánh giá và thẩm định thủ công từng câu bởi chuyên gia.

## 2. Cấu trúc

```text
app.py                       # UI Streamlit
rag_engine/
  auth.py                    # SQLite users / password / roles
  service.py                 # Conversations, queue, events, feedback
  config.py                  # Cấu hình & secrets
  retrieval.py               # Dense candidate retrieval + lexical rerank + MMR
  private_docs.py            # Tài liệu riêng của session
  embeddings.py              # SentenceTransformer
  database.py                # ChromaDB persistent store
  ingestion.py               # Nạp PDF kho chung
  sources.py                 # Quản lý nguồn hệ thống
  generation.py              # Prompt grounded + AI gateway
  handoff.py                 # Quy tắc đề xuất chuyển tiếp
  evaluation.py              # Metric đánh giá tham chiếu
  text_processing.py         # PDF extraction/chunking
ai_server/server.py          # Gateway FastAPI cho Ollama (mới)
requirements.txt
requirements-server.txt
.streamlit/secrets.toml.example
.env.example
evaluation/sample_questions.csv
```

## 3. Cài đặt (Windows PowerShell)

```powershell
git clone <URL_REPO_CUA_BAN>
cd RAG-AI
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Nên dùng Python 3.11. Lần đầu chạy sẽ tải embedding model từ Hugging Face nếu thiếu `embedding_model/`.
- Streamlit Cloud không chạy được model Ollama trên máy cá nhân nếu không có gateway công khai (ví dụ Cloudflare Tunnel).
- PDF scan chỉ chứa ảnh không hỗ trợ OCR trong bản này; OCR trước khi nạp.

### Chạy AI gateway trên máy cài Ollama

```powershell
pip install -r requirements-server.txt
ollama pull qwen3:4b
# Nếu Ollama chưa tự chạy: mở terminal khác rồi gõ: ollama serve
$env:AI_SERVER_API_KEY = "YOUR_NEW_LONG_RANDOM_KEY_AT_LEAST_24_CHARS"
$env:AI_SERVER_MODEL = "qwen3:4b"
python -m uvicorn ai_server.server:app --host 127.0.0.1 --port 8000
```

Kiểm tra `http://127.0.0.1:8000/health`. Đặt `AI_SERVER_API_KEY` này giống khóa trong Streamlit Secrets. Nếu dùng Streamlit Cloud, đưa gateway ra HTTPS qua tunnel **và chỉ công bố gateway FastAPI, không công bố cổng Ollama**.

### Cấu hình ứng dụng Streamlit

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Sửa `.streamlit/secrets.toml` (không commit file này):

```toml
[ai_server]
base_url = "http://127.0.0.1:8000" # Với Streamlit Cloud: URL HTTPS tunnel
api_key = "SAME_LONG_KEY_AS_AI_SERVER_API_KEY"
model = "qwen3:4b"

[auth]
bootstrap_admin_email = "admin@yourdomain.com"
bootstrap_admin_password = "LONG_UNIQUE_ADMIN_PASSWORD"
bootstrap_admin_name = "System Administrator"
staff_invite_code = "SECRET_STAFF_INVITE_CODE_AT_LEAST_12_CHARS"
```

```powershell
streamlit run app.py
```

Mở `http://localhost:8501`.

**Cảnh báo khi triển khai Streamlit Cloud:** `data/` trên đĩa của Streamlit Cloud có thể bị **mất khi restart/redeploy**, không đảm bảo nhiều replica truy cập cùng SQLite/Chroma. Muốn triển khai thực tế và lưu dữ liệu bền vững, phải dùng PostgreSQL cho tài khoản/phiếu/chat và vector DB persistent (Chroma server/pgvector/Qdrant) thay cho SQLite/Chroma local. **Không đưa dữ liệu thật nhạy cảm lên cloud thử nghiệm** nếu chưa có quy trình bảo vệ và quyền truy cập phù hợp. Nếu sử dụng nhiều worker, cần rate-limit bằng shared store, xác thực phiên, CSRF/session hardening, email verification/reset password, logging kiểm toán và backup. Bản này hướng đến đồ án/demo có một instance, không phải hệ thống production đã qua kiểm toán bảo mật.

## 4. Quy trình nghiệp vụ

1. Admin đăng nhập qua cổng nhân viên bằng email/password bootstrap; nếu CSDL đã có tài khoản thì bootstrap admin không được chạy lại.
2. Admin tạo nhân viên, hoặc nhân viên đăng ký bằng mã mời. Khách hàng tự đăng ký.
3. Nhân viên mở **Kho tài liệu hệ thống** và nhập PDF thực tế; hệ thống lấy chữ, chia chunk và lập chỉ mục.
4. Người dùng hỏi đáp theo **Kho hệ thống** hoặc nạp riêng PDF/DOCX/TXT rồi chuyển sang **Tài liệu tải lên**.
5. Hệ thống truy xuất các đoạn gần nhất, rerank theo từ khóa, chọn nguồn đa dạng và gọi AI Gateway tạo câu trả lời có trích dẫn.
6. Nếu câu trả lời không có căn cứ, nguồn yếu hoặc không có trích dẫn, hệ thống _đề xuất_ chuyển nhân viên. Người dùng nhấn **Chuyển nhân viên** để _thực sự tạo phiếu_.
7. Nhân viên vào Hàng đợi, nhấn **Nhận xử lý** → nhập phản hồi → **Hoàn tất**. Người dùng đọc phản hồi ở lịch sử chat và **Yêu cầu của tôi**.
8. Nhân viên xem **Báo cáo** hoặc **Đánh giá** để thu số liệu đồ án.

## 5. Giải thích điểm tin cậy

**Không lấy cosine similarity làm xác suất đúng.** Giá trị cosine `0.45` hoàn toàn không có nghĩa câu trả lời đúng 45%. Chỉ số mới:

```text
semantic_signal = clip((cosine - 0.20) / 0.45, 0, 1)
evidence_index = 0.60 * semantic_signal + 0.40 * lexical_coverage
```

`lexical_coverage` là tỷ lệ từ khóa câu hỏi có mặt trong chunk (có thưởng cho chuỗi liên tục). Công thức là **heuristic công khai, chưa hiệu chỉnh**, dùng để xếp hạng và ra quyết định cần xem xét. Nếu không có kiểm định trên dữ liệu gắn nhãn thì **không được gọi 75% bằng chứng là 75% chính xác**. Mặc định đề xuất chuyển nếu evidence index dưới 0,45, câu trả lời thiếu trích dẫn hoặc nêu thiếu căn cứ. Để tối ưu ngưỡng theo lĩnh vực, tạo ít nhất 100–200 cặp câu hỏi có đáp án thực và câu hỏi cố ý ngoài tài liệu; tối ưu theo precision/recall của chuyển tiếp và mức rủi ro trả lời sai.

## 6. Đánh giá theo đề bài

Tạo file CSV (UTF-8) theo mẫu `evaluation/sample_questions.csv` với cột bắt buộc `question,reference_answer` và tùy chọn `expected_source,should_handoff`. Vào tài khoản nhân viên > **Đánh giá RAG**, nạp CSV, chọn số mẫu, chạy. Sau đó mở phần **Thẩm định thủ công** để đánh dấu Đúng / Sai và tải CSV có nhãn chuyên gia. Công cụ có thể gợi ý ngưỡng phân luồng trên ít nhất 10 câu gắn nhãn đủ cả hai lớp; cần kiểm tra lại ngưỡng trên tập holdout độc lập.

- Exact match / token F1: proxy về độ giống chữ, không thay thế đánh giá ngữ nghĩa.
- Retrieval Recall@5: tỷ lệ câu hỏi có ít nhất một nguồn đúng trong top 5, nếu có nhãn nguồn.
- Handoff precision/recall: đánh giá khả năng đề xuất chuyển tiếp dựa trên `should_handoff`.
- Suggested handoff rate = số câu đề xuất chuyển / số câu hỏi; Actual handoff rate = số phiếu thật sự mở / số câu hỏi (có thể bị ảnh hưởng bởi hành vi khách hàng).
- Dashboard feedback accuracy: tỷ lệ người dùng đánh giá hữu ích, chịu selection bias.
- Nên thêm đánh giá thủ công: tính đúng sự kiện, đủ thông tin, chất lượng trích dẫn, nguy cơ trả lời không có căn cứ; báo cáo số lượng mẫu và khoảng tin cậy.

## 7. Giới hạn và lưu ý an toàn

- Upload ở chế độ riêng được giữ trong RAM của **Streamlit session**, không làm nhiễm kho chung và không lưu nội dung PDF gốc xuống đĩa. Khi phiên kết thúc/restart, cần nạp lại; lịch sử câu hỏi/trả lời vẫn ở SQLite. Dữ liệu của chế độ riêng **vẫn được gửi tới AI Gateway** để tạo câu trả lời.
- Trích dẫn kiểm tra bằng mẫu/nguồn (heuristic), **không phải** kiểm định entailment ngữ nghĩa. Mô hình có thể trích đúng nguồn nhưng diễn giải sai. Cần đánh giá chuyên gia trước khi dùng trong quyết định quyền lợi.
- ChromaDB thực hiện dense candidate search rồi rerank từ khóa + MMR, **chưa phải** full-corpus sparse retrieval/BM25. Muốn mở rộng kho lớn nên thêm sparse index thực sự và cross-encoder reranker.
- Việc xóa/tái lập chỉ mục chưa có transaction xuyên SQLite và Chroma. Không chạy đồng thời hai lần nạp cùng tài liệu.
- Để nhập liệu có dấu/phức tạp, ưu tiên PDF dạng text; OCR là quy trình riêng.
- Không nhúng khóa truy cập hoặc mật khẩu vào source code; thay khóa nếu nghi ngờ lộ.

## 8. Kiểm thử

```powershell
pip install pytest
python -m pytest tests -q
python -m compileall -q app.py rag_engine ai_server
```

Các unit test không cần mô hình Ollama, mạng, ChromaDB hay Streamlit. Phần chạy trực tiếp trên Streamlit và AI Gateway cần được kiểm tra tích hợp trong môi trường có đủ dependencies.

**Nâng cấp dữ liệu từ phiên bản cũ:** Giữ `data/chroma/` cùng tên collection và cùng embedding model để tái sử dụng vector (nếu tương thích phiên bản Chroma). Không tự động nhập `handoff_requests.json`/`metrics.json` cũ vì bản JSON không có định danh người dùng đã xác thực; cần gắn chủ sở hữu hợp lệ trước khi import để tránh lộ thông tin giữa các tài khoản.
