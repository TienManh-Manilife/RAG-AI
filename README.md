# TRỢ LÝ AI CHĂM SÓC KHÁCH HÀNG DỰA TRÊN RAG

## Mục tiêu

Chatbot trả lời câu hỏi từ tài liệu PDF thực tế bằng Retrieval-Augmented
Generation (RAG). Khi không tìm được căn cứ đủ chắc chắn, hệ thống tạo yêu
cầu chuyển cho nhân viên thay vì cố gắng suy đoán.

## Kiến trúc

```text
PDF -> pypdf -> chunking có overlap -> multilingual embedding
    -> ChromaDB persistent -> cosine retrieval -> Ollama LLM
    -> câu trả lời có trích dẫn / hàng đợi nhân viên
```

## Cài đặt và chạy

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama serve
ollama pull qwen3:4b
streamlit run app.py
```

Mặc định Ollama chạy tại `http://127.0.0.1:11434`. Có thể đổi bằng biến môi
trường `OLLAMA_BASE_URL`. Ứng dụng hỗ trợ PDF có text layer; PDF scan cần OCR
trước khi tải lên.

## Các chức năng đã triển khai

- Tải một hoặc nhiều PDF, chuẩn hóa và chia chunk theo ranh giới đoạn/câu.
- Tạo embedding tiếng Việt bằng Sentence-Transformers.
- Lưu index bền vững bằng ChromaDB trong `data/chroma`.
- Trả lời bằng Ollama với prompt chỉ cho phép dùng context được truy xuất.
- Hiển thị tên tài liệu, số trang và nội dung các đoạn nguồn.
- Dedupe và cập nhật idempotent khi nạp lại cùng nội dung tài liệu.
- Xóa đúng nguồn bằng `source_id`.
- Chuyển tiếp khi không có context hoặc similarity cao nhất thấp hơn ngưỡng.
- Lưu yêu cầu chuyển tiếp tại `data/handoff_requests.json`.
- Ghi số liệu vận hành tại `data/metrics.json` và hiển thị tỷ lệ chuyển tiếp.

## Cơ chế chuyển nhân viên

Ngưỡng mặc định là `0.45`, có thể chỉnh trong sidebar. Một câu hỏi được đánh
dấu cần nhân viên nếu không có hit, similarity cao nhất thấp hơn ngưỡng, hoặc
LLM xác định không có đủ context. Người dùng có thể bấm **Chuyển câu hỏi cho
nhân viên**; yêu cầu được lưu trong trang **Hàng đợi nhân viên**.

Đây là mô phỏng handoff phù hợp cho đồ án. Khi triển khai thật, hàm
`create_handoff_request()` có thể được thay bằng API ticket, CRM, email hoặc
webhook.

## Đánh giá

`evaluation/questions.json` là bộ khung để nhập câu hỏi, đáp án chuẩn, trang
nguồn kỳ vọng và nhãn có cần handoff hay không. Cần đánh giá riêng:

- **Recall@K**: hit đúng xuất hiện trong K đoạn truy xuất.
- **Answer correctness**: chấm thủ công 0 (sai), 1 (một phần), 2 (đúng).
- **Citation accuracy**: câu trả lời trích đúng tài liệu/trang.
- **Handoff rate**: số câu chuyển nhân viên / tổng số câu hỏi.
- **Handoff precision/recall**: chất lượng quyết định chuyển tiếp.

Số liệu trong trang Thống kê là số liệu vận hành thực tế, không được gọi là
accuracy nếu chưa chạy trên bộ câu hỏi có đáp án và chấm độc lập.

## Vai trò người–máy

Chatbot xử lý các câu hỏi lặp lại có căn cứ rõ trong tài liệu, giúp giảm tải
nhân viên. Nhân viên xử lý ngoại lệ, câu hỏi liên quan quyền lợi cá nhân,
trường hợp similarity thấp và chịu trách nhiệm cuối cùng đối với câu trả lời.
Thiết kế này ưu tiên không bịa chính sách hơn là cố trả lời mọi câu hỏi.

## Thành viên

- Nguyễn Hải Phong - 21020085
- Trần Anh Dũng - 24020096
- Nguyễn Tiến Mạnh - 24020220
