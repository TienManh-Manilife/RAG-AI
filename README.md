# Trợ lý chăm sóc khách hàng RAG (Docker Local)

> **Đồ án nghiên cứu, 100% dữ liệu doanh nghiệp giả lập.** Tên An Phát, chính sách, email `.example` và các chỉ tiêu trong `knowledge_base/` không phải thông tin của doanh nghiệp thực. Không nên dùng để quyết định quyền lợi thực tế.

## 1. Kiến trúc

```text
Trình duyệt trên máy (http://localhost:8501)
  ↓ [localhost only]
Streamlit /app trong Docker  [Xác thực, phân quyền, hội thoại, phiếu]
  ├── SQLite /app/data/  [Tài khoản, hội thoại, phiếu, tài liệu, embedding, KPI]
  │     └── Truy xuất hybrid BM25 + BGE-M3 cosine + reciprocal rank fusion
  └── Ollama http://ollama:11434 (mạng Docker nội bộ)
        ├── qwen3:4b     [Sinh câu trả lời JSON kèm trích dẫn]
        └── bge-m3       [Vector embedding đa ngôn ngữ]
```

Không có **Cloudflare**, **FastAPI Gateway**, API key gateway hoặc cổng Ollama công khai. Trình duyệt chỉ kết nối Streamlit tại `127.0.0.1:8501` trên máy chủ. Dữ liệu được giữ trong các **Docker named volumes**.

## 2. Yêu cầu

- Windows 10/11 + Docker Desktop (WSL2), hoặc Linux/macOS có Docker Engine + Docker Compose v2.
- Internet **lúc cài lần đầu** để tải Docker images và hai mô hình Ollama; sau khi mô hình đã tải, suy luận diễn ra nội bộ.
- Khuyến nghị tối thiểu 8 GB RAM còn trống cho trải nghiệm CPU; 16 GB trở lên thuận lợi hơn. Model Qwen3 4B chạy CPU có thể chậm; không đảm bảo thời gian đáp ứng cố định.
- Dung lượng đĩa còn trống tối thiểu khoảng 8–12 GB cho image, mô hình và cache, phụ thuộc tag thực tế.

## 3. Cài đặt trên PowerShell

Giải nén dự án. Mở PowerShell tại thư mục chứa `docker-compose.yml`:

```powershell
# Tạo tài khoản admin ban đầu bằng mật khẩu ngẫu nhiên, không sửa trong mã nguồn
.\scripts\setup.ps1

# Khởi động cả Streamlit và Ollama (tự kéo 2 model lần đầu)
docker compose up -d --build

# Xem quá trình tải mô hình và lập chỉ mục
docker compose logs -f pull-models app
```

Script `setup.ps1` tạo `.env` một lần, in mật khẩu admin để chủ máy lưu lại. Email mặc định là `admin@anphat.local`. **Không chia sẻ file `.env` hoặc mật khẩu.** Nếu đã có `.env`, script không ghi đè.

Mở **http://localhost:8501**. Đăng nhập admin bằng email và mật khẩu vừa tạo. Vào **Quản trị** để tạo tài khoản nhân viên. Khách hàng tự đăng ký tại màn hình đăng nhập.

Nếu không muốn dùng script trên Windows, hãy copy `.env.example` thành `.env` và **thay** `BOOTSTRAP_ADMIN_PASSWORD` bằng mật khẩu ngẫu nhiên dài ít nhất 12 ký tự, không bắt đầu bằng `REPLACE_`, `CHANGE_`, `YOUR_`.

### GPU NVIDIA (tùy chọn)

Nếu Docker Desktop/WSL2 đã hỗ trợ NVIDIA Container Toolkit:

```powershell
docker compose -f docker-compose.yml -f compose.gpu.yml up -d --build
```

Không dùng file GPU nếu máy không có GPU NVIDIA được hỗ trợ.

## 4. Các vai trò và nghiệp vụ

| Vai trò       | Phạm vi                                                                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Khách hàng    | Tự đăng ký/đăng nhập, chat nguồn được duyệt, xem lịch sử của mình, phản hồi hữu ích, theo dõi và mở lại phiếu đã hoàn thành, đổi mật khẩu                             |
| Nhân viên     | Tất cả nghiệp vụ tra cứu nội bộ, **upload/cập nhật/xóa tài liệu**, phân quyền nguồn customer/staff, tạo embedding, nhận và xử lý phiếu, phản hồi, xem KPI và đánh giá |
| Quản trị viên | Nghiệp vụ nhân viên, tạo tài khoản staff, khóa/mở tài khoản khác                                                                                                      |

Cơ chế bắt buộc trên tầng dữ liệu: khách không thể gọi API Python để tải tài liệu hay đọc chunk `staff`; truy vấn được lọc _trước_ bước truy xuất. Khách không đọc hội thoại hoặc phiếu của người khác.

**Chuyển tiếp người–máy:** khi phát hiện yêu cầu xác minh giao dịch cá nhân, không tìm được nguồn, điểm liên quan thấp, mô hình xác nhận thiếu căn cứ, thiếu trích dẫn hoặc dịch vụ AI lỗi, hệ thống **tự mở phiếu**. Nhân viên nhận phiếu → xác minh → trả lời → đóng. Khách theo dõi và có thể mở lại phiếu.

## 5. Kho tri thức

`knowledge_base/manifest.json` mô tả danh sách tài liệu và phân quyền:

Tài liệu seed chỉ được nạp **một lần** khi cơ sở dữ liệu được tạo. Sau đó nhân viên có thể xóa, cập nhật và thêm tài liệu; việc xóa không bị seed đè trở lại vào mỗi lần reload. Tải lên hỗ trợ PDF/DOCX/TXT/MD tối đa 15 MB. PDF scan phải OCR trước.

Cập nhật embedding thủ công: vào `Kho tài liệu` → `Cập nhật tất cả embedding còn thiếu`, hoặc chạy:

```powershell
docker compose exec app python -m rag_engine.cli reindex
```

Khi Docker chạy lần đầu, `app` tự bootstrap kho và lập chỉ mục BGE-M3 trước khi mở giao diện. Nếu GPU yếu/CPU chậm, startup có thể lâu; xem `docker compose logs -f app`.

## 6. Kiểm thử và đánh giá

`benchmark/questions_50.csv` và `questions_50.jsonl` chứa **50 câu hỏi gắn nhãn**, gồm **40 có căn cứ** và **10 thiếu căn cứ/cần chuyển tiếp**.

### Kiểm tra trực tiếp Ollama/RAG trước benchmark

```powershell
docker compose exec -w /app app python -m scripts.smoke_test
```

### Chạy nhanh: chỉ kiểm tra retrieval

```powershell
docker compose exec app python -m rag_engine.benchmark --retrieval-only
# So sánh baseline chỉ từ khóa, không gọi embedding:
docker compose exec app python -m rag_engine.benchmark --retrieval-only --lexical-only
```

### Chạy end-to-end 50 câu với Ollama thực

```powershell
docker compose exec app python -m rag_engine.benchmark
```

Hoặc đăng nhập nhân viên vào **Kiểm thử RAG** → chọn `End-to-end` → `50 câu` → chạy. Kết quả thực tế được ghi trong volume `app_data`:

```text
/app/data/reports/benchmark_latest.json
/app/data/reports/benchmark_latest.csv
```

Muốn sao chép báo cáo ra máy:

```powershell
docker compose cp app:/app/data/reports/benchmark_latest.json .\benchmark_latest.json
docker compose cp app:/app/data/reports/benchmark_latest.csv .\benchmark_latest.csv
```

**Các thước đo:**

- **Retrieval Recall@5**: tỷ lệ câu có căn cứ mà nguồn chuẩn nằm trong 5 đoạn truy xuất.
- **Handoff rate**: số câu AI đề nghị nhân viên / số câu xử lý thực tế.
- **Handoff precision / recall / F1**: đánh giá quyết định chuyển tiếp với 10 nhãn cần chuyển và 40 nhãn không cần chuyển.
- **Citation rate**: tỷ lệ câu được trả lời có trích dẫn hợp lệ.
- **Keyphrase coverage**: mức có mặt của các thông tin trọng yếu trong câu trả lời; _chỉ là proxy_, không thay thế thẩm định đúng/sai ngữ nghĩa.
- **Loại lỗi**: retrieval miss, false handoff, missed handoff, missing citation, keyphrase mismatch, system error.
- **KPI hoạt động**: tỷ lệ chuyển đề xuất, tỷ lệ phiếu thực tế, thời gian đóng, phản hồi khách hàng.

Đánh giá khoa học nên bổ sung người chấm độc lập trên câu hỏi do người khác biên soạn, dùng một tập validation để chọn ngưỡng rồi kiểm tra trên hold-out. **Không được gọi `evidence_score` là xác suất câu trả lời đúng.**

Chạy unit tests ngoài Docker (nếu có Python 3.11):

```powershell
pip install -r requirements-dev.txt
python -m pytest -q tests
```

Hoặc chạy trong Docker (dùng một container test tạm, không động vào image production):

```powershell
docker compose run --rm --no-deps --user root app sh -c "pip install -r requirements-dev.txt && python -m pytest -q tests"
```

## 7. Vận hành, bảo trì và sao lưu

```powershell
# Trạng thái các dịch vụ
docker compose ps
# Log ứng dụng
docker compose logs --tail 100 app
# Log mô hình
docker compose logs --tail 100 ollama
# Các model trong Ollama
docker compose exec ollama ollama list
# Cập nhật container sau khi chỉnh code
docker compose up -d --build
# Dừng hệ thống nhưng GIỮ dữ liệu
docker compose down
```

Không sử dụng `docker compose down -v` nếu muốn giữ tài khoản, hội thoại, tài liệu và mô hình đã tải; `-v` **xóa các Docker volumes liên quan**. Để sao lưu, dùng script `scripts/backup.ps1` khi ứng dụng đã dừng, hoặc sao chép backup SQLite qua container sau khi bảo đảm không còn tiến trình ghi. Không phát hành dữ liệu thử nghiệm có PII thật.

### Lỗi thường gặp

- **Không thấy ứng dụng**: chạy `docker compose ps` và `docker compose logs app`. Lần đầu phải tải hai mô hình và tạo embedding.
- **Container pull-models failed**: kiểm tra Internet, DNS, dung lượng, chạy lại `docker compose up -d`.
- **AI offline**: kiểm tra `docker compose exec ollama ollama list`, xem có `qwen3:4b` và `bge-m3`.
- **Hết bộ nhớ**: tăng giới hạn RAM Docker/WSL2 hoặc sử dụng máy mạnh hơn; Qwen3 4B có thể chậm trên CPU.
- **Đã sửa file .env nhưng admin cũ không đổi mật khẩu**: bootstrap admin chỉ tạo khi DB chưa có tài khoản. Đổi mật khẩu ở mục Tài khoản khi đã đăng nhập. Không xóa dữ liệu chỉ để đặt lại mật khẩu.
- **Ứng dụng báo chưa đủ bằng chứng**: xem báo cáo `retrieval_hit_at_5`, kiểm tra scope tài liệu, bổ sung/chỉnh tài liệu và chạy `reindex`.

## 8. Giới hạn hiện tại

- Chưa đánh giá end-to-end với Ollama trên hệ thống đích vì môi trường biên soạn không có Docker daemon/Ollama; phải chạy benchmark trên máy người dùng để có số liệu trả lời thực tế.
- Mô hình có thể sinh câu nghe hợp lý nhưng chưa được tài liệu hỗ trợ. Kiểm tra JSON/citation và chuyển tiếp giảm rủi ro **không bảo đảm loại bỏ hoàn toàn hallucination**.
- Bộ 50 câu dựa trên tài liệu demo, dễ hơn tình huống khách hàng thực. Chấm điểm tự động theo từ khóa không phản ánh hoàn toàn ngữ nghĩa, cần thẩm định chuyên gia.
- Ứng dụng là demo local-only, không có xác minh email, MFA, backup tự động hoặc HTTPS đầu cuối; không nên triển khai Internet hay sử dụng dữ liệu thật khi chưa hoàn thiện bảo mật.

## 9. Tệp chính

```text
app.py                     Streamlit UI, role pages
rag_engine/auth.py          Người dùng, xác thực, admin
rag_engine/service.py       Hội thoại, chuyển tiếp, feedback, metrics
rag_engine/documents.py     PDF/DOCX/TXT/MD, scope customer/staff
rag_engine/retrieval.py     BM25 + BGE-M3 + RRF
rag_engine/generation.py    JSON grounding, citation validation
rag_engine/pipeline.py      Chat và evaluation dùng chung
rag_engine/handoff.py       Ngưỡng và quy tắc cần nhân viên
rag_engine/benchmark.py     Benchmark 50 câu, phân tích lỗi, CSV+JSON
knowledge_base/             16 tài liệu giả lập
benchmark/                  50 câu + báo cáo baseline retrieval
Dockerfile, docker-compose.yml, compose.gpu.yml
scripts/setup.ps1           Tạo file cấu hình admin nội bộ
```

**Nguồn kỹ thuật:** Ollama Docker image: https://hub.docker.com/r/ollama/ollama/ ; Ollama embeddings: https://docs.ollama.com/api/embed ; BGE-M3 multilingual: https://ollama.com/library/bge-m3 .
