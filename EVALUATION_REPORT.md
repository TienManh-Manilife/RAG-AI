# Báo cáo đánh giá bước đầu – RAG An Phát Demo

## Phạm vi

- Bộ dữ liệu **50 câu**, trong đó **40 câu có căn cứ** và **10 câu thiếu căn cứ/cần hỗ trợ**.
- 16 tài liệu giả lập; chỉ 14 tài liệu customer được truy xuất trong benchmark khách hàng.
- Kết quả thực thi ghi nhận ở đây là **retrieval-only / lexical-only**, chưa gọi Qwen3 hoặc BGE-M3. Thời gian mỗi câu tính trên môi trường kiểm thử hiện tại, không suy rộng cho máy Docker của người dùng.

## Kết quả chạy thực trên tập thử

| Thước đo | Giá trị |
|---|---:|
| Tổng ca chạy | 50 |
| Câu có căn cứ | 40 |
| Câu cần chuyển tiếp theo nhãn | 10 |
| Retrieval Recall@5 (BM25 lexical) | **39/40 = 97,5%** |
| Số lỗi truy xuất | **1** |
| Handoff accuracy/precision/recall | **Chưa đo** |
| Accuracy sinh câu trả lời Qwen3 | **Chưa đo** |

Dữ liệu chi tiết lưu trong `benchmark/offline_retrieval_baseline.json`.

## Phân tích lỗi hiện quan sát được

**QA009 – “Hàng bị móp hoặc thiếu phụ kiện cần báo trong bao lâu?”**

Nguồn chuẩn: `03_giao_hang.md` (48 giờ). Phương pháp từ khóa xếp một số đoạn về bảo hành/đổi trả/bảo trì cao hơn đoạn giao nhận, vì câu hỏi dùng từ *móp* khác cách diễn đạt *hư hại* của tài liệu. Đây là trường hợp cần truy xuất ngữ nghĩa bằng BGE-M3 và kiểm tra RRF; không được giả định embedding chắc chắn giải quyết được cho tới khi đo lại.

## Kiểm định cần thực hiện trên máy có Docker

1. `docker compose up -d --build` rồi xác nhận mô hình `qwen3:4b` và `bge-m3` đã tải.
2. `docker compose exec app python -m rag_engine.benchmark` để chạy **50 câu hỏi end-to-end**.
3. Đọc `app_data/reports/benchmark_latest.json` và phân tích confusion matrix TP/FP/FN/TN cho quyết định chuyển tiếp.
4. Phân tích các trường hợp `retrieval_miss`, `false_handoff`, `missed_handoff`, `missing_citation`, `keyphrase_mismatch`, `system_error`.
5. Tổ chức một lượt chấm mù bởi nhân viên/người đánh giá độc lập. Sửa ngưỡng trên validation riêng, sau đó đánh giá tập hold-out mới.

## Nhận xét về phối hợp người–máy

AI phù hợp để tìm nguồn và trả lời câu hỏi có quy tắc rõ ràng, có thể trích dẫn. Nhân viên cần tiếp quản khi tài liệu thiếu điều kiện, thông tin cần kiểm tra trạng thái giao dịch cá nhân, nguồn mâu thuẫn hoặc mô hình không xác minh được. Tự động tạo phiếu giúp tránh bỏ sót khi khách không tự bấm chuyển tiếp, nhưng phải theo dõi **false handoff** vì chuyển quá nhiều sẽ làm tăng tải nhân viên. Cùng lúc phải theo dõi **missed handoff** để tránh AI khẳng định sai trong tình huống thiếu bằng chứng.

**Lưu ý phương pháp:** 50 câu do cùng một người tạo từ chính tài liệu mẫu có thể làm tăng Retrieval Recall. Bộ kiểm thử này chỉ là tập thử nội bộ, không đại diện hiệu năng đối với câu hỏi khách hàng ngoài thực tế.
