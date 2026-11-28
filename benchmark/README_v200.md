# RAG-AI – benchmark nghiên cứu v200

## Phạm vi

Bản cập nhật chỉ thay đổi quy trình benchmark và trang Kiểm thử RAG, không thay đổi hành vi chat thực tế, xác thực, quản lý tài liệu, quyền, sinh câu trả lời hay phiếu hỗ trợ. `hybrid` và `lexical` dùng trực tiếp pipeline/retrieval đang triển khai. Các biến thể khác chỉ tồn tại trong benchmark.

### Dữ liệu

- `questions_50.csv` / `.jsonl` cũ được giữ nguyên; `questions_200.csv` / `.jsonl` gồm QA001–QA050 cũ và QA051–QA200 mới.
- 200 câu, 160 có căn cứ, 40 cần chuyển tiếp. Khách hàng: 118 (90 có căn cứ, 28 cần chuyển); nhân viên: 82 (70 có căn cứ, 12 cần chuyển).
- 150 câu mới: 120 có căn cứ và 30 cần chuyển tiếp, được xây dựng từ hai PDF giả lập, bao phủ các nhóm nghiệp vụ, cả câu hỏi tình huống và câu hỏi trực tiếp. Một số câu khai thác khía cạnh khác hoặc diễn đạt lại cùng điều khoản; 200 câu không tương đương 200 điều khoản độc lập.
- Nhãn gồm câu hỏi, vai trò, đáp án chuẩn, file, trang, điều khoản, nhóm cụm từ cần có, nhãn chuyển tiếp, split, loại câu hỏi và (khi nguồn trùng nội dung) trang tham chiếu thay thế. QA001–QA050 giữ nguyên nhãn gốc. Các trang thay thế được xác định trước khi chạy mô hình, áp dụng cho mọi phương pháp.
- Báo cáo cả `all`, `user`, `staff`, `legacy50`, `new150` và `question_type`. `new150` là tập mở rộng chưa dùng để điều chỉnh kiến trúc, nhưng không phải tập độc lập ngoài corpus. Bộ nhãn mới cần thẩm định độc lập trước khi công bố chính thức. Không chọn câu theo kết quả phương pháp nào thắng.

## Ma trận thực nghiệm

| Method | Điểm xếp hạng | Evidence | Gate trước sinh / cuối | Retrieval-only | E2E |
|---|---|---|---|---|---|
| `lexical` | 1/(50+r_B)+0.03C | C | 0.24 / cấu hình (mặc định 0.38) | Có | Có |
| `dense` | cosine BGE-M3 | S | như trên | Có | Có |
| `standard_hybrid` | RRF BM25+dense, không cộng Evidence | 0.55C+0.45S | như trên | Có | Có |
| `hybrid` | RRF+0.03E | 0.55C+0.45S | như trên | Có | Có |
| `no_gate` | RRF+0.03E | 0.55C+0.45S | bỏ hai ngưỡng | trùng hybrid, không chạy lại | Có |

`standard_hybrid` là **đối chứng xếp hạng RRF thuần trong pipeline dùng chung**, đồng thời là ablation w/o reranking. Không gọi đây là hệ thống Standard Hybrid độc lập hoàn toàn: E vẫn được tính cho định tuyến chung nhằm cô lập hiệu ứng reranking. `lexical` đồng thời là ablation w/o dense, giữ nguyên công thức baseline trước đó. `no_gate` vẫn giữ lọc quyền, chặn xác minh cá nhân, no-context và validation. Dense-only giữ ngưỡng cosine ứng viên 0.45 giống nhánh dense hiện tại. Tất cả dùng Top-5, cùng câu hỏi, kho tri thức, LLM và prompt. Không điều chỉnh riêng bộ chấm hoặc ngưỡng cho EH-RAG để nâng số liệu.

## Hai tầng tách biệt

1. `benchmark_inference.py`: chỉ nhận question/role/method/mode, không đưa nhãn vào prompt; ghi đầu ra pipeline, danh sách hit kèm text, nguồn, trang, metadata trích dẫn, evidence, quyết định và thời gian. Không phải JSON nội bộ chưa qua validation của LLM; pipeline ứng dụng giữ nguyên.
2. `benchmark_run.py`: gọi inference, lưu `predictions.json` nguyên tử sau từng ca (canonical checkpoint), giữ cả ca sai chất lượng. Khi lỗi hạ tầng, ghi lỗi và dừng; ca lỗi chưa được coi là hoàn thành. `predictions.csv` là bản xuất cuối mỗi lượt bình thường. Khi dừng đột ngột, JSON là nguồn chính và có thể chấm ngay. Fingerprint kiểm tra cùng tập nhãn, corpus văn bản, embedding, mã suy luận, model digest, ngưỡng. Scorer không nằm trong fingerprint suy luận.
3. `benchmark_score.py`: chỉ dùng thư viện chuẩn Python; đọc JSON dự đoán+nhãn, tạo `scores.json` và `scores.csv`, không sửa JSON/CSV dự đoán, không gọi Ollama hay truy cập SQLite. Có thể sửa phương pháp chấm rồi chấm lại tất cả cấu hình. Hash scorer và hash raw được ghi vào kết quả. Nhãn đã lưu là snapshot cố định; nếu cần thay nhãn, dùng quy trình kiểm duyệt có version riêng, không chỉnh raw checkpoint bằng tay.
4. `scripts/parse_metrics.py`: tổng hợp điểm (chỉ run hoàn chỉnh theo mặc định) và kiểm tra hash raw. `benchmark_compare.py`: so sánh theo cặp các run tương thích, McNemar chính xác, và sweep threshold evidence-only trên kết quả retrieval đã lưu. Sweep chỉ là diagnostic, không mô phỏng generation/personal guards và không được gọi là hiệu quả handoff thực tế. Các p-value mang tính thăm dò; câu hỏi chung trang có tương quan, chưa hiệu chỉnh nhiều kiểm định.

## Chạy và xem kết quả

Xem `all.txt` ở thư mục gốc. Sau khi build lại image, chạy các lệnh retrieval rồi E2E tuần tự; có thể dùng `--max-new 10` để chia lượt. Cả 9 cấu hình có 1.800 ca (800 retrieval-only và tối đa 1.000 ca E2E, không phải tất cả đều gọi Qwen3). Chưa có phép đo thời gian chạy v200 trên máy người dùng; không chạy lặp nếu không cần. Các kết quả 50 câu trước đây ở `results/` được giữ nguyên, v200 nằm ở `results/v200/`.

Chấm lại trên máy host: `python -m rag_engine.benchmark_score results/v200`. Tổng hợp: `python scripts/parse_metrics.py results/v200`. So sánh: `python -m rag_engine.benchmark_compare results/v200`. Cả ba không cần GPU hay Docker. Nếu chương trình bị đóng đột ngột, có thể chấm trực tiếp `predictions.json` của run một phần; không so sánh như run đủ 200. UI cho phép chấm tay và lưu `expert_reviews.json` gắn với hash từng dự đoán; đây là đánh giá riêng, không tự thay đổi checklist. Chỉ ca có nhãn do người chấm nhập được tính vào expert rate.

## Giới hạn và diễn giải

Recall@K ở đây là tỷ lệ truy vấn có ít nhất một trang chuẩn (hit rate), không phải mức bao phủ mọi đoạn đúng. Citation Rate kiểm tra file/trang chuẩn, không phải entailment. Checklist là proxy dựa trên cụm từ; các câu yes/no có thể khớp cụm từ nhưng vẫn sai nghĩa. Không dùng 0 lỗi trong checkpoint để khẳng định không có lỗi hạ tầng; xem failed_attempts. Kết quả timing v200 đo thời gian suy luận và truy xuất từng ca, không tính chấm/ghi đĩa; không ghép trực tiếp với kết quả thời gian của benchmark v50 cũ. Không thay nhãn sau khi xem kết quả để thiên vị. Không suy ra EH-RAG vượt trội trước khi chạy đủ đối chứng; nếu không thắng, báo cáo đánh đổi thực tế.
