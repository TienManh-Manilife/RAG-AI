# Nghiệm thu chức năng và kịch bản thủ công

Đây là **kế hoạch nghiệm thu**; không được đánh dấu PASS nếu chưa chạy UI Docker trên máy đích.

| Mã | Kịch bản | Kết quả mong đợi |
|---|---|---|
| AUTH-01 | Khách tạo tài khoản, đăng nhập, đổi mật khẩu | Chỉ dùng được tài khoản của chính mình |
| AUTH-02 | Khách thử truy cập trang/quyền upload | Không thể upload ở UI và service backend từ chối |
| AUTH-03 | Quản trị tạo nhân viên và vô hiệu hóa nhân viên | Nhân viên đăng nhập được trước, không đăng nhập được sau khi khóa |
| KB-01 | Nhân viên thêm PDF/TXT/MD/DOCX | Tài liệu xuất hiện trong kho và tìm bằng từ khóa |
| KB-02 | Nhân viên thêm tài liệu staff | Khách không thấy nguồn trong truy xuất/chứng cứ |
| KB-03 | Nhân viên xóa tài liệu seed | Tài liệu biến mất, không tự được seed trở lại trên rerun |
| CHAT-01 | Khách hỏi thời hạn bảo hành | AI trả lời theo `04_bao_hanh.md` có trích dẫn, nếu mô hình đủ căn cứ |
| CHAT-02 | Khách hỏi câu ngoài phạm vi | AI không bịa và mở phiếu chuyển tiếp |
| CHAT-03 | Khách hỏi trạng thái đơn cá nhân | Không suy đoán, chuyển nhân viên |
| CHAT-04 | Khách gửi lời chào | Phản hồi xã giao, không tạo phiếu sai |
| TICKET-01 | Nhân viên tiếp nhận phiếu | Phiếu chuyển sang `in_progress`, gắn staff_id |
| TICKET-02 | Nhân viên khác cố đóng phiếu | Bị từ chối trừ quản trị viên |
| TICKET-03 | Nhân viên gửi phản hồi và đóng | Khách thấy phản hồi chính thức |
| TICKET-04 | Khách mở lại phiếu đã giải quyết | Phiếu trở về pending; khách khác không thể mở lại |
| MET-01 | Nhân viên xem báo cáo | KPI từ dữ liệu thực, không nhầm với accuracy khách quan |
| EVAL-01 | Chạy 50 câu retrieval-only | Xuất `benchmark_latest.json`/`.csv`, ghi lỗi truy xuất |
| EVAL-02 | Chạy 50 câu end-to-end | Có confusion matrix chuyển tiếp, citation/keyphrase proxy và phân tích lỗi |
| EVAL-03 | Chuyên gia đánh giá đúng/sai | Tính accuracy chỉ trên mẫu đã thẩm định |
| NET-01 | Kiểm tra `docker compose ps` | Không public Ollama; Streamlit bind 127.0.0.1 |
| NET-02 | Dừng và chạy lại Compose | Dữ liệu bảo toàn trong volumes |

Chạy `python -m pytest -q tests` trước UI manual acceptance; unit test chỉ chứng minh một phần logic, **không thay thế** xác minh giao diện bằng trình duyệt.
