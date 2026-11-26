"""Conservative human escalation policy. Index is retrieval relevance, not a probability."""
import re
from .config import RAG_MIN_RELEVANCE

# Avoid false statements about live customer-specific information without integration.
PERSONAL = re.compile(
    r'(trạng thái (?:đơn hàng|bảo hành) của tôi|'
    r'kiểm tra đơn hàng của tôi|đơn hàng của tôi (?:đang|ở|đến|bao giờ)|'
    r'(?:cho tôi|cung cấp|tiết lộ|lấy|xem|biết)\s+(?:mật khẩu|mã otp)|'
    r'lịch sử mua hàng của|thông tin khách hàng khác|'
    r'tài khoản của người khác|hoàn tiền cho đơn của tôi|thanh toán của tôi)', re.I)

def personal_handoff(query):
    return bool(PERSONAL.search(query))

def should_handoff(hits, answer_metadata, min_similarity=None):
    t=RAG_MIN_RELEVANCE if min_similarity is None else min_similarity
    best=max((float(x.get('evidence_score') or 0) for x in hits), default=0.0)
    if answer_metadata.get('personal_verification'):
        return True,'Yêu cầu xác minh thông tin giao dịch/cá nhân; nhân viên cần kiểm tra.',best
    if not hits:
        return True,'Không tìm được đoạn nguồn phù hợp.',best
    if answer_metadata.get('service_error'):
        return True,'Dịch vụ AI chưa sẵn sàng; chuyển nhân viên.',best
    if best < t:
        return True,f'Bằng chứng truy xuất dưới ngưỡng liên quan {t:.0%}.',best
    if answer_metadata.get('no_context') or answer_metadata.get('insufficient_evidence'):
        return True,'Mô hình xác nhận tài liệu chưa đủ căn cứ.',best
    if answer_metadata.get('missing_citation') or answer_metadata.get('answer_filtered'):
        return True,'Không xác minh được trích dẫn/định dạng câu trả lời.',best
    return False,'',best
