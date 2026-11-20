"""RAG-AI Enterprise service portal — Streamlit UI."""
import csv
import html
import io
from pathlib import Path

import requests
import streamlit as st

from rag_engine import (
    init_db, login, create_account, get_user, change_password, get_auth_config, get_ai_server_config,
    require_role, new_conversation, get_conversation, list_conversations,
    list_messages, add_message, log_event, submit_feedback, feedback_for_message,
    request_handoff, get_handoff_for_message, list_handoffs, update_handoff, metrics,
    should_handoff, generate_answer, retrieve_context, index_upload, retrieve_uploaded,
    ingest_pdf_bytes, get_collection_stats, list_indexed_sources, delete_source,
    evaluate_rows, bool_value, suggest_evidence_threshold,
)
from rag_engine.followup import contextualize_query

st.set_page_config(page_title='RAG-AI | Cổng tri thức & dịch vụ', page_icon='✦',
                   layout='wide', initial_sidebar_state='expanded')

# CSS intentionally contains no interpolation of user-controlled HTML.
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root {--ink:#18233d;--muted:#67768d;--indigo:#5149bc;--mint:#0f807d;--line:#e2e9f2}
html,body,[class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background:radial-gradient(ellipse 50% 35% at 90% 0%,#ede8ff 0%,transparent 90%),
  radial-gradient(ellipse 55% 22% at 0% 15%,#def8f6 0%,transparent 90%),#f7f9fd;color:var(--ink);}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#ffffff 0%,#f4f7ff 100%);border-right:1px solid var(--line);}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:2.4rem;padding-bottom:2.4rem;max-width:1370px}
h1,h2,h3 {font-family:Manrope,sans-serif!important;letter-spacing:-.035em;color:#18233d}
.hero {background:linear-gradient(116deg,#1a2557 0%,#343b8c 55%,#146c78 100%);
  border-radius:22px;padding:1.7rem 2rem 1.6rem;color:#fff;box-shadow:0 14px 35px #293c7830;
  position:relative;overflow:hidden;margin-bottom:1.3rem;}
.hero:before {content:'';position:absolute;width:360px;height:360px;right:-110px;top:-235px;
  border-radius:50%;border:65px solid #ffffff12;pointer-events:none;}
.hero .eyebrow {font-size:.73rem;letter-spacing:.18em;text-transform:uppercase;color:#b3eee7;font-weight:700;margin-bottom:.35rem;}
.hero h1 {color:white!important;font-size:clamp(1.5rem,2.7vw,2.25rem);line-height:1.25;margin:.2rem 0 .55rem;}
.hero p {color:#dbe4fa;line-height:1.6;max-width:780px;margin:0;font-size:.92rem}
.brand {padding:.5rem 0 1.15rem;border-bottom:1px solid #e4e9f3;margin-bottom:1.4rem;}
.brand-logo {font-size:1.47rem;font-weight:800;letter-spacing:-.055em;color:#2a3a78;font-family:Manrope,sans-serif}
.brand-logo b {color:#008b83}
.brand-small {font-size:.73rem;color:#71819a;letter-spacing:.06em;text-transform:uppercase;margin-top:.27rem}
.helper {font-size:.84rem;color:#69788c;line-height:1.65}
.kicker {font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;color:#667595;font-weight:800;}
.feature {background:#fff;border:1px solid #e5ebf4;border-radius:17px;padding:1.1rem 1.2rem;
  box-shadow:0 6px 20px #17316209;min-height:120px}
.feature strong {color:#283472;font-size:.99rem}
.feature p {color:#64748b;font-size:.83rem;margin:.35rem 0 0;line-height:1.55}
[data-testid="stChatMessage"] {background:#ffffffec;border:1px solid #e4eaf4;border-radius:17px;
  box-shadow:0 4px 13px #13264b09;margin-bottom:.7rem;padding:.85rem 1.1rem}
[data-testid="stMetric"] {background:#fff;border:1px solid #e6eaf3;border-radius:15px;padding:1rem 1.2rem;
  box-shadow:0 4px 15px #101e3908}
[data-testid="stFileUploader"] {background:#ffffffa8;border:1px dashed #9aabcf;border-radius:15px;padding:.5rem}
[data-testid="stForm"] {background:#ffffffd9;border:1px solid #e6e9f5;border-radius:19px;padding:1.25rem 1.45rem;box-shadow:0 10px 30px #1525440c}
.stButton>button,.stFormSubmitButton>button {border-radius:11px;min-height:2.55rem;font-weight:600;transition:all .15s}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:linear-gradient(105deg,#4844b2,#147c85);color:white;border:0;}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover {filter:brightness(1.1);color:white}
.stTextInput input,.stTextArea textarea,.stSelectbox>div {border-radius:11px}
button[data-baseweb="tab"] {font-weight:700}
[data-testid="stAlert"] {border-radius:12px}
hr {border-color:#e2e7f0}
</style>
''', unsafe_allow_html=True)

init_db()
CFG = get_ai_server_config()


def setup_state():
    defaults = {'actor': None, 'page': 'Trò chuyện', 'chat_mode': 'system',
                'active_conversation': None, 'uploaded_indices': {}, 'eval_output': None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def signout():
    for k in ('actor','page','chat_mode','active_conversation','uploaded_indices','eval_output'):
        st.session_state.pop(k, None)
    # Streamlit widget state can hold form values. Clear it upon logout as well.
    for key in list(st.session_state.keys()):
        if key.startswith(('auth_', 'private_', 'chat_', 'eval_', 'admin_')):
            st.session_state.pop(key, None)
    st.rerun()


def hero(title, caption, eyebrow='RAG-AI  ·  KNOWLEDGE SERVICE'):
    st.markdown(f'''<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div>
      <h1>{html.escape(title)}</h1><p>{html.escape(caption)}</p></div>''', unsafe_allow_html=True)


def role_text(role):
    return {'user':'Khách hàng / người học', 'staff':'Nhân viên hỗ trợ', 'admin':'Quản trị viên'}.get(role, role)


def auth_page():
    a,b,c = st.columns([1, 2.35, 1])
    with b:
        hero('Cổng tri thức & hỗ trợ thông minh',
             'Hỏi đáp theo tài liệu chính thức, truy xuất nguồn minh bạch và chuyển tiếp nhân viên khi cần xác minh.',
             '✦  RAG-AI ENTERPRISE')
        tab_in, tab_up = st.tabs(['Đăng nhập', 'Tạo tài khoản'])
        with tab_in:
            with st.form('auth_login_form'):
                st.markdown('#### Chào mừng trở lại')
                profile = st.radio('Cổng đăng nhập', ['Người dùng', 'Nhân viên / quản trị'], horizontal=True)
                email = st.text_input('Email', key='auth_email', placeholder='you@example.com')
                password = st.text_input('Mật khẩu', type='password', key='auth_password')
                submitted = st.form_submit_button('Đăng nhập', type='primary', use_container_width=True)
            if submitted:
                actor = login(email, password)
                allowed = (actor and (profile == 'Người dùng' and actor['role']=='user' or
                                      profile != 'Người dùng' and actor['role'] in {'staff','admin'}))
                if not allowed:
                    st.error('Thông tin đăng nhập hoặc cổng truy cập không hợp lệ. Tài khoản có thể bị khóa tạm thời nếu thử sai nhiều lần.')
                else:
                    st.session_state['actor'] = actor
                    st.session_state['page'] = 'Trò chuyện'
                    st.session_state['active_conversation'] = None
                    st.rerun()
        with tab_up:
            with st.form('auth_signup_form'):
                st.markdown('#### Đăng ký sử dụng dịch vụ')
                account_type = st.radio('Loại tài khoản', ['Người dùng', 'Nhân viên'], horizontal=True)
                name = st.text_input('Họ và tên', key='auth_name')
                email = st.text_input('Email đăng ký', key='auth_new_email')
                pw = st.text_input('Mật khẩu (ít nhất 12 ký tự)', type='password', key='auth_new_pw')
                confirm = st.text_input('Nhập lại mật khẩu', type='password', key='auth_confirm')
                invite = st.text_input('Mã mời nhân viên (bỏ trống nếu là người dùng)', type='password', key='auth_invite')
                submitted = st.form_submit_button('Tạo tài khoản', type='primary', use_container_width=True)
            st.caption('Tài khoản nhân viên cần mã mời do quản trị cấp. Không có đăng ký quản trị tự do.')
            if submitted:
                if pw != confirm:
                    st.error('Hai mật khẩu không trùng khớp.')
                else:
                    try:
                        actor = create_account(name, email, pw,
                                               role='staff' if account_type=='Nhân viên' else 'user',
                                               invite_code=invite)
                        st.session_state['actor'] = actor
                        st.session_state['active_conversation'] = None
                        st.rerun()
                    except (ValueError, PermissionError) as exc:
                        st.error(str(exc))
    st.stop()


setup_state()
actor = st.session_state['actor']
if actor:
    # Revalidate role on every script run to prevent stale or modified session claims.
    actor = get_user(actor['id'])
    if not actor:
        signout()
    st.session_state['actor'] = actor
else:
    auth_page()


@st.cache_data(ttl=15, show_spinner=False)
def gateway_online(url):
    if not url:
        return False
    try:
        r = requests.get(url + '/health', timeout=3)
        return r.ok and r.json().get('status') == 'online'
    except (requests.RequestException, ValueError):
        return False


is_staff = actor['role'] in {'staff', 'admin'}


def nav_button(label, page, icon):
    if st.button(f'{icon}  {label}', key='nav_'+page, use_container_width=True,
                 type='primary' if st.session_state['page'] == page else 'secondary'):
        st.session_state['page'] = page
        st.rerun()


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">✦ RAG<b>–AI</b></div><div class="brand-small">Enterprise knowledge portal</div></div>', unsafe_allow_html=True)
    st.markdown(f'**{actor["name"]}**')
    st.caption(f'{role_text(actor["role"])} · {actor["email"]}')
    st.divider()
    st.markdown('<div class="kicker">Không gian làm việc</div>', unsafe_allow_html=True)
    nav_button('Trò chuyện', 'Trò chuyện', '💬')
    nav_button('Yêu cầu hỗ trợ của tôi', 'Yêu cầu của tôi', '🧾')
    nav_button('Tài khoản & bảo mật', 'Tài khoản', '🛡️')
    if is_staff:
        nav_button('Kho tài liệu hệ thống', 'Kho tài liệu', '📚')
        nav_button('Hàng đợi hỗ trợ', 'Hàng đợi', '🎧')
        nav_button('Báo cáo vận hành', 'Báo cáo', '📊')
        nav_button('Đánh giá RAG', 'Đánh giá', '🧪')
    if actor['role'] == 'admin':
        nav_button('Quản trị tài khoản', 'Quản trị', '🔐')
    st.divider()
    st.markdown('<div class="kicker">Phiên trò chuyện gần đây</div>', unsafe_allow_html=True)
    conversations = list_conversations(actor)
    for c in conversations[:6]:
        tiny = '🏛' if c['mode'] == 'system' else '📎'
        if st.button(tiny + '  ' + c['title'][:36], key='conversation_'+c['id'], use_container_width=True):
            st.session_state['active_conversation'] = c['id']
            st.session_state['chat_mode'] = c['mode']
            st.session_state['page'] = 'Trò chuyện'
            st.rerun()
    st.divider()
    st.caption('● AI Gateway trực tuyến' if gateway_online(CFG['base_url']) else '○ AI Gateway chưa kết nối')
    st.caption('Model: ' + CFG['model'])
    if st.button('Đăng xuất', key='sign_out', use_container_width=True):
        signout()


def source_panel(sources):
    if not sources:
        return
    with st.expander(f'Xem trích dẫn và nguồn tham khảo ({len(sources)})'):
        for src in sources:
            st.markdown(f"**{src.get('source', 'Tài liệu')}** · Trang {src.get('page','?')}")
            st.caption('Cosine: {:.3f} · Chỉ số bằng chứng: {:.0%}'.format(
                float(src.get('similarity',0)), float(src.get('evidence_score',0))))


def render_history(conversation_id):
    messages = list_messages(actor, conversation_id) if conversation_id else []
    last_user = ''
    for message in messages:
        role = message['role']
        with st.chat_message(role):
            st.markdown(message['content'])
            if role == 'user':
                last_user = message['content']
            if role == 'assistant':
                source_panel(message['sources'])
                if message['evidence_score'] is not None:
                    st.caption('Chỉ số phù hợp bằng chứng: **{:.0%}** · Không phải xác suất câu trả lời đúng.'.format(message['evidence_score']))
                if message['needs_handoff']:
                    st.warning('Nên chuyển nhân viên: ' + message['handoff_reason'])
                existing = get_handoff_for_message(actor, message['id'])
                if existing:
                    st.info(f"Yêu cầu {existing['id']} · Trạng thái: {existing['status']}")
                    if existing['staff_reply']:
                        st.success('**Nhân viên phản hồi:** ' + existing['staff_reply'])
                else:
                    vote1, vote2, escalate = st.columns([1,1,3])
                    with vote1:
                        if st.button('👍 Hữu ích', key='good_'+message['id']):
                            submit_feedback(actor, message['id'], True)
                            st.rerun()
                    with vote2:
                        if st.button('👎 Chưa đúng', key='bad_'+message['id']):
                            submit_feedback(actor, message['id'], False)
                            st.rerun()
                    with escalate:
                        if st.button('🎧 Chuyển nhân viên', key='escalate_'+message['id']):
                            request_handoff(actor, conversation_id, message['id'], last_user,
                                            message['content'], message['handoff_reason'] or 'Người dùng đề nghị nhân viên kiểm tra.',
                                            message['sources'])
                            st.rerun()
                if feedback_for_message(actor, message['id']) is not None:
                    st.caption('Đã ghi nhận phản hồi chất lượng của bạn.')
    return messages


def uploaded_document_manager():
    st.markdown('##### 📎 Kho tài liệu riêng của phiên')
    st.caption('Tài liệu tải lên không được thêm vào kho chung. Nội dung được gửi tới AI Gateway khi đặt câu hỏi.')
    files = st.file_uploader('Chọn tệp PDF, DOCX hoặc TXT (tối đa 12 MB/tệp)',
                             type=['pdf','docx','txt'], accept_multiple_files=True,
                             key='private_file_uploader')
    if files and st.button('Lập chỉ mục tài liệu riêng', key='private_index', type='primary'):
        for uploaded in files:
            try:
                content = uploaded.getvalue()
                import hashlib
                signature = hashlib.sha256(content).hexdigest()
                if signature not in st.session_state['uploaded_indices']:
                    with st.spinner('Đang trích xuất và nhúng: ' + uploaded.name):
                        item = index_upload(uploaded.name, content)
                    st.session_state['uploaded_indices'][signature] = item
                    st.success('Đã nạp ' + uploaded.name)
            except Exception as exc:
                st.error(f'{uploaded.name}: {exc}')
        st.rerun()
    for sig, item in list(st.session_state['uploaded_indices'].items()):
        left, right = st.columns([6, 1])
        left.caption(f"📄 {item['filename']} · {item['pages']} trang · {len(item['chunks'])} đoạn")
        if right.button('Xóa', key='private_remove_'+sig):
            del st.session_state['uploaded_indices'][sig]
            st.rerun()
    if st.session_state['uploaded_indices'] and st.button('Xóa toàn bộ tài liệu riêng', key='private_clear'):
        st.session_state['uploaded_indices'] = {}
        st.rerun()


def chat_page():
    hero('Trợ lý tri thức AI', 'Tra cứu thông tin chính sách, quy chế và tài liệu của bạn — với trích dẫn minh bạch và cơ chế hỗ trợ con người.')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature"><strong>🏛 Kho tri thức chính thức</strong><p>Dữ liệu do nhân viên kiểm duyệt và quản lý.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature"><strong>📎 Tài liệu cá nhân</strong><p>Chỉ truy xuất nội dung bạn tải lên trong phiên.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature"><strong>🎧 Human-in-the-loop</strong><p>Nhân viên tiếp nhận khi thiếu bằng chứng.</p></div>', unsafe_allow_html=True)
    st.markdown('#### Chọn phạm vi trả lời')
    mode = st.radio('Nguồn dữ liệu', ['system','upload'],
                    format_func=lambda v: '🏛 Tài liệu có sẵn trên hệ thống' if v == 'system' else '📎 Tài liệu tôi tải lên',
                    horizontal=True, key='chat_mode')
    if mode == 'upload':
        uploaded_document_manager()
        st.divider()
    conv_id = st.session_state['active_conversation']
    conversation = get_conversation(actor, conv_id) if conv_id else None
    if conversation and conversation['mode'] != mode:
        conversation, conv_id = None, None
        st.session_state['active_conversation'] = None
    heading_l, heading_r = st.columns([5,1])
    heading_l.markdown('#### Nội dung trao đổi')
    if heading_r.button('＋ Chat mới', use_container_width=True):
        st.session_state['active_conversation'] = None
        st.rerun()
    messages = render_history(conv_id)
    if not messages:
        st.info('Bạn có thể đặt câu hỏi về học vụ, bảo hành, chính sách hoặc nội dung tài liệu đã chọn.')
    disabled = mode == 'upload' and not st.session_state['uploaded_indices']
    if disabled:
        st.caption('Hãy nạp ít nhất một tài liệu vào kho riêng trước khi bắt đầu hỏi đáp.')
    prompt = st.chat_input('Nhập câu hỏi của bạn...', disabled=disabled,
                           key='chat_prompt_'+mode)
    if prompt:
        if len(prompt) > 3000:
            st.error('Câu hỏi tối đa 3.000 ký tự.')
            return
        if not conv_id:
            conv_id = new_conversation(actor, mode, prompt)
            st.session_state['active_conversation'] = conv_id
        add_message(actor, conv_id, 'user', prompt)
        try:
            with st.spinner('Đang truy xuất nguồn và tổng hợp câu trả lời...'):
                search_query = contextualize_query(prompt, messages)
                if mode == 'system':
                    hits = retrieve_context(search_query, top_k=5)
                else:
                    hits = retrieve_uploaded(search_query, list(st.session_state['uploaded_indices'].values()), top_k=5)
                answer, meta = generate_answer(prompt, hits, model=CFG['model'])
                escalate, reason, score = should_handoff(hits, meta, min_similarity=0.45)
                add_message(actor, conv_id, 'assistant', answer, hits=hits,
                            evidence_score=score, needs_handoff=escalate, reason=reason)
                log_event(actor, mode, score, escalate)
        except Exception as exc:
            import traceback
            traceback.print_exc()

            add_message(
                actor, conv_id, 'assistant',
                'Hệ thống tạm thời chưa xử lý được câu hỏi. '
                'Vui lòng thử lại hoặc liên hệ nhân viên.',
                evidence_score=0,
                needs_handoff=True,
                reason='Lỗi xử lý / kết nối dịch vụ'
            )
            log_event(actor, mode, 0, True)
        st.rerun()


def requests_page():
    hero('Yêu cầu hỗ trợ của tôi', 'Theo dõi các câu hỏi đã chuyển và phản hồi chính thức từ nhân viên.', 'CUSTOMER SUPPORT')
    # Private view: users only see their own handoffs via message ownership.
    related = []
    for c in list_conversations(actor):
        for m in list_messages(actor, c['id']):
            if m['role'] == 'assistant':
                h = get_handoff_for_message(actor, m['id'])
                if h:
                    related.append((c, m, h))
    if not related:
        st.info('Bạn chưa gửi yêu cầu nào đến nhân viên.')
    for conversation, message, ticket in related:
        with st.container(border=True):
            st.markdown(f"**🎧 {ticket['id']}** · {ticket['status']}")
            st.caption('Cuộc trao đổi: ' + conversation['title'])
            if ticket['staff_reply']:
                st.success(ticket['staff_reply'])
            elif ticket['status'] == 'in_progress':
                st.info('Nhân viên đang xử lý yêu cầu.')
            else:
                st.caption('Yêu cầu đang chờ tiếp nhận.')


def documents_page():
    require_role(actor)
    hero('Kho tài liệu chính thức', 'Quản lý nguồn tri thức dùng chung cho tất cả người dùng. Chỉ nhân viên có quyền bổ sung và xóa tài liệu.', 'KNOWLEDGE MANAGEMENT')
    with st.container(border=True):
        st.markdown('#### Nhập tài liệu mới')
        st.caption('Tệp PDF chứa văn bản; bản scan cần OCR trước khi nhập. Tối đa 30 MB/tệp.')
        files = st.file_uploader('Tệp PDF', type=['pdf'], accept_multiple_files=True, key='system_files')
        left, right = st.columns(2)
        chunk = left.slider('Chunk size (ký tự)', 400, 1600, 850, 50)
        overlap = right.slider('Overlap (ký tự)', 50, 300, 140, 10)
        if st.button('Lập chỉ mục vào kho chung', type='primary', disabled=not files):
            for uploaded in files:
                try:
                    with st.spinner('Đang xử lý: '+uploaded.name):
                        result = ingest_pdf_bytes(uploaded.name, uploaded.getvalue(), chunk, overlap)
                    st.success(f"{result['filename']}: {result['pages']} trang, {result['chunks']} đoạn")
                except Exception as exc:
                    st.error(f'{uploaded.name}: {exc}')
    st.markdown('#### Danh mục tài liệu')
    try:
        sources = list_indexed_sources()
    except Exception as exc:
        st.error(f'Không đọc được kho dữ liệu: {exc}')
        return
    if not sources:
        st.info('Kho chưa có tài liệu. Hãy nạp PDF phía trên.')
    for src in sources:
        with st.container(border=True):
            main, action = st.columns([6,1])
            main.markdown(f"**📄 {src['source']}**")
            main.caption(f"{src['pages']} trang · {src['chunks']} đoạn")
            if action.button('Xóa', key='delete_'+src['source_id']):
                st.session_state['delete_pending'] = src['source_id']
                st.rerun()
            if st.session_state.get('delete_pending') == src['source_id']:
                st.warning('Thao tác xóa có hiệu lực với toàn bộ người dùng. Xác nhận xóa tài liệu này?')
                yes, no = st.columns(2)
                if yes.button('Xác nhận', key='yes_'+src['source_id'], type='primary'):
                    delete_source(src['source_id'])
                    st.session_state['delete_pending'] = None
                    st.rerun()
                if no.button('Hủy', key='no_'+src['source_id']):
                    st.session_state['delete_pending'] = None
                    st.rerun()


def staff_queue_page():
    require_role(actor)
    hero('Bàn hỗ trợ nhân viên', 'Tiếp nhận → xử lý → phản hồi người dùng. Mọi yêu cầu đều có trạng thái và nhân viên phụ trách.', 'HUMAN-IN-THE-LOOP')
    filt = st.selectbox('Lọc theo trạng thái', ['Tất cả','pending','in_progress','resolved'])
    rows = list_handoffs(actor, filt if filt != 'Tất cả' else None)
    if not rows:
        st.info('Không có yêu cầu nào phù hợp.')
    for row in rows:
        with st.expander(f"{'🟠' if row['status']=='pending' else '🔵' if row['status']=='in_progress' else '🟢'} {row['id']} · {row['customer_name']} · {row['status']}"):
            st.caption(f"Ngày tạo: {row['created_at']} · Phụ trách: {row['staff_name'] or 'Chưa có'}")
            st.markdown('**Câu hỏi của khách hàng**')
            st.write(row['question'])
            st.markdown('**Câu trả lời AI**')
            st.write(row['ai_answer'])
            st.markdown('**Lý do cần kiểm tra**')
            st.write(row['reason'])
            source_panel(row['sources'])
            if row['status'] == 'pending':
                if st.button('Nhận xử lý', key='take_'+row['id'], type='primary'):
                    try:
                        update_handoff(actor, row['id'], 'take')
                        st.rerun()
                    except (PermissionError, ValueError) as exc:
                        st.error(str(exc))
            elif row['status'] == 'in_progress':
                if row['staff_id'] == actor['id'] or actor['role'] == 'admin':
                    with st.form('resolve_'+row['id']):
                        answer = st.text_area('Phản hồi chính thức', height=110)
                        submit = st.form_submit_button('Gửi phản hồi & hoàn tất', type='primary')
                    if submit:
                        try:
                            update_handoff(actor, row['id'], 'resolve', answer)
                            st.rerun()
                        except (PermissionError, ValueError) as exc:
                            st.error(str(exc))
                else:
                    st.caption('Nhân viên khác đang phụ trách yêu cầu này.')
            else:
                st.success('Phản hồi: ' + row['staff_reply'])
                if actor['role'] == 'admin' and st.button('Mở lại yêu cầu', key='reopen_'+row['id']):
                    update_handoff(actor, row['id'], 'reopen')
                    st.rerun()


def metrics_page():
    require_role(actor)
    hero('Báo cáo hiệu quả dịch vụ', 'Theo dõi tương tác, nhu cầu hỗ trợ và chất lượng phản hồi trên dữ liệu vận hành thực tế.', 'SERVICE ANALYTICS')
    m = metrics(actor)
    a,b,c,d = st.columns(4)
    a.metric('Câu hỏi đã xử lý', m['questions'])
    b.metric('Chuyển tiếp đề xuất', m['suggested'])
    c.metric('Chuyển tiếp thực tế', m['actual_handoffs'])
    d.metric('Đã xử lý xong', m['resolved'])
    a,b,c = st.columns(3)
    a.metric('Tỷ lệ đề xuất chuyển', f"{m['suggested_rate']:.1%}")
    b.metric('Tỷ lệ thực chuyển', f"{m['actual_rate']:.1%}")
    c.metric('Đánh giá hữu ích / đúng từ người dùng',
             f"{m['feedback_accuracy']:.1%}" if m['feedback_accuracy'] is not None else 'Chưa có',
             help='Tính trên những câu trả lời được đánh giá, không phải độ chính xác khách quan.')
    st.caption('Thời gian giải quyết trung bình: ' +
               (f"{m['mean_resolution_hours']:.1f} giờ" if m['mean_resolution_hours'] is not None else 'Chưa có') )
    st.caption(f"Số đánh giá nhận được: {m['feedback_n']}. Chỉ số bằng chứng trung bình: " +
               (f"{m['mean_evidence']:.1%}" if m['mean_evidence'] is not None else 'Chưa có dữ liệu'))
    st.info('Các tỷ lệ này là chỉ số vận hành. Muốn kết luận chất lượng RAG cần bộ kiểm thử có đáp án chuẩn và đánh giá chuyên gia.')
    try:
        stats = get_collection_stats()
        st.markdown('#### Kho tri thức hệ thống')
        a,b = st.columns(2)
        a.metric('Tài liệu', stats['sources'])
        b.metric('Đoạn được lập chỉ mục', stats['chunks'])
    except Exception as exc:
        st.error('Không truy cập được ChromaDB.')


def evaluation_page():
    require_role(actor)
    hero('Phòng kiểm định RAG', 'Đánh giá theo tập câu hỏi tham chiếu. So sánh EM, token F1, Recall@K và khả năng chuyển nhân viên.', 'RAG EVALUATION LAB')
    st.caption('CSV cột bắt buộc: question, reference_answer. Tùy chọn: expected_source, should_handoff (true/false).')
    fixture = st.file_uploader('Bộ dữ liệu đánh giá CSV (UTF-8)', type=['csv'], key='eval_csv')
    max_cases = st.slider('Số trường hợp tối đa trong một lượt', 1, 50, 10)
    if fixture and st.button('Chạy đánh giá', key='eval_run', type='primary'):
        try:
            data = fixture.getvalue().decode('utf-8-sig')
            rows = list(csv.DictReader(io.StringIO(data)))
            if not rows or not {'question','reference_answer'}.issubset(rows[0]):
                raise ValueError('Thiếu cột question hoặc reference_answer.')
            output = []
            bar = st.progress(0)
            for i, row in enumerate(rows[:max_cases]):
                question = str(row.get('question','')).strip()
                if not question:
                    continue
                try:
                    hits = retrieve_context(question, top_k=5)
                    prediction, meta = generate_answer(question, hits, model=CFG['model'])
                    handoff, reason, evidence = should_handoff(hits, meta, min_similarity=.45)
                    output.append({'question': question, 'prediction': prediction,
                                   'reference_answer': str(row.get('reference_answer','')),
                                   'expected_source': str(row.get('expected_source','')).strip(),
                                   'should_handoff': bool_value(row['should_handoff']) if str(row.get('should_handoff','')).strip() else None,
                                   'handoff': handoff, 'evidence_score': evidence,
                                   'hard_handoff': bool(meta.get('no_context') or meta.get('insufficient_evidence') or
                                                        meta.get('missing_citation') or meta.get('answer_filtered') or not hits),
                                   'has_citation': bool(hits) and not meta.get('missing_citation',True),
                                   'hits': hits})
                except Exception:
                    st.warning(f'Không xử lý được ca kiểm thử thứ {i+1}.')
                bar.progress((i+1)/min(len(rows), max_cases))
            for existing_key in list(st.session_state):
                if existing_key.startswith('eval_judge_'):
                    st.session_state.pop(existing_key, None)
            st.session_state['eval_output'] = output
        except (ValueError, UnicodeError) as exc:
            st.error(str(exc))
    output = st.session_state.get('eval_output')
    if output:
        values = evaluate_rows(output)
        cols = st.columns(4)
        cols[0].metric('Số mẫu', values['cases'])
        cols[1].metric('Exact match', f"{values['exact_match']:.1%}")
        cols[2].metric('Token F1', f"{values['token_f1']:.1%}")
        cols[3].metric('Tỷ lệ chuyển tiếp', f"{values['handoff_rate']:.1%}")
        st.write('Tỷ lệ trích dẫn:', f"{values['citation_rate']:.1%}")
        if values['retrieval_recall_at_k'] is not None:
            st.write('Retrieval Recall@5:', f"{values['retrieval_recall_at_k']:.1%}")
        for key in ('handoff_precision','handoff_recall'):
            if values.get(key) is not None:
                st.write(key+':', f"{values[key]:.1%}")
        st.info('EM / token F1 là thước đo trùng khớp chữ, không đủ để kết luận độ đúng ngữ nghĩa. Nên kiểm chứng thêm bằng chuyên gia và bộ câu hỏi có đáp án chuẩn.')
        threshold = suggest_evidence_threshold(output)
        if threshold:
            st.markdown(f"**Ngưỡng chuyển tiếp gợi ý trên tập thử này:** {threshold['threshold']:.0%} "
                        f"(Precision {threshold['precision']:.1%}; Recall {threshold['recall']:.1%}; F1 {threshold['f1']:.1%})")
            st.caption('Đây là mức tối ưu trên dữ liệu đang kiểm thử, có nguy cơ overfit. Cần kiểm tra trên tập holdout độc lập trước khi thay ngưỡng vận hành.')
        st.dataframe([{k:r[k] for k in ('question','reference_answer','prediction','handoff','evidence_score')} for r in output],
                     use_container_width=True, hide_index=True)
        with st.expander('Thẩm định thủ công từng câu trả lời (chuyên gia)'):
            for i, row in enumerate(output):
                st.selectbox(f"Câu {i+1}: {row['question'][:90]}",
                             ['Chưa đánh giá','Đúng','Sai'], key='eval_judge_'+str(i))
        reviewed = [st.session_state.get('eval_judge_'+str(i), 'Chưa đánh giá') for i in range(len(output))]
        checked = [x for x in reviewed if x != 'Chưa đánh giá']
        if checked:
            st.metric('Accuracy do chuyên gia xác nhận',
                      f"{sum(x == 'Đúng' for x in checked)/len(checked):.1%}")
            st.caption(f'Đã kiểm tra {len(checked)}/{len(output)} câu; không tính câu chưa đánh giá.')
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=['question','reference_answer','prediction','handoff','evidence_score','expert_label'])
        writer.writeheader()
        def safe_csv_cell(value):
            text = str(value)
            return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text
        writer.writerows([{**{k:safe_csv_cell(r[k]) for k in writer.fieldnames if k in r},
                           'expert_label': reviewed[i]} for i,r in enumerate(output)])
        st.download_button('Tải kết quả CSV', buf.getvalue().encode('utf-8-sig'),
                           file_name='rag_evaluation.csv', mime='text/csv')


def admin_page():
    require_role(actor, ('admin',))
    hero('Quản trị tài khoản', 'Phân quyền nhân viên và kiểm soát truy cập hệ thống.', 'SECURITY & IDENTITY')
    st.info('Quản trị viên có thể tạo nhân viên trực tiếp; người dùng thông thường tự đăng ký. Không chia sẻ mã mời nhân viên công khai.')
    with st.form('admin_create_staff'):
        name = st.text_input('Tên nhân viên', key='admin_name')
        email = st.text_input('Email nhân viên', key='admin_email')
        pw = st.text_input('Mật khẩu khởi tạo (ít nhất 12 ký tự)', type='password', key='admin_password')
        submit = st.form_submit_button('Tạo nhân viên', type='primary')
    if submit:
        try:
            result = create_account(name, email, pw, role='staff', _privileged=True)
            st.success('Đã tạo tài khoản nhân viên: ' + result['email'])
        except (ValueError, PermissionError) as exc:
            st.error(str(exc))


def account_page():
    hero('Tài khoản & bảo mật', 'Cập nhật mật khẩu truy cập và kiểm tra vai trò tài khoản.', 'ACCOUNT SECURITY')
    st.markdown(f"**Họ tên:** {actor['name']}")
    st.markdown(f"**Email:** {actor['email']}")
    st.markdown(f"**Vai trò:** {role_text(actor['role'])}")
    with st.form('change_own_password'):
        old = st.text_input('Mật khẩu hiện tại', type='password')
        new = st.text_input('Mật khẩu mới (ít nhất 12 ký tự)', type='password')
        new_again = st.text_input('Nhập lại mật khẩu mới', type='password')
        submitted = st.form_submit_button('Đổi mật khẩu', type='primary')
    if submitted:
        if new != new_again:
            st.error('Mật khẩu mới không trùng nhau.')
        else:
            try:
                change_password(actor, old, new)
                st.success('Đã đổi mật khẩu. Vui lòng đăng nhập lại.')
                signout()
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))


page = st.session_state['page']
if page == 'Trò chuyện':
    chat_page()
elif page == 'Yêu cầu của tôi':
    requests_page()
elif page == 'Tài khoản':
    account_page()
elif page == 'Kho tài liệu' and is_staff:
    documents_page()
elif page == 'Hàng đợi' and is_staff:
    staff_queue_page()
elif page == 'Báo cáo' and is_staff:
    metrics_page()
elif page == 'Đánh giá' and is_staff:
    evaluation_page()
elif page == 'Quản trị' and actor['role'] == 'admin':
    admin_page()
else:
    st.session_state['page'] = 'Trò chuyện'
    st.rerun()

st.divider()
st.caption('✦ RAG-AI · Truy xuất có căn cứ · Human-in-the-loop · Quy trình hỗ trợ minh bạch')
