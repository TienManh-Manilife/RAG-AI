"""An Phat Demo: a local-only RAG customer service desk (Streamlit)."""
import html
import io
import json
import os
from pathlib import Path

import streamlit as st

from rag_engine.auth import (init_db,login,create_account,get_user,change_password,
                             require_role,list_accounts,set_account_active)
from rag_engine.documents import init_docs,seed_documents,ingest,list_docs,remove,refresh_embeddings
from rag_engine.pipeline import answer_question
from rag_engine.followup import contextualize_query
from rag_engine.service import (new_conversation,list_conversations,get_conversation,
    list_messages,add_message,log_event,feedback_for_message,submit_feedback,
    request_handoff,get_handoff_for_message,list_my_handoffs,customer_reopen,
    list_handoffs,update_handoff,metrics)
from rag_engine.ollama import ready
from rag_engine.benchmark import read_cases,run_benchmark,write_report
from rag_engine.config import DATA_DIR,CHAT_MODEL,EMBED_MODEL,RAG_MIN_RELEVANCE

st.set_page_config(page_title='An Phát | Trợ lý chăm sóc khách hàng',page_icon='✦',layout='wide',
                   initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#182642;--edge:#e3eaf3;--slate:#62748c}
html,body,[class*=css],.stApp{font-family:'Manrope',sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(circle at 85% 10%,#e9f3ff 0%,transparent 28%),#f6f8fc}
[data-testid=stSidebar]{background:linear-gradient(180deg,#fff,#f5f9ff);border-right:1px solid var(--edge)}
[data-testid=stHeader]{background:transparent}
.block-container{max-width:1350px;padding-top:2rem}
h1,h2,h3{letter-spacing:-.035em!important}
.banner{background:linear-gradient(118deg,#152d56,#343a83 56%,#087c80);border-radius:23px;
 color:#fff;padding:30px 35px;margin-bottom:20px;box-shadow:0 18px 38px #223b701f}
.banner h1{color:#fff!important;font-size:2rem;margin:4px 0 8px}
.banner p{color:#d7e9f5;margin:0;font-size:.91rem;line-height:1.65}
.eyebrow{font-size:.76rem;letter-spacing:.16em;color:#a8f4e7;font-weight:800}
.panel{background:white;border:1px solid var(--edge);border-radius:16px;padding:18px;
 box-shadow:0 6px 22px #24334b0a;min-height:110px}
.panel h4{margin:0 0 8px;color:#233b69;font-size:.97rem}
.panel p{color:var(--slate);font-size:.84rem;line-height:1.6;margin:0}
[data-testid=stChatMessage],[data-testid=stForm],[data-testid=stMetric]{background:#fff;
 border:1px solid var(--edge);border-radius:16px;padding:13px 18px;box-shadow:0 4px 12px #233c5908}
.stButton>button,.stFormSubmitButton>button{border-radius:11px;min-height:2.5rem;font-weight:700}
.stButton>button[kind=primary],.stFormSubmitButton>button[kind=primary]{background:#344b99;border:0}
[data-testid=stAlert]{border-radius:12px}
</style>''',unsafe_allow_html=True)
init_db();init_docs();seed_documents()

DEFAULT={'actor':None,'page':'Trò chuyện','active_conversation':None,'evaluated':None}
for k,v in DEFAULT.items():st.session_state.setdefault(k,v)

def banner(title,desc,tag='AN PHÁT · CUSTOMER INTELLIGENCE'):
    st.markdown(f'<div class="banner"><div class="eyebrow">{html.escape(tag)}</div>'
                f'<h1>{html.escape(title)}</h1><p>{html.escape(desc)}</p></div>',unsafe_allow_html=True)

def logout():
    st.session_state.clear();st.rerun()

def auth_screen():
    a,b,c=st.columns([1,2.15,1])
    with b:
        banner('Hệ thống hỗ trợ khách hàng','Câu trả lời dựa trên tài liệu, có trích dẫn và nhân viên xử lý khi chưa đủ căn cứ.', '✦ AN PHÁT DEMO · NỘI BỘ')
        st.info('Đồ án minh họa sử dụng chính sách của doanh nghiệp giả lập. Không nhập thông tin cá nhân nhạy cảm.')
        tin,tup=st.tabs(['Đăng nhập','Đăng ký khách hàng'])
        with tin:
            with st.form('login'):
                email=st.text_input('Email đăng nhập')
                password=st.text_input('Mật khẩu',type='password')
                ok=st.form_submit_button('Đăng nhập',type='primary',use_container_width=True)
            if ok:
                actor=login(email,password)
                if actor:
                    st.session_state.actor=actor;st.rerun()
                st.error('Email hoặc mật khẩu không đúng, hoặc tài khoản đang bị khóa.')
        with tup:
            with st.form('signup'):
                name=st.text_input('Họ tên')
                email=st.text_input('Email mới')
                pw=st.text_input('Mật khẩu (tối thiểu 12 ký tự)',type='password')
                pw2=st.text_input('Nhập lại mật khẩu',type='password')
                go=st.form_submit_button('Tạo tài khoản khách hàng',type='primary',use_container_width=True)
            if go:
                if pw!=pw2: st.error('Mật khẩu nhập lại không khớp.')
                else:
                    try:
                        st.session_state.actor=create_account(name,email,pw)
                        st.rerun()
                    except (ValueError,PermissionError) as exc: st.error(str(exc))
            st.caption('Nhân viên chỉ được tạo bởi quản trị viên, không thể tự đăng ký.')
    st.stop()

actor=st.session_state.actor
if actor:
    actor=get_user(actor['id'])
    if not actor:logout()
    st.session_state.actor=actor
else:auth_screen()
staff=actor['role'] in ('staff','admin')

with st.sidebar:
    st.markdown('## ✦ AN PHÁT')
    st.caption('RAG CUSTOMER SERVICE · LOCAL EDITION')
    st.divider()
    st.markdown(f'**{html.escape(actor["name"])}**')
    st.caption({'user':'Khách hàng','staff':'Nhân viên hỗ trợ','admin':'Quản trị viên'}[actor['role']])
    menus=['Trò chuyện','Yêu cầu của tôi','Tài khoản']
    if staff: menus+=['Kho tài liệu','Hàng đợi nhân viên','Thống kê & KPI','Kiểm thử RAG']
    if actor['role']=='admin':menus+=['Quản trị']
    for name in menus:
        if st.button(name, key='nav:'+name,use_container_width=True,
                     type='primary' if st.session_state.page==name else 'secondary'):
            st.session_state.page=name;st.rerun()
    st.divider()
    st.caption('Hội thoại gần đây')
    for con in list_conversations(actor)[:5]:
        if con['mode']!='system':continue
        if st.button('💬 '+con['title'][:30],key='history:'+con['id'],use_container_width=True):
            st.session_state.active_conversation=con['id']
            st.session_state.page='Trò chuyện';st.rerun()
    st.divider()
    status=ready()
    st.caption(('🟢' if status['online'] and status['chat'] else '🔴')+' Ollama: '+CHAT_MODEL)
    st.caption(('🟢' if status['embed'] else '🟠')+' Embedding: '+EMBED_MODEL)
    if st.button('Đăng xuất',use_container_width=True):logout()

def display_sources(hits):
    if not hits:return
    with st.expander('Nguồn được truy xuất ('+str(len(hits))+')'):
        for h in hits:
            st.markdown(f"**{h.get('title',h.get('source',''))}** — {h.get('section','')} · Trang {h.get('page', '?')}")
            st.caption('Chỉ số liên quan bằng chứng: '+f"{100*float(h.get('evidence_score') or 0):.0f}%")

def chat_page():
    banner('Trợ lý tri thức AI','Khách hàng hỏi từ kho tài liệu đã được nhân viên phê duyệt. Hệ thống tự tạo phiếu khi không có đủ căn cứ.')
    left,right=st.columns([4,1])
    left.markdown('#### Hội thoại của bạn')
    if right.button('＋ Tạo hội thoại',use_container_width=True):
        st.session_state.active_conversation=None;st.rerun()
    conv_id=st.session_state.active_conversation
    conversation=get_conversation(actor,conv_id) if conv_id else None
    if conversation and conversation['mode']!='system':
        conversation=None
    if not conversation:conv_id=None
    messages=list_messages(actor,conv_id) if conv_id else []
    if not messages:
        c1,c2,c3=st.columns(3)
        c1.markdown('<div class="panel"><h4>📚 Nguồn có kiểm duyệt</h4><p>Chỉ đọc nội dung khách hàng được phép xem.</p></div>',unsafe_allow_html=True)
        c2.markdown('<div class="panel"><h4>🧾 Trích dẫn minh bạch</h4><p>Mỗi câu trả lời được gắn với mục tài liệu liên quan.</p></div>',unsafe_allow_html=True)
        c3.markdown('<div class="panel"><h4>🎧 Nhân viên hỗ trợ</h4><p>Tự động mở phiếu khi chưa đủ bằng chứng.</p></div>',unsafe_allow_html=True)
        st.info('Ví dụ: “Thiết bị được bảo hành bao lâu?” hoặc “Đổi trả sản phẩm lỗi trong mấy ngày?”')
    for message in messages:
        with st.chat_message('user' if message['role']=='user' else 'assistant'):
            st.markdown(message['content'])
            if message['role']=='assistant':
                display_sources(message['sources'])
                if message['evidence_score'] is not None:
                    st.caption(f"Mức liên quan của nguồn: {message['evidence_score']:.0%}. Đây không phải xác suất câu trả lời đúng.")
                ticket=get_handoff_for_message(actor,message['id'])
                if ticket:
                    st.warning('Phiếu chuyển tiếp: '+ticket['id']+' · '+ticket['status'])
                    if ticket['staff_reply']:st.success('Nhân viên: '+ticket['staff_reply'])
                elif message['needs_handoff']:
                    st.warning(message['handoff_reason'])
                if feedback_for_message(actor,message['id']) is None:
                    a,b=st.columns([1,1])
                    if a.button('👍 Hữu ích',key='up:'+message['id']):
                        submit_feedback(actor,message['id'],True);st.rerun()
                    if b.button('👎 Chưa đúng',key='down:'+message['id']):
                        submit_feedback(actor,message['id'],False);st.rerun()
                else:st.caption('Đã ghi nhận đánh giá của bạn.')
                if not ticket and not message['needs_handoff']:
                    if st.button('🎧 Vẫn muốn nhờ nhân viên kiểm tra',key='askstaff:'+message['id']):
                        request_handoff(actor,conv_id,message['id'],'','','Khách hàng yêu cầu kiểm tra thêm.',[])
                        st.rerun()
    prompt=st.chat_input('Nhập câu hỏi về chính sách, bảo hành, đổi trả, giao hàng…')
    if prompt:
        if len(prompt)>3000:st.error('Tối đa 3.000 ký tự.');return
        if not conv_id:
            conv_id=new_conversation(actor,'system',prompt[:120]);st.session_state.active_conversation=conv_id
        add_message(actor,conv_id,'user',prompt)
        query=contextualize_query(prompt,messages)
        with st.spinner('Đang tìm bằng chứng và tổng hợp phản hồi…'):
            result=answer_question(query,actor=actor,history=messages)
        smalltalk=bool(result['meta'].get('smalltalk'))
        msg_id=add_message(actor,conv_id,'assistant',result['answer'],hits=result['hits'],
            evidence_score=None if smalltalk else result['evidence_score'],
            needs_handoff=result['handoff'],reason=result['reason'])
        if not smalltalk:
            log_event(actor,'system',result['evidence_score'],result['handoff'])
        if result['handoff']:
            request_handoff(actor,conv_id,msg_id,prompt,result['answer'],result['reason'],result['hits'])
        st.rerun()

def customer_tickets():
    banner('Phiếu hỗ trợ của tôi','Theo dõi kết quả chuyển tiếp từ AI sang nhân viên, mở lại nếu cần kiểm tra bổ sung.','SERVICE DESK · CUSTOMER')
    rows=list_my_handoffs(actor)
    if not rows:st.info('Chưa có phiếu. Khi AI thiếu căn cứ, phiếu được tự động tạo.')
    for r in rows:
        with st.expander(f"{'🟠' if r['status']=='pending' else '🔵' if r['status']=='in_progress' else '🟢'} {r['question'][:80]} — {r['status']}"):
            st.caption('Mã phiếu '+r['id']+' · Tạo '+r['created_at'])
            st.write('**Câu hỏi:**',r['question']);st.write('**Lý do:**',r['reason'])
            if r['staff_reply']:
                st.success('Phản hồi nhân viên: '+r['staff_reply'])
                if st.button('Yêu cầu xem xét lại',key='reopen_customer:'+r['id']):
                    customer_reopen(actor,r['id']);st.rerun()
            else:st.info('Đang chờ xử lý.' if r['status']=='pending' else 'Nhân viên đang phụ trách.')

def docs_page():
    require_role(actor)
    banner('Kho tri thức doanh nghiệp','Chỉ nhân viên được thêm, cập nhật hoặc xóa tài liệu. Khách hàng chỉ truy cập nguồn đã phê duyệt.','KNOWLEDGE MANAGEMENT')
    with st.container(border=True):
        st.markdown('#### Nhập tài liệu chính thức')
        files=st.file_uploader('Tải lên PDF / DOCX / TXT / MD (≤15 MB)',type=['pdf','docx','txt','md'],accept_multiple_files=True)
        audience=st.radio('Phạm vi hiển thị', ['customer','staff'],format_func=lambda x:'Khách hàng có thể tra cứu' if x=='customer' else 'Chỉ nội bộ nhân viên',horizontal=True)
        if st.button('Phê duyệt & lập chỉ mục',disabled=not files,type='primary'):
            success=0
            for f in files:
                try:
                    r=ingest(actor,f.name,f.getvalue(),audience)
                    success+=1;st.success(f"{f.name}: {r['chunks']} đoạn")
                except Exception as exc:st.error(f'{f.name}: {exc}')
            if success:st.info('Tài liệu đã có thể tìm bằng từ khóa. Nhấn cập nhật embedding để kích hoạt truy xuất ngữ nghĩa.')
        if st.button('Cập nhật tất cả embedding còn thiếu'):
            with st.spinner('Đang mã hóa tài liệu bằng BGE-M3…'):
                try:st.success(str(refresh_embeddings()))
                except Exception as exc:st.error('Không lập chỉ mục được: '+str(exc))
    st.markdown('#### Danh sách tài liệu')
    docs=list_docs(actor)
    a,b,c=st.columns(3)
    a.metric('Tài liệu',len(docs));b.metric('Đoạn',sum(d['chunks_count'] for d in docs));c.metric('Đã nhúng',sum(d['embedded_count'] or 0 for d in docs))
    for d in docs:
        with st.container(border=True):
            a,b=st.columns([6,1]);a.markdown(f"**{d['title']}**")
            a.caption(f"{d['filename']} · {'Khách hàng' if d['audience']=='customer' else 'Nội bộ'} · {d['chunks_count']} đoạn · {d['embedded_count'] or 0} vector")
            if b.button('Xóa',key='delete_req:'+d['id']):st.session_state['delete_doc']=d['id']
            if st.session_state.get('delete_doc')==d['id']:
                st.warning('Xóa tài liệu sẽ loại bỏ mọi trích dẫn mới liên quan. Xác nhận?')
                l,r=st.columns(2)
                if l.button('Xác nhận xóa',key='yes:'+d['id'],type='primary'):
                    remove(actor,d['id']);st.session_state.delete_doc=None;st.rerun()
                if r.button('Hủy',key='no:'+d['id']):st.session_state.delete_doc=None;st.rerun()

def staff_queue():
    require_role(actor)
    banner('Bàn hỗ trợ nhân viên','Tiếp nhận · Xác minh · Trả lời · Đóng phiếu. Các thao tác có kiểm tra quyền sở hữu.','HUMAN IN THE LOOP')
    filt=st.selectbox('Bộ lọc',['Tất cả','pending','in_progress','resolved'],format_func=lambda s:{'pending':'Chờ xử lý','in_progress':'Đang xử lý','resolved':'Đã giải quyết'}.get(s,s))
    rows=list_handoffs(actor,filt if filt!='Tất cả' else None)
    if not rows:st.info('Không có phiếu phù hợp.')
    for r in rows:
        with st.expander(f"{'🟠' if r['status']=='pending' else '🔵' if r['status']=='in_progress' else '🟢'} {r['customer_name']} · {r['question'][:75]} · {r['status']}"):
            st.caption('Phiếu '+r['id']+' · '+r['created_at']+' · Phụ trách: '+(r['staff_name'] or 'Chưa có'))
            st.write('**Câu hỏi khách hàng:**',r['question'])
            st.write('**Phản hồi AI:**',r['ai_answer'])
            st.write('**Lý do chuyển tiếp:**',r['reason'])
            display_sources(r['sources'])
            if r['status']=='pending':
                if st.button('Nhận xử lý',key='take:'+r['id'],type='primary'):
                    try:update_handoff(actor,r['id'],'take');st.rerun()
                    except (ValueError,PermissionError) as exc:st.error(str(exc))
            elif r['status']=='in_progress' and (r['staff_id']==actor['id'] or actor['role']=='admin'):
                with st.form('staff_reply:'+r['id']):
                    response=st.text_area('Phản hồi chính thức')
                    if st.form_submit_button('Gửi & đóng phiếu',type='primary'):
                        try:update_handoff(actor,r['id'],'resolve',response);st.rerun()
                        except (ValueError,PermissionError) as exc:st.error(str(exc))
            elif r['status']=='resolved':
                st.success(r['staff_reply'])
                if st.button('Mở lại phiếu',key='reopen_staff:'+r['id']):
                    update_handoff(actor,r['id'],'reopen');st.rerun()

def metrics_page():
    require_role(actor)
    banner('Phân tích hiệu quả dịch vụ','Số liệu nghiệp vụ được tính trực tiếp từ hội thoại, phiếu hỗ trợ và đánh giá thực.','OPERATIONS & QUALITY')
    m=metrics(actor)
    a,b,c,d=st.columns(4)
    a.metric('Câu hỏi',m['questions']);b.metric('Đề xuất chuyển',m['suggested']);c.metric('Phiếu tạo',m['actual_handoffs']);d.metric('Đã đóng',m['resolved'])
    a,b,c=st.columns(3)
    a.metric('Tỷ lệ đề xuất',f"{m['suggested_rate']:.1%}")
    b.metric('Tỷ lệ chuyển thực',f"{m['actual_rate']:.1%}")
    c.metric('Phản hồi tích cực',f"{m['feedback_accuracy']:.1%}" if m['feedback_accuracy'] is not None else 'N/A')
    st.caption(f"Thời gian xử lý TB: {m['mean_resolution_hours']:.2f} giờ" if m['mean_resolution_hours'] is not None else 'Chưa có phiếu được xử lý xong.')
    st.warning('Tỷ lệ phản hồi tích cực không đồng nghĩa độ chính xác RAG. Độ chính xác cần đánh giá trên bộ kiểm thử và thẩm định của chuyên gia.')

def benchmark_page():
    require_role(actor)
    banner('Kiểm thử chất lượng RAG','50 câu có nhãn chuẩn: 40 câu trả lời được và 10 câu cần từ chối/chuyển nhân viên.','EVALUATION LAB')
    rows=read_cases()
    a,b,c=st.columns(3)
    a.metric('Câu hỏi chuẩn',len(rows))
    b.metric('Có căn cứ',sum(not x['should_handoff'] for x in rows))
    c.metric('Thiếu căn cứ',sum(x['should_handoff'] for x in rows))
    st.caption('Chấm Retrieval Recall@5, handoff precision/recall/F1, tỷ lệ trích dẫn và độ phủ cụm từ khóa. Độ phủ từ khóa KHÔNG phải accuracy ngữ nghĩa.')
    mode=st.radio('Chế độ chạy',['retrieval','full'],format_func=lambda x:'Chỉ đánh giá truy xuất (nhanh)' if x=='retrieval' else 'End-to-end: chạy LLM cho từng câu (chậm)',horizontal=True)
    count=st.slider('Số câu chạy',1,50,50)
    if st.button('Chạy bộ kiểm thử',type='primary'):
        progress=st.progress(0)
        with st.spinner('Đang thực thi trên dữ liệu và model hiện tại…'):
            report=run_benchmark(rows[:count],retrieval_only=(mode=='retrieval'),progress=lambda i,n:progress.progress(i/n))
        write_report(report)
        st.session_state.evaluated=report
    report=st.session_state.evaluated
    if not report:
        f=DATA_DIR/'reports'/'benchmark_latest.json'
        if f.is_file():
            try:report=json.loads(f.read_text(encoding='utf-8'))
            except (ValueError,OSError):pass
    if report:
        summary=report['summary']
        st.markdown('### Kết quả đã đo · '+report['mode'])
        cols=st.columns(4)
        cols[0].metric('Đã xử lý',summary['cases'])
        cols[1].metric('Retrieval Recall@5',f"{summary['retrieval_recall_at_5']:.1%}" if summary['retrieval_recall_at_5'] is not None else 'N/A')
        cols[2].metric('Tỷ lệ chuyển tiếp',f"{summary['handoff_rate']:.1%}" if summary.get('handoff_rate') is not None else 'N/A')
        cols[3].metric('Recall chuyển tiếp',f"{summary['handoff_recall']:.1%}" if summary.get('handoff_recall') is not None else 'N/A')
        if report['mode']=='end_to_end':
            a,b,c=st.columns(3)
            a.metric('Precision chuyển',f"{summary['handoff_precision']:.1%}" if summary.get('handoff_precision') is not None else 'N/A')
            b.metric('F1 chuyển',f"{summary['handoff_f1']:.1%}" if summary.get('handoff_f1') is not None else 'N/A')
            c.metric('Độ phủ từ khóa (proxy)',f"{summary['keyphrase_coverage_proxy']:.1%}" if summary.get('keyphrase_coverage_proxy') is not None else 'N/A')
        if report['mode']=='end_to_end':
            a,b=st.columns(2)
            a.metric('Tỷ lệ đạt checklist tự động (proxy)',
                f"{summary['auto_test_pass_rate_proxy']:.1%}" if summary.get('auto_test_pass_rate_proxy') is not None else 'N/A')
            b.metric('Đạt checklist trên câu có căn cứ (proxy)',
                f"{summary['answer_accuracy_checklist_proxy']:.1%}" if summary.get('answer_accuracy_checklist_proxy') is not None else 'N/A')
        st.markdown('**Phân tích lỗi:** '+str(summary['error_counts']))
        st.dataframe([{k:(str(r[k]) if isinstance(r[k],list) else r[k]) for k in ('id','question','retrieval_hit_at_5','pred_handoff','term_coverage','errors')} for r in report['cases']],
                     use_container_width=True,hide_index=True)
        if report['mode']=='end_to_end':
            with st.expander('Thẩm định bởi chuyên gia (không tự suy ra từ điểm retrieval)'):
                st.caption('Đánh dấu Đúng/Sai sau khi đối chiếu câu trả lời với nguồn chuẩn. Chỉ các mẫu đã gắn nhãn mới được tính vào expert accuracy.')
                for record in report['cases']:
                    st.selectbox(record['id']+' · '+record['question'][:85],
                                 ['Chưa đánh giá','Đúng','Sai','Không áp dụng'],
                                 index=['Chưa đánh giá','Đúng','Sai','Không áp dụng'].index(
                                     record.get('expert_label','Chưa đánh giá')) if record.get('expert_label','Chưa đánh giá') in
                                         ['Chưa đánh giá','Đúng','Sai','Không áp dụng'] else 0,
                                 key='expert:'+record['id'])
                if st.button('Lưu kết quả chuyên gia'):
                    for record in report['cases']:
                        record['expert_label']=st.session_state.get('expert:'+record['id'],'Chưa đánh giá')
                    labels=[r['expert_label'] for r in report['cases'] if r.get('expert_label') in ('Đúng','Sai')]
                    report['summary']['expert_reviewed']=len(labels)
                    report['summary']['expert_accuracy']=sum(x=='Đúng' for x in labels)/len(labels) if labels else None
                    st.session_state.evaluated=report
                    write_report(report)
                    st.success(f'Đã lưu {len(labels)} đánh giá hợp lệ.')
            if report['summary'].get('expert_accuracy') is not None:
                st.metric('Accuracy do chuyên gia xác nhận',f"{report['summary']['expert_accuracy']:.1%}")
                st.caption(f"Số ca được chuyên gia kiểm tra: {report['summary'].get('expert_reviewed',0)}")
        raw=json.dumps(report,ensure_ascii=False,indent=2)
        st.download_button('Tải báo cáo JSON',raw,file_name='benchmark_latest.json',mime='application/json')
        st.info('Báo cáo chỉ phản ánh kết quả của lần chạy thực. Các ca sai từ khóa cần được chuyên gia xem xét trước khi kết luận sai ngữ nghĩa.')

def admin_page():
    require_role(actor,('admin',))
    banner('Quản trị hệ thống','Tạo tài khoản nhân viên, kích hoạt hoặc vô hiệu hóa tài khoản, phân định quyền truy cập.','IDENTITY & ACCESS')
    with st.form('staff_create'):
        st.markdown('#### Tạo nhân viên')
        name=st.text_input('Họ tên nhân viên')
        email=st.text_input('Email nhân viên')
        pw=st.text_input('Mật khẩu tạm (ít nhất 12 ký tự)',type='password')
        if st.form_submit_button('Tạo nhân viên',type='primary'):
            try:
                create_account(name,email,pw,role='staff',_privileged=True)
                st.success('Đã tạo tài khoản nhân viên.')
            except (ValueError,PermissionError) as exc:st.error(str(exc))
    st.markdown('#### Danh sách người dùng')
    accounts=list_accounts(actor)
    for user in accounts:
        with st.container(border=True):
            left,right=st.columns([5,1]);left.markdown(f"**{user['name']}** · {user['email']}")
            left.caption(f"Vai trò: {user['role']} · {'Đang hoạt động' if user['active'] else 'Đã khóa'}")
            if user['id']!=actor['id']:
                if right.button('Khóa' if user['active'] else 'Mở khóa',key='active:'+user['id']):
                    set_account_active(actor,user['id'],not bool(user['active']));st.rerun()

def account_page():
    banner('Tài khoản & bảo mật','Quản lý thông tin cá nhân và mật khẩu đăng nhập.','ACCOUNT SECURITY')
    st.write(f"**Họ tên:** {actor['name']}")
    st.write(f"**Email:** {actor['email']}")
    with st.form('pw_change'):
        old=st.text_input('Mật khẩu hiện tại',type='password')
        new=st.text_input('Mật khẩu mới (≥12 ký tự)',type='password')
        repeat=st.text_input('Xác nhận mật khẩu mới',type='password')
        if st.form_submit_button('Đổi mật khẩu'):
            if new!=repeat:st.error('Xác nhận mật khẩu không trùng.')
            else:
                try:change_password(actor,old,new);st.success('Đã đổi mật khẩu, hãy đăng nhập lại.');logout()
                except (ValueError,PermissionError) as exc:st.error(str(exc))

ROUTES={'Trò chuyện':chat_page,'Yêu cầu của tôi':customer_tickets,'Tài khoản':account_page,
        'Kho tài liệu':docs_page,'Hàng đợi nhân viên':staff_queue,'Thống kê & KPI':metrics_page,
        'Kiểm thử RAG':benchmark_page,'Quản trị':admin_page}
if st.session_state.page not in menus:st.session_state.page='Trò chuyện'
ROUTES[st.session_state.page]()
st.divider();st.caption('An Phát Demo · Ứng dụng nội bộ · Dữ liệu doanh nghiệp giả lập · Không sử dụng kết quả kiểm thử như xác suất đúng của từng câu.')
