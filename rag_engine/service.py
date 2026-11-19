"""Multi-user conversations, auditable events, handoff lifecycle, and feedback."""
import json
import secrets
from datetime import datetime
from .auth import db, require_role, utcnow


def _id():
    return secrets.token_hex(12)


def _source_summary(hits):
    # Persist only source references, not uploaded full-text document contents.
    return [dict(source=str(h.get('source', ''))[:200], page=h.get('page'),
                 similarity=round(float(h.get('similarity', 0)), 4),
                 evidence_score=round(float(h.get('evidence_score', 0)), 4)) for h in hits[:10]]


def new_conversation(actor, mode: str, title: str) -> str:
    if mode not in {'system', 'upload'}:
        raise ValueError('Chế độ chat không hợp lệ.')
    conversation_id, now = _id(), utcnow()
    with db() as c:
        c.execute('INSERT INTO conversations VALUES (?,?,?,?,?,?)',
                  (conversation_id, actor['id'], title.strip()[:120] or 'Cuộc trò chuyện mới', mode, now, now))
    return conversation_id


def list_conversations(actor):
    with db() as c:
        return [dict(r) for r in c.execute(
            'SELECT id,title,mode,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 100',
            (actor['id'],)).fetchall()]


def get_conversation(actor, conversation_id: str):
    with db() as c:
        row = c.execute('SELECT * FROM conversations WHERE id=? AND user_id=?',
                        (conversation_id, actor['id'])).fetchone()
    return dict(row) if row else None


def list_messages(actor, conversation_id: str):
    if not get_conversation(actor, conversation_id):
        raise PermissionError('Không có quyền đọc cuộc trò chuyện này.')
    with db() as c:
        rows = c.execute('SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at, rowid',
                         (conversation_id,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item['sources'] = json.loads(item.pop('sources_json'))
        result.append(item)
    return result


def add_message(actor, conversation_id: str, role: str, content: str,
                hits=None, evidence_score=None, needs_handoff=False, reason='') -> str:
    if not get_conversation(actor, conversation_id):
        raise PermissionError('Không có quyền gửi vào cuộc trò chuyện này.')
    if role not in {'user', 'assistant'}:
        raise ValueError('Vai trò tin nhắn không hợp lệ.')
    message_id, now = _id(), utcnow()
    with db() as c:
        c.execute('''INSERT INTO messages(id, conversation_id, role, content, sources_json,
                   evidence_score, needs_handoff, handoff_reason, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)''',
                  (message_id, conversation_id, role, content[:15000],
                   json.dumps(_source_summary(hits or []), ensure_ascii=False), evidence_score,
                   int(needs_handoff), reason[:1000], now))
        c.execute('UPDATE conversations SET updated_at=? WHERE id=?', (now, conversation_id))
    return message_id


def log_event(actor, mode, evidence_score, handoff_suggested):
    with db() as c:
        c.execute('INSERT INTO events VALUES (?,?,?,?,?,?)',
                  (_id(), actor['id'], mode, float(evidence_score), int(handoff_suggested), utcnow()))


def feedback_for_message(actor, message_id):
    with db() as c:
        row = c.execute('''SELECT f.correct FROM feedback f JOIN messages m ON m.id=f.message_id
                           JOIN conversations c ON c.id=m.conversation_id
                           WHERE f.message_id=? AND c.user_id=?''', (message_id, actor['id'])).fetchone()
        return row['correct'] if row else None


def submit_feedback(actor, message_id, correct: bool):
    with db() as c:
        row = c.execute('''SELECT m.role FROM messages m JOIN conversations v
                           ON v.id=m.conversation_id WHERE m.id=? AND v.user_id=?''',
                        (message_id, actor['id'])).fetchone()
        if not row or row['role'] != 'assistant':
            raise PermissionError('Không có quyền đánh giá câu trả lời này.')
        c.execute('''INSERT INTO feedback VALUES (?,?,?,?) ON CONFLICT(message_id)
                     DO UPDATE SET correct=excluded.correct, created_at=excluded.created_at''',
                  (message_id, actor['id'], int(correct), utcnow()))


def request_handoff(actor, conversation_id, message_id, question, ai_answer, reason, hits):
    if not get_conversation(actor, conversation_id):
        raise PermissionError('Không có quyền chuyển tiếp cuộc trò chuyện này.')
    handoff_id, now = _id(), utcnow()
    with db() as c:
        row = c.execute('SELECT role FROM messages WHERE id=? AND conversation_id=?',
                        (message_id, conversation_id)).fetchone()
        if not row or row['role'] != 'assistant':
            raise ValueError('Tin nhắn AI không hợp lệ.')
        # Ticket contents are reconstructed from persisted, owned chat records.
        stored_ai = c.execute('SELECT content,sources_json,handoff_reason FROM messages WHERE id=?',
                              (message_id,)).fetchone()
        stored_question = c.execute("""SELECT content FROM messages
                               WHERE conversation_id=? AND role='user' AND rowid <
                                   (SELECT rowid FROM messages WHERE id=?)
                               ORDER BY rowid DESC LIMIT 1""", (conversation_id, message_id)).fetchone()
        question = stored_question['content'] if stored_question else ''
        ai_answer = stored_ai['content']
        hits = json.loads(stored_ai['sources_json'])
        if stored_ai['handoff_reason']:
            reason = stored_ai['handoff_reason']
        old = c.execute('SELECT id FROM handoffs WHERE message_id=?', (message_id,)).fetchone()
        if old:
            return old['id']
        c.execute('''INSERT INTO handoffs(id,user_id,conversation_id,message_id,question,ai_answer,
                    reason,sources_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,'pending',?,?)''',
                  (handoff_id, actor['id'], conversation_id, message_id,
                   question[:8000], ai_answer[:10000], reason[:1000],
                   json.dumps(_source_summary(hits or []), ensure_ascii=False), now, now))
    return handoff_id


def get_handoff_for_message(actor, message_id):
    with db() as c:
        r = c.execute('SELECT id,status,staff_reply FROM handoffs WHERE message_id=? AND user_id=?',
                      (message_id, actor['id'])).fetchone()
    return dict(r) if r else None


def list_handoffs(actor, status=None):
    require_role(actor)
    params = []
    sql = '''SELECT h.*,u.name AS customer_name,u.email AS customer_email,
              s.name AS staff_name FROM handoffs h JOIN users u ON h.user_id=u.id
              LEFT JOIN users s ON h.staff_id=s.id'''
    if status and status in {'pending', 'in_progress', 'resolved'}:
        sql += ' WHERE h.status=?'
        params.append(status)
    sql += ' ORDER BY CASE h.status WHEN \'pending\' THEN 0 WHEN \'in_progress\' THEN 1 ELSE 2 END, h.created_at DESC LIMIT 300'
    with db() as c:
        rows = [dict(r) for r in c.execute(sql, params).fetchall()]
    for r in rows:
        r['sources'] = json.loads(r.pop('sources_json'))
    return rows


def update_handoff(actor, handoff_id, action, reply=''):
    require_role(actor)
    if action not in {'take', 'resolve', 'reopen'}:
        raise ValueError('Thao tác không hợp lệ.')
    with db() as c:
        row = c.execute('SELECT * FROM handoffs WHERE id=?', (handoff_id,)).fetchone()
        if not row:
            raise ValueError('Không tìm thấy yêu cầu.')
        if action == 'take':
            if row['status'] != 'pending':
                raise ValueError('Yêu cầu đã được tiếp nhận.')
            c.execute('''UPDATE handoffs SET status='in_progress',staff_id=?, updated_at=?
                         WHERE id=? AND status='pending' ''', (actor['id'], utcnow(), handoff_id))
        elif action == 'resolve':
            if row['status'] != 'in_progress' or (row['staff_id'] != actor['id'] and actor['role'] != 'admin'):
                raise PermissionError('Chỉ nhân viên tiếp nhận mới được trả lời.')
            if not reply.strip() or len(reply) > 10000:
                raise ValueError('Phản hồi phải có nội dung hợp lệ.')
            c.execute('''UPDATE handoffs SET status='resolved',staff_reply=?,updated_at=? WHERE id=?''',
                      (reply.strip(), utcnow(), handoff_id))
        elif action == 'reopen':
            if row['status'] != 'resolved':
                raise ValueError('Chỉ mở lại yêu cầu đã hoàn tất.')
            c.execute('''UPDATE handoffs SET status='pending',staff_id=NULL,staff_reply='',updated_at=? WHERE id=?''',
                      (utcnow(), handoff_id))


def metrics(actor):
    require_role(actor)
    with db() as c:
        events = c.execute('SELECT count(*) AS total,coalesce(sum(handoff_suggested),0) AS suggested,avg(evidence_score) AS mean_score FROM events').fetchone()
        actual = c.execute('SELECT count(*) FROM handoffs').fetchone()[0]
        resolved = c.execute("SELECT count(*) FROM handoffs WHERE status='resolved'").fetchone()[0]
        feedback = c.execute('SELECT count(*) AS n,coalesce(sum(correct),0) AS correct FROM feedback').fetchone()
        system_docs = c.execute("SELECT count(*) FROM events WHERE mode='system'").fetchone()[0]
        closed_tickets = c.execute("SELECT created_at,updated_at FROM handoffs WHERE status='resolved'").fetchall()
    resolution_hours = [(datetime.fromisoformat(x['updated_at']) - datetime.fromisoformat(x['created_at'])).total_seconds()/3600
                        for x in closed_tickets]
    n = events['total']
    return {
        'questions': n, 'suggested': events['suggested'],
        'suggested_rate': events['suggested']/n if n else 0,
        'actual_handoffs': actual, 'actual_rate': actual/n if n else 0,
        'resolved': resolved,
        'mean_resolution_hours': sum(resolution_hours)/len(resolution_hours) if resolution_hours else None,
        'feedback_n': feedback['n'],
        'feedback_accuracy': feedback['correct']/feedback['n'] if feedback['n'] else None,
        'mean_evidence': events['mean_score'], 'system_questions': system_docs,
    }
