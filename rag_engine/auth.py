"""SQLite-backed users, role-based access and password verification."""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from .config import DATA_DIR, SQLITE_PATH, get_auth_config

PBKDF2_ROUNDS = 600_000
MAX_FAILURES = 5
LOCK_MINUTES = 15
ROLES = {'user', 'staff', 'admin'}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA busy_timeout = 15000')
    conn.execute('PRAGMA journal_mode = WAL')
    return conn


def init_db() -> None:
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS users(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','staff','admin')),
            created_at TEXT NOT NULL, failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations(
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
            title TEXT NOT NULL, mode TEXT NOT NULL CHECK(mode IN ('system','upload')),
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_conversations_user ON conversations(user_id, updated_at);
        CREATE TABLE IF NOT EXISTS messages(
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
            role TEXT NOT NULL CHECK(role IN ('user','assistant','staff')),
            content TEXT NOT NULL, sources_json TEXT NOT NULL DEFAULT '[]',
            evidence_score REAL, needs_handoff INTEGER NOT NULL DEFAULT 0,
            handoff_reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, created_at);
        CREATE TABLE IF NOT EXISTS handoffs(
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            message_id TEXT UNIQUE REFERENCES messages(id),
            question TEXT NOT NULL, ai_answer TEXT NOT NULL,
            reason TEXT NOT NULL, sources_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','in_progress','resolved')),
            staff_id TEXT REFERENCES users(id), staff_reply TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_handoffs_user ON handoffs(user_id, created_at);
        CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
            mode TEXT NOT NULL, evidence_score REAL NOT NULL,
            handoff_suggested INTEGER NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback(
            message_id TEXT PRIMARY KEY REFERENCES messages(id),
            user_id TEXT NOT NULL REFERENCES users(id),
            correct INTEGER NOT NULL CHECK(correct IN (0,1)),
            created_at TEXT NOT NULL
        );
        ''')
    cfg = get_auth_config()
    # Bootstrap admin only when there are no accounts; prevents a future secret
    # misconfiguration from promoting an existing user silently.
    if (cfg['bootstrap_admin_email'] and cfg['bootstrap_admin_password']
            and not cfg['bootstrap_admin_password'].startswith(('REPLACE_', 'CHANGE_', 'YOUR_'))):
        with db() as c:
            total = c.execute('SELECT count(*) FROM users').fetchone()[0]
        if total == 0:
            create_account(cfg['bootstrap_admin_name'], cfg['bootstrap_admin_email'],
                           cfg['bootstrap_admin_password'], role='admin', _privileged=True)


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PBKDF2_ROUNDS)
    return f'pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}'


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, hex_digest = encoded.split('$')
        if algorithm != 'pbkdf2_sha256' or not (100_000 <= int(iterations) <= 2_000_000):
            return False
        computed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(computed, bytes.fromhex(hex_digest))
    except (ValueError, TypeError):
        return False


def create_account(name: str, email: str, password: str, *, role: str = 'user',
                   invite_code: str = '', _privileged: bool = False) -> dict:
    name, email = name.strip(), email.strip().lower()
    if len(name) < 2 or len(name) > 100:
        raise ValueError('Họ tên phải có từ 2 đến 100 ký tự.')
    if len(email) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        raise ValueError('Email không hợp lệ.')
    if len(password) < 12 or len(password) > 128:
        raise ValueError('Mật khẩu cần có từ 12 đến 128 ký tự.')
    if role not in ROLES:
        raise ValueError('Vai trò không hợp lệ.')
    if role == 'admin' and not _privileged:
        raise PermissionError('Không được tự đăng ký tài khoản quản trị.')
    if role == 'staff' and not _privileged:
        allowed = get_auth_config()['staff_invite_code']
        if len(allowed) < 12 or allowed.startswith(('REPLACE_', 'CHANGE_', 'YOUR_')) or not hmac.compare_digest(
            hashlib.sha256(invite_code.encode()).digest(), hashlib.sha256(allowed.encode()).digest()
        ):
            raise PermissionError('Mã mời nhân viên không hợp lệ hoặc chưa cấu hình.')
    user_id = secrets.token_hex(16)
    try:
        with db() as c:
            c.execute('INSERT INTO users(id, name, email, password_hash, role, created_at) VALUES (?,?,?,?,?,?)',
                      (user_id, name, email, _hash_password(password), role, utcnow()))
    except sqlite3.IntegrityError as exc:
        raise ValueError('Email này đã được đăng ký.') from exc
    return {'id': user_id, 'name': name, 'email': email, 'role': role}


def login(email: str, password: str) -> dict | None:
    email = email.strip().lower()
    with db() as c:
        row = c.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if row is None:
            # Keep roughly comparable verification cost for unknown accounts.
            _verify_password(password, _DUMMY_HASH)
            return None
        locked = row['locked_until']
        if locked and datetime.fromisoformat(locked) > datetime.now(timezone.utc):
            return None
        if not _verify_password(password, row['password_hash']):
            failures = row['failed_attempts'] + 1
            lock_until = (datetime.now(timezone.utc) + timedelta(minutes=LOCK_MINUTES)).isoformat() if failures >= MAX_FAILURES else None
            c.execute('UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?',
                      (failures, lock_until, row['id']))
            return None
        c.execute('UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?', (row['id'],))
        return {'id': row['id'], 'name': row['name'], 'email': row['email'], 'role': row['role']}


# Precomputed once per process; never derived from any real user secret.
_DUMMY_HASH = _hash_password('not_a_real_user_password')


def require_role(actor: dict, roles=('staff', 'admin')) -> None:
    if not actor or actor.get('role') not in roles:
        raise PermissionError('Bạn không có quyền thực hiện thao tác này.')


def get_user(user_id: str) -> dict | None:
    with db() as c:
        row = c.execute('SELECT id, name, email, role FROM users WHERE id=?', (user_id,)).fetchone()
        return dict(row) if row else None


def change_password(actor: dict, current_password: str, new_password: str) -> None:
    """Change own credential after verifying the current password."""
    if not actor or not actor.get('id'):
        raise PermissionError('Cần đăng nhập để đổi mật khẩu.')
    if not 12 <= len(new_password) <= 128:
        raise ValueError('Mật khẩu mới cần có từ 12 đến 128 ký tự.')
    with db() as c:
        row = c.execute('SELECT password_hash FROM users WHERE id=?', (actor['id'],)).fetchone()
        if not row or not _verify_password(current_password, row['password_hash']):
            raise PermissionError('Mật khẩu hiện tại không chính xác.')
        if hmac.compare_digest(current_password, new_password):
            raise ValueError('Mật khẩu mới phải khác mật khẩu hiện tại.')
        c.execute('UPDATE users SET password_hash=?, failed_attempts=0, locked_until=NULL WHERE id=?',
                  (_hash_password(new_password), actor['id']))
