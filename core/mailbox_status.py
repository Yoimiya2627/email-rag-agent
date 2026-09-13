"""Read account-scoped metadata only; no credentials, message text or writes."""
from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import time

import config.settings as cfg
from core.mail_accounts import get_data_dir
from core.mail_capabilities import CHAT_SCOPE_EXPLANATION


@contextmanager
def _read_db(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        yield connection
    finally:
        connection.close()


def _timestamp(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def _counts(path, account_id):
    if not path.is_file():
        return {'state': 'not_synced'}
    with _read_db(path) as db:
        if [row[0] for row in db.execute('SELECT account_id FROM binding')] != [account_id]:
            raise ValueError('binding mismatch')
        folders, remote = db.execute('SELECT COUNT(*), COALESCE(SUM(remote_count),0) FROM folders').fetchone()
        counts = dict(db.execute('SELECT status, COUNT(*) FROM messages WHERE present=1 GROUP BY status').fetchall())
        row = db.execute("SELECT value FROM meta WHERE key='last_run'").fetchone()
    last = None
    if row:
        try:
            raw = json.loads(row[0])
            if (isinstance(raw, dict) and _timestamp(raw.get('finished'))
                    and type(raw.get('failed')) is int and raw['failed'] >= 0):
                last = {'finished': raw['finished'], 'failed': raw['failed']}
        except (ValueError, TypeError):
            pass
    if not folders and not counts and not row:
        return {'state': 'not_synced'}
    parsed, failed = counts.get('parsed', 0), counts.get('failed', 0)
    return {'state': 'available', 'parsed': parsed, 'failed': failed,
            'not_downloaded': max(0, remote-parsed-failed), 'folder_count': folders,
            'last_completed': last, 'completion_record_unavailable': bool(row and last is None)}


def _schedule(owner, account_id):
    if not Path(cfg.MAIL_SCHEDULE_PATH).is_file():
        return {'state': 'not_configured', 'enabled': False}
    with _read_db(cfg.MAIL_SCHEDULE_PATH) as db:
        row = db.execute('''SELECT enabled, interval_seconds, failure_count
                            FROM mail_schedules WHERE owner=? AND account_id=?''', (owner, account_id)).fetchone()
    if row is None:
        return {'state': 'not_configured', 'enabled': False}
    return {'state': 'available', 'enabled': bool(row['enabled']),
            'interval_seconds': row['interval_seconds'], 'failure_count': row['failure_count'],
            'worker_enabled': cfg.MAIL_SCHEDULER_ENABLED}


def _latest_job(owner, account_id, credential_version):
    if not Path(cfg.JOB_STORE_PATH).is_file():
        return {'state': 'no_record'}
    with _read_db(cfg.JOB_STORE_PATH) as db:
        row = db.execute('''SELECT status, updated,
                json_extract(request,'$.credential_version') AS credential_version
            FROM jobs WHERE owner=? AND kind='imap_sync'
                AND json_extract(request,'$.account_id')=?
            ORDER BY updated DESC, created DESC, id DESC LIMIT 1''', (owner, account_id)).fetchone()
    if row is None:
        return {'state': 'no_record'}
    allowed = {'queued', 'running', 'succeeded', 'failed', 'incomplete', 'cancelled', 'interrupted'}
    return {'state': row['status'] if row['status'] in allowed else 'unavailable',
            'updated': _timestamp(row['updated']),
            'current_authorization': row['credential_version'] == credential_version}


def read_mailbox_status(owner, account_id=None, provider=None):
    """Owner comes from authenticated server context, never query text/model args."""
    result = {'account_state': 'unavailable', 'checked_at': time.time(),
              'live_connection_verified': False, 'ai_read_enabled': False, 'local_only': True}
    if provider == 'multiple':
        return result | {'account_state': 'provider_selection_required'}
    try:
        if Path(cfg.MAIL_ACCOUNTS_PATH).is_file():
            with _read_db(cfg.MAIL_ACCOUNTS_PATH) as db:
                # In this schema every account is a 163 account. Do not SELECT
                # address, display_name or encrypted_code for status questions.
                rows = [dict(row) for row in db.execute(
                    'SELECT id, credential_version FROM mail_accounts WHERE owner=? ORDER BY id', (owner,))]
        else:
            rows = []
        if account_id is not None:
            account = next((row for row in rows if row['id'] == account_id), None)
            if account is None:
                return result | {'account_state': 'not_accessible'}
        elif len(rows) == 1:
            account = rows[0]
        else:
            return result | {'account_state': 'selection_required' if rows else 'not_configured'}
        if provider is not None and provider != '163':
            return result | {'account_state': 'provider_mismatch', 'provider': '163', 'requested_provider': provider}
        result.update(account_state='configured', provider='163')
        for key, read in (
            ('local_sync', lambda: _counts(get_data_dir(cfg.IMAP_DATA_ROOT, owner, account['id'])/'mail.sqlite3', account['id'])),
            ('schedule', lambda: _schedule(owner, account['id'])),
            ('latest_job', lambda: _latest_job(owner, account['id'], account['credential_version'])),
        ):
            try:
                result[key] = read()
            except (OSError, sqlite3.Error, ValueError, TypeError):
                result[key] = {'state': 'unavailable'}
        return result
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return result


def _when(timestamp):
    try:
        return datetime.fromtimestamp(timestamp).astimezone().strftime('%m-%d %H:%M:%S')
    except (TypeError, ValueError, OSError, OverflowError):
        return '时间未知'


def format_mailbox_status(value):
    state = value.get('account_state')
    notices = {
        'not_configured': '尚未找到已保存的邮箱账号。请先在「邮箱设置」中添加账号。',
        'selection_required': '有多个邮箱账号，请先在左侧选择要查询的邮箱。',
        'provider_selection_required': '请一次查询一个邮箱，并在左侧选中对应账号。QQ 邮箱目前尚未接入。',
        'not_accessible': '所选邮箱不存在或当前用户无权访问，请重新选择账号。',
        'provider_mismatch': '当前选中的是 163 邮箱，你问的是其他邮箱。QQ 邮箱目前尚未接入，不能用 163 的状态回答。',
        'unavailable': '暂时无法读取邮箱配置，不能确认连接或同步情况。请检查本地服务和存储后重试。',
    }
    if state != 'configured':
        return notices.get(state, notices['unavailable']) + '\n\n' + CHAT_SCOPE_EXPLANATION
    lines = ['当前 163 邮箱的账号配置已保存。']
    local, schedule, job = (value.get(key, {}) for key in ('local_sync', 'schedule', 'latest_job'))
    if local.get('state') == 'available':
        lines.append(f"本地已解析 {local['parsed']} 封邮件；失败 {local['failed']} 封，待下载 {local['not_downloaded']} 封。数量仅覆盖已扫描文件夹的本地快照。")
        last = local.get('last_completed')
        if last:
            detail = f"，其中 {last['failed']} 封处理失败" if last['failed'] else ''
            lines.append('最近一次完成的同步记录：' + _when(last['finished']) + detail + '。')
        else:
            lines.append('尚无法确认最近一次完成的同步时间。')
    elif local.get('state') == 'not_synced':
        lines.append('尚无本地同步记录，不能确认已收取数量。请在「邮箱设置」中测试连接并同步。')
    else:
        lines.append('本地同步状态暂时无法读取，不能确认邮件数量。')
    job_labels = {'queued': '最近的同步任务正在排队', 'running': '最近的同步任务正在运行',
                  'succeeded': '最近的同步任务已结束', 'failed': '最近的同步任务失败',
                  'incomplete': '最近的同步任务未完成', 'cancelled': '最近的同步任务已停止',
                  'interrupted': '最近的同步任务被中断'}
    if job.get('state') in job_labels:
        lines.append(job_labels[job['state']] + '（' + _when(job.get('updated')) + '）。')
        if not job.get('current_authorization'):
            lines.append('这条任务未绑定当前授权版本，不能证明当前授权有效。')
    elif job.get('state') == 'unavailable':
        lines.append('最近任务状态暂时无法读取，不能据旧同步记录判断最近是否失败。')
    if schedule.get('state') == 'unavailable':
        lines.append('自动同步设置暂时无法读取。')
    elif schedule.get('enabled'):
        if not schedule.get('worker_enabled'):
            lines.append('自动同步计划已保存，但后台同步服务未启用。')
        elif schedule.get('failure_count', 0) >= 8:
            lines.append('自动同步因连续失败已暂停，请检查网络与授权后重新保存设置。')
        elif schedule.get('failure_count', 0):
            lines.append('自动同步近期有失败，正在等待重试。')
        else:
            lines.append(f"自动同步已启用，正常间隔为 {schedule['interval_seconds']//60} 分钟。")
    else:
        lines.append('自动同步未开启。')
    lines.append('本次只查询本地记录，未进行实时连接测试。')
    lines.append(CHAT_SCOPE_EXPLANATION)
    return '\n\n'.join(lines)
