"""Local mailbox management; credentials and mail never enter a model call."""
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Any

import config.settings as cfg
from api.security import Identity, require_identity
from core.mail_accounts import MailAccountStore, get_data_dir
from core.mail_sync import MailSyncStore, ImapSyncError, sync_mailbox
from agents.runtime import current_run


def accounts():
    return MailAccountStore(cfg.MAIL_ACCOUNTS_PATH)


def mailbox(owner, account_id):
    accounts().get(owner, account_id)
    return MailSyncStore(get_data_dir(cfg.IMAP_DATA_ROOT, owner, account_id), account_id)


def provider_for(owner, account_id):
    from agents.imap_readonly import ImapReadOnlyProvider
    account, code = accounts().credentials(owner, account_id)
    return account, ImapReadOnlyProvider(account['address'], code, timeout=30,
                                       max_message_bytes=cfg.IMAP_MAX_MESSAGE_BYTES)


_ERRORS = {
    'authentication_failed':'163 登录失败，请核对邮箱地址、客户端授权码及 IMAP 服务开关。',
    'imap_authentication_failed':'163 登录失败，请核对客户端授权码及 IMAP 服务开关。',
    'account_sync_busy':'这个账号已有同步任务，请等待完成或先停止该任务。',
    'credential_binding_changed':'账号授权已更新，请重新提交同步任务。',
    'unsupported_secure_storage':'当前系统尚未配置受支持的凭据加密存储。',
    'secure_storage_failed':'无法读取本机加密授权，请在当前 Windows 用户下重新配置账号。',
    'folder_not_available':'所选文件夹已变化，请重新测试连接并选择文件夹。',
}


def mailbox_error(exc):
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, KeyError):
        return HTTPException(404, '邮箱或邮件不存在')
    if isinstance(exc, PermissionError):
        return HTTPException(403, '无权访问该邮箱')
    code = getattr(exc, 'code', '')
    if not code and str(exc) in _ERRORS:
        code = str(exc)
    # Do not return exception messages, server responses or validation inputs.
    return HTTPException(409 if isinstance(exc, (ValueError,ImapSyncError)) else 502,
                         _ERRORS.get(code, '邮箱操作未完成，请检查 IMAP 服务、授权码和网络后重试。'))


class SyncRequest(BaseModel):
    folders: list[str] = Field(default_factory=lambda:['INBOX'], min_length=1, max_length=30)
    max_messages: int = Field(default=100, ge=1, le=2000)
    retry_failed: bool = False
    operation_key: str | None = Field(default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z0-9._:-]+$')


def run_mail_sync(job):
    request, owner = job['request'], job['owner']
    account_id = request['account_id']
    account, provider = provider_for(owner, account_id)
    def binding():
        current = accounts().get(owner, account_id)
        if current['credential_version'] != request['credential_version']:
            raise ImapSyncError('credential_binding_changed')
    binding()
    run = current_run()
    if run:
        run.checkpoint({'safe':True, 'kind':'imap_sync', 'account_id':account_id})
        run.progress('imap_connecting')
    with provider:
        return sync_mailbox(provider, mailbox(owner, account_id), request['folders'],
                            max_messages=request['max_messages'], parse_timeout=cfg.IMAP_PARSE_TIMEOUT,
                            retry_failed=request.get('retry_failed',False), check_binding=binding)


def mailbox_router(admission, manager):
    router = APIRouter(prefix='/mailboxes', tags=['local-mailboxes'])

    @router.get('')
    def list_accounts(identity: Identity = Depends(require_identity)):
        try:
            return {'accounts': accounts().list(identity.owner_id), 'local_only':True}
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.post('')
    def configure(payload: Any = Body(None), identity: Identity = Depends(require_identity)):
        # Manual input validation deliberately avoids echoing a malformed body
        # containing an authorization code in a Pydantic error response.
        try:
            if not isinstance(payload,dict) or set(payload)-{'address','authorization_code','display_name'}:
                raise ValueError('invalid_account_input')
            return accounts().upsert(identity.owner_id, payload.get('address'),
                                     payload.get('authorization_code'), payload.get('display_name',''))
        except (TypeError,ValueError):
            raise HTTPException(422, '请填写完整的 163 邮箱地址和有效客户端授权码。') from None
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.post('/{account_id}/connect')
    def connect(account_id: str, identity: Identity = Depends(require_identity)):
        try:
            with admission.slot():
                account, provider = provider_for(identity.owner_id, account_id)
                with provider:
                    description = provider.describe_account()
                    folders = provider.list_folders()
            return {'account':account, 'connection':description, 'folders':folders,
                    'read_only':True, 'local_only':True}
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.post('/{account_id}/sync',status_code=202)
    def sync(account_id: str, request: SyncRequest, identity: Identity = Depends(require_identity)):
        try:
            if any(not isinstance(v,str) or not v or len(v)>1024 or any(ord(c)<32 for c in v) for v in request.folders):
                raise ValueError('invalid_folder')
            account = accounts().get(identity.owner_id, account_id)
            payload = request.model_dump(exclude={'operation_key'})
            payload.update(account_id=account_id,credential_version=account['credential_version'])
            return manager().submit(identity.owner_id,'imap_sync',payload,request.operation_key)
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.get('/{account_id}/report')
    def report(account_id: str, identity: Identity = Depends(require_identity)):
        try:
            return mailbox(identity.owner_id, account_id).report()
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.get('/{account_id}/messages')
    def messages(account_id: str, offset:int=Query(0,ge=0), limit:int=Query(25,ge=1,le=100),
                 failures_only:bool=False, identity: Identity = Depends(require_identity)):
        try:
            return mailbox(identity.owner_id, account_id).messages(offset=offset,limit=limit,failures_only=failures_only)
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.get('/{account_id}/messages/{key}')
    def message(account_id: str, key: str, identity: Identity = Depends(require_identity)):
        try:
            return mailbox(identity.owner_id, account_id).message(key)
        except Exception as exc:
            raise mailbox_error(exc) from None

    @router.get('/{account_id}/messages/{key}/raw')
    def raw(account_id: str, key: str, identity: Identity = Depends(require_identity)):
        try:
            store = mailbox(identity.owner_id, account_id)
            row = store.message(key)
            path = store.raw_path(row['raw_sha256'])
            import hashlib
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['raw_sha256']:
                raise KeyError('Raw capture missing or changed')
            return FileResponse(path, media_type='application/octet-stream', filename=key+'.eml',
                                headers={'X-Content-Type-Options':'nosniff','Cache-Control':'no-store'})
        except Exception as exc:
            raise mailbox_error(exc) from None
    return router
