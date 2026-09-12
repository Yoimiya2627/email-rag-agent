"""Answer mailbox-access questions from local connection state, without an LLM."""
import re
import unicodedata


def is_mailbox_access_question(query):
    text = unicodedata.normalize('NFKC', query).casefold().strip().strip('?!？！。.')
    compact = re.sub(r'\s+', '', text)
    target = (r'(?:我(?:的)?)?(?:(?:网易)?163|qq)?(?:的)?'
              r'(?:邮箱(?:里|里面|中)?(?:的)?(?:邮件|内容)?|邮件(?:内容)?)')
    chinese = (r'(?:请问[,，]?)?(?:你)?(?:现在|目前)?'
               r'(?:可以|能|能不能|可不可以|是否能|是否可以|已经能|已经可以)'
               r'(?:直接)?(?:看到|看见|看得到|看|读取|读到|查看|访问)' + target + r'(?:了)?(?:吗|么|没有|没)?')
    english = (r'(?:can|could) you (?:already |now |directly )?(?:see|read|access) '
               r'my (?:(?:163|qq) )?(?:emails?|mailbox|inbox)(?: now| yet)?')
    return bool(re.fullmatch(chinese, compact) or re.fullmatch(english, text))


def mailbox_access_answer(accounts_response, account_id, get):
    limitation = ('当前 AI 对话还不能读取这些真实邮件。本次状态检查没有向模型发送邮件内容。'
                  '看邮件请打开左侧「查看邮件」；AI 对话目前使用单独导入的资料。')
    if accounts_response is None:
        return '暂时无法读取邮箱连接状态，不能确认本地同步情况。' + limitation
    accounts = accounts_response.get('accounts', [])
    account = next((item for item in accounts if item['id'] == account_id), None)
    if account is None:
        return '目前没有选中已连接的邮箱。可在「邮箱设置」中添加账号。' + limitation
    report = get('/mailboxes/' + account['id'] + '/report')
    parsed = (report or {}).get('parsed')
    if type(parsed) is int and parsed >= 0:
        local = f'应用已经在本地收取并解析了当前邮箱的 {parsed} 封邮件。'
    else:
        local = '当前邮箱已保存连接配置，但暂时无法确认本地同步数量。'
    return local + '\n\n' + limitation
