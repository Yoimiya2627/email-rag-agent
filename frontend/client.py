"""Pure HTTP/SSE result handling, independent of Streamlit rendering."""
from dataclasses import dataclass, field


def http_error(response) -> dict:
    try:
        detail = response.json().get('detail')
    except (ValueError, AttributeError):
        detail = None
    status = response.status_code
    messages = {409: '会话正在执行，或操作与当前状态冲突，请查看任务状态。',
                413: '输入或证据超出预算，请缩小本次请求。',
                422: '请求字段不符合要求，请检查输入。',
                429: '当前任务已达并发上限，请稍后再试。',
                503: '所选模式尚未就绪，请检查服务状态。',
                504: '操作超时，请先核对任务或审批状态，再决定后续操作。'}
    return {'error': detail[:2000] if isinstance(detail, str) else messages.get(status, '请求失败，请查看服务状态。'),
            'status_code': status}


@dataclass
class StreamAccumulator:
    answer: str = ''
    sources: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    intent: str = ''
    error: str | None = None
    done: bool = False

    def feed(self, event):
        if event == '[DONE]':
            self.done = True
            return
        if not isinstance(event, dict):
            raise ValueError('Invalid stream event')
        if isinstance(event.get('token'), str):
            self.answer += event['token']
        if isinstance(event.get('sources'), list):
            self.sources = event['sources']
        if isinstance(event.get('metadata'), dict):
            self.metadata.update(event['metadata'])
        if isinstance(event.get('intent'), str):
            self.intent = event['intent']
        if isinstance(event.get('error'), str):
            self.error = event['error'][:2000]

    def result(self):
        metadata = dict(self.metadata)
        if self.error or not self.done or not self.answer.strip():
            metadata.setdefault('status', 'incomplete' if self.answer.strip() else 'empty_model_response')
            if metadata.get('status') == 'success':
                metadata['status'] = 'incomplete'
            metadata['completion_status'] = 'incomplete' if self.answer.strip() else 'error'
        output = {'answer': self.answer, 'sources': self.sources, 'intent': self.intent,
                  'metadata': metadata, '_streamed': True}
        if self.error:
            output['warning'] = self.error
        elif not self.done:
            output['warning'] = '流式连接中断，已显示的文字是不完整结果。请先查看会话状态。'
        elif not self.answer.strip():
            output['warning'] = '模型未返回有效回答，任务尚未完成。'
        return output
