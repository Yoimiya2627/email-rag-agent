"""
AnalyzerAgent: computes statistical summaries (top senders, label distribution,
daily volume) and uses LLM to interpret results in natural language.
"""
import json
import logging
from collections import Counter
from typing import Any, Dict

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse
from core.embedder import get_all_chunks
import config.settings as cfg

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    def __init__(self):
        self._client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)

    def _compute_stats(self) -> Dict[str, Any]:
        chunks = get_all_chunks()
        # Deduplicate by email_id to avoid counting chunks multiple times
        seen: Dict[str, dict] = {}
        for chunk in chunks:
            meta = chunk["metadata"]
            eid = meta.get("email_id", "")
            if eid and eid not in seen:
                seen[eid] = meta

        emails = list(seen.values())
        total = len(emails)

        sender_counts = Counter(m.get("sender", "unknown") for m in emails)
        top5_senders = sender_counts.most_common(5)

        label_counts: Counter = Counter()
        for m in emails:
            raw = m.get("labels", "[]")
            try:
                labels = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                labels = []
            for label in labels:
                label_counts[label] += 1

        daily_counts: Counter = Counter()
        for m in emails:
            date_str = m.get("date", "")
            if date_str and len(date_str) >= 10:
                daily_counts[date_str[:10]] += 1

        return {
            "total_emails": total,
            "top5_senders": [{"sender": s, "count": c} for s, c in top5_senders],
            "label_distribution": dict(label_counts.most_common(10)),
            "daily_counts": [
                {"date": d, "count": c}
                for d, c in sorted(daily_counts.items())[-30:]
            ],
        }

    def run(self, request: AgentRequest) -> AgentResponse:
        stats = self._compute_stats()
        stats_json = json.dumps(stats, ensure_ascii=False, indent=2)

        resp = self._client.chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是邮件数据分析专家。根据统计数据，用清晰易懂的语言回答用户的分析问题，"
                        "并给出有价值的洞察。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"邮件统计数据如下：\n{stats_json}\n\n用户问题：{request.query}",
                },
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        return AgentResponse(
            answer=resp.choices[0].message.content,
            sources=[],
            metadata=stats,
        )
