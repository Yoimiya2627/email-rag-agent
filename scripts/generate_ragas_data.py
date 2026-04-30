"""
Generate 100 Q&A test pairs from emails.json for RAGAS evaluation.
Output: data/ragas_testset.json
Each item: {question, ground_truth, contexts, email_ids}
"""
import json
import random
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET = 100
BATCH = 5
EMAIL_PATH = Path(__file__).parent.parent / "data" / "emails.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "ragas_testset.json"

_QA_SYSTEM = """你是一位测试数据生成专家。根据给定的邮件内容，生成高质量的问答对。

要求：
1. 问题要真实、多样（事实型/推理型/摘要型各占1/3）
2. 答案必须完全基于邮件内容，不能臆造
3. 每个问题针对不同的信息点

严格按JSON数组格式返回（不要任何解释）：
[
  {
    "question": "具体问题",
    "ground_truth": "基于邮件内容的准确答案（2-4句话）",
    "email_ids": ["涉及的邮件id列表"]
  }
]"""


def generate_qa_batch(client: OpenAI, emails: list, count: int) -> list:
    emails_text = "\n\n".join(
        f"邮件ID: {e['id']}\n发件人: {e['sender']}\n日期: {e['date']}\n"
        f"主题: {e['subject']}\n内容: {e['body'][:300]}"
        for e in emails
    )
    prompt = f"请根据以下{len(emails)}封邮件，生成{count}个问答对：\n\n{emails_text}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=cfg.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": _QA_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=3000,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
                if "```" in raw:
                    raw = raw[: raw.index("```")]
            items = json.loads(raw)
            valid = []
            for item in items:
                if all(k in item for k in ("question", "ground_truth", "email_ids")):
                    valid.append(item)
            return valid
        except Exception as exc:
            logger.warning(f"Attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


def main():
    with open(EMAIL_PATH, encoding="utf-8") as f:
        all_emails = json.load(f)
    logger.info(f"Loaded {len(all_emails)} emails")

    # Resume
    existing = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    logger.info(f"Resuming from {len(existing)} existing Q&A pairs")

    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    total = len(existing)

    try:
        while total < TARGET:
            # Sample diverse emails: pick from different parts of the dataset
            sample = random.sample(all_emails, min(BATCH * 2, len(all_emails)))
            batch = generate_qa_batch(client, sample, BATCH)
            if batch:
                existing.extend(batch)
                total = len(existing)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                logger.info(f"  {total}/{TARGET} Q&A pairs generated")
            else:
                logger.warning("Empty batch, retrying...")
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info(f"Interrupted. Saved {total} Q&A pairs.")

    logger.info(f"Done! Total: {len(existing)} Q&A pairs → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
