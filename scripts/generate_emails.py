"""
Batch email generator using DeepSeek API.
Generates diverse Chinese business emails and saves to data/emails.json.
Supports resume: if output file already has emails, continues from where it left off.
"""
import json
import os
import sys
import time
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_COUNT = 5000
BATCH_SIZE = 25
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "emails.json"

# ── Diversity pools ────────────────────────────────────────────────────────────
DEPARTMENTS = ["产品部", "技术部", "销售部", "市场部", "财务部", "人力资源部",
               "客服中心", "运营部", "法务部", "战略部", "数据分析团队", "采购部"]

SENDERS = [
    "张伟 <zhangwei@company.com>", "李娜 <lina@company.com>",
    "陈刚 <chengang@company.com>", "王强 <wangqiang@company.com>",
    "刘敏 <liumin@company.com>", "赵磊 <zhaolei@company.com>",
    "孙晓 <sunxiao@company.com>", "周婷 <zhongting@company.com>",
    "吴浩 <wuhao@company.com>", "郑丽 <zhengli@company.com>",
    "人力资源部 <hr@company.com>", "客服中心 <support@company.com>",
    "财务部 <finance@company.com>", "监控系统 <alerts@company.com>",
    "数据分析团队 <data@company.com>", "采购部 <procurement@company.com>",
    "市场部 <marketing@company.com>", "法务部 <legal@company.com>",
    "运营部 <ops@company.com>", "战略部 <strategy@company.com>",
]

RECIPIENTS_POOL = [
    "product-team@company.com", "tech-lead@company.com",
    "all-staff@company.com", "management@company.com",
    "finance@company.com", "hr@company.com",
    "cto@company.com", "ceo@company.com",
    "ops@company.com", "sales@company.com",
    "on-call@company.com", "legal@company.com",
]

LABEL_POOL = [
    ["产品", "会议"], ["技术", "紧急"], ["HR", "全员"], ["财务", "合同"],
    ["客服", "表扬"], ["数据", "月报"], ["采购", "供应商"], ["市场", "活动"],
    ["OKR", "规划"], ["告警", "生产环境"], ["入职", "HR"], ["法务", "合规"],
    ["销售", "客户"], ["运营", "增长"], ["技术债务", "架构"], ["安全", "紧急"],
    ["培训", "HR"], ["绩效", "管理层"], ["项目", "里程碑"], ["预算", "财务"],
    ["战略", "管理层"], ["合同", "法务"], ["投诉", "客服"], ["发布", "技术"],
]

SCENARIOS = [
    # 技术类
    "生产环境故障告警和处理",
    "代码审查意见和回复",
    "技术方案评审讨论",
    "系统性能优化方案",
    "安全漏洞发现和修复",
    "数据库迁移计划",
    "API接口设计讨论",
    "微服务架构重构",
    "CI/CD流水线问题",
    "技术债务清理计划",
    "第三方依赖升级",
    "容器化部署方案",
    # 产品类
    "新功能需求讨论",
    "用户反馈整理和优先级",
    "竞品分析报告",
    "产品路线图评审",
    "A/B测试结果分析",
    "用户体验改进建议",
    "版本发布计划",
    "需求变更通知",
    "用户调研结果",
    "产品数据周报",
    # 业务类
    "客户投诉处理",
    "合同续签谈判",
    "新客户开拓进展",
    "销售数据月报",
    "市场活动策划",
    "合作伙伴对接",
    "采购申请审批",
    "供应商评估报告",
    "客户成功案例",
    "商务合作意向",
    # 管理类
    "季度OKR回顾",
    "团队绩效评估",
    "新员工入职安排",
    "员工离职交接",
    "团队建设活动",
    "招聘需求提交",
    "薪资调整申请",
    "培训计划通知",
    "跨部门协作问题",
    "预算申请和审批",
    # 财务类
    "月度财务报告",
    "费用报销审批",
    "年度预算规划",
    "资金使用情况",
    "发票和付款处理",
    # 运营类
    "增长数据周报",
    "用户生命周期分析",
    "渠道效果对比",
    "促销活动效果复盘",
    "内容运营计划",
]


def make_prompt(batch_id: int, count: int, used_ids: set) -> str:
    scenarios = random.sample(SCENARIOS, min(count, len(SCENARIOS)))
    start_num = batch_id * count + 16  # offset from existing 15 emails
    date_base = random.randint(1, 28)
    month = random.choice(["01", "02", "03", "04"])
    year = "2026"

    return f"""请生成 {count} 封中文商务邮件，以JSON数组格式返回。

要求：
1. 每封邮件覆盖不同场景，本批次场景参考（可以扩展）：{', '.join(scenarios[:8])}
2. 邮件要真实、专业，有具体细节（数字、日期、人名、决策内容）
3. 部分邮件应该是同一话题的回复链（使用相同thread_id）
4. 日期范围：{year}-{month}-01 到 {year}-{month}-{date_base:02d}
5. id格式：email_{start_num:04d} 到 email_{start_num + count - 1:04d}

严格按以下JSON格式，只返回JSON数组，不要任何解释：
[
  {{
    "id": "email_{start_num:04d}",
    "subject": "邮件主题",
    "sender": "姓名 <email@company.com>",
    "recipients": ["recipient@company.com"],
    "date": "{year}-{month}-{date_base:02d}",
    "body": "邮件正文（可以用\\n换行，禁止在正文中使用未转义的双引号）",
    "labels": ["标签1", "标签2"],
    "thread_id": "thread_XXX或null"
  }}
]

注意：body字段中不得出现未转义的双引号字符，如需引用请用「」或【】代替。"""


def generate_batch(client: OpenAI, batch_id: int, count: int, used_ids: set) -> list:
    prompt = make_prompt(batch_id, count, used_ids)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=cfg.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
                if "```" in raw:
                    raw = raw[:raw.index("```")]
            emails = json.loads(raw)
            if not isinstance(emails, list):
                raise ValueError("Response is not a list")
            # Validate and deduplicate ids
            valid = []
            for e in emails:
                if not all(k in e for k in ("id", "subject", "sender", "recipients", "date", "body")):
                    continue
                if e["id"] in used_ids:
                    e["id"] = e["id"] + f"_b{batch_id}"
                used_ids.add(e["id"])
                if not isinstance(e.get("recipients"), list):
                    e["recipients"] = [e["recipients"]]
                if not isinstance(e.get("labels"), list):
                    e["labels"] = []
                valid.append(e)
            return valid
        except Exception as exc:
            logger.warning(f"Batch {batch_id} attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return []


def main():
    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)

    # Load existing emails
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    used_ids = {e["id"] for e in existing}
    logger.info(f"Starting with {len(existing)} existing emails, target: {TARGET_COUNT}")

    batch_id = len(existing) // BATCH_SIZE
    total = len(existing)

    try:
        while total < TARGET_COUNT:
            remaining = TARGET_COUNT - total
            count = min(BATCH_SIZE, remaining)
            logger.info(f"Batch {batch_id+1} | {total}/{TARGET_COUNT} emails...")

            batch = generate_batch(client, batch_id, count, used_ids)
            if batch:
                existing.extend(batch)
                total += len(batch)
                # Save after every batch
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                logger.info(f"  -> Got {len(batch)} emails, total: {total}")
            else:
                logger.warning(f"  -> Batch {batch_id} returned 0 emails, skipping")

            batch_id += 1
            time.sleep(0.5)  # Rate limit courtesy

    except KeyboardInterrupt:
        logger.info(f"Interrupted. Saved {total} emails so far.")

    logger.info(f"Done! Total emails: {len(existing)}")


if __name__ == "__main__":
    main()
