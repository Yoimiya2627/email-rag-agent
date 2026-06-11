.PHONY: help install install-lock index api ui run mcp eval agent-eval agent-eval-smoke agent-eval-full gmail-agent-testset agent-eval-real agent-eval-real-gate trace-summary eval-all latency reranker-latency gmail-preflight gmail-sync gmail-sync-index gmail-gold-template gmail-gold-quality context-recall context-recall-real test compile verify clean

# Prefer the project venv if present. Override with `make PYTHON=python3 install`.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

help:
	@echo "Email RAG Agent - common tasks"
	@echo ""
	@echo "  make install     pip install + preload bge-m3 (~5 min first time, ~570MB)"
	@echo "  make install-lock pip install from requirements.lock"
	@echo "  make index       index data/emails.json into ChromaDB"
	@echo "  make run         start API on :8000 + Streamlit on :8501 (Ctrl-C stops both)"
	@echo "  make mcp         start standalone MCP server on :8001"
	@echo "  make api         API only — handy when iterating on backend"
	@echo "  make ui          Streamlit only — handy when iterating on frontend"
	@echo "  make eval        run RAGAS on V2 only (~3 min, recommended config)"
	@echo "  make agent-eval  run agent task evaluation"
	@echo "  make agent-eval-smoke run offline gate for current smoke eval results"
	@echo "  make agent-eval-full  run strict 100+ task offline gate"
	@echo "  make gmail-agent-testset build private real Gmail agent testset"
	@echo "  make agent-eval-real run private real Gmail agent task evaluation"
	@echo "  make agent-eval-real-gate run offline gate for real Gmail agent eval"
	@echo "  make trace-summary summarize agent trace JSONL"
	@echo "  make eval-all    run all 7 ablation versions (~30 min)"
	@echo "  make latency     measure end-to-end latency"
	@echo "  make reranker-latency measure reranker-only latency"
	@echo "  make gmail-preflight check local readiness for real Gmail data"
	@echo "  make gmail-sync  sync Gmail read-only messages to local ignored JSON"
	@echo "  make gmail-sync-index sync Gmail read-only messages and rebuild index"
	@echo "  make gmail-gold-template build real-mail gold annotation template"
	@echo "  make gmail-gold-quality check real-mail gold label quality"
	@echo "  make context-recall evaluate retrieval against gold chunk labels"
	@echo "  make context-recall-real evaluate V2 top10 retrieval against real-mail gold labels"
	@echo "  make test        run unit tests"
	@echo "  make verify      compile, test, and run smoke EvalOps gate"
	@echo "  make clean       remove chroma_db and __pycache__ (model + eval results kept)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) scripts/preload_model.py

install-lock:
	$(PYTHON) -m pip install -r requirements.lock

index:
	$(PYTHON) scripts/index_emails.py

api:
	$(PYTHON) -m api.main

ui:
	$(PYTHON) -m streamlit run frontend/app.py

run:
	@echo "Starting API (:8000) + Streamlit (:8501). Ctrl-C stops both."
	@trap 'kill 0' INT TERM; \
	  $(PYTHON) -m api.main & \
	  $(PYTHON) -m streamlit run frontend/app.py & \
	  wait

mcp:
	$(PYTHON) mcp_server.py --transport streamable-http

eval:
	$(PYTHON) scripts/run_ragas_eval.py --versions V2

agent-eval:
	$(PYTHON) scripts/run_agent_eval.py

agent-eval-smoke:
	$(PYTHON) scripts/check_agent_eval_gate.py --input data/eval_results/agent_eval.json --min-tasks 1 --min-task-success-rate 0.80 --min-tool-accuracy 0.80 --max-forbidden-tool-violation-rate 0.01 --max-max-steps-reached-rate 0.05

agent-eval-full:
	$(PYTHON) scripts/check_agent_eval_gate.py --input data/eval_results/agent_eval.json --min-tasks 100 --min-task-success-rate 0.80 --min-tool-accuracy 0.80 --max-forbidden-tool-violation-rate 0.01 --max-max-steps-reached-rate 0.05

gmail-agent-testset:
	$(PYTHON) scripts/build_real_agent_testset.py --gold data/real_emails/gold_chunks.real.json --output data/real_emails/agent_testset.real.json --limit 30

agent-eval-real:
	ENABLE_AGENT_TRACE=true AGENT_TRACE_LOG_PATH=data/traces/agent_traces.real.jsonl $(PYTHON) scripts/run_agent_eval.py --testset-path data/real_emails/agent_testset.real.json --output data/eval_results/agent_eval.real.json --report-output data/eval_results/agent_eval.real_report.md --trace-input data/traces/agent_traces.real.jsonl

agent-eval-real-gate:
	$(PYTHON) scripts/check_agent_eval_gate.py --input data/eval_results/agent_eval.real.json --min-tasks 30 --min-task-success-rate 0.75 --min-tool-accuracy 0.80 --max-forbidden-tool-violation-rate 0.00 --max-max-steps-reached-rate 0.05

trace-summary:
	$(PYTHON) scripts/summarize_agent_traces.py

eval-all:
	$(PYTHON) scripts/run_ragas_eval.py

latency:
	$(PYTHON) scripts/measure_latency.py

reranker-latency:
	$(PYTHON) scripts/measure_reranker_latency.py

gmail-preflight:
	$(PYTHON) scripts/gmail_phase2a_preflight.py

gmail-sync:
	$(PYTHON) scripts/sync_gmail_readonly.py

gmail-sync-index:
	$(PYTHON) scripts/sync_gmail_readonly.py --index --clear-index

gmail-gold-template:
	$(PYTHON) scripts/build_real_gmail_gold_template.py

gmail-gold-quality:
	$(PYTHON) scripts/check_real_gold_quality.py

context-recall:
	$(PYTHON) scripts/evaluate_context_recall.py

context-recall-real: gmail-gold-quality
	$(PYTHON) scripts/evaluate_context_recall.py --gold data/real_emails/gold_chunks.real.json --output data/eval_results/context_recall.real.json --versions V2 --top-n 10 --fetch-k 80

test:
	$(PYTHON) -m pytest tests/ -v

compile:
	$(PYTHON) -m compileall -q api agents core config frontend models scripts mcp_server.py

verify: compile test agent-eval-smoke

clean:
	rm -rf chroma_db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
