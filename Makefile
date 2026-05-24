.PHONY: help install index api ui run mcp eval agent-eval trace-summary eval-all latency reranker-latency test clean

# Prefer the project venv if present. Override with `make PYTHON=python3 install`.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

help:
	@echo "Email RAG Agent - common tasks"
	@echo ""
	@echo "  make install     pip install + preload bge-m3 (~5 min first time, ~570MB)"
	@echo "  make index       index data/emails.json into ChromaDB"
	@echo "  make run         start API on :8000 + Streamlit on :8501 (Ctrl-C stops both)"
	@echo "  make mcp         start standalone MCP server on :8001"
	@echo "  make api         API only — handy when iterating on backend"
	@echo "  make ui          Streamlit only — handy when iterating on frontend"
	@echo "  make eval        run RAGAS on V2 only (~3 min, recommended config)"
	@echo "  make agent-eval  run agent task evaluation"
	@echo "  make trace-summary summarize agent trace JSONL"
	@echo "  make eval-all    run all 7 ablation versions (~30 min)"
	@echo "  make latency     measure end-to-end latency"
	@echo "  make reranker-latency measure reranker-only latency"
	@echo "  make test        run unit tests"
	@echo "  make clean       remove chroma_db and __pycache__ (model + eval results kept)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) scripts/preload_model.py

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

trace-summary:
	$(PYTHON) scripts/summarize_agent_traces.py

eval-all:
	$(PYTHON) scripts/run_ragas_eval.py

latency:
	$(PYTHON) scripts/measure_latency.py

reranker-latency:
	$(PYTHON) scripts/measure_reranker_latency.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf chroma_db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
