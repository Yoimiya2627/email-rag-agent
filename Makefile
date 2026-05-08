.PHONY: help install index api ui run eval eval-all latency test clean

# Override with `make PYTHON=python3 install` etc. if needed.
PYTHON ?= python

help:
	@echo "Email RAG Agent - common tasks"
	@echo ""
	@echo "  make install     pip install + preload bge-m3 (~5 min first time, ~570MB)"
	@echo "  make index       index data/emails.json into ChromaDB"
	@echo "  make run         start API on :8000 + Streamlit on :8501 (Ctrl-C stops both)"
	@echo "  make api         API only — handy when iterating on backend"
	@echo "  make ui          Streamlit only — handy when iterating on frontend"
	@echo "  make eval        run RAGAS on V2 only (~3 min, recommended config)"
	@echo "  make eval-all    run all 6 ablation versions (~30 min)"
	@echo "  make latency     measure end-to-end latency"
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

eval:
	$(PYTHON) scripts/run_ragas_eval.py --versions V2

eval-all:
	$(PYTHON) scripts/run_ragas_eval.py

latency:
	$(PYTHON) scripts/measure_latency.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf chroma_db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
