# SKEIN — Structural Knowledge and Enterprise Intelligence Network
# Makefile — common development tasks
#
# Usage:
#   make test           run all 135 tests (no LLM needed)
#   make test-unit      unit tests only (fastest)
#   make test-load      load/stress tests (slower)
#   make lint           ruff linting
#   make typecheck      mypy type checking
#   make server         start production server (DryRun mode)
#   make docker-build   build Docker image
#   make k8s-deploy     deploy to Kubernetes
#   make clean          remove generated files

.PHONY: test test-unit test-integration test-system test-scenarios test-load \
        lint typecheck server server-dry docker-build docker-run \
        k8s-deploy k8s-status clean install install-dev help

PYTHON    := python3
SRC_DIRS  := framework agents platform scripts
TEST_DIRS := tests

# ── Testing ────────────────────────────────────────────────────────────────────

test:
	@echo "Running all 135 tests..."
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

test-unit:
	@echo "Running unit tests..."
	$(PYTHON) -m unittest discover -s tests/unit -p "test_*.py" -v

test-integration:
	@echo "Running integration tests..."
	$(PYTHON) -m unittest tests.integration.test_framework_integration -v

test-system:
	@echo "Running system tests..."
	$(PYTHON) -m unittest tests.system.test_multi_agent_system -v

test-scenarios:
	@echo "Running procurement scenario tests..."
	$(PYTHON) -m unittest tests.scenarios.test_procurement_scenarios -v

test-load:
	@echo "Running load/stress tests (takes ~30s)..."
	$(PYTHON) -m unittest tests.load.test_stress_load -v

test-retry:
	@echo "Running retry + circuit breaker tests..."
	$(PYTHON) -m unittest tests.unit.test_retry_circuit -v

test-memory:
	@echo "Running memory tests..."
	$(PYTHON) -m unittest tests.unit.test_memory -v

test-quick:
	@echo "Running fastest subset (unit only)..."
	$(PYTHON) -m unittest tests.unit.test_retry_circuit tests.unit.test_memory -v

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check $(SRC_DIRS) $(TEST_DIRS) \
		|| echo "ruff not installed — pip install ruff"

lint-fix:
	@command -v ruff >/dev/null 2>&1 && ruff check --fix $(SRC_DIRS) $(TEST_DIRS) \
		|| echo "ruff not installed — pip install ruff"

typecheck:
	@command -v mypy >/dev/null 2>&1 && mypy framework agents --ignore-missing-imports \
		|| echo "mypy not installed — pip install mypy"

# ── Server ────────────────────────────────────────────────────────────────────

server:
	@echo "Starting SKEIN server (config: config/config.yaml)..."
	$(PYTHON) -m scripts.server --config config/config.yaml

server-dry:
	@echo "Starting SKEIN server in DryRun mode (no LLM calls)..."
	$(PYTHON) -m scripts.server --config config/config.yaml --dry-run

health:
	@echo "Checking health endpoints..."
	curl -s http://localhost:8080/health | python3 -m json.tool
	curl -s http://localhost:8080/ready  | python3 -m json.tool

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -f deploy/docker/Dockerfile -t skein-framework:latest .

docker-run:
	docker run -p 8080:8080 \
		-e LLM_PROVIDER=ollama \
		-e LLM_BASE_URL=http://host.docker.internal:11434 \
		-e AGENT_DRY_RUN=true \
		skein-framework:latest

docker-compose-up:
	docker-compose -f deploy/docker/docker-compose.yml up -d

docker-compose-down:
	docker-compose -f deploy/docker/docker-compose.yml down

# ── Kubernetes ────────────────────────────────────────────────────────────────

k8s-deploy:
	kubectl apply -f deploy/kubernetes/deployment.yaml
	kubectl rollout status deployment/skein-agents -n skein

k8s-status:
	kubectl get all -n skein

k8s-logs:
	kubectl logs -n skein -l app=skein --tail=50 -f

k8s-delete:
	kubectl delete -f deploy/kubernetes/deployment.yaml

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install pyyaml requests

install-anthropic: install
	pip install anthropic

install-openai: install
	pip install openai

install-langchain: install
	pip install langchain langchain-core langchain-anthropic

install-langgraph: install install-langchain
	pip install langgraph

install-databricks: install
	pip install databricks-sdk mlflow pyspark

install-fabric: install
	pip install azure-identity azure-storage-file-datalake

install-dev:
	pip install pyyaml requests mypy ruff

install-all: install install-anthropic install-langchain install-langgraph

# ── Synthetic data ────────────────────────────────────────────────────────────

generate-data:
	$(PYTHON) data/synthetic/generate_all.py

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "SKEIN — Structural Knowledge and Enterprise Intelligence Network"
	@echo ""
	@echo "Testing:"
	@echo "  make test              All 135 tests"
	@echo "  make test-unit         Unit tests only (fastest)"
	@echo "  make test-integration  Integration tests"
	@echo "  make test-system       Multi-agent system tests"
	@echo "  make test-scenarios    Procurement scenario tests"
	@echo "  make test-load         Load/stress tests (slower)"
	@echo ""
	@echo "Server:"
	@echo "  make server            Start with config/config.yaml"
	@echo "  make server-dry        Start in DryRun mode (no LLM)"
	@echo "  make health            Check /health and /ready"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      Build image"
	@echo "  make docker-run        Run container (DryRun)"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-deploy        Apply manifests"
	@echo "  make k8s-status        Show all resources"
	@echo "  make k8s-logs          Tail pod logs"
	@echo ""
	@echo "Install:"
	@echo "  make install           Core only (pyyaml requests)"
	@echo "  make install-anthropic + Anthropic Claude"
	@echo "  make install-langchain + LangChain"
	@echo "  make install-databricks + Databricks SDK"
	@echo "  make install-fabric    + Azure SDK (MS Fabric)"
	@echo ""
