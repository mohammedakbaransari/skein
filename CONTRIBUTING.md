# Contributing to SKEIN

Thank you for your interest in contributing. SKEIN is an independent research project and welcomes contributions that advance its research goals.

## What Contributions Are Welcome

- **New agent implementations** — additional structural intelligence agents addressing procurement gaps
- **Bug fixes** — correctness issues in existing agent logic or framework infrastructure
- **Test coverage** — additional unit, integration, or scenario tests
- **Platform adapters** — new deployment targets (AWS SageMaker, Google Vertex, Snowflake, etc.)
- **Documentation** — clearer explanations, usage examples, architecture diagrams

## What to Discuss First

Open a GitHub issue before starting large changes:
- New agent types (to align with the 15-mystery research framework)
- Breaking changes to framework interfaces
- New platform adapter designs

## Development Setup

```bash
git clone https://github.com/mohammedakbaransari/skein
cd skein
pip install -r requirements.txt
make test                     # must be 417/417 before you start
```

## Before Submitting a PR

1. **All 417 tests pass**: `make test`
2. **Your new code has tests**: unit tests at minimum, scenario tests if adding an agent
3. **No secrets in code**: API keys, passwords, tokens — use a configured `SecretsProvider`; environment variables are development/bootstrap only
4. **Agent metadata is complete**: every agent has `METADATA`, `mystery_refs`, and `tags`
5. **observe() is a pure function**: no LLM calls, no I/O, no side effects in `observe()`

## Adding a New Agent

1. Create `agents/<domain>/<agent_name>.py`
2. Add `AgentMetadata` to `framework/agents/catalogue.py`
3. Inherit from `StructuralAgent` or `DecisionAgent`
4. Implement `observe()`, `reason()`, `parse_findings()`
5. Add unit tests in `tests/unit/`
6. Register in `scripts/server.py`, or use `PluginAgentRegistry` for deployment-time/per-tenant plugins

The `observe()` contract is strict: it must be a pure function. No LLM calls, no network I/O, no file I/O. It extracts and structures data from `task.payload` only. This makes every agent trivially unit-testable without mocking.

## Adding a Security, Multi-Tenancy, or Auth Control

The `framework/security/`, `framework/multitenancy/`, `framework/auth/`, and `framework/adapters/` packages follow one convention: **provider-neutral and backward-compatible by default**. A new control must:

1. Do nothing (no-op / pass-through) unless explicitly configured — existing single-tenant/no-config callers and tests must not change behaviour.
2. Be independently unit-testable without a live LLM, database, or cluster (see `tests/unit/test_security_controls.py`, `test_tenant_context.py`, `test_api_keys.py` for the pattern).
3. Fail closed on malformed input (reject, don't silently ignore) but never crash the caller — catch and log at the boundary.
4. Keep cloud/vendor SDK imports inside a concrete adapter; core agents, orchestration, and governance depend on portable contracts.
5. Add normal, failure, tenant-boundary, and concurrency tests where applicable.

Before submitting, also run:

```bash
python -m unittest tests.chaos.test_failure_injection -v
python -m unittest tests.load.test_stress_load -v
python -m compileall -q framework agents scripts tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how these packages fit into the request path, and [docs/SECURITY.md](docs/SECURITY.md) for what is and isn't covered today.

## Code Style

- Python 3.11+
- Type annotations on all public functions
- Docstrings on classes and non-trivial functions
- `ruff` for formatting: `make lint-fix`
- `mypy` for types: `make typecheck`

## Licence

By contributing, you agree your contributions are licensed under the MIT Licence.
