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
pip install pyyaml requests   # minimum for tests
make test                     # must be 135/135 before you start
```

## Before Submitting a PR

1. **All 135 tests pass**: `make test`
2. **Your new code has tests**: unit tests at minimum, scenario tests if adding an agent
3. **No secrets in code**: API keys, passwords, tokens — use environment variables only
4. **Agent metadata is complete**: every agent has `METADATA`, `mystery_refs`, and `tags`
5. **observe() is a pure function**: no LLM calls, no I/O, no side effects in `observe()`

## Adding a New Agent

1. Create `agents/<domain>/<agent_name>.py`
2. Add `AgentMetadata` to `framework/agents/catalogue.py`
3. Inherit from `StructuralAgent` or `DecisionAgent`
4. Implement `observe()`, `reason()`, `parse_findings()`
5. Add unit tests in `tests/unit/`
6. Register in `scripts/server.py`

The `observe()` contract is strict: it must be a pure function. No LLM calls, no network I/O, no file I/O. It extracts and structures data from `task.payload` only. This makes every agent trivially unit-testable without mocking.

## Code Style

- Python 3.11+
- Type annotations on all public functions
- Docstrings on classes and non-trivial functions
- `ruff` for formatting: `make lint-fix`
- `mypy` for types: `make typecheck`

## Licence

By contributing, you agree your contributions are licensed under the MIT Licence.
