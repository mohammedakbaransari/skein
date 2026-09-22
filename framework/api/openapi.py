"""Static OpenAPI 3.x contract for the task-submission API (R8).

Kept hand-written and small deliberately — this is the authoritative,
versioned contract callers/SDKs build against; a contract test asserts
every route the live handler actually serves is declared here (and vice
versa) so an undeclared breaking change fails CI instead of shipping.
"""

from __future__ import annotations

from typing import Any, Dict

OPENAPI_VERSION = "1.0.0"

_TASK_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "succeeded": {"type": "boolean"},
        "findings": {"type": "array"},
        "error": {"type": ["string", "null"]},
    },
}

ROUTES: Dict[str, Dict[str, Any]] = {
    ("POST", "/v1/tasks"): {
        "summary": "Submit a task for synchronous execution",
        "responses": {"200": _TASK_RESULT_SCHEMA, "400": {}, "401": {}, "403": {}, "413": {}, "429": {}, "422": {}},
    },
    ("POST", "/v1/tasks/async"): {
        "summary": "Submit a task for asynchronous execution",
        "responses": {"202": {"type": "object", "properties": {"job_id": {"type": "string"}}}},
    },
    ("GET", "/v1/tasks/async/{job_id}"): {
        "summary": "Poll an asynchronously submitted task",
        "responses": {"200": _TASK_RESULT_SCHEMA, "202": {}, "404": {}},
    },
    ("POST", "/v1/workflows"): {
        "summary": "Submit and execute a workflow DAG",
        "responses": {"200": {"type": "object"}, "400": {}, "401": {}, "403": {}, "422": {}},
    },
    ("POST", "/v1/workflows/async"): {
        "summary": "Submit a workflow DAG for asynchronous execution",
        "responses": {"202": {"type": "object", "properties": {"job_id": {"type": "string"}}}, "400": {}, "401": {}, "403": {}},
    },
    ("GET", "/v1/findings"): {
        "summary": "Query persisted findings for a tenant",
        "responses": {"200": {"type": "object"}, "400": {}, "401": {}, "403": {}, "404": {}},
    },
    ("POST", "/v1/findings/{finding_id}/review"): {
        "summary": "Transition a finding's human-review state",
        "responses": {"200": {}, "400": {}, "404": {}},
    },
}


def build_openapi_spec() -> Dict[str, Any]:
    paths: Dict[str, Any] = {}
    for (method, path), route in ROUTES.items():
        paths.setdefault(path, {})[method.lower()] = {
            "summary": route["summary"],
            "responses": {code: {"description": code} for code in route["responses"]},
        }
    return {
        "openapi": "3.0.3",
        "info": {"title": "SKEIN Task API", "version": OPENAPI_VERSION},
        "paths": paths,
    }
