"""Role-based authorization policy over a normalized `Principal` (R11).

Closes the gap where a `Principal` was propagated but never actually
enforced: the task API's bearer-key auth answers "which tenant", this
module answers "is this principal allowed to perform this action".
"""

from __future__ import annotations

from typing import Iterable, Optional


class RoleAuthorizationError(PermissionError):
    """Raised when a principal lacks a role required for an action."""


class AuthorizationPolicy:
    """Minimal RBAC check: does the principal hold at least one required role?"""

    def require_any_role(self, principal, required_roles: Iterable[str], action: str) -> None:
        if principal is None:
            return  # no identity adapter configured — legacy tenant-only auth applies
        required = set(required_roles)
        if not required.intersection(principal.roles):
            raise RoleAuthorizationError(
                f"principal {principal.subject_id!r} lacks a required role for {action!r} "
                f"(has {list(principal.roles)!r}, needs one of {sorted(required)!r})"
            )
