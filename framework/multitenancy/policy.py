"""Provider-neutral tenant deployment and capability policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class TenantPolicy:
    """Policy inputs shared by isolation, residency, and capability adapters."""

    isolation_profile: str = "logical"
    allowed_regions: List[str] = field(default_factory=list)
    processing_region: str = ""
    storage_region: str = ""
    backup_region: str = ""
    allowed_subprocessors: List[str] = field(default_factory=list)
    cross_region_transfer_allowed: bool = False
    retention_policy: str = "standard"
    workflow_limits: Dict[str, int] = field(default_factory=dict)
    audit_profile: str = "standard"
    networking_profile: str = "standard"
    support_profile: str = "standard"

    def __post_init__(self) -> None:
        if self.isolation_profile not in {"logical", "physical"}:
            raise ValueError("isolation_profile must be 'logical' or 'physical'")
        if self.processing_region and self.allowed_regions and self.processing_region not in self.allowed_regions:
            raise ValueError("processing_region must be in allowed_regions")
        if self.storage_region and self.allowed_regions and self.storage_region not in self.allowed_regions:
            raise ValueError("storage_region must be in allowed_regions")
        if self.backup_region and not self.cross_region_transfer_allowed and self.storage_region and self.backup_region != self.storage_region:
            raise ValueError("backup_region cannot differ without cross-region transfer permission")