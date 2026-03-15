"""
framework/tools/base.py
========================
Tool infrastructure for ToolAgent subclasses.

Tools are discrete, named operations that agents can invoke:
  - ERP data fetchers (PO acknowledgement times, goods receipts)
  - Market data APIs (commodity prices, tariff schedules)
  - Document parsers (contract extraction, specification analysis)
  - External system connectors (SAP, Coupa, Oracle)

TOOL REGISTRY:
  A process-level registry of available tools.
  Tools are registered at startup and looked up by name.

TOOL INTERFACE:
  Every tool implements BaseTool with a single invoke() method.
  Tools are stateless after construction — safe for concurrent use.

ADDING A NEW TOOL (3 steps):
  1. Subclass BaseTool
  2. Define TOOL_NAME and invoke()
  3. Call ToolRegistry.register(MyTool())
"""

from __future__ import annotations

import abc
import logging
import threading
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseTool protocol
# ---------------------------------------------------------------------------

class BaseTool(abc.ABC):
    """
    Abstract base for all SKEIN framework tools.

    Thread-safety: tools must be stateless after construction.
    Multiple concurrent invoke() calls must be safe.
    """

    TOOL_NAME: str = ""          # Unique tool identifier — required on subclass
    DESCRIPTION: str = ""        # Human-readable tool description
    VERSION: str = "1.0.0"

    @abc.abstractmethod
    def invoke(self, arguments: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given arguments.

        Args:
            arguments: Key-value parameters for this tool.

        Returns:
            Tool output (any JSON-serialisable value).

        Raises:
            ToolInvocationError: On unrecoverable tool failure.
        """

    def validate_arguments(self, arguments: Dict[str, Any]) -> None:
        """
        Validate arguments before invoke(). Override in subclasses.
        Raises ValueError on invalid arguments.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.TOOL_NAME}, v={self.VERSION})"


# ---------------------------------------------------------------------------
# Tool exceptions
# ---------------------------------------------------------------------------

class ToolInvocationError(RuntimeError):
    """Raised when a tool call fails in a non-retryable way."""


class ToolNotFoundError(KeyError):
    """Raised when a requested tool is not in the registry."""


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Process-level registry for tools.

    Thread-safe: RLock protects all read/write operations.

    Usage:
        # Register at startup
        ToolRegistry.register(ERPConnectorTool())

        # Lookup by name (used by ToolAgent._execute_tool)
        tool = ToolRegistry.get("erp_po_acknowledgement")
        result = tool.invoke({"supplier_id": "SUP-001"})
    """

    _registry: Dict[str, BaseTool] = {}
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """Register a tool instance. Raises ValueError on duplicate name."""
        if not tool.TOOL_NAME:
            raise ValueError(f"{tool.__class__.__name__} must define TOOL_NAME")
        with cls._lock:
            if tool.TOOL_NAME in cls._registry:
                raise ValueError(
                    f"Tool '{tool.TOOL_NAME}' is already registered. "
                    f"Use ToolRegistry.replace() to update."
                )
            cls._registry[tool.TOOL_NAME] = tool
            log.info("Registered tool: %s v%s", tool.TOOL_NAME, tool.VERSION)

    @classmethod
    def replace(cls, tool: BaseTool) -> None:
        """Register or replace a tool (for hot-reload scenarios)."""
        if not tool.TOOL_NAME:
            raise ValueError(f"{tool.__class__.__name__} must define TOOL_NAME")
        with cls._lock:
            cls._registry[tool.TOOL_NAME] = tool

    @classmethod
    def get(cls, tool_name: str) -> Optional[BaseTool]:
        """Return tool by name, or None if not found."""
        with cls._lock:
            return cls._registry.get(tool_name)

    @classmethod
    def get_or_raise(cls, tool_name: str) -> BaseTool:
        """Return tool by name, or raise ToolNotFoundError."""
        tool = cls.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )
        return tool

    @classmethod
    def list_tools(cls) -> Dict[str, str]:
        """Return {tool_name: description} for all registered tools."""
        with cls._lock:
            return {name: t.DESCRIPTION for name, t in cls._registry.items()}

    @classmethod
    def clear(cls) -> None:
        """Remove all tools. Use in tests only."""
        with cls._lock:
            cls._registry.clear()


# ---------------------------------------------------------------------------
# Built-in stub tools (for testing and development)
# ---------------------------------------------------------------------------

class EchoTool(BaseTool):
    """
    Development stub tool that echoes its arguments back.
    Useful for testing ToolAgent subclasses without real connectors.
    """
    TOOL_NAME   = "echo"
    DESCRIPTION = "Returns its input arguments as output. Development use only."

    def invoke(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {"echoed": arguments, "tool": self.TOOL_NAME}


class ERPSupplierTransactionTool(BaseTool):
    """
    Stub ERP connector that returns synthetic supplier transaction data.
    Replace with real SAP/Oracle/Coupa connector in production.

    Real implementation would:
      - Connect to SAP via RFC or OData API
      - Query ME2M (purchase orders), EKBE (delivery), MSEG (goods movements)
      - Return normalised records matching the supplier_transactions schema
    """
    TOOL_NAME   = "erp_supplier_transactions"
    DESCRIPTION = "Fetch supplier transaction records from ERP system."

    def invoke(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        supplier_id = arguments.get("supplier_id", "UNKNOWN")
        months      = arguments.get("months", 6)
        log.info("ERP tool called for supplier=%s months=%d", supplier_id, months)
        # Stub: return empty — replace with real ERP connector
        return {
            "supplier_id": supplier_id,
            "months_requested": months,
            "records": [],
            "source": "stub — replace with real ERP connector",
        }
