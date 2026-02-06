"""QC IO error hierarchy.

This module defines a small set of typed exceptions used by the QC I/O subsystem.
All exceptions raised from `simpnmr.io.qc` should derive from `QCError`.

The intent is to make error handling predictable across backends (ORCA, Gaussian etc.)
while keeping error semantics explicit:
- UnsupportedFileError: cannot determine backend/file type.
- ParseError: file/backend recognized, but parsing failed due to unexpected format.
- DataNotFoundError: expected data item is not present.
- MissingSectionError: expected section/block is not present.
- ReaderContractError: developer error (implementation violated the gateway contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class QCError(Exception):
    """Base class for all QC I/O errors.

    Attributes:
        message: Human-readable error message.
        path: Optional path to the file being parsed.
        backend: Optional backend identifier (e.g., "orca", "gaussian").
        kind: Optional semantic kind (e.g., "spin", "hfc", "shield", "gtensor").
        section: Optional section identifier used by the backend (e.g., ORCA property).
        details: Optional free-form structured details for debugging/logging.
    """

    message: str
    path: str | Path | None = None
    backend: str | None = None
    kind: str | None = None
    section: str | None = None
    details: dict[str, Any] | None = None

    def format_context(self) -> str:
        """Return a compact context string for debugging/logging."""

        parts: list[str] = []
        if self.backend:
            parts.append(f"backend={self.backend}")
        if self.kind:
            parts.append(f"kind={self.kind}")
        if self.section:
            parts.append(f"section={self.section}")
        if self.path is not None:
            parts.append(f"path={self.path}")
        return " | ".join(parts)

    def to_log_extra(self) -> dict[str, Any]:
        """Return structured fields suitable for logging `extra=...`."""

        return {
            "backend": str(self.backend) if self.backend is not None else "",
            "kind": str(self.kind) if self.kind is not None else "",
            "section": str(self.section) if self.section is not None else "",
            "path": str(self.path) if self.path is not None else "",
        }

    def __str__(self) -> str:
        """Return the user-facing error message (ValueError-style)."""

        return self.message


@dataclass(frozen=True, slots=True)
class UnsupportedFileError(QCError):
    """Raised when the QC backend/file type cannot be determined."""


@dataclass(frozen=True, slots=True)
class ParseError(QCError):
    """Raised when the backend is known but the file is unexpected or malformed."""


@dataclass(frozen=True, slots=True)
class DataNotFoundError(QCError):
    """Raised when an expected data item is missing from an otherwise valid file."""


@dataclass(frozen=True, slots=True)
class MissingSectionError(QCError):
    """Raised when an expected block is missing (e.g., ORCA property section)."""


@dataclass(frozen=True, slots=True)
class ReaderContractError(QCError):
    """Raised when a backend implementation violates the reader contract.

    This indicates a developer error: e.g., a backend class does not define a required
    attribute/method that the gateway expects.
    """


__all__ = [
    "QCError",
    "UnsupportedFileError",
    "ParseError",
    "DataNotFoundError",
    "MissingSectionError",
    "ReaderContractError",
]
