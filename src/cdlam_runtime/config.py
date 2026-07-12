"""Load a path-portable CD-LAM runtime profile."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a runtime profile is incomplete or malformed."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated accessors for one JSON runtime profile."""

    profile_path: Path
    workspace: Path
    document: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> RuntimeConfig:
        profile = Path(path).expanduser().resolve()
        if not profile.is_file():
            raise ConfigError(f"runtime profile is missing: {profile}")
        try:
            document = json.loads(profile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"invalid runtime profile {profile}: {exc}") from exc
        if not isinstance(document, dict):
            raise ConfigError("runtime profile must contain a JSON object")
        if document.get("schema_version") != 1:
            raise ConfigError("runtime profile schema_version must be 1")
        workspace_value = document.get("workspace", "..")
        if not isinstance(workspace_value, str) or not workspace_value.strip():
            raise ConfigError("workspace must be a non-empty path string")
        workspace = Path(os.path.expandvars(workspace_value)).expanduser()
        if not workspace.is_absolute():
            workspace = profile.parent / workspace
        return cls(profile, workspace.resolve(), document)

    def table(self, name: str) -> dict[str, Any]:
        value = self.document.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"runtime profile section {name!r} must be an object")
        return value

    @staticmethod
    def _expected_name(expected: type | tuple[type, ...]) -> str:
        if isinstance(expected, tuple):
            return " or ".join(item.__name__ for item in expected)
        return expected.__name__

    def value(self, section: str, name: str, expected: type | tuple[type, ...]) -> Any:
        value = self.table(section).get(name)
        if isinstance(value, bool) and expected is int:
            raise ConfigError(
                f"{section}.{name} must be {self._expected_name(expected)}"
            )
        if not isinstance(value, expected):
            raise ConfigError(
                f"{section}.{name} must be {self._expected_name(expected)}"
            )
        return value

    def optional_value(
        self,
        section: str,
        name: str,
        expected: type | tuple[type, ...],
        default: Any,
    ) -> Any:
        value = self.table(section).get(name, default)
        if isinstance(value, bool) and expected is int:
            raise ConfigError(
                f"{section}.{name} must be {self._expected_name(expected)}"
            )
        if not isinstance(value, expected):
            raise ConfigError(
                f"{section}.{name} must be {self._expected_name(expected)}"
            )
        return value

    def path(self, name: str, *, required: bool = True) -> Path | None:
        value = self.table("paths").get(name)
        if value in (None, "") and not required:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"paths.{name} must be a non-empty path string")
        expanded = Path(os.path.expandvars(value)).expanduser()
        if not expanded.is_absolute():
            expanded = self.workspace / expanded
        # A virtual-environment Python is commonly a symlink to its base
        # interpreter. Dereferencing that final component discards the venv's
        # pyvenv.cfg and silently launches the host environment instead.
        if name in {"python", "torchrun"}:
            return Path(os.path.abspath(expanded))
        return expanded.resolve()

    def positive_int(self, section: str, name: str, default: int) -> int:
        value = self.optional_value(section, name, int, default)
        if value < 1:
            raise ConfigError(f"{section}.{name} must be positive")
        return value

    def nonnegative_int(self, section: str, name: str, default: int) -> int:
        value = self.optional_value(section, name, int, default)
        if value < 0:
            raise ConfigError(f"{section}.{name} must be nonnegative")
        return value

    def positive_float(self, section: str, name: str, default: float) -> float:
        value = self.optional_value(section, name, (int, float), default)
        result = float(value)
        if result <= 0:
            raise ConfigError(f"{section}.{name} must be positive")
        return result
