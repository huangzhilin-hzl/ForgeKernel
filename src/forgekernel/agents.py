"""Build the headless agent commands from a task's per-role settings."""

from __future__ import annotations

import shlex
import sys

from forgekernel import provider as provider_mod
from forgekernel.codex_agent import MODULE as CODEX_MODULE
from forgekernel.paths import repo_root

AGENT_SCRIPT = "scripts/agent-claude.sh"


def agent_command(spec, role: str, env: dict | None = None) -> str:
    """The shell command the loop runs for one ``role`` turn (prompt on stdin). A provider
    other than ``native`` rides as environment assignments in front of it: the endpoint and
    the name of the variable that holds the key, never the key."""
    grant = provider_mod.grant(spec.backend, spec.provider, env)
    if spec.backend == "claude":
        argv = ["bash", str(repo_root() / AGENT_SCRIPT)]
    elif spec.backend == "codex":
        argv = [sys.executable, "-m", CODEX_MODULE]
    else:
        raise ValueError(f"backend {spec.backend!r} is not supported")
    argv += [spec.model, spec.effort or "", str(spec.timeout_s)]
    prefix = ["env", *(f"{k}={v}" for k, v in grant.items())] if grant else []
    return " ".join(shlex.quote(a) for a in [*prefix, *argv])
