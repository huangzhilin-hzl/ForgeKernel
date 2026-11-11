"""Headless Codex launcher and per-turn sandbox configuration."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

from forgekernel import skills
from forgekernel.loop.agent_policy import AgentPolicy, HOOK_SCRIPT, _git_exclude, _git_metadata_dirs
from forgekernel.paths import repo_root

MODULE = "forgekernel.codex_agent"
PROFILE = "forgekernel"
SKILLS_PATH = Path(".agents/skills")
CONTROL_DIRS = (".git", ".agents", ".codex", ".claude")


def matches(command: str) -> bool:
    return f" -m {MODULE} " in command


def _string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def permission_profile(policy: AgentPolicy, role: str, workspace: Path) -> str:
    # A source-mode teacher submits and profiles reference GPU jobs. The student
    # uses the driver's job results and can run entirely offline.
    offline = policy.offline(role)
    if not offline and (policy.no_cluster(role) or policy.deny_urls(role)):
        raise ValueError('This agent role needs URL or command restrictions supported by '
                         'Claude Code. Select backend = "claude" for this role.')
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    protected = [repo_root() / "local.env", codex_home.expanduser().resolve()]
    workspace = workspace.resolve()
    writable = [path.resolve() for path in policy.allow_write(role)]
    read_only = {HOOK_SCRIPT, *policy._deny_write_paths(workspace)}
    for root in (workspace, *writable):
        for name in CONTROL_DIRS:
            path = root / name
            read_only.update((path, path.resolve()))
        read_only.update(_git_metadata_dirs(root))

    filesystem: dict[str, str] = {":root": "read"}
    filesystem.update((str(path), "write") for path in writable
                      if not any(path.is_relative_to(protected) for protected in read_only))
    filesystem.update((str(path), "read") for path in sorted(read_only))
    filesystem.update((str(path), "deny") for path in [*policy._deny_paths(role, workspace),
                                                     *protected])
    entries = ", ".join(f"{_string(path)}={_string(access)}" for path, access in filesystem.items())
    return (f'permissions.{PROFILE}={{extends=":workspace", filesystem={{{entries}}}, '
            f'network={{enabled={str(not offline).lower()}}}}}')


def _prepare_skill(skill: Path) -> None:
    content = skill.read_text(encoding="utf-8")
    if content.startswith("---\n"):
        return
    name = skill.parent.name
    paragraph = content.partition("\n\n")[2].partition("\n\n")[0]
    description = " ".join(line.lstrip("> ") for line in paragraph.splitlines())
    header = f"---\nname: {_string(name)}\ndescription: {_string(description or name)}\n---\n"
    skill.write_text(header + content, encoding="utf-8")


def prepare(command: str, policy: AgentPolicy, role: str, workspace: Path) -> str:
    profile = permission_profile(policy, role, workspace)
    skills.mirror(workspace, relpath=SKILLS_PATH, prepare=_prepare_skill)
    _git_exclude(workspace, ".agents/")
    return f"{command} {shlex.quote(profile)}"


def main() -> None:
    if len(sys.argv) != 5 or os.environ.get("FK_AGENT_DRIVER") == "goal":
        sys.exit("ForgeKernel's Codex launcher requires a headless turn with a role policy")
    _, model, effort, timeout, profile = sys.argv
    argv = ["codex", "exec", "--ignore-user-config", "--strict-config",
            "--skip-git-repo-check", "--ephemeral",
            "--disable", "apps", "--disable", "plugins", "--disable", "hooks",
            "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
            "-c", "mcp_servers={}", "-c", f'default_permissions="{PROFILE}"', "-c", profile,
            "-c", 'shell_environment_policy.exclude=["OPENAI_API_KEY","ANTHROPIC_API_KEY",'
                  '"CLAUDE_CODE_OAUTH_TOKEN","FK_REF_KEY*"]']
    if model != "default":
        argv += ["--model", model]
    if effort:
        argv += ["-c", f"model_reasoning_effort={_string(effort)}"]
    helper = repo_root() / "scripts" / "timeout-run.py"
    os.execv(sys.executable, [sys.executable, str(helper), timeout, *argv, "-"])


if __name__ == "__main__":
    main()
