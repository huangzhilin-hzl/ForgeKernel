import shlex
import subprocess
import tomllib

import pytest

from forgekernel import codex_agent
from forgekernel.agents import agent_command
from forgekernel.loop.agent_policy import AgentPolicy
from forgekernel.loop.cli import RealBackend
from forgekernel.task import AgentSpec


def test_codex_turn_applies_the_policy_for_each_role(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reference = tmp_path / "reference"
    policy = AgentPolicy(oracle_mode="source", repo_root=tmp_path,
                         ref_source_paths=(reference,))
    spec = AgentSpec(backend="codex", model="default", effort=None, timeout_s=10,
                     provider="native")
    backend = RealBackend("rmsnorm", workspace, agent_command(spec, "student"),
                          "test", "test@localhost", agent_policy=policy)
    source = tmp_path / "source" / "probe"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# probe\n\nInspect probe output.\n")
    monkeypatch.setattr(codex_agent.skills, "skill_dirs", lambda: [source])
    profiles = {}

    def turn(cmd, prompt, role):
        assert prompt.endswith("test prompt")
        profiles[role] = tomllib.loads(shlex.split(cmd)[-1])["permissions"]["forgekernel"]
        return 0, "test reply", ""

    monkeypatch.setattr(backend, "_agent_turn", turn)
    assert backend.run_agent("test prompt") == "test reply"
    assert backend.run_agent("test prompt", role="teacher") == "test reply"
    assert profiles["student"]["filesystem"][str(reference)] == "deny"
    assert str(reference) not in profiles["teacher"]["filesystem"]
    assert profiles["student"]["network"]["enabled"] is False
    assert profiles["teacher"]["network"]["enabled"] is True
    skill = workspace / ".agents/skills/probe/SKILL.md"
    assert skill.read_text().startswith('---\nname: "probe"\ndescription: "Inspect probe output."\n---\n')
    assert (source / "SKILL.md").read_text() == "# probe\n\nInspect probe output.\n"


def test_codex_protects_worktree_metadata_from_extra_write_roots(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@localhost", "-c", "core.hooksPath=/dev/null",
                    "commit", "-qm", "initial", "--allow-empty"], check=True)
    workspace = tmp_path / "workspace"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach",
                    str(workspace)], check=True)
    common = repo / ".git"
    metadata = common / "worktrees" / workspace.name
    policy = AgentPolicy(oracle_mode="source", repo_root=repo,
                         extra_allow_write=(workspace / ".agents/skills", metadata,
                                            common / "hooks"))

    profile = tomllib.loads(codex_agent.permission_profile(policy, "teacher", workspace))
    filesystem = profile["permissions"]["forgekernel"]["filesystem"]
    assert filesystem[str(repo)] == "write"
    for path in (workspace / ".git", workspace / ".agents", workspace / ".codex",
                 workspace / ".claude", metadata, common):
        assert filesystem[str(path)] == "read"
    assert str(common / "hooks") not in filesystem
    assert str(workspace / ".agents/skills") not in filesystem


def test_codex_fails_closed_for_online_url_restrictions(tmp_path):
    policy = AgentPolicy(oracle_mode="blackbox", repo_root=tmp_path,
                         ref_source_urls=("github.com/org/reference",))
    with pytest.raises(ValueError, match="URL or command restrictions"):
        codex_agent.permission_profile(policy, "teacher", tmp_path)
