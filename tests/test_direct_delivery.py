"""forgekernel.submit.direct: local, SSH and Kubernetes GPU job delivery."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from forgekernel.submit import cli as submit_cli
from forgekernel.submit import jobs as ds
from forgekernel.submit import direct
from forgekernel.submit.config import load_config

ENTRY_OK = """#!/bin/bash
echo "entry task=$1 gpu=$CUDA_VISIBLE_DEVICES timeout=$TASK_TIMEOUT key=${FK_REF_KEY:-none}"
ls
echo "task=$1 finished"
"""


def _git(d: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


def _op(base: Path, name: str = "op", *, delivery: str = "local", entry: str = ENTRY_OK,
        timeout: int = 60, exec_table: str = 'gpu = "2"\n', env: str = "") -> Path:
    d = base / "operators" / name
    d.mkdir(parents=True)
    (d / "gpu_entry.sh").write_text(entry)
    (d / "kernel.py").write_text("x = 1\n")
    (d / "submit.toml").write_text(
        f'[job]\ntimeout = {timeout}\n[repo]\ndelivery = "{delivery}"\nbranch = "main"\n'
        f"[exec]\n{exec_table}[env]\nA = \"1\"\n{env}")
    _git(d, "init", "-q", "-b", "main")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    return d


@pytest.fixture(autouse=True)
def _locks(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))


def _run(cfg, mode="perf", rid="1"):
    ds.jobs_dir(Path(cfg.op_dir)).joinpath(rid).mkdir(parents=True, exist_ok=True)
    rc = ds.runner_main(cfg, rid, mode, cfg.branch)
    log = ds.log_path(Path(cfg.op_dir), rid).read_text()
    return rc, ds.job_status(Path(cfg.op_dir), rid)["status"], log


# -- config ---------------------------------------------------------------------------------


def test_a_local_tree_needs_no_cluster_fields(tmp_path):
    _op(tmp_path)
    cfg = load_config("op", base_dir=tmp_path)
    assert (cfg.delivery, cfg.exec_gpu, cfg.timeout, cfg.branch) == ("local", "2", 60, "main")
    assert cfg.env == {"A": "1"}


def test_an_ssh_tree_needs_a_host_and_an_absolute_workdir(tmp_path):
    _op(tmp_path, delivery="ssh", exec_table='host = "box"\nworkdir = "rel"\n')
    with pytest.raises(ValueError, match="ssh delivery needs"):
        load_config("op", base_dir=tmp_path)


# -- local ----------------------------------------------------------------------------------


def test_local_run_succeeds_on_a_fresh_tree(tmp_path):
    d = _op(tmp_path)
    stale = d / direct.LOCAL_TREE / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("left over")
    rc, status, log = _run(load_config("op", base_dir=tmp_path))
    assert (rc, status) == (0, ds.STATUS_SUCCEEDED)
    assert "entry task=perf gpu=2 timeout=60 key=none" in log
    assert ds.BEGIN_MARKER in log and "stale.txt" not in log and "kernel.py" in log
    assert log.rstrip().endswith("task=perf finished")


def test_local_run_failure_is_failed(tmp_path):
    _op(tmp_path, entry="#!/bin/bash\necho boom\nexit 3\n")
    rc, status, log = _run(load_config("op", base_dir=tmp_path))
    assert (rc, status) == (1, ds.STATUS_FAILED)
    assert "exited rc=3" in log


def test_local_run_past_its_deadline_is_killed(tmp_path, monkeypatch):
    _op(tmp_path, entry="#!/bin/bash\necho start\nsleep 30\n", timeout=1)
    monkeypatch.setattr(direct, "RUN_HEADROOM_S", 0)
    t0 = time.monotonic()
    rc, status, log = _run(load_config("op", base_dir=tmp_path))
    assert status == ds.STATUS_FAILED and "exceeded" in log
    assert time.monotonic() - t0 < 20


def test_a_stopped_local_job_ends_its_whole_session(tmp_path):
    """gpu_entry's `timeout` moves the task into a process group of its own, so stopping
    the job's group would leave the task running on the GPU: the whole session goes."""
    entry = ('#!/bin/bash\n'
             'python3 -c "import os, time; os.setpgid(0, 0); time.sleep(47.3)" &\nwait\n')
    _op(tmp_path, entry=entry)
    cfg = load_config("op", base_dir=tmp_path)

    class Interrupted(subprocess.Popen):
        def wait(self, timeout=None):
            if timeout is not None and not getattr(self, "_hit", False):
                self._hit = True
                time.sleep(1.0)            # started, and its child left the group
                raise KeyboardInterrupt
            return super().wait(timeout)

    runner = direct.DirectRunner(cfg, popen=Interrupted)
    with open(tmp_path / "log.txt", "w") as lf, pytest.raises(KeyboardInterrupt):
        runner._run_local(direct._payload(cfg, "main"), "main", "perf", lf)
    left = subprocess.run(["pgrep", "-f", r"time\.sleep\(47\.3\)"], capture_output=True,
                          text=True).stdout
    assert left.strip() == ""


def test_cmd_mode_reaches_the_entry_cmd_task(tmp_path):
    _op(tmp_path, entry='#!/bin/bash\necho "args: $*"\n')
    _, status, log = _run(load_config("op", base_dir=tmp_path), mode="cmd:python3 probe.py")
    assert status == ds.STATUS_SUCCEEDED and "args: cmd python3 probe.py" in log


def test_the_gpu_lock_makes_a_second_run_wait(tmp_path, monkeypatch):
    calls = []

    class _Log:
        def write(self, s):
            calls.append(s)

        def flush(self):
            pass

    got = threading.Event()

    def _second():
        with direct.gpu_lock("0", _Log(), poll_s=0.05):
            got.set()

    with direct.gpu_lock("0"):
        t = threading.Thread(target=_second)
        t.start()
        time.sleep(0.3)
        assert not got.is_set()
    t.join(timeout=5)
    assert got.is_set()
    assert any("in use by another forgekernel run" in c for c in calls)


# -- ssh ------------------------------------------------------------------------------------


class _FakeSsh:
    """Stands in for subprocess.run / Popen: records argv, plays a scripted remote."""

    def __init__(self, outcomes, *, remote_exit=True):
        self.outcomes = list(outcomes)   # per exec: (rc, began)
        self.runs: list = []
        self.execs: list = []
        self.stdin_text = ""
        self.remote_exit = remote_exit

    def run(self, argv, input=None, capture_output=False, timeout=None):
        self.runs.append((argv, input))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, stdout=None, stderr=None, text=None, stdin=None, **kw):
        rc, began = self.outcomes.pop(0)
        if began:
            stdout.write(f"{ds.BEGIN_MARKER}\nremote ok\n")
            if argv[0] == "kubectl" and self.remote_exit:
                stdout.write(f"{direct.REMOTE_EXIT_MARKER}\n")
        else:
            stdout.write("ssh: connect to host box port 22: Connection refused\n")
        stdout.flush()
        self.execs.append(argv)
        fake = self

        class _P:
            pid = 0

            class stdin:  # noqa: N801 -- mimics Popen.stdin
                @staticmethod
                def write(s):
                    fake.stdin_text += s

                @staticmethod
                def close():
                    pass

            @staticmethod
            def wait(timeout=None):
                return rc
        return _P()


def _ssh_cfg(tmp_path, env=""):
    _op(tmp_path, delivery="ssh",
        exec_table='host = "box"\nworkdir = "/data/fk"\ngpu = "1"\nenv_file = "/opt/env.sh"\n',
        env=env)
    return load_config("op", base_dir=tmp_path)


def _ssh_run(cfg, fake):
    log = Path(cfg.op_dir) / "log.txt"
    with open(log, "a+", encoding="utf-8") as lf:
        status = direct.DirectRunner(cfg, run=fake.run, popen=fake.popen,
                                     sleep=lambda s: None).run_mode("main", "perf", lf)
    return status, log.read_text()


def test_ssh_run_syncs_the_tree_then_runs_under_the_host_gpu_lock(tmp_path):
    cfg = _ssh_cfg(tmp_path, env='FK_REF_KEY = "s3cret"\n')
    fake = _FakeSsh([(0, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_SUCCEEDED
    tree = f"/data/fk/{tmp_path.name}/op"
    (argv, payload), = fake.runs
    assert argv[0] == "ssh" and "box" in argv and f"tar xJ -C {tree}" in argv[-1]
    assert payload[:6] == b"\xfd7zXZ\x00"              # xz-compressed git archive
    remote = fake.execs[0][-1]
    assert remote.startswith(f"cd {tree}") and ". /opt/env.sh" in remote
    assert "export CUDA_VISIBLE_DEVICES=1" in remote and "export A=1" in remote
    assert "setsid --wait sh -c" in remote and "/data/fk/.gpu1.lock bash gpu_entry.sh perf" in remote
    assert "s3cret" not in remote and fake.stdin_text == "s3cret\n"


def test_ssh_that_never_delivers_is_a_transport_error_after_retries(tmp_path):
    cfg = _ssh_cfg(tmp_path)
    fake = _FakeSsh([(255, False)] * direct.TRANSPORT_ATTEMPTS)
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED
    assert ds.TRANSPORT_ERROR_MARKER in log
    assert len(fake.execs) == direct.TRANSPORT_ATTEMPTS


def test_ssh_transport_blip_then_success(tmp_path):
    cfg = _ssh_cfg(tmp_path)
    status, log = _ssh_run(cfg, _FakeSsh([(255, False), (0, True)]))
    assert status == ds.STATUS_SUCCEEDED and "retrying" in log


def test_ssh_payload_failure_is_charged_without_retry(tmp_path):
    cfg = _ssh_cfg(tmp_path)
    fake = _FakeSsh([(1, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED and "exited rc=1" in log
    assert ds.TRANSPORT_ERROR_MARKER not in log and len(fake.execs) == 1


def _kubectl_cfg(tmp_path, env=""):
    _op(tmp_path, delivery="kubectl", env=env,
        exec_table='kubeconfig = "/local/kube config"\nnamespace = "ml"\npod = "gpu-worker"\n'
                   'container = "workspace"\nworkdir = "/data/fk"\ngpu = "1"\n')
    return load_config("op", base_dir=tmp_path)


@pytest.fixture
def kube_context(monkeypatch):
    run = subprocess.run
    selected = {"context": "dev", "server": "https://dev.example.invalid"}

    def config_view(argv, **kwargs):
        if argv[0] != "kubectl":
            return run(argv, **kwargs)
        assert argv[-5:] == ["config", "view", "--minify", "-o", "json"]
        result = {"current-context": selected["context"],
                  "clusters": [{"cluster": {"server": selected["server"]}}]}
        return subprocess.CompletedProcess(argv, 0, json.dumps(result), "")

    monkeypatch.setattr(subprocess, "run", config_view)
    return selected


def test_kubectl_streams_archive_and_secret_on_the_pinned_target(tmp_path, monkeypatch, kube_context):
    cfg = _kubectl_cfg(tmp_path, env='FK_REF_KEY = "synthetic-key"\n')
    fake = _FakeSsh([(0, True)])
    direct_runner = direct.DirectRunner

    def upload(argv, **kwargs):
        kube_context.update(context="prod", server="https://prod.example.invalid")
        return fake.run(argv, **kwargs)

    monkeypatch.setattr(direct, "DirectRunner",
                        lambda c: direct_runner(c, run=upload, popen=fake.popen))
    _, status, _ = _run(cfg)
    assert status == ds.STATUS_SUCCEEDED
    prefix = ["kubectl", "--kubeconfig", "/local/kube config", "--context", "dev",
              "--server", "https://dev.example.invalid", "--namespace", "ml",
              "exec", "gpu-worker", "--container", "workspace"]
    (upload, payload), = fake.runs
    assert upload[:-1] == [*prefix, "--stdin", "--", "sh", "-c"]
    assert "tar xJ -C " in upload[-1] and payload[:6] == b"\xfd7zXZ\x00"
    assert fake.execs[0][:-1] == [*prefix, "--stdin", "--", "sh", "-c"]
    assert "export CUDA_VISIBLE_DEVICES=1" in fake.execs[0][-1]
    assert "synthetic-key" not in fake.execs[0][-1] and fake.stdin_text == "synthetic-key\n"
    expected = "kubectl:https://dev.example.invalid:dev:ml/gpu-worker:workspace:gpu=1"
    assert ds.job_status(Path(cfg.op_dir), "1")["cluster"] == expected
    assert cfg.cluster == expected
    new_cfg = load_config("op", base_dir=tmp_path)
    assert new_cfg.cluster == "kubectl:https://prod.example.invalid:prod:ml/gpu-worker:workspace:gpu=1"
    new_cfg.exec_gpu = "2"
    assert new_cfg.cluster.endswith(":gpu=2")


def test_kubectl_stream_loss_after_start_is_a_transport_failure(tmp_path, kube_context):
    cfg = _kubectl_cfg(tmp_path)
    fake = _FakeSsh([(1, True)], remote_exit=False)
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_FAILED and ds.TRANSPORT_ERROR_MARKER in log
    assert len(fake.execs) == 1
    assert "--stdin" not in fake.execs[0]
    cleanup, _ = fake.runs[-1]
    assert cleanup[:7] == fake.execs[0][:7]
    assert "pkill -TERM" in cleanup[-1]


def test_kubectl_remote_exit_is_a_kernel_failure(tmp_path, kube_context):
    status, log = _ssh_run(_kubectl_cfg(tmp_path), _FakeSsh([(7, True)]))
    assert status == ds.STATUS_FAILED and "exited rc=7" in log
    assert ds.TRANSPORT_ERROR_MARKER not in log


def test_missing_kubectl_is_reported_as_transport_failure(tmp_path, monkeypatch):
    cfg = _kubectl_cfg(tmp_path)
    run = subprocess.run

    def missing(argv, **kwargs):
        if argv[0] == "kubectl":
            raise FileNotFoundError(2, "No such file or directory", "kubectl")
        return run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", missing)
    _, status, log = _run(cfg, "calibrate")
    assert status == ds.STATUS_FAILED
    assert ds.TRANSPORT_ERROR_MARKER in log
    assert "cannot resolve kubectl context" in log and "No such file or directory" in log


@pytest.mark.parametrize("phase", ["upload", "job"])
def test_kubectl_spawn_errors_are_transport_failures(tmp_path, kube_context, phase):
    fake = _FakeSsh([(0, True)])

    def spawn_error(*args, **kwargs):
        raise PermissionError("permission denied")

    if phase == "upload":
        fake.run = spawn_error
    else:
        fake.popen = spawn_error
    status, log = _ssh_run(_kubectl_cfg(tmp_path), fake)
    assert status == ds.STATUS_FAILED
    assert f"{ds.TRANSPORT_ERROR_MARKER} cannot start kubectl {phase}:" in log


# -- the submit facade ----------------------------------------------------------------------


def test_submit_status_logs_route_through_the_job_files(tmp_path, monkeypatch, capsys):
    _op(tmp_path)
    cfg = load_config("op", base_dir=tmp_path)

    def _spawn(c, mode, branch):
        rid = ds.alloc_job_id(Path(c.op_dir))
        ds.runner_main(c, rid, mode, branch)
        return rid

    monkeypatch.setattr(ds, "spawn_runner", _spawn)
    assert submit_cli.main(["op", "submit", "--mode", "perf", "--base-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    job = out.split("job=")[1].split()[0]
    assert submit_cli.main(["op", "status", job, "--base-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == ds.STATUS_SUCCEEDED
    assert submit_cli.main(["op", "logs", job, "--base-dir", str(tmp_path)]) == 0
    assert "task=perf finished" in capsys.readouterr().out.splitlines()
    assert cfg.delivery == "local"


def test_submit_dry_run_says_where_it_would_run(tmp_path, capsys):
    _op(tmp_path)
    assert submit_cli.main(["op", "submit", "--dry-run", "--base-dir", str(tmp_path)]) == 0
    assert "local delivery: would run gpu_entry all" in capsys.readouterr().out


def test_kubectl_dry_run_names_pod_without_resolving_context(tmp_path, monkeypatch, capsys):
    cfg = _kubectl_cfg(tmp_path)

    def unexpected_resolution(cfg):
        pytest.fail("dry run must not need kubectl")

    monkeypatch.setattr(type(cfg), "kubectl_target", property(unexpected_resolution))
    assert submit_cli.main(["op", "submit", "--dry-run", "--base-dir", str(tmp_path)]) == 0
    assert "on ml/gpu-worker:workspace (GPU 1)" in capsys.readouterr().out
