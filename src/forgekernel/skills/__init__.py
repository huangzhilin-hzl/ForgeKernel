"""The repository's skills/ as one importable package.

A skill is a directory under the repository's ``skills/``: a ``SKILL.md`` saying what it is
and how an agent uses it, and the scripts it ships (as in ForgeTrain). The harness uses the
same scripts as a library -- it parses their verdict lines, and the driver seeds them into a
tree's ``notes/`` so a GPU job runs them as ``python3 notes/<tool>.py`` -- so every skill
directory is on this package's path: ``from forgekernel.skills import sass_tools``.

Inside a tree the scripts sit side by side in ``notes/`` and import each other bare
(``import sass_tools``); here the bare import fails and each falls back to this package.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from forgekernel.paths import skills_dir

SKILL_FILE = "SKILL.md"
# Where Claude Code looks for a project's skills, relative to the agent's working directory.
MIRROR_RELPATH = Path(".claude") / "skills"


def skill_dirs() -> list[Path]:
    root = skills_dir()
    return sorted(d for d in root.iterdir() if (d / SKILL_FILE).is_file()) if root.is_dir() else []


def mirror(workspace: Path, *, relpath: Path = MIRROR_RELPATH,
           prepare: Callable[[Path], None] | None = None) -> list[str]:
    """Copy each skill into the agent's workspace, replacing the previous copy."""
    if relpath.is_absolute() or not relpath.parts or ".." in relpath.parts:
        raise ValueError(f"skill mirror path must be relative to the workspace: {relpath}")
    sources = skill_dirs()
    if not sources:
        return []
    workspace = Path(workspace).resolve()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        # Pin each directory so a workspace symlink cannot redirect host-side writes.
        dst_fd = os.open(workspace, flags)
        stack.callback(os.close, dst_fd)
        for part in relpath.parts:
            try:
                os.mkdir(part, dir_fd=dst_fd)
            except FileExistsError:
                pass
            try:
                dst_fd = os.open(part, flags, dir_fd=dst_fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ValueError(f"unsafe skill mirror directory: {workspace / relpath}") from exc
                raise
            stack.callback(os.close, dst_fd)

        existing = set()
        for source in sources:
            try:
                info = os.stat(source.name, dir_fd=dst_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"unsafe skill mirror directory: {workspace / relpath / source.name}")
            existing.add(source.name)

        # Prepare complete copies before replacing any existing directory.
        stage = Path(stack.enter_context(tempfile.TemporaryDirectory(
            dir=workspace, prefix=".forgekernel-skills-")))
        for source in sources:
            copy = stage / source.name
            shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            if prepare is not None:
                prepare(copy / SKILL_FILE)
        for source in sources:
            if source.name in existing:
                shutil.rmtree(source.name, dir_fd=dst_fd)
            os.rename(stage / source.name, source.name, dst_dir_fd=dst_fd)
    return [source.name for source in sources]


__path__ = [str(d) for d in skill_dirs()]
