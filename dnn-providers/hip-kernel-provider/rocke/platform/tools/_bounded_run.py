# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run a command under a time limit, and when the limit is hit, end everything
# the command started -- not just the command. Shared by the tools under this
# directory that give a compiler a time budget.
#
# Not a CLI entry point -- hence the leading underscore, as with _hostcaps.py.

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def run_bounded(
    argv: list[str], *, timeout: float, cwd: Path | str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `argv` to completion, capturing text output, within `timeout` seconds.

    Behaves like ``subprocess.run(argv, capture_output=True, text=True,
    check=False, timeout=timeout, cwd=cwd)`` with one difference: on timeout
    every process the command started is killed, then
    ``subprocess.TimeoutExpired`` is raised as usual.

    Why `subprocess.run` is not enough: it kills only the process it started.
    The clang driver does the compile in a child (`clang -cc1`), so a compile
    that never finishes -- llvm20 on `raw.ptr.buffer.load.async.lds` at -O0 is
    the live case -- leaves that child behind, orphaned and spinning at full
    CPU for as long as the host stays up. Every arch-domain sweep added one per
    hanging probe.

    The command therefore runs as the leader of its own process group (a new
    session on POSIX, a new process group on Windows), and a timeout ends the
    whole group. A descendant that deliberately leaves the group escapes this;
    clang does not.
    """
    if os.name == "nt":
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        group = {"start_new_session": True}

    with subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        **group,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:
            # Timeout, or an interrupt while waiting: either way nothing may
            # outlive this call.
            _kill_group(proc)
            proc.communicate()
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill `proc` and every process in the group it leads."""
    if os.name == "nt":
        # A Windows process group cannot be killed as a unit; `taskkill /T`
        # walks the tree from the leader instead.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        proc.kill()
        return
    try:
        # The leader of a new session leads its process group, so the group id
        # is the leader's pid. The group outlives the leader as long as any
        # member is alive, which is exactly the case this exists for.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # every member had already exited
