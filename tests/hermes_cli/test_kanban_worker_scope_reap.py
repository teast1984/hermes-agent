"""A finished Kanban run's transient systemd scope must not outlive the run.

``_restart_safe_worker_argv`` wraps a dispatched worker in
``hermes-worker-kanban-<task>-run-<run>.scope`` (``outlives_parent=True``) and
discards the unit name. When the run finishes, a descendant that survived the
worker's exit — a double-forked Python ``http.server`` reparented into the
cgroup — keeps the scope active, so ``systemd-run --collect`` never collects it
and the server stays bound on 0.0.0.0. The dispatcher tracks only the worker
PID, so nothing ever stops the scope: the terminal-worker reaper must tear a
closed run's scope down after the grace window, mirroring how
``ProcessRegistry.kill_process`` stops a scope on the kill path.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def scope_teardown(monkeypatch):
    """A systemd unit-hosted dispatcher: the only context in which a worker is
    scope-wrapped (``restart_safe_gateway_child_argv`` requires INVOCATION_ID).
    Records every ``_stop_systemd_unit`` call so the test can assert teardown."""
    monkeypatch.setenv("INVOCATION_ID", "test-dispatch-unit")
    stopped: list[str] = []
    monkeypatch.setattr(
        "tools.process_registry._stop_systemd_unit",
        lambda unit: stopped.append(unit) or True,
    )
    return stopped


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _finished_run(conn, proc, *, ended_ago: int = 600) -> tuple[str, int]:
    """A host-local run closed ``ended_ago`` seconds ago, retaining its worker's
    evidence (pid + spawn fingerprint) exactly as a real completion leaves it."""
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def test_finished_run_scope_is_torn_down_when_worker_already_exited(conn, scope_teardown):
    """The run finished and the worker exited normally; the scope must still be
    torn down so a descendant that survived the exit cannot keep the scope (and
    its 0.0.0.0-bound server) alive. Today the reaper clears the dead worker's
    evidence and never touches the scope."""
    worker = _sleeper()
    try:
        tid, run_id = _finished_run(conn, worker)
        worker.kill()
        worker.wait()

        assert kbd.reap_terminal_workers(conn) == []
        assert scope_teardown == [f"hermes-worker-kanban-{tid}-run-{run_id}.scope"]
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_fresh_finished_run_scope_is_left_alone_until_grace_passes(conn, scope_teardown):
    """Scope teardown rides the same grace window as the worker signal: a run
    closed moments ago may still have its worker finalising INSIDE the scope, so
    stopping the scope (which kills every process in the cgroup) must wait —
    the worker PID is signalled first, never the scope."""
    worker = _sleeper()
    try:
        tid, run_id = _finished_run(conn, worker, ended_ago=0)

        assert kbd.reap_terminal_workers(conn) == []
        assert scope_teardown == [], "a finalizing worker's scope must not be stopped"

        conn.execute(
            "UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
            (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id),
        )
        assert kbd.reap_terminal_workers(conn) == [tid]
        assert scope_teardown == [f"hermes-worker-kanban-{tid}-run-{run_id}.scope"]
    finally:
        worker.kill()
        worker.wait()