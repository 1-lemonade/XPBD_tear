"""Host package orchestrator (tools/u4_joint.py): process monitoring and the
lock, tested on whatever machine runs them (the host package runs this file on Windows: the real process-tree
memory monitor is exercised there). No platform-specific skips: a skip stops the host package.

  monitor  a probe and its child (32 + 128 MiB, touched) are both counted; a sampling failure, or a sample that
           counts no process while the step runs, ends the step ("memory monitor unavailable"), never a degraded mode;
           the memory cap ends the whole tree
  tests    the host tests gate: exit 0, readable junit, no failure / error / skip, the expected count
  lock     src/ = SHA256SUMS and SHA256SUMS + package files = BASELINE.json; any change is reported
  walk      the Windows Toolhelp process walk, simulated: only ERROR_NO_MORE_FILES ends it; a failed first call,
           FALSE with any other code (0 included) or a walk without the running step process raises, and the step stops
  hang      a host test that runs too long writes every thread's stack into the step log and goes on
"""
import importlib.util
import json
import os
import pathlib
import sys
import time

import pytest

WORK = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("u4_joint_under_test", WORK / "tools" / "u4_joint.py")
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
SLEEPER = [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
PROBE = [sys.executable, "-B", str(WORK / "tools" / "u4_joint.py"), "--monitor-probe", "parent"]


def _run(tmp_path, cmd, **kw):
    opts = dict(cwd=tmp_path, env=ENV, timeout=60, mem_cap=4 << 30, poll=0.2)
    opts.update(kw)
    return J.run_step("t", cmd, tmp_path / "step.log", **opts)


# ------------------------------------------------------------------ memory monitor
def test_monitor_counts_the_probe_and_its_child(tmp_path):
    r = _run(tmp_path, PROBE)
    ok, detail = J.monitor_preflight(r)
    assert ok, (detail, r["monitor_error"], r["killed"], (tmp_path / "step.log").read_text()[-2000:])
    assert r["max_tree_processes"] >= 2 and r["peak_tree_rss_bytes"] >= (J.PROBE_CHILD_MIB << 20)
    assert r["exit"] == 0 and r["monitor_samples"] >= 3


@pytest.mark.parametrize("failure", ["raises", "counts_nothing"])
def test_an_unmeasurable_tree_stops_the_step(failure, tmp_path, monkeypatch):
    def broken(proc):
        if failure == "raises":
            raise OSError("injected: the process tree cannot be read")
        return 0, 0, 0                                         # a sample that sees no process while it runs
    monkeypatch.setattr(J, "tree_rss", broken)
    t0 = time.monotonic()
    r = _run(tmp_path, SLEEPER)
    assert r["killed"] is not None and r["killed"].startswith("memory monitor unavailable"), r["killed"]
    assert r["monitor_error"] and r["monitor_samples"] == 0
    assert r["exit"] is not None and r["exit"] != 0 and time.monotonic() - t0 < 30   # ended, not waited out
    assert not J.monitor_preflight(r)[0]


def test_the_memory_cap_ends_the_tree(tmp_path):
    t0 = time.monotonic()
    r = _run(tmp_path, PROBE, mem_cap=64 << 20)                # the 128 MiB child exceeds it
    assert r["killed"] is not None and r["killed"].startswith("memory budget"), r
    assert r["exit"] is not None and r["exit"] != 0 and time.monotonic() - t0 < 30
    assert r["kill_record"]


def test_the_wall_cap_ends_the_tree(tmp_path):
    r = _run(tmp_path, SLEEPER, timeout=1.0)
    assert r["killed"] == "wall budget: 1 s" and r["exit"] is not None and r["exit"] != 0 and r["wall_s"] < 30


# ------------------------------------------------------------------ host tests gate
def test_the_host_tests_gate():
    good = dict(tests=48, failures=0, errors=0, skipped=0)
    assert J.tests_gate(0, good, 48) == (True, "passed")
    for exit_code, junit, why in ((1, dict(good, failures=1), "1 failures"), (0, dict(good, skipped=1), "1 skipped"),
                                  (1, dict(good, errors=2), "2 errors"), (0, dict(good, tests=47), "47 tests"),
                                  (5, dict(good, tests=0), "pytest exit 5"), (0, dict(error="no file"), "unreadable")):
        ok, reason = J.tests_gate(exit_code, junit, 48)
        assert not ok and why in reason, (junit, reason)


def test_junit_counts(tmp_path):
    x = tmp_path / "j.xml"
    x.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="0" failures="1" skipped="2" '
                 'tests="5"></testsuite></testsuites>')
    assert J._junit_counts(x) == dict(tests=5, failures=1, errors=0, skipped=2)
    assert "error" in J._junit_counts(tmp_path / "missing.xml")


# ------------------------------------------------------------------ lock
def _package(tmp_path):
    pkg, src = tmp_path / "pkg", tmp_path / "pkg" / "src"
    (src / "tools").mkdir(parents=True)
    (src / "tools" / "a.py").write_text("x = 1\n")
    (src / "SHA256SUMS").write_text(f"{J.sha256_file(src / 'tools' / 'a.py')}  tools/a.py\n")
    files = {}
    for n in J.PACKAGE_FILES:
        (pkg / n).write_text(n)
        files[n] = J.sha256_file(pkg / n)
    (pkg / "BASELINE.json").write_text(json.dumps(dict(baseline="t", sha256sums=J.sha256_file(src / "SHA256SUMS"),
                                                       files=files)))
    return pkg, src


def test_the_lock_reports_every_change(tmp_path, monkeypatch):
    pkg, src = _package(tmp_path)
    monkeypatch.setattr(J, "WORK", src)
    monkeypatch.setattr(J, "PKG", pkg)
    assert J.check_lock()["ok"]
    (src / "tools" / "a.py").write_text("x = 2\n")             # a locked file changed
    r = J.check_lock()
    assert not r["ok"] and r["changed"] == ["tools/a.py"]
    (src / "tools" / "a.py").write_text("x = 1\n")
    (src / "tools" / "b.py").write_text("")                    # an extra file
    assert J.check_lock()["extra"] == ["tools/b.py"]
    (src / "tools" / "b.py").unlink()
    (pkg / "run.sh").write_text("changed")                     # a package file changed
    assert J.check_lock()["package_files_differ"] == ["run.sh"]
    (pkg / "run.sh").write_text("run.sh")
    (src / "SHA256SUMS").write_text((src / "SHA256SUMS").read_text() + "\n")   # the list itself rewritten
    r = J.check_lock()
    assert not r["ok"] and r["sha256sums_matches"] is False


# ------------------------------------------------------------------ Windows process walk , simulated
def _fake_api(entries, end_error, first_fails=False):
    """Process32FirstW / Process32NextW / GetLastError stand-ins: the entries, then FALSE with end_error."""
    state = dict(i=0, err=0)

    def call():
        if first_fails and state["i"] == 0:
            state["err"] = 5                                   # e.g. ERROR_ACCESS_DENIED
            return None
        if state["i"] < len(entries):
            state["i"] += 1
            return entries[state["i"] - 1]
        state["err"] = end_error
        return None
    return call, call, lambda: state["err"]


TREE = [(4, 0), (100, 4), (200, 100), (300, 200), (301, 200), (900, 4)]   # 200 = step, 300/301 its children


@pytest.mark.parametrize("case", ["complete", "error_mid_walk", "false_with_code_0", "first_call_fails"])
def test_the_windows_process_walk_ends_only_on_no_more_files(case):
    if case == "complete":
        assert J.walk_processes(*_fake_api(TREE, J.ERROR_NO_MORE_FILES)) == dict(TREE)
        return
    if case == "error_mid_walk":                               # FALSE after 3 entries, an API error: not the end
        first, nxt, err = _fake_api(TREE[:3], 31)              # ERROR_GEN_FAILURE
        pattern = "error 31, not ERROR_NO_MORE_FILES"
    elif case == "false_with_code_0":                          # FALSE with no code at all: not the end either
        first, nxt, err = _fake_api(TREE[:4], 0)
        pattern = "error 0, not ERROR_NO_MORE_FILES"
    else:
        first, nxt, err = _fake_api(TREE, J.ERROR_NO_MORE_FILES, first_fails=True)
        pattern = "Process32FirstW failed"
    with pytest.raises(OSError, match=pattern):
        J.walk_processes(first, nxt, err)


def test_an_incomplete_walk_would_have_missed_the_children():
    """Why the check matters: had the walk stopped after (200, 100), 300 and 301 would not be counted."""
    assert J.descendants(dict(TREE), 200) == {200, 300, 301}
    assert J.descendants(dict(TREE[:3]), 200) == {200}
    J.check_leader_in_walk(dict(TREE), 200, leader_alive=True)
    J.check_leader_in_walk(dict(TREE[:2]), 200, leader_alive=False)      # the step already ended: nothing missed
    with pytest.raises(OSError, match="does not list the running step process 200"):
        J.check_leader_in_walk(dict(TREE[:2]), 200, leader_alive=True)


def test_a_walk_failure_stops_the_step(tmp_path, monkeypatch):
    """The walk's OSError reaches run_step like any monitor failure: the step is ended, never continued."""
    def tree_rss(proc):
        return J.walk_processes(*_fake_api(TREE[:2], 31))
    monkeypatch.setattr(J, "tree_rss", tree_rss)
    r = _run(tmp_path, SLEEPER)
    assert r["killed"] is not None and "not ERROR_NO_MORE_FILES" in r["killed"], r["killed"]
    assert r["exit"] is not None and r["exit"] != 0 and r["monitor_samples"] == 0


# ------------------------------------------------------------------ host tests step: hang traceback 
HUNG = """import time


def test_hangs_here():
    time.sleep(3)
"""


def test_a_hung_test_writes_its_stack_into_the_step_log(tmp_path):
    """The tests step
    has pytest's faulthandler_timeout: a test running longer writes the stacks of all threads into the step log and
    goes on (diagnosis only). Run through run_step, the same log redirection as the host package; the inner pytest
    has its own pytest.ini so no outer configuration or conftest is picked up."""
    assert "faulthandler_timeout=240" in J.tests_command([], "j.xml", "tmp", [])
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_hung.py").write_text(HUNG)
    cmd = J.tests_command([sys.executable, "-B"], tmp_path / "j.xml", tmp_path / "bt", ["test_hung.py"], traceback_s=1)
    r = _run(tmp_path, cmd)
    log = (tmp_path / "step.log").read_text(errors="replace")
    assert r["exit"] == 0 and r["killed"] is None, log[-3000:]
    assert "Timeout (0:00:01)!" in log and "test_hangs_here" in log, log[-3000:]
    assert J._junit_counts(tmp_path / "j.xml") == dict(tests=1, failures=0, errors=0, skipped=0)
