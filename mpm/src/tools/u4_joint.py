"""Portable fixed/translation MPM-XPBD candidate runner.

New candidate identity; not the original locked baseline. Every step retains the
source lock, input digests, subprocess budgets, memory monitor and numerical gates.
Runtime logs contain local paths and must be reviewed separately before sharing.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import signal
import subprocess
import sys
import time

WORK = pathlib.Path(__file__).resolve().parents[1]          # src/ in the package (Work_place/ in the project)
PKG = WORK.parent
MESH_NAME = "cloth52-v1-20261109"
MESH_SHA256 = "e6b155ccf4bb7f18ad4ef5be058c291194815c7996af343f6161498dc4cfe198"
MANIFEST_SHA256 = "45a469a9c1d68655b9cacf0431cc153706bc3683b25d0ba6cac3e8a90bd5dca5"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ memory of a process tree
def _linux_tree_rss(pid):
    """Resident bytes of every process in the session led by pid (start_new_session), the leader's VmHWM and the
    number of processes counted. A process that ends while it is being read is skipped (not an error)."""
    total, hwm, count = 0, 0, 0
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) != pid:                          # session id
                continue
            seen = False
            for line in (d / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    seen = True
                elif line.startswith("VmHWM:") and int(d.name) == pid:
                    hwm = int(line.split()[1]) * 1024
            count += seen
        except (OSError, IndexError, ValueError):
            continue
    return total, hwm, count


def _windows_tree_rss(proc):
    """Working set of the process and its descendants (Toolhelp snapshot + GetProcessMemoryInfo), the process's own
    peak working set and the number of processes counted. Strict: a snapshot that fails, a
    descendant that cannot be opened for any reason other than having ended (access denied, ...) or whose memory
    cannot be read raises, and the step is stopped; a descendant that ended in between is skipped."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    class PE32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    psapi, k32 = ctypes.WinDLL("psapi", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE32)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def info(handle):
        c = PMC()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c

    own = info(wintypes.HANDLE(int(proc._handle)))
    total, peak, count = int(own.WorkingSetSize), int(own.PeakWorkingSetSize), 1
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)                # TH32CS_SNAPPROCESS
    if not snap or snap == ctypes.c_void_p(-1).value:          # INVALID_HANDLE_VALUE: the step is stopped
        raise OSError(f"CreateToolhelp32Snapshot failed (error {ctypes.get_last_error()})")
    try:
        e = PE32()
        e.dwSize = ctypes.sizeof(e)

        def step(fn):                                          # (pid, parent pid), or None when the call returned FALSE
            ctypes.set_last_error(0)                           # no stale code can pass for ERROR_NO_MORE_FILES
            return (int(e.th32ProcessID), int(e.th32ParentProcessID)) if fn(snap, ctypes.byref(e)) else None
        parent = walk_processes(lambda: step(k32.Process32FirstW), lambda: step(k32.Process32NextW),
                                ctypes.get_last_error)
    finally:
        k32.CloseHandle(snap)
    check_leader_in_walk(parent, proc.pid, proc.poll() is None)
    tree = descendants(parent, proc.pid)
    for p in tree - {proc.pid}:
        h = k32.OpenProcess(0x1000 | 0x0010, False, p)        # QUERY_LIMITED_INFORMATION | VM_READ
        if not h:
            err = ctypes.get_last_error()
            if err == 87:                                      # ERROR_INVALID_PARAMETER: the process has ended
                continue
            raise OSError(f"OpenProcess({p}) failed (error {err}): its memory cannot be counted")
        try:
            total += int(info(h).WorkingSetSize)              # raises OSError: not counted -> the step stops
            count += 1
        finally:
            k32.CloseHandle(h)
    return total, peak, count


ERROR_NO_MORE_FILES = 18


def walk_processes(first, nxt, last_error):
    """Toolhelp process walk -> {pid: parent pid}. Process32FirstW and Process32NextW return FALSE both at the end of
    the list and on failure; only FALSE from NextW with ERROR_NO_MORE_FILES is the end. A failed FirstW, or FALSE
    with any other code (0 included), raises: an API failure must not pass for a complete walk and leave children
    uncounted. first() / nxt(): (pid, parent pid), or None for FALSE; last_error(): the code."""
    entry = first()
    if entry is None:
        raise OSError(f"Process32FirstW failed (error {last_error()}): no process walk")
    parent = {}
    while entry is not None:
        parent[entry[0]] = entry[1]
        entry = nxt()
    err = last_error()
    if err != ERROR_NO_MORE_FILES:
        raise OSError(f"Process32NextW returned FALSE with error {err}, not ERROR_NO_MORE_FILES "
                      f"({ERROR_NO_MORE_FILES}): the walk stopped after {len(parent)} processes, incomplete")
    return parent


def check_leader_in_walk(parent, leader, leader_alive):
    """A complete walk that does not list the running leader cannot find its descendants either."""
    if leader_alive and leader not in parent:
        raise OSError(f"the process walk does not list the running step process {leader}: incomplete")


def descendants(parent, leader):
    """{leader} and every process whose chain of parents reaches it."""
    tree, grew = {leader}, True
    while grew:
        new = {p for p, q in parent.items() if q in tree and p not in tree}
        tree |= new
        grew = bool(new)
    return tree


def tree_rss(proc):
    return _windows_tree_rss(proc) if os.name == "nt" else _linux_tree_rss(proc.pid)


def _windows_leader_rss(proc):
    """The leader's own working set and peak (only to record the leader's true peak after the step; never used to
    enforce the cap: a tree that cannot be counted stops the step)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t)] + [(n, ctypes.c_size_t) for n in (
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(proc._handle)), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(c.WorkingSetSize), int(c.PeakWorkingSetSize)


def kill_tree(proc):
    """End the process tree; never waits without a limit. Returns a record of what was tried."""
    tried = []
    if os.name == "nt":
        try:
            r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=60)
            tried.append(dict(taskkill=r.returncode, stderr=r.stderr.decode(errors="replace")[-300:]))
        except Exception as e:  # noqa: BLE001
            tried.append(dict(taskkill_error=f"{type(e).__name__}: {e}"))
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            tried.append(dict(killpg="SIGKILL"))
        except OSError as e:
            tried.append(dict(killpg_error=str(e)))
    for attempt in range(2):
        try:
            proc.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            tried.append(dict(still_alive_after_s=30 * (attempt + 1)))
            try:
                proc.kill()                                    # the leader at least
            except OSError as e:
                tried.append(dict(kill_error=str(e)))
    return tried


def run_step(name, cmd, log, *, cwd, env, timeout, mem_cap, poll):
    """One step as its own process tree, output to log; wall and memory budgets enforced."""
    kw = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else dict(start_new_session=True)
    rec = dict(step=name, cmd=[str(c) for c in cmd], log=str(log), timeout_s=timeout, memory_cap_bytes=mem_cap,
               started_utc=utc(), exit=None, wall_s=None, peak_tree_rss_bytes=0, peak_leader_bytes=0, killed=None,
               kill_record=None, monitor_error=None, monitor_samples=0, max_tree_processes=0, pid=None)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"$ {' '.join(rec['cmd'])}\n")
        fh.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, **kw)
        rec["pid"] = p.pid
        while True:
            try:
                rec["exit"] = p.wait(timeout=max(0.01, min(poll, timeout - (time.monotonic() - t0))))
                break
            except subprocess.TimeoutExpired:
                pass
            # the memory cap is either measured or the step stops -- a sampling error, or a
            # sample that counts no process while the leader is alive, ends the tree (exit 5); no degraded mode
            try:
                total, lead, count = tree_rss(p)
                if count < 1 and p.poll() is None:
                    raise OSError("the sample counted no process while the step is running")
            except Exception as e:  # noqa: BLE001
                total = None
                rec["monitor_error"] = rec["monitor_error"] or f"{type(e).__name__}: {e}"
                if rec["killed"] is None:
                    rec["killed"] = f"memory monitor unavailable: {rec['monitor_error']}"
            if total is not None:
                rec["monitor_samples"] += 1
                rec["max_tree_processes"] = max(rec["max_tree_processes"], count)
                rec["peak_tree_rss_bytes"] = max(rec["peak_tree_rss_bytes"], total)
                rec["peak_leader_bytes"] = max(rec["peak_leader_bytes"], lead)
                if total > mem_cap and rec["killed"] is None:
                    rec["killed"] = f"memory budget: process tree {total} B > {mem_cap} B"
            if rec["killed"] is None and time.monotonic() - t0 > timeout:
                rec["killed"] = f"wall budget: {timeout:.0f} s"
            if rec["killed"] is not None:
                rec["kill_record"] = kill_tree(p)
                rec["exit"] = p.poll()                     # None: the leader did not end (recorded, not waited on)
                break
    rec["wall_s"] = time.monotonic() - t0
    if os.name == "nt":
        try:                                                   # the leader's true peak (handle still open)
            rec["peak_leader_bytes"] = max(rec["peak_leader_bytes"], _windows_leader_rss(p)[1])
        except Exception:  # noqa: BLE001
            pass
    print(f"[u4] {name}: exit {rec['exit']} in {rec['wall_s']:.1f} s, peak tree {rec['peak_tree_rss_bytes'] / 2**30:.2f} GB"
          + (f", KILLED ({rec['killed']})" if rec["killed"] else ""), flush=True)
    return rec


# ------------------------------------------------------------------ lock and inputs
PACKAGE_FILES = ("run.ps1", "run.sh", "environment.yml", "RELEASE.json")


def check_lock():
    """src/ = src/SHA256SUMS, and SHA256SUMS and the package files = BASELINE.json (written when the package was
    locked). Records the digest of BASELINE.json: the value to compare with interface specification(the external anchor)."""
    sums, base = WORK / "SHA256SUMS", PKG / "BASELINE.json"
    if not sums.exists() or not base.exists():
        return dict(ok=False, reason=f"{sums if not sums.exists() else base} missing")
    want = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            want[rel.strip().lstrip("*")] = digest
    have = {}
    for p in sorted(WORK.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts:
            have[p.relative_to(WORK).as_posix()] = sha256_file(p)
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    changed = sorted(k for k in set(want) & set(have) if want[k] != have[k])
    b = json.loads(base.read_text(encoding="utf-8"))
    pkg_bad = sorted(n for n in PACKAGE_FILES if not (PKG / n).is_file() or sha256_file(PKG / n) != b["files"].get(n))
    sums_ok = sha256_file(sums) == b.get("sha256sums")
    return dict(ok=not (missing or extra or changed or pkg_bad) and sums_ok, baseline=b.get("baseline"), files=len(want),
                missing=missing, extra=extra, changed=changed, sha256sums=sha256_file(sums), sha256sums_matches=sums_ok,
                package_files_differ=pkg_bad, baseline_json_sha256=sha256_file(base),
                pycache=sorted(str(p) for p in WORK.rglob("__pycache__")))


def find_inputs(args, pins):
    root = PKG.parent
    rec = dict(package_dir=str(PKG.resolve()), shared_root=str(root.resolve()), looked=[])
    mesh = pathlib.Path(args.mesh_dir) if args.mesh_dir else PKG / "data" / MESH_NAME
    inte = pathlib.Path(args.inte) if args.inte else None
    xpbd = pathlib.Path(args.xpbd_root) if args.xpbd_root else root
    rec.update(mesh_dir=None if mesh is None else str(mesh.resolve()), xpbd_root=str(xpbd.resolve()),
               integration_dir=None if inte is None else str(inte.resolve()))
    problems = []
    # Fail closed before any process or physical step. Only the locked package config is accepted.
    try:
        import importlib.util
        plan = json.loads((WORK / "tools" / "u4_cases.json").read_text(encoding="utf-8"))
        if plan.get("config") != "scenes/configs/coupled_u4.json":
            raise ValueError("configuration must be the fixed package scene")
        config_path = (WORK / plan["config"]).resolve(strict=True)
        if not config_path.is_relative_to(WORK.resolve()):
            raise ValueError("configuration escapes the locked source tree")
        checker = (WORK / "tools" / "u4_check.py").resolve(strict=True)
        if not checker.is_relative_to(WORK.resolve()):
            raise ValueError("transform checker escapes the locked source tree")
        spec = importlib.util.spec_from_file_location("public_transform_check", checker)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not module.matches_public_transform(config["contact"]["surface"]["transform"]):
            raise ValueError("configuration does not match the fixed public transform contract")
        rec["public_transform_preflight"] = dict(ok=True)
    except Exception as error:
        rec["public_transform_preflight"] = dict(ok=False, error=str(error))
        problems.append("public transform preflight: " + str(error))

    if mesh is None or not (mesh / "mesh.npz").is_file():
        problems.append("published mesh not found (give --mesh-dir)")
    else:
        rec["mesh_sha256"] = sha256_file(mesh / "mesh.npz")
        rec["manifest_sha256"] = sha256_file(mesh / "manifest.json")
        if rec["mesh_sha256"] != MESH_SHA256 or rec["manifest_sha256"] != MANIFEST_SHA256:
            problems.append("mesh digests differ from the published cloth52-v1-20261109")
    rec["xpbd_files"] = {}
    for rel, digest in pins.items():
        p = xpbd / rel
        got = sha256_file(p) if p.is_file() else None
        rec["xpbd_files"][rel] = got
        if got != digest:
            problems.append(f"XPBD file {rel}: {got} != pinned {digest}")
    # : XPBD's contract vectors, read by three host tests from beside the XPBD
    # root; missing they would skip (the tests gate stops, exit 3), changed they would fail: pinned here instead
    vec = PKG.joinpath(*CONTRACT_VECTORS_REL)
    rec["contract_vectors"] = dict(path=str(vec.resolve()), sha256=sha256_file(vec) if vec.is_file() else None)
    want = read_vectors_pin()
    if rec["contract_vectors"]["sha256"] != want:
        problems.append(f"XPBD contract vectors {vec}: {rec['contract_vectors']['sha256']} != pinned {want} "
                        f"(tests/test_adapter.py VECTORS_SHA256)")
    rec["problems"] = problems
    return rec, (None if problems else (mesh.resolve(), xpbd.resolve(), None if inte is None else inte.resolve()))


CONTRACT_VECTORS_REL = ("data", "contract-vectors.json")


def read_vectors_pin():
    """VECTORS_SHA256 from tests/test_adapter.py without importing it: the one place the contract-vector digest is kept."""
    import ast
    tree = ast.parse((WORK / "tests" / "test_adapter.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "VECTORS_SHA256" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("VECTORS_SHA256 not found in tests/test_adapter.py")


def read_pins():
    """PINNED_XPBD from coupling/driver.py without importing it (the orchestrator stays standard library only)."""
    import ast
    tree = ast.parse((WORK / "coupling" / "driver.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PINNED_XPBD" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("PINNED_XPBD not found in coupling/driver.py")


# ------------------------------------------------------------------ smoke (its own process)
def smoke_mpm():
    sys.path[:0] = [str(WORK), str(WORK / "scenes")]
    import numpy as np
    from mpm import Solver, load_config
    from mpm.runtime import init_taichi
    import sand_column
    cfg = load_config(str(WORK / "scenes" / "configs" / "sand_column.json"), ["grid.n=16"], sand_column.SCENE_DEFAULTS)
    rt = init_taichi("cpu", cfg["precision"], 1)
    pos, _ = sand_column.initial_positions(cfg)
    sim = Solver(cfg, pos)
    sim.advance(1)
    x = sim.P.x.to_numpy()
    print(json.dumps(dict(runtime=rt, substeps=int(sim.step[None]), particles=len(x), finite=bool(np.isfinite(x).all()),
                          python=sys.version.split()[0], numpy=np.__version__), default=str))
    return 0 if np.isfinite(x).all() and int(sim.step[None]) == 1 else 1


def smoke_xpbd(xpbd_root, mesh_dir):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(WORK))
    from coupling import xpbd_service                         # standard library + NumPy (not coupling.driver: Taichi)
    X, files = xpbd_service.import_xpbd(pathlib.Path(xpbd_root), read_pins())
    m = xpbd_service.load_mesh(str(mesh_dir), MESH_SHA256, MANIFEST_SHA256)
    forbidden = xpbd_service.forbidden_loaded()
    print(json.dumps(dict(xpbd_files=files, forbidden=forbidden, vertices=int(len(m["arrays"]["vertex_ids"])),
                          triangles=int(len(m["arrays"]["triangles"])),
                          manifest_topology_version=m.get("manifest_topology_version"))))
    return 0 if not forbidden else 1


# ------------------------------------------------------------------ memory monitor pre-flight (its own process tree)
TESTS_TRACEBACK_S = 240     # a host test running longer than this dumps every thread's stack into the step log


def tests_command(py, junit, basetemp, tests, traceback_s=TESTS_TRACEBACK_S):
    """The host tests step (gate: tests_gate). pytest's faulthandler_timeout -- a test that runs longer than
    traceback_s writes the stacks of all threads to the step log and goes on (diagnosis only; the step's wall budget
    still ends a hung step). The traceback identifies stalled test threads."""
    return list(py) + ["-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={junit}",
                       f"--basetemp={basetemp}", "-o", "tmp_path_retention_policy=failed",
                       "-o", f"faulthandler_timeout={traceback_s}", *tests]


PROBE_PARENT_MIB, PROBE_CHILD_MIB, PROBE_HOLD_S = 32, 128, 4.0


def _touched(mib):
    b = bytearray(mib << 20)
    b[::4096] = b"\x01" * len(range(0, len(b), 4096))       # every page resident (calloc'd pages are lazy)
    return b


def monitor_probe(role):
    """parent: hold 32 MiB and start a child (a grandchild of the orchestrator) that holds 128 MiB; both stay for
    PROBE_HOLD_S. The orchestrator samples the tree meanwhile (monitor_preflight)."""
    hold = _touched(PROBE_PARENT_MIB if role == "parent" else PROBE_CHILD_MIB)
    if role == "parent":
        child = subprocess.Popen([sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "--monitor-probe", "child"])
        time.sleep(PROBE_HOLD_S)
        code = child.wait(timeout=60)
        print(json.dumps(dict(role=role, held_mib=PROBE_PARENT_MIB, child_exit=code, held=len(hold))))
        return 0 if code == 0 else 1
    time.sleep(PROBE_HOLD_S)
    print(json.dumps(dict(role=role, held_mib=PROBE_CHILD_MIB, held=len(hold))))
    return 0


def monitor_preflight(rec):
    """The probe's record: both processes counted at least once, and at least the child's 128 MiB seen."""
    ok = (rec["exit"] == 0 and rec["killed"] is None and rec["monitor_error"] is None and rec["monitor_samples"] >= 1
          and rec["max_tree_processes"] >= 2 and rec["peak_tree_rss_bytes"] >= PROBE_CHILD_MIB << 20)
    return ok, dict(samples=rec["monitor_samples"], max_tree_processes=rec["max_tree_processes"],
                    peak_tree_mib=rec["peak_tree_rss_bytes"] >> 20, required=dict(processes=2, mib=PROBE_CHILD_MIB))


def tests_gate(exit_code, junit, expected):
    """the host tests pass only with exit 0, a readable junit, no failure, error or skip,
    and exactly the expected number of tests."""
    if "error" in junit:
        return False, f"junit unreadable: {junit['error']}"
    bad = []
    if exit_code != 0:
        bad.append(f"pytest exit {exit_code}")
    for k in ("failures", "errors", "skipped"):
        if junit.get(k, 0):
            bad.append(f"{junit[k]} {k}")
    if junit.get("tests") != expected:
        bad.append(f"{junit.get('tests')} tests, expected {expected}")
    return not bad, "; ".join(bad) or "passed"


# ------------------------------------------------------------------ main
def main():
    try:
        return _main()
    except _Stop as e:
        return e.code


class _Stop(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xpbd-root")
    ap.add_argument("--mesh-dir")
    ap.add_argument("--inte", help="shared directory for --publish (default: where the mesh was found)")
    ap.add_argument("--results", default=str(PKG / "results"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--rehearse-windows", type=int)
    ap.add_argument("--skip-tests", action="store_true", help="rehearsal only")
    ap.add_argument("--smoke", choices=["mpm", "xpbd"], help=argparse.SUPPRESS)
    ap.add_argument("--monitor-probe", choices=["parent", "child"], help=argparse.SUPPRESS)
    ap.add_argument("--smoke-args", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.monitor_probe:
        return monitor_probe(args.monitor_probe)
    if args.smoke == "mpm":
        return smoke_mpm()
    if args.smoke == "xpbd":
        return smoke_xpbd(*args.smoke_args)

    base = json.loads((WORK / "tools" / "u4_cases.json").read_text(encoding="utf-8"))
    B = base["budgets"]
    t_pkg = time.monotonic()
    run = pathlib.Path(args.results) / f"run-{utc()}"
    run.mkdir(parents=True)
    rehearsal = args.rehearse_windows is not None or args.skip_tests
    summary = dict(baseline=base["baseline"], package=base["package"], unit=base["unit"], run_dir=str(run.resolve()),
                   started_utc=utc(), rehearsal=rehearsal, rehearse_windows=args.rehearse_windows,
                   skip_tests=args.skip_tests, publish_requested=bool(args.publish), budgets=B,
                   host=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                             logical_cpus=os.cpu_count(), python=sys.version, executable=sys.executable),
                   steps=[], cases={}, exit=None, outcome=None)

    def save():
        (run / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    def finish(code, outcome):
        summary.update(exit=code, outcome=outcome, finished_utc=utc(), wall_s=time.monotonic() - t_pkg)
        save()
        print(f"[u4] {outcome} -> exit {code}; {run / 'run_summary.json'}", flush=True)
        return code

    def lock_stop(name):
        raise _Stop(finish(4, f"the locked files changed before step {name}: stopped"))

    lock = check_lock()
    summary["lock"] = lock
    if not lock["ok"]:
        return finish(4, "the package differs from its locked baseline (src/SHA256SUMS or BASELINE.json): not run; "
                         f"changed {lock.get('changed')}, missing {lock.get('missing')}, extra {lock.get('extra')}, "
                         f"package files {lock.get('package_files_differ')}, SHA256SUMS matches "
                         f"{lock.get('sha256sums_matches')}{'; ' + lock['reason'] if lock.get('reason') else ''}")
    pins = read_pins()
    inputs, found = find_inputs(args, pins)
    summary["inputs"] = inputs
    (run / "inputs.json").write_text(json.dumps(inputs, indent=1), encoding="utf-8")
    if found is None:
        return finish(4, f"inputs: {'; '.join(inputs['problems'])}")
    mesh_dir, xpbd_root, inte = found
    if args.publish and inte is None:                          # fail fast, before the joint cases 
        return finish(4, "publication requested but no shared directory (give -InteDir / --inte)")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPM_XPBD_WORK=str(xpbd_root),
               MPM_PUBLISHED_MESH=str(mesh_dir))
    py = [sys.executable, "-B"]

    def step(name, cmd, log):
        left = B["package_wall_s"] - (time.monotonic() - t_pkg)
        if left <= 0:
            return None
        again = check_lock()                                   # before every step 
        if not again["ok"]:
            summary["lock_before_" + name] = again
            lock_stop(name)
        rec = run_step(name, cmd, run / "logs" / log, cwd=WORK, env=env, timeout=min(B["step_wall_s"], left),
                       mem_cap=B["memory_bytes"], poll=B["memory_poll_s"])
        summary["steps"].append(rec)
        save()
        return rec

    # memory monitor pre-flight : the 4 GiB cap must be measurable on this machine before anything heavy runs
    rec = step("monitor_preflight", py + [str(WORK / "tools" / "u4_joint.py"), "--monitor-probe", "parent"],
               "monitor_preflight.log")
    if rec is None:
        return finish(5, "budget stop before the memory monitor pre-flight")
    ok, detail = monitor_preflight(rec)
    summary["monitor_preflight"] = dict(ok=ok, **detail)
    if not ok:
        return finish(4, f"the memory monitor cannot count the process tree on this machine ({detail}; "
                         f"{rec['monitor_error'] or rec['killed']}): stopped before any joint step")

    # smoke: both sides, nothing joint yet
    for side, extra in (("mpm", []), ("xpbd", [str(xpbd_root), str(mesh_dir)])):
        rec = step(f"smoke_{side}", py + [str(WORK / "tools" / "u4_joint.py"), "--smoke", side, "--smoke-args", *extra],
                   f"smoke_{side}.log")
        if rec is None or rec["killed"]:
            return finish(5, f"budget or memory-monitor stop at smoke_{side}")
        if rec["exit"] != 0:
            return finish(4, f"smoke_{side} failed (exit {rec['exit']}): see {rec['log']}")

    # host tests : a failure or skip stops the package before the joint cases; the evidence stays
    if not args.skip_tests:
        junit = run / "tests" / "junit.xml"
        junit.parent.mkdir(parents=True, exist_ok=True)
        rec = step("tests", tests_command(py, junit, run / "tests" / "tmp", base["tests"]), "tests.log")
        if rec is None or rec["killed"]:
            return finish(5, "budget or memory-monitor stop at the host tests")
        rec["junit"] = _junit_counts(junit)
        ok, why = tests_gate(rec["exit"], rec["junit"], int(base["tests_expected"]))
        rec["gate"] = dict(passed=ok, reason=why)
        save()
        if not ok:
            return finish(3, f"host tests did not pass ({why}): stopped before the joint cases; evidence kept in "
                             f"logs/tests.log, tests/junit.xml and tests/tmp/ (failed tests' directories)")

    # joint cases, in the locked order
    broke = gate_failed = False
    for case in base["cases"]:
        name = case["name"]
        tier = base["tiers"][case["tier"]]
        W = int(case["windows"]) if args.rehearse_windows is None else min(int(case["windows"]), args.rehearse_windows)
        stream = f"u4-{name}-{os.urandom(8).hex()}"
        sets = [*tier["sets"], *case["sets"], f"scene.windows={W}",
                f"scene.sheet_velocity={json.dumps(case['velocity_world'])}", f"scene.stream_id={json.dumps(stream)}",
                f"scene.xpbd.snapshot.mesh_path={mesh_dir.as_posix()}"]
        cmd = py + [str(WORK / base["scene"]), "--config", str(WORK / base["config"]), "--arch", base["arch"],
                    "--threads", str(tier["threads"]), "--out", str(run / "cases"), "--name", name]
        for s in sets:
            cmd += ["--set", s]
        rec = step(f"case_{name}", cmd, f"case_{name}.log")
        if rec is None or rec["killed"]:
            summary["cases"][name] = dict(scene=rec, check=None, passed=False)
            return finish(5, f"budget or memory-monitor stop at case {name} ({rec['killed'] if rec else 'package wall'})")
        crec = step(f"check_{name}", py + [str(WORK / "tools" / "u4_check.py"), "--run", str(run / "cases" / name),
                                           "--case", name, "--cases", str(WORK / "tools" / "u4_cases.json"),
                                           "--out", str(run / "checks" / f"{name}.json"), "--scene-exit", str(rec["exit"]),
                                           "--mpm-pid", str(rec["pid"])] + (["--rehearsal"] if rehearsal else []),
                    f"check_{name}.log")
        if crec is None or crec["killed"]:
            summary["cases"][name] = dict(scene=rec["exit"], check=None, passed=False)
            return finish(5, f"budget or memory-monitor stop at the check of {name}")
        chk = run / "checks" / f"{name}.json"
        passed = None
        if chk.exists():
            c = json.loads(chk.read_text(encoding="utf-8"))
            passed = c.get("passed")
            summary["cases"][name] = dict(scene_exit=rec["exit"], check_exit=crec["exit"], passed=passed,
                                          failed=[k for k, g in {**c["gates"], **c["case_gates"]}.items()
                                                  if g["passed"] is False],
                                          timing=c.get("timing"), peak_tree_rss_bytes=rec["peak_tree_rss_bytes"],
                                          windows=W)
        else:
            summary["cases"][name] = dict(scene_exit=rec["exit"], check_exit=crec["exit"], passed=False, failed=["(no check)"])
        if rec["exit"] not in (0, 4) or crec["exit"] not in (0, 2) or not chk.exists():
            broke = True
        elif not passed:
            gate_failed = True
        save()

    pre = ((1, "a scene or check process broke (see steps)") if broke else
           (2, "every step ran; a gate failed (see cases)") if gate_failed else
           (0, "every step ran and every gate passed" + (" (REHEARSAL: not a U4 result)" if rehearsal else "")))
    summary.update(exit_before_publish=pre[0], outcome_before_publish=pre[1])
    save()
    publish_failed = False
    if args.publish:
        if inte is None:
            publish_failed = True
            summary["publish"] = dict(error="no shared directory (give --inte)")
        else:
            rec = step("publish", py + [str(WORK / "tools" / "u4_publish.py"), "--run", str(run), "--inte", str(inte),
                                        "--src", str(WORK)], "publish.log")
            if rec is None or rec["killed"]:
                return finish(5, "budget or memory-monitor stop at the publication")
            publish_failed = rec["exit"] != 0
            pj = run / "publish.json"
            summary["publish"] = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else dict(exit=rec["exit"])
    if pre[0] == 0 and publish_failed:
        return finish(6, "every case passed; the publication failed")
    return finish(*pre)


def _junit_counts(path):
    try:
        import xml.etree.ElementTree as ET
        r = ET.parse(path).getroot()
        s = r if r.tag == "testsuite" else r.find("testsuite")
        return {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    except Exception as e:  # noqa: BLE001
        return dict(error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
