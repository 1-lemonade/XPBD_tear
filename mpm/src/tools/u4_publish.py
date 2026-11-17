"""P2-U4: publish a host run to the shared directory . Standard library only.

    python -B tools/u4_publish.py --run results/run-<UTC> --inte <shared directory> --src <package src>

Run     <inte>/coupling/runs/u4-<UTC>-host/        the whole run directory; MPM (the driver) owns it; the
                                                   subdirectory xpbd/ is reserved for XPBD and not created here
                                                   (a rehearsal run: u4-<UTC>-rehearsal, never -host)
Adapter <inte>/coupling/adapter/<baseline>-<sha12>/ the whole package src/ (importable with an XPBD root and the
                                                   mesh: coupling/ needs mpm/) and the package's BASELINE.json;
                                                   sha12 = first 12 hex of the SHA-256 of src/SHA256SUMS
Each: copy into <parent>/.staging-<name>; read back every file against the SHA-256 of its source; write
manifest.json last (complete: true); rename to <name> (same parent: atomic on one volume); read back again.
A target that exists is never overwritten (exit 5); an adapter snapshot that exists with the same file digests
is reused (immutable, identical) after its files are read back. Readers skip .staging-* and directories without complete: true. A failed
read back leaves the staging directory as it is (exit 1), for the owner to handle. Writes <run>/publish.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import sys



def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def listing(root, patterns=None):
    root = pathlib.Path(root)
    if patterns is None:
        files = [p for p in root.rglob("*") if p.is_file()]
    else:
        files = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file()})
    return {p.relative_to(root).as_posix(): p for p in sorted(files) if "__pycache__" not in p.parts}


def readback(root, want):
    root = pathlib.Path(root)
    have = {k: v for k, v in listing(root).items() if k != "manifest.json"}
    bad = sorted(k for k in want if k not in have or sha256_file(have[k]) != want[k])
    extra = sorted(set(have) - set(want))
    return dict(ok=not bad and not extra, bad=bad[:20], extra=extra[:20], files=len(want))


def publish(files, parent, name, manifest):
    """files: rel -> source path. Returns a record; raises FileExistsError when the target exists."""
    parent = pathlib.Path(parent)
    dest, stage = parent / name, parent / f".staging-{name}"
    if dest.exists() or stage.exists():
        raise FileExistsError(str(dest if dest.exists() else stage))
    want = {rel: sha256_file(p) for rel, p in files.items()}
    parent.mkdir(parents=True, exist_ok=True)
    for rel, p in files.items():
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, stage / rel)
    rb = readback(stage, want)
    rec = dict(target=str(dest.resolve()), staging=str(stage.resolve()), staged_readback=rb, published=False)
    if not rb["ok"]:
        return rec
    man = dict(manifest, files=want, file_count=len(want), complete=True, written_utc=utc())
    (stage / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    os.rename(stage, dest)
    rec.update(published=True, published_readback=readback(dest, want),
               manifest_sha256=sha256_file(dest / "manifest.json"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inte", required=True)
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    run, inte, src = pathlib.Path(args.run), pathlib.Path(args.inte), pathlib.Path(args.src)
    base = json.loads((src / "tools" / "u4_cases.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    out = dict(started_utc=utc(), inte=str(inte.resolve()), run=None, adapter=None, exit=None)
    code = 0
    run_id = f"u4-{run.name[len('run-'):]}-{'rehearsal' if summary.get('rehearsal') else 'host'}"
    common = dict(baseline=base["baseline"], unit=base["unit"], package=base["package"], owner="MPM (driver side)",
                  retention="No retention policy is asserted by this tool; review generated data before sharing.",
                  delete_rule="Retention and deletion are operator responsibilities; existing targets are not overwritten.")
    try:
        files = {k: v for k, v in listing(run).items() if k not in ("publish.json", "logs/publish.log")}   # still open
        out["run"] = publish(files, inte / "coupling" / "runs", run_id, dict(
            common, format="mpm-u4-joint-run-v1", run_id=run_id, source_run_dir=str(run.resolve()),
            run_outcome=summary.get("outcome_before_publish"), run_exit=summary.get("exit_before_publish"),
            rehearsal=summary.get("rehearsal"),
            xpbd_subdirectory="xpbd/ is reserved for XPBD (written by XPBD only; not created by MPM)"))
        if not out["run"]["published"] or not out["run"]["published_readback"]["ok"]:
            code = 1
    except FileExistsError as e:
        out["run"] = dict(error=f"target exists, never overwritten: {e}")
        code = 5
    sums = src / "SHA256SUMS"
    version = f"{base['baseline']}-{sha256_file(sums)[:12]}"
    files = {f"src/{k}": v for k, v in listing(src).items()}
    files["BASELINE.json"] = src.parent / "BASELINE.json"
    parent = inte / "coupling" / "adapter"
    try:
        out["adapter"] = publish(files, parent, version, dict(common, format="mpm-u4-adapter-snapshot-v1",
                                                               version=version, sha256sums=sha256_file(sums)))
        if not out["adapter"]["published"] or not out["adapter"]["published_readback"]["ok"]:
            code = code or 1
    except FileExistsError:
        dest = parent / version
        man = json.loads((dest / "manifest.json").read_text(encoding="utf-8")) if (dest / "manifest.json").exists() else {}
        want = {k: sha256_file(p) for k, p in files.items()}
        rb = readback(dest, want)                              # the files themselves, not only the manifest
        same = man.get("complete") is True and man.get("files") == want and rb["ok"]
        out["adapter"] = dict(target=str(dest.resolve()), reused=same, existing_readback=rb,
                              note="exists; identical snapshot reused (read back)" if same
                              else "exists with different or damaged content: never overwritten")
        if not same:
            code = code or 5
    out.update(exit=code, finished_utc=utc())
    (run / "publish.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(dict(run=(out["run"] or {}).get("target"), adapter=(out["adapter"] or {}).get("target"), exit=code)))
    return code


if __name__ == "__main__":
    sys.exit(main())
