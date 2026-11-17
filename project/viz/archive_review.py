"""Export one to three verified cloth or synthetic exchange archives to an offline review."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

import numpy as np

from ..cloth_archive import load_run, sha256
from ..coupling.exchange_archive import read_archive
from .cloth_replay import (MAX_ARCHIVE_BYTES, MAX_EXPANDED_BYTES, MAX_FILES,
                           MAX_HTML_BYTES, MAX_RECORDS, MAX_TRIANGLE_INSTANCES,
                           MAX_VERTEX_INSTANCES, ReplayLimitError, _limit, encode_payload)
from .review_cloth import cloth_case
from .review_exchange import exchange_case

MAX_CASES = 3
MAX_EXCHANGE_BYTES = 8 * 1024 * 1024
MAX_QUERY_PAIRS = 1_000_000
TEMPLATE = Path(__file__).with_name("archive_review.html")


def _plain(path):
    info = path.lstat()
    if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"linked archive member: {path}")
    return info


def _fingerprint(paths):
    """Check the shared byte budget before hashing regular source files."""
    result = []
    total = 0
    for path in paths:
        members = sorted(path.iterdir()) if path.is_dir() else [path]
        _limit("MAX_FILES", len(members), MAX_FILES)
        records = {}
        for member in members:
            info = _plain(member)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"archive requires regular files: {member}")
            total += info.st_size
            _limit("MAX_ARCHIVE_BYTES", total, MAX_ARCHIVE_BYTES)
            if not path.is_dir():
                _limit("MAX_EXCHANGE_BYTES", info.st_size, MAX_EXCHANGE_BYTES)
            records[member.name] = dict(size=info.st_size, sha256=sha256(member))
        result.append(records)
    return result


def _cloth_sizes(path, fingerprint, totals):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    totals["frames"] += len(records)
    _limit("MAX_RECORDS", totals["frames"], MAX_RECORDS)
    expected = {"manifest.json", *(f"record_{i:06d}.npz" for i in range(len(records)))}
    if set(fingerprint) != expected:
        raise ValueError("cloth directory members differ from manifest record sequence")
    for filename in sorted(expected - {"manifest.json"}):
        with zipfile.ZipFile(path / filename) as archive:
            for info in archive.infolist():
                totals["expanded"] += info.file_size
                _limit("MAX_EXPANDED_BYTES", totals["expanded"], MAX_EXPANDED_BYTES)
                # Reject an inflated NPY shape before np.load can allocate it.
                with archive.open(info) as entry:
                    ver = np.lib.format.read_magic(entry)
                    if ver not in ((1, 0), (2, 0)):
                        raise ValueError("unsupported NPY member format")
                    header = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = header(entry)
                    if (dtype.hasobject or any(dim < 0 for dim in shape)
                            or entry.tell() + math.prod(shape) * dtype.itemsize != info.file_size):
                        raise ValueError(f"invalid archived array size: {filename}/{info.filename}")
                    if info.filename in ("vertex_ids.npy", "triangle_ids.npy"):
                        key = "vertices" if info.filename == "vertex_ids.npy" else "triangles"
                        totals[key] += math.prod(shape)
        _geometry_limits(totals)


def _geometry_limits(totals):
    _limit("MAX_VERTEX_INSTANCES", totals["vertices"], MAX_VERTEX_INSTANCES)
    _limit("MAX_TRIANGLE_INSTANCES", totals["triangles"], MAX_TRIANGLE_INSTANCES)


def _exchange_sizes(path, totals):
    windows = json.loads(path.read_text(encoding="utf-8"))["windows"]
    for i, window in enumerate(windows):
        start, end = window["start"], window["end"]
        # The solved frame has the opening topology. Only a cut adds a commit frame.
        states = [start] * (2 if i == 0 else 1)
        if window["cloth"]["event"] is not None:
            states.append(end)
        totals["frames"] += len(states)
        _limit("MAX_RECORDS", totals["frames"], MAX_RECORDS)
        totals["vertices"] += sum(len(state["vertex_ids"]) for state in states)
        totals["triangles"] += sum(len(state["triangle_ids"]) for state in states)
        _geometry_limits(totals)
        peer = window["peer"]
        totals["query_pairs"] += len(peer["ticks"]) * len(peer["particle_ids"]) * len(start["triangles"])
        _limit("MAX_QUERY_PAIRS", totals["query_pairs"], MAX_QUERY_PAIRS)


def _browser_numbers(value):
    # JavaScript Numbers must preserve every stable ID, version and integer parameter.
    if isinstance(value, int) and abs(value) >= 2**53:
        raise ValueError("integer outside the browser's exact range")
    if isinstance(value, (list, tuple, dict)):
        for item in (value.values() if isinstance(value, dict) else value):
            _browser_numbers(item)


def export_review(cases, output):
    """Read validated source archives in the requested order and publish without overwrite."""
    if not cases:
        raise ValueError("select at least one archive")
    _limit("MAX_CASES", len(cases), MAX_CASES)
    paths = []
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output already exists: {target}")
    target = target.resolve()
    if not target.parent.is_dir():
        raise ValueError("output parent must already exist")
    for label, source in cases:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("each archive needs a nonempty label")
        path = Path(source)
        _plain(path)
        path = path.resolve(strict=True)
        if target.is_relative_to(path if path.is_dir() else path.parent):
            raise ValueError("output must be outside each source archive directory")
        paths.append(path)

    before = _fingerprint(paths)
    totals = dict(frames=0, vertices=0, triangles=0, expanded=0, query_pairs=0)
    for path, fingerprint in zip(paths, before):
        if path.is_dir():
            _cloth_sizes(path, fingerprint, totals)
        else:
            _exchange_sizes(path, totals)

    datasets = []
    array_bytes = 0
    for (label, _), path, fingerprint in zip(cases, paths, before):
        if path.is_dir():
            run = load_run(path)
            array_bytes += sum(a.nbytes for record in run.records for a in record.arrays.values())
            _limit("MAX_EXPANDED_BYTES", array_bytes, MAX_EXPANDED_BYTES)
            dataset = cloth_case(run, fingerprint)
        else:
            dataset = exchange_case(read_archive(path))
        dataset.update(label=label, source_files=fingerprint)
        datasets.append(dataset)
    payload = dict(schema="xpbd-archive-review-v1", cases=datasets)
    _browser_numbers(payload)
    html = TEMPLATE.read_text(encoding="utf-8").replace("@@DATA@@", encode_payload(payload)).encode("utf-8")
    _limit("MAX_HTML_BYTES", len(html), MAX_HTML_BYTES)
    if _fingerprint(paths) != before:
        raise ValueError("source archive changed during review export")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".archive-review-", delete=False) as out:
            temporary = Path(out.name)
            out.write(html)
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(output=str(target), sha256=sha256(target), bytes=len(html),
                frames=[len(case["frames"]) for case in datasets], limits_usage=totals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, nargs=2, metavar=("LABEL", "ARCHIVE"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = export_review(args.case, args.out)
    except ReplayLimitError as error:
        print(f"Review limit: {error}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError,
            RecursionError, EOFError, zipfile.BadZipFile) as error:
        print(f"Review rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
