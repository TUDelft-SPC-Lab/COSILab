#!/usr/bin/env python3
"""Check which project folders are fully duplicated inside 4TU_upload.

For every folder under ROOT (default: the COSILab project share), check whether
the content of each file in it also exists somewhere in REF (default:
ROOT/4TU_upload). Matching is by file content (MD5), not by name or path, so
renamed or reorganised copies still count as duplicates.

The tool only reads and reports. It never deletes anything.

How it avoids hashing everything
--------------------------------
1. Walk both trees and record file sizes (cheap metadata only).
2. A source file whose size does not occur in REF cannot be a copy, so it is
   marked MISSING without reading it.
3. Files whose sizes do match get a quick hash: MD5 of the first and last 1 MiB.
   Files of 2 MiB or less are hashed whole at this step, so they are done.
4. Only files whose (size, quick hash) occur in both trees get a full MD5.

Hashes are cached in a SQLite file keyed by (path, size, mtime), so an
interrupted run (e.g. a Slurm time limit) resumes where it stopped.

Outputs (in --out)
------------------
summary.csv  one row per folder: verdict, file/byte counts, bytes covered.
files.csv    one row per source file: status, reason, and the matching REF path.
hash_cache.sqlite  the hash cache (safe to delete; only makes reruns faster).

Per-file status
---------------
COVERED           identical content exists in REF.
COVERED_HARDLINK  same inode as a REF file (deleting it frees no space).
COVERED_QUICK     only with --quick-only: size and first/last 1 MiB match.
EMPTY             zero-byte file (nothing to lose).
SYMLINK           a symlink (no data of its own; not followed).
MISSING           content not found in REF. `reason` says why.
PROTECTED         a REF symlink points at this file/dir. Deleting it breaks REF.
ERROR             could not be read.

Folder verdict
--------------
FULLY_COVERED  every file is COVERED/COVERED_HARDLINK/EMPTY/SYMLINK and the
               whole folder could be listed. Candidate for deletion.
QUICK_COVERED  as above but relying on --quick-only matches; rerun without
               --quick-only before deleting.
PARTIAL        some bytes covered, some not.
NOT_COVERED    nothing covered.
NO_FILES       folder contains no regular files.

Example
-------
    python misc/check_4tu_coverage.py --out ~/4tu_coverage --workers 8
    python misc/check_4tu_coverage.py --folders raw_video --summary-depth 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

DEFAULT_ROOT = "/tudelft.net/staff-umbrella/neon/cosilab_project"
DEFAULT_REF_NAME = "4TU_upload"
QUICK_BYTES = 1 << 20  # 1 MiB from each end
CHUNK = 8 << 20

OK_STATUSES = {"COVERED", "COVERED_HARDLINK", "EMPTY", "SYMLINK"}


@dataclass
class FileRec:
    path: str
    size: int
    mtime_ns: int
    dev_ino: tuple[int, int]


@dataclass
class WalkResult:
    files: list[FileRec] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)
    errors: list[tuple[str, str, bool]] = field(default_factory=list)  # (path, msg, is_dir)


def walk(top: str, skip: set[str]) -> WalkResult:
    """List regular files under `top` without following symlinks."""
    res = WalkResult()
    stack = [top]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError as e:
            res.errors.append((d, str(e), True))
            continue
        with it:
            for e in it:
                try:
                    if e.is_symlink():
                        res.symlinks.append(e.path)
                    elif e.is_dir(follow_symlinks=False):
                        if e.path not in skip:
                            stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                        res.files.append(
                            FileRec(e.path, st.st_size, st.st_mtime_ns, (st.st_dev, st.st_ino))
                        )
                    # sockets, fifos, devices: ignored
                except OSError as ex:
                    res.errors.append((e.path, str(ex), False))
    return res


# --------------------------------------------------------------------------- hashing


def quick_hash(rec: FileRec) -> tuple[str, str | None]:
    """MD5 of first+last QUICK_BYTES. For small files this is the full MD5."""
    h = hashlib.md5()
    with open(rec.path, "rb") as f:
        if rec.size <= 2 * QUICK_BYTES:
            data = f.read()
            if len(data) != rec.size:
                raise OSError(f"size changed while reading ({len(data)} != {rec.size})")
            h.update(data)
            d = h.hexdigest()
            return d, d
        h.update(f.read(QUICK_BYTES))
        f.seek(rec.size - QUICK_BYTES)
        h.update(f.read(QUICK_BYTES))
    return h.hexdigest(), None


def full_md5(rec: FileRec) -> str:
    h = hashlib.md5()
    buf = bytearray(CHUNK)
    view = memoryview(buf)
    n_total = 0
    with open(rec.path, "rb", buffering=0) as f:
        while n := f.readinto(buf):
            h.update(view[:n])
            n_total += n
    if n_total != rec.size:
        raise OSError(f"size changed while reading ({n_total} != {rec.size})")
    return h.hexdigest()


class HashCache:
    """SQLite cache of quick/full hashes keyed by (path, size, mtime_ns).

    Only the main thread touches it; worker threads just compute hashes.
    """

    def __init__(self, path: str):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS h ("
            "path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, quick TEXT, md5 TEXT)"
        )
        self.mem: dict[str, tuple[int, int, str | None, str | None]] = {
            p: (s, m, q, d) for p, s, m, q, d in self.db.execute("SELECT * FROM h")
        }
        self._last_commit = time.monotonic()

    def get(self, rec: FileRec, kind: str) -> str | None:
        row = self.mem.get(rec.path)
        if row is None or row[0] != rec.size or row[1] != rec.mtime_ns:
            return None
        return row[2] if kind == "quick" else row[3]

    def put(self, rec: FileRec, quick: str | None = None, md5: str | None = None) -> None:
        row = self.mem.get(rec.path)
        if row is not None and row[0] == rec.size and row[1] == rec.mtime_ns:
            quick = quick or row[2]
            md5 = md5 or row[3]
        self.mem[rec.path] = (rec.size, rec.mtime_ns, quick, md5)
        self.db.execute(
            "INSERT OR REPLACE INTO h VALUES (?,?,?,?,?)",
            (rec.path, rec.size, rec.mtime_ns, quick, md5),
        )
        if time.monotonic() - self._last_commit > 30:
            self.commit()

    def commit(self) -> None:
        self.db.commit()
        self._last_commit = time.monotonic()

    def close(self) -> None:
        self.commit()
        self.db.close()


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PiB"


def run_hash_phase(
    recs: list[FileRec], kind: str, cache: HashCache, workers: int, errors: dict[str, str]
) -> None:
    """Fill the cache with `kind` ('quick' or 'md5') hashes for `recs`."""
    todo = [r for r in recs if r.path not in errors and cache.get(r, kind) is None]
    cost = (lambda r: min(r.size, 2 * QUICK_BYTES)) if kind == "quick" else (lambda r: r.size)
    total_bytes = sum(cost(r) for r in todo)
    print(
        f"[{kind}] {len(recs) - len(todo)} cached, {len(todo)} to hash ({human(total_bytes)})",
        file=sys.stderr,
    )
    if not todo:
        return
    fn = quick_hash if kind == "quick" else full_md5
    done_files = done_bytes = 0
    t0 = last = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, r): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                res = fut.result()
            except OSError as e:
                errors[r.path] = str(e)
            else:
                if kind == "quick":
                    cache.put(r, quick=res[0], md5=res[1])
                else:
                    cache.put(r, md5=res)
            done_files += 1
            done_bytes += cost(r)
            now = time.monotonic()
            if now - last > 15 or done_files == len(todo):
                last = now
                rate = done_bytes / max(now - t0, 1e-9)
                eta = (total_bytes - done_bytes) / rate if rate else 0
                print(
                    f"[{kind}] {done_files}/{len(todo)} files, "
                    f"{human(done_bytes)}/{human(total_bytes)}, "
                    f"{human(rate)}/s, ETA {eta / 60:.0f} min",
                    file=sys.stderr,
                )
    cache.commit()


# --------------------------------------------------------------------------- main


def group_of(rel: str, depth: int) -> str:
    parts = rel.split(os.sep)
    g = parts[: min(depth, len(parts) - 1)]  # always a directory
    return os.sep.join(g) if g else "(files directly in root)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for statuses and verdicts.",
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="project root (default: %(default)s)")
    ap.add_argument(
        "--ref", default=None, help=f"reference tree (default: ROOT/{DEFAULT_REF_NAME})"
    )
    ap.add_argument(
        "--folders", nargs="*", default=None,
        help="only check these top-level folders under ROOT (default: all except REF)",
    )
    ap.add_argument(
        "--exclude", nargs="*", default=[],
        help="top-level names under ROOT to skip (in addition to REF)",
    )
    ap.add_argument(
        "--summary-depth", type=int, default=1,
        help="aggregate summary.csv at this folder depth below ROOT (default: 1)",
    )
    ap.add_argument("--workers", type=int, default=8, help="parallel hashing threads")
    ap.add_argument(
        "--quick-only", action="store_true",
        help="skip full MD5; match on size + first/last 1 MiB only (fast survey, NOT for deleting)",
    )
    ap.add_argument("--out", default="4tu_coverage", help="output directory (default: %(default)s)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    ref = os.path.abspath(args.ref or os.path.join(root, DEFAULT_REF_NAME))
    if not os.path.isdir(root) or not os.path.isdir(ref):
        sys.exit(f"root or ref is not a directory: {root} / {ref}")
    os.makedirs(args.out, exist_ok=True)

    # ---- choose source folders
    excluded = set(args.exclude)
    if args.folders is not None:
        tops = [os.path.join(root, f.rstrip("/")) for f in args.folders]
    else:
        tops = sorted(
            e.path for e in os.scandir(root)
            if e.is_dir(follow_symlinks=False) and e.name not in excluded
        )
        root_files = [
            e.path for e in os.scandir(root)
            if not e.is_dir(follow_symlinks=False) and e.name not in excluded
        ]
    # never scan REF or our own output directory (it changes while we run)
    out_abs = os.path.abspath(args.out)
    skip_dirs = {ref, out_abs}
    tops = [t for t in tops if os.path.abspath(t) not in skip_dirs]

    # ---- walk REF
    print(f"Indexing reference tree {ref} ...", file=sys.stderr)
    ref_walk = walk(ref, skip=set())
    print(
        f"  {len(ref_walk.files)} files, {human(sum(r.size for r in ref_walk.files))}, "
        f"{len(ref_walk.symlinks)} symlinks, {len(ref_walk.errors)} unreadable entries",
        file=sys.stderr,
    )
    ref_sizes: dict[int, list[FileRec]] = defaultdict(list)
    for r in ref_walk.files:
        ref_sizes[r.size].append(r)
    ref_inodes = {r.dev_ino: r.path for r in ref_walk.files}
    ref_names = {os.path.basename(r.path) for r in ref_walk.files}

    # REF symlinks pointing into ROOT: their targets must not be deleted.
    real_root = os.path.realpath(root)
    protected: list[str] = []
    for link in ref_walk.symlinks:
        target = os.path.realpath(link)
        if target == real_root or target.startswith(real_root + os.sep):
            protected.append(target)
    if protected:
        print(
            f"  WARNING: {len(protected)} symlinks in REF point into ROOT; "
            "their targets are marked PROTECTED",
            file=sys.stderr,
        )
    prot_prefixes = tuple(p + os.sep for p in protected)
    prot_exact = set(protected)

    def is_protected(path: str) -> bool:
        real = os.path.join(real_root, os.path.relpath(path, root))
        return real in prot_exact or real.startswith(prot_prefixes)

    # ---- walk sources
    src = WalkResult()
    if args.folders is None:
        for p in root_files:
            st = os.lstat(p)
            if os.path.islink(p):
                src.symlinks.append(p)
            elif os.path.isfile(p):
                src.files.append(FileRec(p, st.st_size, st.st_mtime_ns, (st.st_dev, st.st_ino)))
    for t in tops:
        print(f"Indexing {t} ...", file=sys.stderr)
        w = walk(t, skip=skip_dirs)
        src.files += w.files
        src.symlinks += w.symlinks
        src.errors += w.errors
    print(
        f"  {len(src.files)} source files, {human(sum(r.size for r in src.files))}",
        file=sys.stderr,
    )

    # ---- decide what needs hashing
    status: dict[str, tuple[str, str, str]] = {}  # path -> (status, reason, match)
    need_quick: list[FileRec] = []
    for r in src.files:
        if is_protected(r.path):
            status[r.path] = ("PROTECTED", "target of a symlink in REF", "")
        elif r.size == 0:
            status[r.path] = ("EMPTY", "", "")
        elif r.dev_ino in ref_inodes:
            status[r.path] = ("COVERED_HARDLINK", "", ref_inodes[r.dev_ino])
        elif r.size not in ref_sizes:
            status[r.path] = ("MISSING", "no REF file of this size", "")
        else:
            need_quick.append(r)

    cand_sizes = {r.size for r in need_quick}
    ref_cand = [r for r in ref_walk.files if r.size in cand_sizes]
    cache = HashCache(os.path.join(args.out, "hash_cache.sqlite"))
    errors: dict[str, str] = {}
    try:
        run_hash_phase(need_quick + ref_cand, "quick", cache, args.workers, errors)

        ref_by_quick: dict[tuple[int, str], list[FileRec]] = defaultdict(list)
        for r in ref_cand:
            q = cache.get(r, "quick")
            if q:
                ref_by_quick[(r.size, q)].append(r)

        need_full: list[FileRec] = []
        ref_full: dict[str, FileRec] = {}
        for r in need_quick:
            if r.path in errors:
                status[r.path] = ("ERROR", errors[r.path], "")
                continue
            key = (r.size, cache.get(r, "quick"))
            if key not in ref_by_quick:
                status[r.path] = ("MISSING", "same size in REF but content differs", "")
            elif args.quick_only:
                status[r.path] = ("COVERED_QUICK", "", ref_by_quick[key][0].path)
            else:
                need_full.append(r)
                for rr in ref_by_quick[key]:
                    ref_full[rr.path] = rr

        if not args.quick_only:
            # small files already have md5 from the quick pass; this only hashes big ones
            run_hash_phase(need_full + list(ref_full.values()), "md5", cache, args.workers, errors)
            ref_by_md5: dict[str, str] = {}
            for rr in ref_full.values():
                d = cache.get(rr, "md5")
                if d:
                    ref_by_md5.setdefault(d, rr.path)
            for r in need_full:
                d = cache.get(r, "md5")
                if r.path in errors or d is None:
                    status[r.path] = ("ERROR", errors.get(r.path, "hash failed"), "")
                elif d in ref_by_md5:
                    status[r.path] = ("COVERED", "", ref_by_md5[d])
                else:
                    status[r.path] = ("MISSING", "same size in REF but content differs", "")
    finally:
        cache.close()

    # ---- per-file report
    files_csv = os.path.join(args.out, "files.csv")
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if args.summary_depth == 1:  # make empty top-level folders show up too
        for t in tops:
            groups[os.path.relpath(t, root)]
    with open(files_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["group", "rel_path", "size", "status", "reason", "same_name_in_ref", "ref_match"])
        for r in sorted(src.files, key=lambda r: r.path):
            rel = os.path.relpath(r.path, root)
            g = group_of(rel, args.summary_depth)
            st, reason, match = status[r.path]
            wr.writerow([
                g, rel, r.size, st, reason,
                "yes" if st == "MISSING" and os.path.basename(r.path) in ref_names else "",
                os.path.relpath(match, root) if match else "",
            ])
            c = groups[g]
            c["files"] += 1
            c["bytes"] += r.size
            c[f"n_{st}"] += 1
            c[f"b_{st}"] += r.size
        for p in sorted(src.symlinks):
            rel = os.path.relpath(p, root)
            g = group_of(rel, args.summary_depth)
            wr.writerow([g, rel, 0, "SYMLINK", f"-> {os.readlink(p)}", "", ""])
            groups[g]["n_SYMLINK"] += 1
        for p, msg, is_dir in src.errors:
            rel = os.path.relpath(p, root)
            g = group_of(os.path.join(rel, "_") if is_dir else rel, args.summary_depth)
            wr.writerow([g, rel, "", "ERROR", f"could not list: {msg}", "", ""])
            groups[g]["n_ERROR"] += 1

    # ---- per-folder summary
    def verdict(c: dict[str, int]) -> str:
        if c["files"] == 0 and c["n_ERROR"] == 0:
            return "NO_FILES"
        bad = c["n_MISSING"] + c["n_PROTECTED"] + c["n_ERROR"]
        if bad == 0:
            return "QUICK_COVERED" if c["n_COVERED_QUICK"] else "FULLY_COVERED"
        if c["b_COVERED"] + c["b_COVERED_HARDLINK"] + c["b_COVERED_QUICK"] > 0:
            return "PARTIAL"
        return "NOT_COVERED"

    cols = [
        "group", "verdict", "files", "bytes", "covered_bytes", "missing_files",
        "missing_bytes", "protected_files", "errors", "hardlinked_bytes",
        "empty_files", "symlinks", "reclaimable_bytes",
    ]
    rows = []
    for g in sorted(groups):
        c = groups[g]
        v = verdict(c)
        covered = c["b_COVERED"] + c["b_COVERED_QUICK"] + c["b_COVERED_HARDLINK"]
        rows.append({
            "group": g, "verdict": v, "files": c["files"], "bytes": c["bytes"],
            "covered_bytes": covered,
            "missing_files": c["n_MISSING"], "missing_bytes": c["b_MISSING"],
            "protected_files": c["n_PROTECTED"], "errors": c["n_ERROR"],
            "hardlinked_bytes": c["b_COVERED_HARDLINK"],
            "empty_files": c["n_EMPTY"], "symlinks": c["n_SYMLINK"],
            # space freed by deleting the whole folder (hardlinks free nothing)
            "reclaimable_bytes": (c["bytes"] - c["b_COVERED_HARDLINK"])
            if v in ("FULLY_COVERED", "QUICK_COVERED") else 0,
        })
    summary_csv = os.path.join(args.out, "summary.csv")
    with open(summary_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    # ---- console table
    w = max([len(r["group"]) for r in rows] + [5])
    print(
        f"\n{'folder':<{w}}  {'verdict':<13}  {'size':>10}  {'covered':>10}  "
        f"{'missing':>10}  {'miss_files':>10}  notes"
    )
    for r in rows:
        notes = []
        if r["protected_files"]:
            notes.append(f"{r['protected_files']} PROTECTED (REF symlinks point here)")
        if r["errors"]:
            notes.append(f"{r['errors']} unreadable")
        print(
            f"{r['group']:<{w}}  {r['verdict']:<13}  {human(r['bytes']):>10}  "
            f"{human(r['covered_bytes']):>10}  {human(r['missing_bytes']):>10}  "
            f"{r['missing_files']:>10}  {'; '.join(notes)}"
        )
    total_reclaim = sum(r["reclaimable_bytes"] for r in rows)
    print(f"\nReclaimable by deleting FULLY_COVERED folders: {human(total_reclaim)}")
    if args.quick_only:
        print("NOTE: --quick-only was used; rerun without it before deleting anything.")
    if ref_walk.errors:
        print(f"NOTE: {len(ref_walk.errors)} entries in REF could not be read; they were not used as matches.")
    print(f"Wrote {summary_csv} and {files_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
