# misc

## check_4tu_coverage.py

Finds which folders under the project share
(`/tudelft.net/staff-umbrella/neon/cosilab_project`) are already fully
duplicated in `4TU_upload`, so they can be deleted to free space.

- Files are matched by content (MD5), not by name, so renamed or moved copies
  still count.
- It only reads and reports. It never deletes anything.
- Files whose size does not appear in `4TU_upload` are never read. Files with a
  matching size get a quick hash (first and last 1 MiB) first, and a full MD5
  only if the quick hash also matches.
- Hashes are cached in `<out>/hash_cache.sqlite`, so a rerun after a time-out
  continues where it stopped.

```bash
# fast first look (not proof of duplication)
python misc/check_4tu_coverage.py --quick-only --out ~/4tu_coverage

# full check, needed before deleting anything (reuses the cache above)
python misc/check_4tu_coverage.py --out ~/4tu_coverage --workers 8

# break one big folder down by subfolder
python misc/check_4tu_coverage.py --folders raw_data --summary-depth 3 --out ~/4tu_coverage
```

On DAIC, run the full check as a job (arguments are passed through to the script;
resubmit the same command to resume after a time-out):

```bash
mkdir -p /home/zli33/linuxhome/slurm_outputs/misc   # once
sbatch misc/job_scripts/check_4tu_coverage_daic.sh
sbatch misc/job_scripts/check_4tu_coverage_daic.sh --folders raw_data --summary-depth 2
```

Outputs:

- `summary.csv`: one row per folder, with a verdict (`FULLY_COVERED`,
  `PARTIAL`, `NOT_COVERED`, ...) and `reclaimable_bytes`.
- `files.csv`: one row per file, with its status, why it is `MISSING`, and the
  matching path in `4TU_upload`. For `MISSING` files, `same_name_in_ref=yes`
  means a file with that name exists in `4TU_upload` but its content differs,
  for example because it was re-encoded or redacted.

Before deleting a `FULLY_COVERED` folder:

- It must come from a full run, not `--quick-only`.
- No folder marked `PROTECTED` may be deleted: a symlink in `4TU_upload`
  points to it.
- A file that was processed before upload (redacted, trimmed, re-encoded) is
  reported as `MISSING` even though a derived version is in `4TU_upload`.
  Decide on those yourself; the script cannot.
