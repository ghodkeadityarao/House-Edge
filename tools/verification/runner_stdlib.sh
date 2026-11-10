#!/bin/bash
# Imitate the target runner: Python 3.13 standard library only (no NumPy installed), two CPUs,
# 16 GiB, read-only /submission and /data, separate scratch /tmp, writable /output, no network.
set -e
SP="$1"; UVW="$2"; ENTRY="$3"; COPY_TO="$4"
KIT="$5"
BASE=$(mktemp -d /tmp/he_stdlib_XXXXXX)
mkdir -p "$BASE/uv" "$BASE/py" "$BASE/submission" "$BASE/data" "$BASE/output" "$BASE/scratch"
cd "$BASE/uv"; python3 -m zipfile -e "$UVW" .
UV=$(find "$BASE/uv" -name uv -type f | head -1); chmod +x "$UV"
UV_PYTHON_INSTALL_DIR="$BASE/py" "$UV" python install 3.13 2>&1 | tail -1
PY=$(find "$BASE/py" -type f -name 'python3.13' | head -1)
"$PY" --version
if "$PY" -s -c 'import numpy' 2>/dev/null; then echo "numpy unexpectedly installed"; exit 1; else echo "numpy: not installed"; fi
python3 -m zipfile -e "$ENTRY" "$BASE/submission"
cp -r "$KIT/public/test" "$BASE/data/test"; cp "$KIT/public/metadata.json" "$BASE/data/"
chmod -R a-w "$BASE/submission" "$BASE/data"
cd "$BASE/submission"
ulimit -v 17179869184
START=$(date +%s)
RUN="TMPDIR=$BASE/scratch OMP_NUM_THREADS=2 taskset -c 0,1 $PY -s predict.py --data $BASE/data --out $BASE/output/submission.csv"
if unshare -rn true 2>/dev/null; then echo "network: disabled"; unshare -rn bash -c "$RUN"; else echo "network: could not disable"; bash -c "$RUN"; fi
echo "elapsed seconds: $(( $(date +%s) - START ))"
echo "scratch used: $(du -sm "$BASE/scratch" | cut -f1) MiB (limit 128)"
echo "output files: $(ls "$BASE/output")"
echo "files written into submission: $(find "$BASE/submission" -newer "$BASE/data/metadata.json" -type f | wc -l)"
cp "$BASE/output/submission.csv" "$COPY_TO"
