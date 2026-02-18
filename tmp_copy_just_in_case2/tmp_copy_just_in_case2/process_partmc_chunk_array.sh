#!/bin/bash
#SBATCH -A m1657
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -L scratch
#SBATCH --output=nersc_partmc_process_array_%x.%A_%a.log
#SBATCH --array=0-49%20

set -euo pipefail

: "${ENSEMBLE_TAG:?ERROR: ENSEMBLE_TAG not set (e.g. ensemble_47912345)}"

CFS_BASEDIR=/global/cfs/cdirs/m1657/lfierce/partmc_output
CHUNK_ID="${SLURM_ARRAY_TASK_ID}"
CHUNK_DIR="$CFS_BASEDIR/${ENSEMBLE_TAG}/chunk_${CHUNK_ID}"

if [[ ! -d "$CHUNK_DIR" ]]; then
  echo "ERROR: Missing chunk directory: $CHUNK_DIR" >&2
  exit 1
fi

# Pick the newest tar in the chunk directory (robust if you rerun chunk)
CHUNK_ARCHIVE="$(ls -1t "$CHUNK_DIR"/*.tar 2>/dev/null | head -n 1 || true)"
if [[ -z "${CHUNK_ARCHIVE:-}" || ! -s "$CHUNK_ARCHIVE" ]]; then
  echo "ERROR: No .tar archive found (or empty) in $CHUNK_DIR" >&2
  echo "Debug listing:" >&2
  ls -lah "$CHUNK_DIR" >&2 || true
  exit 1
fi

echo "Processing: ENSEMBLE_TAG=$ENSEMBLE_TAG CHUNK_ID=$CHUNK_ID"
echo "Using CHUNK_ARCHIVE=$CHUNK_ARCHIVE"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}"

PROCESS_SCRIPT=/global/homes/l/lfierce/ambrs-project/part2pop/process_partmc_chunk.sh
exec sbatch --wait \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --export=ALL,CHUNK_ARCHIVE="$CHUNK_ARCHIVE",ENSEMBLE_TAG="$ENSEMBLE_TAG",CHUNK_ID="$CHUNK_ID" \
  "$PROCESS_SCRIPT"
