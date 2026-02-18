#!/bin/bash
#SBATCH -A m1657
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -L scratch
#SBATCH --output=nersc_partmc_process_%x.%j.log

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ambrs_mosaic

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Try to raise the open-file limit to the hard limit (or 4096 minimum)
HARD_NOFILE="$(ulimit -Hn)"
echo "Hard nofile limit: $HARD_NOFILE"
if [[ "$HARD_NOFILE" =~ ^[0-9]+$ ]]; then
  # set soft to hard
  ulimit -Sn "$HARD_NOFILE" || true
fi

echo "ulimit -n (soft): $(ulimit -Sn)"
echo "ulimit -n (hard): $(ulimit -Hn)"


: "${CHUNK_ARCHIVE:?ERROR: CHUNK_ARCHIVE not set (full path to tar)}"
: "${ENSEMBLE_TAG:?ERROR: ENSEMBLE_TAG not set}"
: "${CHUNK_ID:?ERROR: CHUNK_ID not set}"

REPO_ROOT=/global/homes/l/lfierce/ambrs-project/part2pop

ARCHIVE="$CHUNK_ARCHIVE"
if [[ ! -s "$ARCHIVE" ]]; then
  echo "ERROR: Archive not found or empty: $ARCHIVE" >&2
  exit 1
fi

# Put final outputs next to the tarball
CHUNK_CFS_DIR="$(dirname "$ARCHIVE")"
CFS_OUTDIR="$CHUNK_CFS_DIR/postproc"
mkdir -p "$CFS_OUTDIR"

SCRATCH_BASE="$PSCRATCH/partmc_processing/${SLURM_JOB_ID}/${ENSEMBLE_TAG}/chunk_${CHUNK_ID}"
UNPACK_DIR="$SCRATCH_BASE/unpacked"
PROCESSED_DIR="$SCRATCH_BASE/processed_data"
ANALYSIS_DIR="$SCRATCH_BASE/analysis"
mkdir -p "$UNPACK_DIR" "$PROCESSED_DIR" "$ANALYSIS_DIR"

echo "Unpacking $ARCHIVE -> $UNPACK_DIR"
tar -xf "$ARCHIVE" -C "$UNPACK_DIR"

# -----------------------------------------------------------------------------
# Locate partmc_runs dir robustly
# We expect something like:
#   .../partmc_output/<run_name>/partmc_runs
# where <run_name> might be ensemble_...._chunk...
# -----------------------------------------------------------------------------
mapfile -t PARTMC_RUNS_CANDIDATES < <(
  find "$UNPACK_DIR" -type d -path '*/partmc_output/*/partmc_runs' 2>/dev/null | sort
)

if [[ ${#PARTMC_RUNS_CANDIDATES[@]} -eq 0 ]]; then
  echo "ERROR: Could not locate '*/partmc_output/*/partmc_runs' inside $UNPACK_DIR" >&2
  echo "Debug: candidate partmc_runs dirs:" >&2
  find "$UNPACK_DIR" -type d -name partmc_runs | head -n 50 >&2 || true
  exit 1
fi

# If multiple matches, prefer one that mentions this chunk id (common if tar includes chunk path)
PARTMC_RUNS_DIR=""
for cand in "${PARTMC_RUNS_CANDIDATES[@]}"; do
  if [[ "$cand" == *"chunk_${CHUNK_ID}"* ]]; then
    PARTMC_RUNS_DIR="$cand"
    break
  fi
done

# Otherwise take the first (should be fine for a single-chunk tar)
if [[ -z "$PARTMC_RUNS_DIR" ]]; then
  PARTMC_RUNS_DIR="${PARTMC_RUNS_CANDIDATES[0]}"
fi

if [[ ! -d "$PARTMC_RUNS_DIR" ]]; then
  echo "ERROR: Selected PARTMC_RUNS_DIR is not a directory: $PARTMC_RUNS_DIR" >&2
  exit 1
fi

ENSEMBLE_DIR="$(dirname "$PARTMC_RUNS_DIR")"
ENSEMBLE_DIRNAME="$(basename "$ENSEMBLE_DIR")"
echo "Found PARTMC_RUNS_DIR: $PARTMC_RUNS_DIR"
echo "Ensemble directory name: $ENSEMBLE_DIRNAME"

cd "$REPO_ROOT"

# ---- Auto-detect max timestep from one member out/ directory ----
REQUESTED_TIMESTEPS=(1 2 3 4 5 6 7 13 25 37 49 61 73 97 121 145)

SAMPLE_OUTDIR="$(find "$PARTMC_RUNS_DIR" -type d -path '*/out' | head -n 1)"
if [[ -z "${SAMPLE_OUTDIR:-}" || ! -d "$SAMPLE_OUTDIR" ]]; then
  echo "ERROR: Could not find any 'out' directory under $PARTMC_RUNS_DIR" >&2
  exit 1
fi

MAX_T="$(find "$SAMPLE_OUTDIR" -maxdepth 1 -type f -name '*.nc' \
  | sed -E 's/.*_0*([0-9]+)\.nc$/\1/' \
  | sort -n \
  | tail -n 1)"

if [[ -z "${MAX_T:-}" || ! "$MAX_T" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Could not detect integer MAX_T from $SAMPLE_OUTDIR" >&2
  exit 1
fi

TIMESTEPS=()
for t in "${REQUESTED_TIMESTEPS[@]}"; do
  if (( t <= MAX_T )); then
    TIMESTEPS+=("$t")
  fi
done

if [[ ${#TIMESTEPS[@]} -eq 0 ]]; then
  echo "ERROR: No timesteps selected. MAX_T=$MAX_T, requested: ${REQUESTED_TIMESTEPS[*]}" >&2
  exit 1
fi

echo "Using timesteps: ${TIMESTEPS[*]} (MAX_T=$MAX_T)"

# 1) Save pool outputs
SAVE_OUTBASE="$PROCESSED_DIR/$ENSEMBLE_DIRNAME"
mkdir -p "$SAVE_OUTBASE"

PYTHONPATH="$REPO_ROOT/src" python scripts/run_save_pool.py \
  --ensemble-roots "$PARTMC_RUNS_DIR" \
  --output-base "$SAVE_OUTBASE" \
  --timesteps "${TIMESTEPS[@]}" \
  --num-processes ${SLURM_CPUS_PER_TASK} \
  --rewrite \
  --include-bc-uncoated

# 2) Run analysis pool: discover all saved-root directories via .npz
mapfile -t SAVED_ROOTS < <(
  find "$SAVE_OUTBASE" -type f -name 'part2pop_population_*.npz' -printf '%h\n' \
  | sort -u
)

if [[ ${#SAVED_ROOTS[@]} -eq 0 ]]; then
  echo "ERROR: No saved population directories found under $SAVE_OUTBASE" >&2
  exit 1
fi

PYTHONPATH="$REPO_ROOT/src" python scripts/run_analysis_pool.py \
  --saved-roots "${SAVED_ROOTS[@]}" \
  --output-base "$ANALYSIS_DIR" \
  --num-processes ${SLURM_CPUS_PER_TASK} \
  --thresholds 0.1 0.3 1.0 \
  --temperature 298.15

# 3) Archive outputs to CFS (next to chunk tar)
PROCESSED_ARCHIVE="processed_${ENSEMBLE_TAG}_chunk${CHUNK_ID}_${SLURM_JOB_ID}.tar"
tar -cf "$CFS_OUTDIR/$PROCESSED_ARCHIVE" -C "$PROCESSED_DIR" .
sync

ANALYSIS_ARCHIVE="analysis_${ENSEMBLE_TAG}_chunk${CHUNK_ID}_${SLURM_JOB_ID}.tar"
tar -cf "$CFS_OUTDIR/$ANALYSIS_ARCHIVE" -C "$ANALYSIS_DIR" .
sync

cp -f "nersc_partmc_process_${SLURM_JOB_ID}.log" "$CFS_OUTDIR/" 2>/dev/null || true
echo "Done. Archives in: $CFS_OUTDIR"
