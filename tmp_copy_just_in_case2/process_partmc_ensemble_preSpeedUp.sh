```bash
#!/bin/bash
#SBATCH -A m1657
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -L scratch
#SBATCH --output=nersc_partmc_pool_save.%j.log

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ambrs_mosaic

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -n 4096

echo "Processing job cpus-per-task: ${SLURM_CPUS_PER_TASK:-unknown}"

# Required input from submission
: "${ENSEMBLE_ID:?ERROR: ENSEMBLE_ID not set. Submit with: sbatch --export=ALL,ENSEMBLE_ID=<jobid> ...}"

REPO_ROOT=/global/homes/l/lfierce/ambrs-project/part2pop

# Match run script output layout
ENSEMBLE_NAME="ensemble_${ENSEMBLE_ID}"
DATASET_ROOT=/global/cfs/cdirs/m1657/lfierce/partmc_output
DATASET_DIR="$DATASET_ROOT/$ENSEMBLE_NAME"
ARCHIVE="$DATASET_DIR/${ENSEMBLE_NAME}.tar"

# Fail fast if archive missing/empty
if [[ ! -s "$ARCHIVE" ]]; then
  echo "ERROR: Archive not found or empty: $ARCHIVE" >&2
  echo "Directory listing for: $DATASET_DIR" >&2
  ls -lah "$DATASET_DIR" >&2 || true
  exit 1
fi

SCRATCH_BASE="$PSCRATCH/partmc_processing/${SLURM_JOB_ID}"
UNPACK_DIR="$SCRATCH_BASE/unpacked"
PROCESSED_DIR="$SCRATCH_BASE/processed_data"
ANALYSIS_DIR="$SCRATCH_BASE/analysis"
mkdir -p "$UNPACK_DIR" "$PROCESSED_DIR" "$ANALYSIS_DIR"

echo "Unpacking ensemble archive: $ARCHIVE -> $UNPACK_DIR"
tar -xf "$ARCHIVE" -C "$UNPACK_DIR"

# -----------------------------------------------------------------------------
# Locate the correct partmc_runs directory robustly:
# choose a partmc_runs dir whose immediate children include numeric member dirs.
# This prevents accidentally selecting the wrong level and saving only 1 dataset.
# -----------------------------------------------------------------------------
PARTMC_RUNS_DIR=""
while IFS= read -r cand; do
  # must be a dir and have at least one numeric child directory
  if [[ -d "$cand" ]] && ls -1 "$cand" 2>/dev/null | grep -Eq '^[0-9]+$'; then
    PARTMC_RUNS_DIR="$cand"
    break
  fi
done < <(find "$UNPACK_DIR" -type d -path '*/partmc_output/*/partmc_runs' 2>/dev/null | sort)

if [[ -z "${PARTMC_RUNS_DIR:-}" ]]; then
  echo "ERROR: Could not find a partmc_runs directory with numeric member dirs under $UNPACK_DIR" >&2
  echo "Debug: all partmc_runs candidates:" >&2
  find "$UNPACK_DIR" -type d -name partmc_runs | head -n 50 >&2 || true
  exit 1
fi

RUN_DIR="$(dirname "$PARTMC_RUNS_DIR")"
RUN_NAME="$(basename "$RUN_DIR")"
echo "Selected PARTMC_RUNS_DIR: $PARTMC_RUNS_DIR"
echo "Selected RUN_NAME:        $RUN_NAME"
echo "Member dirs (sample):"
ls -1 "$PARTMC_RUNS_DIR" | head -n 30 || true

cd "$REPO_ROOT"

# Desired timesteps (what you'd like, when available)
REQUESTED_TIMESTEPS=(1 2 3 4 5 6 7 13 25 37 49 61 73 97 121 145)

# Detect max timestep from one representative out/ directory
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

echo "Detected max timestep in outputs: $MAX_T (sampled from $SAMPLE_OUTDIR)"

# Filter requested timesteps to those <= MAX_T
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

echo "Using timesteps: ${TIMESTEPS[*]}"

# -----------------------------------------------------------------------------
# 1) Save pool outputs (to scratch)
# IMPORTANT: run_save_pool.py writes to output_base/<ensemble_root.name>.
# Here, ensemble_root.name == "partmc_runs". To avoid collisions and to match
# the run identity, we set output_base to include RUN_NAME.
# Final layout: $PROCESSED_DIR/$RUN_NAME/partmc_runs/<member>/part2pop_population_*.npz
# -----------------------------------------------------------------------------
SAVE_OUTBASE="$PROCESSED_DIR/$RUN_NAME"
mkdir -p "$SAVE_OUTBASE"

echo "Saving processed populations under: $SAVE_OUTBASE"

PYTHONPATH="$REPO_ROOT/src" python scripts/run_save_pool.py \
  --ensemble-roots "$PARTMC_RUNS_DIR" \
  --output-base "$SAVE_OUTBASE" \
  --timesteps "${TIMESTEPS[@]}" \
  --num-processes "${SLURM_CPUS_PER_TASK}" \
  --rewrite \
  --include-bc-uncoated

echo "Processed data write complete."
echo "DEBUG: first few saved files:"
find "$SAVE_OUTBASE" -type f -name 'part2pop_population_*.npz' | head -n 50 || true

# -----------------------------------------------------------------------------
# 2) Run analysis pool (to scratch)
# Find ALL directories that contain part2pop_population_*.npz
# -----------------------------------------------------------------------------
mapfile -t SAVED_ROOTS < <(
  find "$SAVE_OUTBASE" -type f -name 'part2pop_population_*.npz' -printf '%h\n' \
  | sort -u
)

if [[ ${#SAVED_ROOTS[@]} -eq 0 ]]; then
  echo "ERROR: No saved population directories found under $SAVE_OUTBASE" >&2
  exit 1
fi

echo "Running analysis for ${#SAVED_ROOTS[@]} saved roots -> $ANALYSIS_DIR"

PYTHONPATH="$REPO_ROOT/src" python scripts/run_analysis_pool.py \
  --saved-roots "${SAVED_ROOTS[@]}" \
  --output-base "$ANALYSIS_DIR" \
  --num-processes "${SLURM_CPUS_PER_TASK}" \
  --thresholds 0.1 0.3 1.0 \
  --temperature 298.15

echo "Analysis outputs written to $ANALYSIS_DIR"

# -----------------------------------------------------------------------------
# 3) Bundle outputs to CFS
# -----------------------------------------------------------------------------
CFS_PROCESSED_DIR="$DATASET_DIR/processed_${SLURM_JOB_ID}"
mkdir -p "$CFS_PROCESSED_DIR"

PROCESSED_ARCHIVE="processed_${ENSEMBLE_NAME}_${SLURM_JOB_ID}.tar"
tar -cf "$CFS_PROCESSED_DIR/$PROCESSED_ARCHIVE" -C "$PROCESSED_DIR" .
sync
echo "Processed archive saved to: $CFS_PROCESSED_DIR/$PROCESSED_ARCHIVE"

ANALYSIS_ARCHIVE="analysis_${ENSEMBLE_NAME}_${SLURM_JOB_ID}.tar"
tar -cf "$CFS_PROCESSED_DIR/$ANALYSIS_ARCHIVE" -C "$ANALYSIS_DIR" .
sync
echo "Analysis archive saved to: $CFS_PROCESSED_DIR/$ANALYSIS_ARCHIVE"

# Optional: save this job's Slurm log alongside the archives
cp -f "nersc_partmc_pool_save.${SLURM_JOB_ID}.log" "$CFS_PROCESSED_DIR/" 2>/dev/null || true

echo "All results archived under: $CFS_PROCESSED_DIR"
```
