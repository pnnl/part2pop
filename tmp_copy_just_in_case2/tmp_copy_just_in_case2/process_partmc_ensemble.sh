#!/bin/bash
#SBATCH -A m1657
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

: "${ENSEMBLE_ID:?ERROR: ENSEMBLE_ID not set. Use: sbatch --export=ALL,ENSEMBLE_ID=<jobid> ...}"

PART2POP_ROOT=/global/homes/l/lfierce/ambrs-project/part2pop
RUNROOT="$PSCRATCH/ambrs_runs/${ENSEMBLE_ID}"
PARTMC_OUT="$RUNROOT/partmc_output"
PROC_OUT="$RUNROOT/processed_data"

if [[ ! -d "$PARTMC_OUT" ]]; then
  echo "ERROR: Missing $PARTMC_OUT" >&2
  exit 1
fi

mkdir -p "$PROC_OUT"

SAVE_NPROC="${SAVE_NPROC:-${SLURM_CPUS_PER_TASK}}"
TIMESTEPS_STR="${TIMESTEPS:-"1 2 3 4 5 6 7 13 25 37 49 61 73 97 121 145"}"

cd "$PART2POP_ROOT"
PYTHONPATH="$PART2POP_ROOT/src" python scripts/save_partmc_ensemble.py \
  --ensemble-root "$PARTMC_OUT" \
  --output-root "$PROC_OUT" \
  --timesteps $TIMESTEPS_STR \
  --repeat 1 \
  --temperature 298.15 \
  --num-processes "$SAVE_NPROC" \
  --rewrite

# Archive to CFS
CFS_BASEDIR=/global/cfs/cdirs/m1657/lfierce/partmc_output
CFS_OUTDIR="$CFS_BASEDIR/ensemble_${ENSEMBLE_ID}"
mkdir -p "$CFS_OUTDIR"

ARCHIVE="processed_only_ensemble_${ENSEMBLE_ID}_procjob_${SLURM_JOB_ID}.tar"
tar -cf "$CFS_OUTDIR/$ARCHIVE" -C "$RUNROOT" processed_data
sync
echo "Saved processed archive to: $CFS_OUTDIR/$ARCHIVE"
