#!/usr/bin/env bash
#SBATCH --job-name=ga-subfamily
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-ga-subfamily-%j.out
#SBATCH --error=logs/slurm-ga-subfamily-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv-idun/bin/activate

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_NAME="${RUN_NAME:-ga_subfamily_selection_v1}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ser-feature-subfamily-selection}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
FAMILIES="${FAMILIES:-mfcc_raw prosody_energy prosody_pitch rhythm_pauses tonality representations ssl_hubert ssl_wav2vec bert tfidf}"
FAMILY_SELECTION_JSON="${FAMILY_SELECTION_JSON:-}"
INCLUDE_XXX="${INCLUDE_XXX:-0}"
REQUIRE_AGREEMENT="${REQUIRE_AGREEMENT:-1}"
EXCLUDED_EMOTIONS="${EXCLUDED_EMOTIONS:-sur fea oth dis}"
POPULATION_SIZE="${POPULATION_SIZE:-48}"
GENERATIONS="${GENERATIONS:-30}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-random_forest}"
EMBEDDING_BLOCK_SIZE="${EMBEDDING_BLOCK_SIZE:-128}"
VALIDATION_MODE="${VALIDATION_MODE:-loso}"
GROUP_COLUMN="${GROUP_COLUMN:-session}"
RESUME="${RESUME:-1}"

echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Run name: ${RUN_NAME}"
echo "Report to: ${REPORT_TO}"
echo "Primary W&B project: ${WANDB_PROJECT}"
echo "Primary W&B entity: ${WANDB_ENTITY:-<default>}"
echo "Secondary W&B key set: $([[ -n "${WANDB_SECONDARY_API_KEY:-}" ]] && echo yes || echo no)"
echo "Secondary W&B project: ${WANDB_SECONDARY_PROJECT:-<default primary/mirror>}"
echo "Secondary W&B entity: ${WANDB_SECONDARY_ENTITY:-<default primary/mirror>}"
echo "Validation mode: ${VALIDATION_MODE}"

cmd=(
  python feature_selection/ga/ga_subfamily_selection.py
  --run-name "${RUN_NAME}"
  --report-to "${REPORT_TO}"
  --wandb-project "${WANDB_PROJECT}"
  --population-size "${POPULATION_SIZE}"
  --generations "${GENERATIONS}"
  --evaluator-model "${EVALUATOR_MODEL}"
  --embedding-block-size "${EMBEDDING_BLOCK_SIZE}"
  --validation-mode "${VALIDATION_MODE}"
  --group-column "${GROUP_COLUMN}"
  --excluded-emotions
)

for emotion in ${EXCLUDED_EMOTIONS}; do
  cmd+=("${emotion}")
done

if [[ -n "${FAMILY_SELECTION_JSON}" ]]; then
  cmd+=(--family-selection-json "${FAMILY_SELECTION_JSON}")
else
  cmd+=(--families)
  for family in ${FAMILIES}; do
    cmd+=("${family}")
  done
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  cmd+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${INCLUDE_XXX}" == "1" ]]; then
  cmd+=(--include-xxx)
else
  cmd+=(--no-include-xxx)
fi
if [[ "${REQUIRE_AGREEMENT}" == "1" ]]; then
  cmd+=(--require-agreement)
else
  cmd+=(--no-require-agreement)
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
else
  cmd+=(--no-resume)
fi

printf 'Command:\n  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

echo "End: $(date)"
