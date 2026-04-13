#!/bin/bash
#SBATCH --job-name=exhaustive-family
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-exhaustive-family-%j.out
#SBATCH --error=logs/slurm-exhaustive-family-%j.err

set -euo pipefail

mkdir -p logs

cd "${SLURM_SUBMIT_DIR}"

module purge
module load Python/3.11.5-GCCcore-13.2.0

if [[ -d ".venv-idun" ]]; then
  source .venv-idun/bin/activate
elif [[ -d ".venv" ]]; then
  source .venv/bin/activate
else
  echo "No virtual environment found (.venv-idun or .venv)." >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

RUN_NAME="${RUN_NAME:-exhaustive_feature_family_selection_without_xxx_v1}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ser-feature-family-selection}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
INCLUDE_XXX="${INCLUDE_XXX:-0}"
REQUIRE_AGREEMENT="${REQUIRE_AGREEMENT:-1}"
EXCLUDED_EMOTIONS="${EXCLUDED_EMOTIONS:-sur fea oth dis}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-random_forest}"
TEST_SIZE="${TEST_SIZE:-0.20}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.05}"
RANDOM_STATE="${RANDOM_STATE:-42}"
FAMILIES="${FAMILIES:-mfcc_raw prosody_energy prosody_pitch ssl_hubert ssl_wav2vec bert tfidf tonality rhythm_pauses representations}"

CLI_ARGS=(
  --run-name "${RUN_NAME}"
  --report-to "${REPORT_TO}"
  --evaluator-model "${EVALUATOR_MODEL}"
  --test-size "${TEST_SIZE}"
  --alpha "${ALPHA}"
  --beta "${BETA}"
  --random-state "${RANDOM_STATE}"
  --families ${FAMILIES}
  --excluded-emotions ${EXCLUDED_EMOTIONS}
)

if [[ "${REPORT_TO}" == "wandb" ]]; then
  CLI_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CLI_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

if [[ "${INCLUDE_XXX}" == "1" ]]; then
  CLI_ARGS+=(--include-xxx)
else
  CLI_ARGS+=(--no-include-xxx)
fi

if [[ "${REQUIRE_AGREEMENT}" == "1" ]]; then
  CLI_ARGS+=(--require-agreement)
else
  CLI_ARGS+=(--no-require-agreement)
fi

python feature_family_selection/exhaustive_search/exhaustive_family_selection.py "${CLI_ARGS[@]}"
