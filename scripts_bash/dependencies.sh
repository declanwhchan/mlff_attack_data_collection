#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project}"
MLFF_ATTACK_REPO="${MLFF_ATTACK_REPO:-https://github.com/TRustworthy-AI-Tools-for-Science/mlff_attack.git}"

cd "$PROJECT_DIR"

module load gcc/12.3 python/3.11 arrow

if [ ! -d "$PROJECT_DIR/mlff_attack" ]; then
  git clone "$MLFF_ATTACK_REPO" "$PROJECT_DIR/mlff_attack"
else
  echo "Using existing $PROJECT_DIR/mlff_attack"
fi

setup_env() {
  local env_name="$1"
  local extra_package="$2"
  local env_path="$PROJECT_DIR/$env_name"

  if [ ! -d "$env_path" ]; then
    python -m venv "$env_path" --system-site-packages
  else
    echo "Using existing $env_path"
  fi

  source "$env_path/bin/activate"

  python -m pip install --upgrade pip setuptools wheel

  python -m pip install -e "$PROJECT_DIR/mlff_attack"

  python -m pip install "$extra_package"

  python -m pip install \
    pymatgen \
    spglib \
    pytest \
    boto3 \
    botocore \
    s3transfer

  python -m pip install --no-deps mp-api

  python - <<'PY'
import boto3
import botocore
from mp_api.client import MPRester

print("boto3", boto3.__version__)
print("botocore", botocore.__version__)
print("mp_api import OK")
PY

  deactivate
}

setup_env ".venv-mace" "mace-torch"
setup_env ".venv-uma" "fairchem-core"

echo "Environment setup complete."