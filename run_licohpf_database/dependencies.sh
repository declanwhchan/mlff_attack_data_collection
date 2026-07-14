#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/project}"
MLFF_REPO="$PROJECT_ROOT/mlff_attack"
DATA_REPO="$PROJECT_ROOT/mlff_attack_data_collection"

MACE_ENV="$PROJECT_ROOT/.venv-mace"
UMA_ENV="$PROJECT_ROOT/.venv-uma"
CHGNET_ENV="$PROJECT_ROOT/.venv-chgnet"
MTP_ENV="$PROJECT_ROOT/.venv-mtp"

MLIP_ROOT="$PROJECT_ROOT/mlip-3"
MLIP_BIN="$MLIP_ROOT/bin/mlp"
MTP_BIN="$MTP_ENV/bin/mlp"

echo "Project root: $PROJECT_ROOT"

for required_dir in "$MLFF_REPO" "$DATA_REPO"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "ERROR: Missing directory: $required_dir" >&2
        exit 1
    fi
done

create_python_env() {
    local env_path="$1"

    if [[ ! -x "$env_path/bin/python" ]]; then
        echo "Creating Python environment: $env_path"
        python3 -m venv "$env_path"
    fi

    "$env_path/bin/python" -m pip install --upgrade pip setuptools wheel
    "$env_path/bin/python" -m pip install -e "$MLFF_REPO"
}

create_python_env "$MACE_ENV"
create_python_env "$UMA_ENV"
create_python_env "$CHGNET_ENV"

if [[ ! -x "$MLIP_BIN" ]]; then
    echo "ERROR: Missing MTP executable:"
    echo "       $MLIP_BIN"
    echo "Install/build MLIP-3 separately; do not use pip for MTP."
    exit 1
fi

if [[ ! -d "$MTP_ENV" ]]; then
    echo "Creating MTP Python environment: $MTP_ENV"
    python3 -m venv "$MTP_ENV"
fi

mkdir -p "$MTP_ENV/bin"

if [[ ! -x "$MTP_BIN" ]]; then
    cp "$MLIP_BIN" "$MTP_BIN"
    chmod u+x "$MTP_BIN"
fi

cat > "$DATA_REPO/.env" <<EOF
PROJECT_ROOT=$PROJECT_ROOT
MLFF_REPO=$MLFF_REPO
DATA_REPO=$DATA_REPO

MACE_ENV=$MACE_ENV
UMA_ENV=$UMA_ENV
CHGNET_ENV=$CHGNET_ENV
MTP_ENV=$MTP_ENV

MACE_PYTHON=$MACE_ENV/bin/python
UMA_PYTHON=$UMA_ENV/bin/python
CHGNET_PYTHON=$CHGNET_ENV/bin/python
MTP_PYTHON=$MTP_ENV/bin/python

MLIP_ROOT=$MLIP_ROOT
MLP_BIN=$MTP_BIN

MACE_MH_MODEL=$DATA_REPO/mace-mh-1.model
UMA_MODEL=$DATA_REPO/uma-s-1p1.pt
CHGNET_MODEL=$DATA_REPO/pot.almtp
MACE_MODEL=$DATA_REPO/MACE_model.model
EOF

for model_file in \
    "$DATA_REPO/mace-mh-1.model" \
    "$DATA_REPO/uma-s-1p1.pt" \
    "$DATA_REPO/pot.almtp" \
    "$DATA_REPO/MACE_model.model"; do

    if [[ ! -f "$model_file" ]]; then
        echo "WARNING: Model file not found: $model_file"
    fi
done

echo
echo "Environment layout:"
echo "  $MLFF_REPO"
echo "  $DATA_REPO"
echo "  $DATA_REPO/.env"
echo "  $MACE_ENV"
echo "  $UMA_ENV"
echo "  $CHGNET_ENV"
echo "  $MLIP_BIN"
echo "  $MTP_BIN"

echo
echo "Validation:"
"$MACE_ENV/bin/python" --version
"$UMA_ENV/bin/python" --version
"$CHGNET_ENV/bin/python" --version
"$MTP_ENV/bin/python" --version
"$MTP_BIN" list | head -10

echo
echo "DEPENDENCIES SETUP PASSED"