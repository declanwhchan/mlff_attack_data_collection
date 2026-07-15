#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project}"
MLFF_ATTACK_DIR="$PROJECT_DIR/mlff_attack"
DATA_REPO="$PROJECT_DIR/mlff_attack_data_collection"

MACE_ENV="$PROJECT_DIR/.venv-mace"
UMA_ENV="$PROJECT_DIR/.venv-uma"
CHGNET_ENV="$PROJECT_DIR/.venv-chgnet"
MTP_ENV="$PROJECT_DIR/.venv-mtp"

MLIP_DIR="$PROJECT_DIR/mlip-3"
MLIP_BIN_DIR="$MLIP_DIR/bin"
MLIP_COMMAND="$MLIP_BIN_DIR/mlp"

module load gcc/12.3 python/3.11 arrow

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory does not exist:"
    echo "$PROJECT_DIR"
    exit 1
fi

if [ ! -d "$MLFF_ATTACK_DIR" ]; then
    echo "ERROR: Uploaded mlff_attack repository is missing:"
    echo "$MLFF_ATTACK_DIR"
    exit 1
fi

if [ ! -f "$MLFF_ATTACK_DIR/pyproject.toml" ] && \
   [ ! -f "$MLFF_ATTACK_DIR/setup.py" ]; then
    echo "ERROR: $MLFF_ATTACK_DIR is not a valid"
    echo "Python repository."
    exit 1
fi

if [ ! -d "$DATA_REPO" ]; then
    echo "ERROR: Data-collection repository is missing:"
    echo "$DATA_REPO"
    exit 1
fi

required_model_files=(
    "$DATA_REPO/mace-mh-1.model"
    "$DATA_REPO/uma-s-1p1.pt"
    "$DATA_REPO/pot.almtp"
    "$DATA_REPO/pot.almtp.elements"
    "$DATA_REPO/MACE_model.model"
)

for model_file in "${required_model_files[@]}"; do
    if [ ! -f "$model_file" ]; then
        echo "ERROR: Required model file is missing:"
        echo "$model_file"
        exit 1
    fi
done

create_python_environment() {
    local environment_path="$1"
    local primary_package="$2"
    local import_name="$3"

    echo
    echo "Preparing environment:"
    echo "$environment_path"

    if [ ! -x "$environment_path/bin/python" ]; then
        echo "Creating environment."
        python -m venv \
            "$environment_path" \
            --system-site-packages
    else
        echo "Using existing environment."
    fi

    local environment_python
    environment_python="$environment_path/bin/python"

    "$environment_python" -m pip install \
        --upgrade \
        pip \
        setuptools \
        wheel

    "$environment_python" -m pip install \
        "$primary_package"

    "$environment_python" -m pip install \
        ase \
        numpy \
        pandas \
        scipy \
        matplotlib \
        seaborn \
        pymatgen \
        spglib \
        pyarrow \
        psutil \
        pytest \
        boto3 \
        botocore \
        s3transfer

    "$environment_python" -m pip install \
        mp-api

    "$environment_python" -m pip install \
        --no-deps \
        --editable \
        "$MLFF_ATTACK_DIR"

    "$environment_python" - \
        "$import_name" <<'PY'
import importlib
import sys


module_name = sys.argv[1]
module = importlib.import_module(module_name)

print("Python:", sys.executable)
print("Imported:", module.__name__)
PY
}

create_python_environment \
    "$MACE_ENV" \
    "mace-torch" \
    "mace"

create_python_environment \
    "$UMA_ENV" \
    "fairchem-core" \
    "fairchem"

create_python_environment \
    "$CHGNET_ENV" \
    "chgnet" \
    "chgnet"

echo
echo "Validating uploaded MTP environment:"
echo "$MTP_ENV"

if [ ! -x "$MTP_ENV/bin/python" ]; then
    echo "ERROR: Uploaded MTP Python is missing:"
    echo "$MTP_ENV/bin/python"
    echo
    echo "Upload and unpack env-mtp.tar.gz first."
    exit 1
fi

if [ ! -x "$MTP_ENV/bin/mlp" ]; then
    echo "ERROR: Uploaded MTP executable is missing:"
    echo "$MTP_ENV/bin/mlp"
    echo
    echo "Do not install MTP using pip."
    exit 1
fi

MTP_PYTHON="$MTP_ENV/bin/python"
MTP_EXECUTABLE="$MTP_ENV/bin/mlp"

"$MTP_EXECUTABLE" list | head -n 10

echo
echo "Installing only the local mlff_attack Python"
echo "wrapper into .venv-mtp."
echo "The MLIP executable itself is not installed"
echo "or modified by pip."

"$MTP_PYTHON" -m pip install \
    --no-deps \
    --editable \
    "$MLFF_ATTACK_DIR"

"$MTP_PYTHON" - <<'PY'
import sys
import mlff_attack

print("MTP Python:", sys.executable)
print("mlff_attack:", mlff_attack.__file__)
PY

mkdir -p "$MLIP_BIN_DIR"

if [ -e "$MLIP_COMMAND" ] || \
   [ -L "$MLIP_COMMAND" ]; then
    if [ ! -x "$MLIP_COMMAND" ]; then
        echo "ERROR: Existing MLIP command is not executable:"
        echo "$MLIP_COMMAND"
        exit 1
    fi

    echo "Using existing MLIP command:"
    echo "$MLIP_COMMAND"
else
    ln -s "$MTP_EXECUTABLE" "$MLIP_COMMAND"

    echo "Created MLIP command link:"
    echo "$MLIP_COMMAND"
    echo "-> $MTP_EXECUTABLE"
fi

echo
echo "Running environment checks."

"$MACE_ENV/bin/python" - <<'PY'
import ase
import mace
import mlff_attack
import mp_api
import pymatgen
import spglib
import torch

print("MACE environment passed")
print("CUDA build:", torch.version.cuda)
PY

"$UMA_ENV/bin/python" - <<'PY'
import ase
import fairchem
import mlff_attack
import pymatgen
import spglib
import torch

print("UMA environment passed")
PY

"$CHGNET_ENV/bin/python" - <<'PY'
import ase
import chgnet
import mlff_attack
import pymatgen
import spglib
import torch

print("CHGNet environment passed")
PY

PATH="$MTP_ENV/bin:$MLIP_BIN_DIR:$PATH" \
    "$MTP_PYTHON" - <<'PY'
from pathlib import Path
import shutil
import sys

import ase
import mlff_attack
import numpy

mlp = shutil.which("mlp")

if mlp is None:
    raise SystemExit(
        "ERROR: mlp is unavailable in the MTP environment"
    )

print("MTP environment passed")
print("Python:", sys.executable)
print("mlp:", Path(mlp).resolve())
PY

echo
echo "Dependency setup complete."
echo
echo "Expected project layout:"
echo "$PROJECT_DIR/"
echo "  mlff_attack/"
echo "  mlff_attack_data_collection/"
echo "  .venv-mace/"
echo "  .venv-uma/"
echo "  .venv-chgnet/"
echo "  .venv-mtp/"
echo "  mlip-3/bin/mlp"
