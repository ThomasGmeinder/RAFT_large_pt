#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
WHEEL_DIR="${SCRIPT_DIR}/.wheels"

TRITON_WHL="triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"
TORCH_WHL="torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
VISION_WHL="torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
BASE_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"

echo "=== RAFT Large — ROCm 7.2.1 environment setup ==="

# --- virtual environment ---
if [ -d "${VENV_DIR}" ]; then
    echo "Existing venv found at ${VENV_DIR}, reusing it."
else
    echo "Creating Python 3.12 venv at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel

# --- AMD ROCm wheels ---
mkdir -p "${WHEEL_DIR}"

for WHL in "${TRITON_WHL}" "${TORCH_WHL}" "${VISION_WHL}"; do
    # The filename on disk has the literal '+' (URL-decoded)
    LOCAL_NAME="$(python3 -c "import urllib.parse, sys; print(urllib.parse.unquote(sys.argv[1]))" "${WHL}")"
    if [ -f "${WHEEL_DIR}/${LOCAL_NAME}" ]; then
        echo "Wheel already downloaded: ${LOCAL_NAME}"
    else
        echo "Downloading ${LOCAL_NAME} ..."
        wget -q --show-progress -O "${WHEEL_DIR}/${LOCAL_NAME}" "${BASE_URL}/${WHL}"
    fi
done

echo "Installing triton + PyTorch + torchvision (ROCm 7.2.1) ..."
pip install "${WHEEL_DIR}"/triton-*.whl "${WHEEL_DIR}"/torch-*.whl "${WHEEL_DIR}"/torchvision-*.whl

# --- pip dependencies ---
echo "Installing remaining dependencies ..."
pip install "numpy<2" opencv-python matplotlib

# --- verify ---
echo ""
echo "=== Verification ==="
python3 -c "
import torch, torchvision
print(f'PyTorch  : {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'HIP/ROCm : {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device 0 : {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: No GPU detected — inference will fall back to CPU.')
"

echo ""
echo "Setup complete.  Activate with:  source ${VENV_DIR}/bin/activate"
