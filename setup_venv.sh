#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${SCRIPT_DIR}/.wheels"

usage() {
    cat <<'EOF'
Usage: setup_venv.sh <7.2.1|7.2.4|10.0.0>

  7.2.1   PyTorch wheels from repo.radeon.com (rocm-rel-7.2.1)
  7.2.4   PyTorch wheels from repo.radeon.com for the ROCm 7.2.4
          install on this machine (/opt/rocm-7.2.4)
  10.0.0  gfx1151 Python wheels only, from the TheRock index.
          Does not use a system ROCm install.
EOF
}

if [[ $# -ne 1 || "${1}" == "-h" || "${1}" == "--help" ]]; then
    usage
    [[ $# -eq 1 ]] || exit 2
    exit 0
fi

ROCM_VERSION="$1"
VENV_DIR="${SCRIPT_DIR}/.venv-rocm-${ROCM_VERSION}"

system_rocm_version() {
    local dest
    dest="$(readlink -f /opt/rocm 2>/dev/null || true)"
    if [[ -z "${dest}" || ! -d "${dest}" ]]; then
        return 0
    fi
    basename "${dest}" | sed -E 's/^rocm-//'
}

SYSTEM_ROCM="$(system_rocm_version)"
if [[ -n "${SYSTEM_ROCM}" ]]; then
    echo "System ROCm: ${SYSTEM_ROCM} ($(readlink -f /opt/rocm))"
else
    echo "System ROCm: none"
fi

case "${ROCM_VERSION}" in
    7.2.1)
        INSTALL_KIND="manylinux"
        BASE_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
        TRITON_WHL="triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"
        TORCH_WHL="torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
        VISION_WHL="torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
        VERSION_TAG="+rocm7.2.1"
        ;;
    7.2.4)
        INSTALL_KIND="manylinux"
        BASE_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4"
        TRITON_WHL="triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp312-cp312-linux_x86_64.whl"
        TORCH_WHL="torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp312-cp312-linux_x86_64.whl"
        VISION_WHL="torchvision-0.24.0%2Brocm7.2.4.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
        VERSION_TAG="+rocm7.2.4"
        if [[ "${SYSTEM_ROCM}" != "7.2.4" ]]; then
            echo "error: ROCm 7.2.4 is not the install on this machine (found: ${SYSTEM_ROCM:-none})" >&2
            exit 1
        fi
        ;;
    10.0.0)
        INSTALL_KIND="therock"
        ROCM_INDEX="https://stable.repo.amd.com/rocm/whl-next/"
        TORCH_SPEC='torch[device-gfx1151]==2.13.0+rocm10.0.0'
        VISION_SPEC='torchvision[device-gfx1151]==0.28.0+rocm10.0.0'
        VERSION_TAG="+rocm10.0.0"
        ;;
    *)
        echo "error: unsupported ROCm version '${ROCM_VERSION}'" >&2
        usage >&2
        exit 2
        ;;
esac

echo "=== RAFT Large — ROCm ${ROCM_VERSION} environment setup ==="

torch_version() {
    "${VENV_DIR}/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || true
}

# uv venv records an absolute path to the interpreter it found. A venv created
# in the devcontainer points at /usr/local/bin/python3.12 and at
# /workspaces/RAFT_large_pt, neither of which exists on the host. Relink to
# whatever `uv python find 3.12` resolves to here, and rewrite console-script
# shebangs, instead of deleting the installed wheels.
repair_interpreter() {
    local py home ver script first n=0
    py="$(uv python find 3.12)"
    if [[ ! -x "${py}" ]]; then
        echo "error: uv python find 3.12 returned '${py}', which is not executable" >&2
        exit 1
    fi
    echo "Linking ${VENV_DIR}/bin/python -> ${py}"
    ln -sfn "${py}" "${VENV_DIR}/bin/python"
    home="$(dirname "${py}")"
    ver="$("${py}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
    python3 - "${VENV_DIR}/pyvenv.cfg" "${home}" "${ver}" <<'PY'
import sys
path, home, ver = sys.argv[1:]
lines = []
for line in open(path):
    if line.startswith("home = "):
        lines.append(f"home = {home}\n")
    elif line.startswith("version_info = "):
        lines.append(f"version_info = {ver}\n")
    else:
        lines.append(line)
open(path, "w").writelines(lines)
PY
    local target="${VENV_DIR}/bin/python"
    for script in "${VENV_DIR}/bin/"*; do
        [[ -f "${script}" && ! -L "${script}" ]] || continue
        first="$(head -n 1 "${script}" 2>/dev/null || true)"
        if [[ "${first}" == '#!'*python* && "${first}" != "#!${target}" ]]; then
            python3 - "${script}" "${target}" <<'PY'
import pathlib, sys
path, target = sys.argv[1:]
file = pathlib.Path(path)
data = file.read_bytes()
nl = data.find(b"\n")
if nl < 0:
    raise SystemExit(0)
file.write_bytes(b"#!" + target.encode() + data[nl:])
PY
            n=$((n + 1))
        fi
    done
    echo "Rewrote ${n} script shebangs to ${target}"
    repair_activate
}

# The activate scripts hard-code the venv path they were created at.
recorded_venv_path() {
    sed -n "s/^VIRTUAL_ENV='\(.*\)'$/\1/p" "${VENV_DIR}/bin/activate" 2>/dev/null | head -n 1
}

repair_activate() {
    local old
    old="$(recorded_venv_path)"
    if [[ -z "${old}" || "${old}" == "${VENV_DIR}" ]]; then
        return 0
    fi
    echo "Rewriting activate scripts: ${old} -> ${VENV_DIR}"
    python3 - "${VENV_DIR}/bin" "${old}" "${VENV_DIR}" <<'PY'
import pathlib, sys
bindir, old, new = sys.argv[1:]
for path in pathlib.Path(bindir).glob("activate*"):
    if path.is_symlink() or not path.is_file():
        continue
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
PY
}

verify() {
    echo ""
    echo "=== Verification ==="
    "${VENV_DIR}/bin/python" -c "
import importlib.metadata as metadata
import torch, torchvision
print(f'PyTorch  : {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
try:
    print(f'ROCm pkg : {metadata.version(\"rocm\")}')
except metadata.PackageNotFoundError:
    pass
print(f'HIP build: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device 0 : {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: No GPU detected — inference will fall back to CPU.')
"
    echo ""
    echo "Setup complete.  Activate with:  source ${VENV_DIR}/bin/activate"
}

# Cursor and the Python extension discover ${workspace}/.venv only.
link_default_venv() {
    ln -sfn ".venv-rocm-${ROCM_VERSION}" "${SCRIPT_DIR}/.venv"
}

if [ -d "${VENV_DIR}" ]; then
    if ! "${VENV_DIR}/bin/python" -c 'import sys' >/dev/null 2>&1; then
        echo "Venv interpreter does not run. Repairing it in place."
        repair_interpreter
    else
        repair_activate
    fi
    current="$(torch_version)"
    if [[ "${current}" == *"${VERSION_TAG}"* ]]; then
        echo "ROCm ${ROCM_VERSION} already present in ${VENV_DIR} (${current})."
        link_default_venv
        verify
        exit 0
    fi
    echo "Existing venv is '${current:-unknown}', recreating for ROCm ${ROCM_VERSION} ..."
    rm -rf "${VENV_DIR}"
fi

PY_BIN="$(uv python find 3.12)"
echo "Creating Python 3.12 venv at ${VENV_DIR} (interpreter ${PY_BIN}) ..."
uv venv --python "${PY_BIN}" "${VENV_DIR}"

unquote() {
    python3 -c "import urllib.parse, sys; print(urllib.parse.unquote(sys.argv[1]))" "$1"
}

wheel_is_complete() {
    python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).namelist()' "$1" >/dev/null 2>&1
}

if [[ "${INSTALL_KIND}" == "manylinux" ]]; then
    mkdir -p "${WHEEL_DIR}"
    local_wheels=()
    for WHL in "${TRITON_WHL}" "${TORCH_WHL}" "${VISION_WHL}"; do
        LOCAL_NAME="$(unquote "${WHL}")"
        LOCAL_PATH="${WHEEL_DIR}/${LOCAL_NAME}"
        local_wheels+=("${LOCAL_PATH}")
        if [[ -f "${LOCAL_PATH}" ]] && wheel_is_complete "${LOCAL_PATH}"; then
            echo "Wheel already downloaded: ${LOCAL_NAME}"
            continue
        fi
        if [[ -f "${LOCAL_PATH}" ]]; then
            echo "Removing incomplete wheel: ${LOCAL_NAME}"
            rm -f "${LOCAL_PATH}"
        fi
        echo "Downloading ${LOCAL_NAME} ..."
        wget -q --show-progress -O "${LOCAL_PATH}.partial" "${BASE_URL}/${WHL}"
        mv "${LOCAL_PATH}.partial" "${LOCAL_PATH}"
        if ! wheel_is_complete "${LOCAL_PATH}"; then
            rm -f "${LOCAL_PATH}"
            echo "error: downloaded wheel is not a complete archive: ${LOCAL_NAME}" >&2
            exit 1
        fi
    done
    echo "Installing triton + PyTorch + torchvision (ROCm ${ROCM_VERSION}) ..."
    uv pip install --python "${VENV_DIR}/bin/python" --system-certs "${local_wheels[@]}"
else
    echo "Installing ${TORCH_SPEC} and ${VISION_SPEC} ..."
    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        --system-certs \
        --index "${ROCM_INDEX}" \
        "${TORCH_SPEC}" \
        "${VISION_SPEC}"
fi

echo "Installing remaining dependencies ..."
uv pip install --python "${VENV_DIR}/bin/python" --system-certs "numpy<2" opencv-python matplotlib jupyterlab

link_default_venv
verify
