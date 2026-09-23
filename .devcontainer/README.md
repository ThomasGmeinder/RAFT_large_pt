# RAFT Large devcontainer

VS Code opens this repo with Python 3.12, `uv`, Jupyter, and the ROCm 10.0.0
virtualenv from `setup_venv.sh`.

## Base image

`mcr.microsoft.com/devcontainers/python:3.12-bookworm`.

The documented stack is the ROCm 10.0.0 Python wheels (`torch[device-gfx1151]`).
Those wheels ship the ROCm userspace and do not use a system ROCm install.
A `rocm/dev-ubuntu` 7.2 image would put host ROCm libraries next to those
wheels. This machine's issue log already shows that mix breaking MIOpen Find.

## GPU devices

`devcontainer.json` passes `/dev/kfd` and `/dev/dri`. The numeric groups are
this host:

| Group | GID | Check on another machine |
|-------|-----|--------------------------|
| `video` | 44 | `getent group video` |
| `render` | 992 | `getent group render` |

Replace `--group-add` if those GIDs differ, then rebuild.

## First open

Dev Containers: **Reopen in Container**. `postCreateCommand` runs
`bash setup_venv.sh 10.0.0`, then installs `requirements.txt` into that venv
and registers the `raft-large` Jupyter kernel.

RAFT Large weights download on the first inference into the named volume
`raft-large-pt-torch-cache` (`/home/vscode/.cache/torch`). They are not baked
into the image.

`Geisskopf_Gap_Jump.mp4` is gitignored. The devcontainer bind-mounts the
working tree, so a copy already in the repo root is visible. A fresh clone
needs the video placed there before the notebook smoke cell.

## Smoke test

Inside the container, after post-create finishes:

```bash
.venv-rocm-10.0.0/bin/python infer_optical_flow.py \
  --video Geisskopf_Gap_Jump.mp4 \
  --frame 156 \
  --output outputs/playbook/flow_pair.png \
  --compile False
```

Frame 156 is about 5.2 seconds into the 30 fps sample. The notebook
`notebooks/raft_large_playbook.ipynb` is the colleague walkthrough. Select the
**RAFT Large (ROCm 10)** kernel.
