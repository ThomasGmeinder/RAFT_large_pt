# Applying `project_playbook_generation` to RAFT Large

The skill at `~/.agents/skills/project_playbook_generation/SKILL.md` turns an existing demo into a guided Jupyter notebook. This note records how that skill was applied to this repository, then the prompts that followed and what the agent did with each one.

## What the skill required

A colleague playbook, not a dump of commands. The notebook has to answer what the demo does, which inputs it needs, how to run a short smoke path, how to see the result, and how to change the input. It must not invent a devcontainer, and it must not commit weights or generated media. After writing the notebook, validate it.

## How it was applied

This repository already lives on the machine where the notebook was written. There is no `.devcontainer/`. Discovery used `README.md`, `infer_optical_flow.py`, and `requirements.txt`. The demo is one script, so it stayed one notebook:

`notebooks/raft_large_playbook.ipynb`

`FAST_MODE = True` runs one frame pair with `torch.compile` off. The whole video is a later cell. The README points at the notebook. Generated files go to `outputs/playbook/`, which is gitignored.

Jupyter was not installed, so `nbconvert` could not execute the notebook. Validation was a JSON check plus the smoke command in `.venv-rocm-10.0.0`, which wrote `outputs/playbook/flow_pair.png`. The cells below were added or changed after that first run.

## Where it landed

Branch `notebook/raft-large-playbook`, pushed to `origin` and `xilinx`.

| Commit | What it contains |
|--------|------------------|
| `592ee14` | Notebook, README pointer, `outputs/` and `.ipynb_checkpoints/` ignored |
| `4145caf` | JupyterLab in `setup_venv.sh` and `requirements.txt` |

Executed notebook output was cleared before the commit, so the git file does not contain the embedded MP4.

## Prompt log

### Which skills mention playbook generation

> which skills do you see regarding playbook generation

The agent found `~/.agents/skills/project_playbook_generation/SKILL.md`. It converts an existing demo into a guided Jupyter notebook and names RAFT optical flow as an example. It assumes Jupyter already exists and does not create `.devcontainer/` unless asked.

### Create the notebook

> apply the skill to this repo

The agent added `notebooks/raft_large_playbook.ipynb`, a README pointer, and `outputs/` in `.gitignore`. It reported the missing devcontainer, JSON validation, and a successful smoke run. `jupyter lab` was not installed yet.

### JupyterLab was missing

The terminal showed:

```text
jupyter lab notebooks/raft_large_playbook.ipynb
Jupyter command `jupyter-lab` not found.
```

Apt `jupyter-core` only installed the dispatcher. The agent installed JupyterLab 4.6.4 into `.venv-rocm-10.0.0` and added it to `setup_venv.sh` and `requirements.txt`.

### The kernel was not in the repository root

The first code cell raised:

```text
FileNotFoundError: Run this notebook with the repository root as the working directory.
```

Jupyter starts the kernel in `notebooks/`. The config cell now walks upward until it finds `infer_optical_flow.py`.

### Watch the generated video

> nice it worked. the only thing that is missing in the ipynb is a way to watch the generated video at the end

The agent added a **Watch the video** cell after the full-video run. It pointed a `<video>` tag at `/files/outputs/playbook/optical_flow.mp4`.

> I reloaded the notebook but do not see a watch the video cell

The cell was already on disk. A browser refresh keeps Jupyter’s open copy. The agent said to close the tab without saving and open the file again.

> player not working

The screenshot showed a player stuck at 0:00. The MP4 is valid H.264, about 6 seconds. JupyterLab does not serve that `/files/` URL to a video tag. The cell now uses `IPython.display.Video(..., embed=True)`.
