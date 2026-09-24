# Containerizing RAFT Large after `project_playbook_generation`

The skill at `~/.agents/skills/project_playbook_generation/SKILL.md` turns an existing demo into a guided Jupyter notebook. It must not invent a `.devcontainer/`. This note records the container that was added so that notebook can run, then the prompts that followed.

The git release did not reach GitHub. `git push` stopped on `Permission denied (publickey)`. This container has no GitHub SSH key.

## What the skill required

A colleague notebook, not a container. Discovery used the repository as it already lived on the machine. The skill assumes Jupyter exists and leaves `.devcontainer/` out unless a later request asks for it. Weights and generated media stay uncommitted.

That boundary is why the playbook report says there is no `.devcontainer/`. The container in this note is the follow-up, not part of that skill's output.

`~/.agents/skills/` is not present inside the image. The skill file lives on the host. The container cannot reread it, and it cannot authenticate to GitHub either.

## How it was applied

`.devcontainer/` opens the repository with Python 3.12, `uv`, Jupyter, and the ROCm 10.0.0 wheels from `setup_venv.sh`. The base image is `mcr.microsoft.com/devcontainers/python:3.12-bookworm`. Those wheels ship their own ROCm userspace. A `rocm/dev-ubuntu` 7.2 image was not used, because mixing host ROCm libraries with the ROCm 10 wheels already breaks MIOpen Find on this machine.

`devcontainer.json` passes `/dev/kfd` and `/dev/dri`, plus this host's `video` (44) and `render` (992) groups. `postCreateCommand` runs `bash setup_venv.sh 10.0.0`, installs `requirements.txt` with `uv`, and registers the **RAFT Large (ROCm 10)** kernel. Torch weights go to the named volume `raft-large-pt-torch-cache` at `/home/vscode/.cache/torch`.

The first container create failed before that setup finished. Later cells of `notebooks/raft_large_playbook.ipynb` ran only after the fixes below. The one-pair smoke image is `outputs/playbook/flow_pair.png`. The whole-video file is `outputs/playbook/optical_flow.mp4`. Both paths are gitignored.

## Where it landed

Local `master` only. `origin/master` does not have this commit. The tag is local only.

| Commit | What it contains |
|--------|------------------|
| `8be3f88` | `.devcontainer/`, `.dockerignore`, README pointer, notebook instructions, `.venv` symlink in `setup_venv.sh`, VAAPI probe that falls back to libx264 |

Annotated tag `v1.0.0` points at `8be3f88` with the message `RAFT Large ROCm 10 playbook`.

Left untracked on purpose: `.venv` (a symlink to `.venv-rocm-10.0.0`), `.claude/`, and `AGENT_TODO.md`.

## Permission denied (publickey)

This is the failure that matters for the release.

`origin` is `git@github.com:ThomasGmeinder/RAFT_large_pt.git`. The repository is already on GitHub. `master` on the remote matched the parent of `8be3f88` before the release commit. Pushing that commit and the tag was the whole release step.

From inside the container:

```text
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```

This container has no GitHub SSH key. `git push origin master` and `git push origin v1.0.0` both die on that line. The commit and the tag were created first, so they exist here and nowhere else. A host that has the key can publish them:

```bash
git push origin master
git push origin v1.0.0
```

Nothing in `.devcontainer/` copies `~/.ssh` into the image, and this session did not create a key or change git remotes. The same isolation shows up as a missing `~/.agents/skills/` directory: host credentials and host skills are outside the container unless they are mounted.

## Prompt log

### First create could not build the venv

The devcontainer terminal ended with:

```text
System ROCm: none
=== RAFT Large — ROCm 10.0.0 environment setup ===
Existing venv is 'unknown', recreating for ROCm 10.0.0 ...
Creating Python 3.12 venv at /workspaces/RAFT_large_pt/.venv-rocm-10.0.0 ...
error: Failed to initialize cache at `/home/vscode/.cache/uv`
  cause: failed to create directory `/home/vscode/.cache/uv`: Permission denied
         (os error 13)
```

Docker creates missing parents of a volume mount as root. The torch cache is mounted at `/home/vscode/.cache/torch`, so `/home/vscode/.cache` was `root:root` and `uv` could not create `.cache/uv`. The image now creates that directory as `vscode` before the mount. `postCreateCommand` still runs `sudo chown` because an already-created volume stays root-owned. A `uv` venv has no `pip` module, so the requirements install uses `uv pip`.

### The notebook kernel was not listed

> how to run ipynb now in container from Cursor ?

The registered kernel is **RAFT Large (ROCm 10)**, interpreter `.venv-rocm-10.0.0/bin/python`.

> No existing Jupyter server available for selection !!

That list is for a Jupyter process that is already running. This container does not start one. Cursor starts the kernel from the virtualenv.

> .venv-rocm-10.0.0/bin/python not available for selection. only Global Env

Cursor treats a project environment named `.venv` as the selectable one. The ROCm tree is `.venv-rocm-10.0.0`, so the picker showed Global Env. `setup_venv.sh` now links `.venv` at the environment it just built. After a window reload, the notebook uses `.venv`.

### The first cell looked stuck

> ipynb is hanging after I click play on the first cell despite that it is a markdown cell

The kernel was busy in the full-video cell. `FAST_MODE` was false, so that cell had started `infer_optical_flow.py` with `torch.compile` on. A markdown cell queued behind it stays on the spinner until the kernel is idle.

### The full-video cell exited 1

The notebook cell showed only:

```text
CalledProcessError: Command '['/workspaces/RAFT_large_pt/.venv-rocm-10.0.0/bin/python', 'infer_optical_flow.py', '--video', '/workspaces/RAFT_large_pt/Geisskopf_Gap_Jump.mp4', '--output', '/workspaces/RAFT_large_pt/outputs/playbook/optical_flow.mp4', '--param_dtype', 'fp16', '--resize', '376x672']' returned non-zero exit status 1.
```

`run()` uses `subprocess.run(..., check=True)`, so the traceback stops at the exit code. The script itself died earlier:

```text
BrokenPipeError: [Errno 32] Broken pipe
```

`ffmpeg` closed the pipe because VAAPI never started:

```text
Failed to initialise VAAPI connection: -1 (unknown libva error).
Failed to set value '/dev/dri/renderD128' for option 'vaapi_device': Input/output error
```

`/dev/dri/renderD128` exists and `libva` is installed. There is no `*_drv_video.so`. `VaapiWriter.is_available()` used to return true when the device node existed. It now encodes a one-frame probe and, on failure, the script prints the reason and writes H.264 with libx264. The rerun saved `outputs/playbook/optical_flow.mp4` (191 pairs, 27.7 ms each). The user confirmed the notebook worked after that.

### Release with git

> how to release this on github ?
> i meant release with git. run the proposed commands

The agent committed `8be3f88` and tagged `v1.0.0`, then pushed. The push is the step that failed. The error is the one in **Permission denied (publickey)** above. The release is local until those two pushes run on a machine that has the GitHub SSH key.
