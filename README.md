# Spark workloads

Queue mixed container workloads on four NVIDIA GB10 nodes. Saved profiles divide the nodes between jobs and Imagination Studio. Model search, recipe discovery, memory fit, and download plans use the same inventory.

Use this controller when inference, media, and batch jobs share a cluster. A single persistent model server does not need it.

## Install

Use Linux with Python 3.12 or newer, Git, and SSH on the controller. Each node needs Bash, Python 3, Git, Docker, the NVIDIA Container Toolkit, and `nvidia-smi`. Model downloads need `hf` on the download node and `rsync` on each node.

```sh
git clone https://github.com/StefMVP/spark-workloads.git
cd spark-workloads
python3 tools/install "$HOME/.local/opt/spark-workloads"
export PATH="$HOME/.local/opt/spark-workloads/bin:$PATH"
mkdir -p "$HOME/.config/spark-workloads"
cp examples/inventory.json "$HOME/.config/spark-workloads/inventory.json"
```

The installer needs network access for Python dependencies. It refuses an existing install directory. Use a separate prefix if you already have a `spark` command.

Edit the inventory and your SSH configuration. [Configure the cluster](docs/configuration.md) gives the node, storage, image, and Studio requirements. The example addresses are fictional.

## Plan and run

These commands use no node connection:

```sh
spark profile list
spark profile apply jobs-studio-pair
models fit example/model --size-gib 140 --offline
```

Stage your container image on each job node before use. The default image name is `spark-workload:local`. To activate four job nodes, run:

```sh
spark switch jobs
spark --as batch job submit --cpu-only --mem 4 --cpus 2 --image spark-workload:local -- bash -lc 'printf "hello\n"'
spark --as inference job submit --gpu --mem 40 --image spark-workload:local -- python /mnt/data/inference.py
spark --as media job submit --gpu --mem 16 --image spark-workload:local -- python /mnt/data/media.py
spark job queue
spark job status
spark job logs JOB_ID
```

Use an image with the dependencies for each command. A switch can stop and restart jobs. It also runs one GPU check on each idle job node.

| Profile | Job nodes | Studio nodes |
|---|---|---|
| `jobs` | All four | None |
| `jobs-studio` | 1, 2, 4 | 3 |
| `jobs-studio-pair` | 1, 4 | 3 and 2 |
| `studio-4` | None | All four |

To search and plan model storage, run:

```sh
models search Qwen
models recipes --source eugr
models pull example/model --to /models --no-sync
```

`models pull` writes a pinned download plan and checks disk space. Add `--yes` to download. Configure ordered `sync_routes` before you omit `--no-sync`. Model fit is an estimate of weight placement, not proof that a runtime can serve the model.

Jobs restart after a node loss unless you use `--once`. Make restartable jobs safe to run again. Write results to `/mnt/data`; finished workspaces are removed. [Run jobs and inspect state](docs/jobs.md) covers repository jobs, secrets, queue ticks, and recovery.
