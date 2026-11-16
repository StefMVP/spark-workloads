# Run jobs and inspect state

## Submit a committed repository

```sh
spark --as batch job submit --repo /path/to/repository --ref HEAD --cpu-only -- python job.py
spark --as batch job submit --repo /path/to/repository --shards 8 -- python job.py --shard '{shard}' --of '{nshards}'
```

The controller sends the commit through Git over SSH. The node checks it out under its job directory. Commit changes first. `--allow-dirty` includes a tracked diff; it does not include untracked files. Repository and workspace names use letters, digits, dots, underscores, and hyphens. A workspace is one directory name.

The router places each entry on a job node with enough memory, CPU capacity, and GPU share. Higher priorities run first. A smaller job can pass a job that does not fit. `--mem` is a reservation estimate, not a container memory limit.

Use `--env-file-secret /absolute/path/to/file` for `KEY=VALUE` secrets. Values pass through SSH input to a container FIFO on tmpfs. State stores the file path and variable names. A job can still print its own secrets, so control its output. Protect the controller state, patches, and logs as private data.

## Keep the queue active

A submit tries to dispatch immediately. Later work needs ticks. Run `spark tick` from the same install and state path, or install the user timer:

```sh
spark tick --install
spark util --hours 6
spark clean
```

The timer runs every 30 seconds through the user systemd manager. It needs the configured SSH account and credentials. `spark clean` is a preview. Add `--yes` to remove eligible finished workspaces and old build, pip, and uv caches.

## Inspect a failed operation

```sh
spark profile status
spark profile status --operation /absolute/path/to/receipt.json
spark profile resume /absolute/path/to/receipt.json --yes
spark job status JOB_ID
spark job logs JOB_ID
spark job cancel JOB_ID
```

A receipt-only status command reads saved state. General status can reconcile containers and queue state. Keep the same `SPARK_STATE` during recovery. Exit 3 means pending; exit 1 means refused or failed. Do not resubmit a queued job.

Use `spark switch PROFILE` for profile changes. It holds the queue, stops restartable jobs, applies placement, and checks the result. A failure after effects attempts to restore the earlier profile. A running `--once` job delays the switch until it finishes or the switch budget ends. Do not override an unknown owner or an unfinished operation.

Job status records the exact container identity and node boot ID. After a lost node, restartable work returns to the queue. This does not preserve container memory or a job's uncommitted results. Each job must save checkpoints to `/mnt/data` and tolerate another attempt.

For `--blender`, install native Blender on each eligible job node first. The router mounts the target of `~/.local/bin/blender` and the node's Cycles cache. The workload image needs the matching CUDA toolchain. This installer does not install Blender.
