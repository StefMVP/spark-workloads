# Configure the cluster

Use one Linux controller and one state writer for a four-node GB10 cluster. Other controller copies need the same state file and file lock. Do not put the state on storage without POSIX file locking and atomic rename.

1. Edit `~/.config/spark-workloads/inventory.json`. Keep aliases `spark1` through `spark4`. Set each hostname, LAN address, free disk estimate, and memory size. `models fit --offline` uses the stored disk estimate.
2. Adapt `examples/ssh_config` in `~/.ssh/config`. Each alias must reach the same node as its inventory entry. Use the account that owns Docker, the Studio services, and `~/spark`. Configure host keys and authentication before use. SSH uses batch mode.
3. Create `/models` and `~/spark/data` on each node. Give the SSH user access. Jobs mount `/models` read-only and `~/spark/data` at `/mnt/data`. The controller creates `~/spark/jobs`, `~/spark/git`, and `~/spark/groups`. These remote paths are fixed.
4. Build or load your workload images on every job node. Use ARM64 images with Bash. GPU jobs need a compatible CUDA runtime. `spark switch` checks the image in the profile. Edit the installed `lib/spark-workloads/profiles/jobs*.yaml` image names if needed. Images are never pulled by the router.
5. If you use Studio profiles, install the guarded Imagination Studio services first. The head is `spark3`; the other nodes are workers. The service name is `imagination-studio.service`. Each local HTTP endpoint is `127.0.0.1:7870`. Register workers with their `sparkN` aliases. Worker identity, guard, code, and readiness checks must pass. The upstream reference is in `third-party.json`. Studio installation is separate from this installer.

The `jobs` profile can run without Studio. A foreign GPU process blocks profile activation. Stop that process through its owner before a switch.

The topology is a fixed ring: 1-2-3-4-1. Multi-node jobs require direct ring links, the expected RoCE interface names, and `ring_ips` for each adjacent peer. Do not submit a multi-node job on a LAN-only inventory. The example supports single-node placement; it has no fabric addresses.

For model copies, add ordered `sync_routes` entries with `from`, `to_alias`, and `to_ip`. Start from the download host. Each route must start at a node that already has the files and add one destination. Cover all four nodes. The source node needs SSH access to the destination. The target alias controls the SSH user and host key; `to_ip` selects the link address. Empty routes require `models pull --no-sync`.

| Setting | Default |
|---|---|
| `SPARK_INVENTORY` | `~/.config/spark-workloads/inventory.json` |
| `SPARK_STATE` | `~/.local/state/spark-workloads/cluster.json` |
| `SPARK_UTIL_JSON` | `util.json` beside the state file |
| `SPARK_AS` | Current user and controller hostname |

Plans, queue patches, receipts, measurements, and router records stay beside the state file. Remote cache defaults are `~/.cache/uv`, `~/.cache/pip`, and `~/.cache/cycles`. No model weights or cache data are installed by this project.

The container network uses the host network. Keep Docker's local DNS stub at `172.17.0.1`; do not set a public DNS resolver in a container.

Private installed-model profiles, cache attestations, client model settings, and model staging queues are not supplied. Run inference as a container job with a staged image and weights. Automatic asset transfer and training-specific profiles are outside the supported profile set.
