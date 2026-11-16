"""Spark jobs (`spark job submit|status|queue|logs|wait|set|cancel`): one docker container per job, placed by the router.

Agents never name a node. Every job enters one queue; `dispatch` asks tools/lib/router.py where each waiting job
starts, from measured free memory, cores and GPU share on the job nodes of the applied profile.
A job runs only under the operation that owns the node's GPU lease. The durable order is
Controller.admit_job, job_start_permission, docker run, observe_job_start, settle_job.

Secret environment (`--env-file-secret PATH`): values go from the lab file to the job process through ssh stdin and a
FIFO on a tmpfs that only the container has. They are not in the job record, the docker config, the image, the logs
written by this tool, or a file on the Spark. Only the variable names and the lab file paths are recorded.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path

import router
from clustercontrol import Refused

KIND = "vision"
SECRET_DIR = "/run/spark-secret"
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEFAULT_IMAGE = "spark-workload:local"
HOLD_S = 420            # a switch holds the queue for at most switch.MAX_SECONDS; a dead switch loses the hold here
CLASSES = {"io": (1, 1), "cpu": (4, 4), "gpu": (8, 2), "blender": (16, 16)}     # default GiB and cores of each job class
# Blender jobs mount the native executable, X libraries and Cycles kernel cache.
# The workload image supplies the CUDA compiler; denoising uses the booked CPUs.
BLENDER_MOUNTS = ('B="$(dirname "$(readlink -f "$HOME/.local/bin/blender")")"; [ -x "$B/blender" ] || exit 72\n'
                  'mkdir -p "$HOME/.cache/cycles" || exit 69\n'
                  'BM="-v $B:/opt/blender:ro -v $HOME/.cache/cycles:/work/.cache/cycles -v /usr/lib/aarch64-linux-gnu/libSM.so.6:/opt/blender-libs/libSM.so.6:ro '
                  '-v /usr/lib/aarch64-linux-gnu/libICE.so.6:/opt/blender-libs/libICE.so.6:ro -e LD_LIBRARY_PATH=/opt/blender-libs"\n')
BLENDER_PATH = "/opt/blender"
TIDY_MAX = 10           # finished containers removed per node in one pass
JOB_LOG = "job.log"     # the output of a removed container, in the job folder on its node
SWEEP_AGE_S = 7200      # a job folder that no live, waiting or stopped-restartable job owns loses its workspace after this
JOB_DIR = re.compile(r"vi\d{4}-\d{6}-[0-9a-f]{4}")    # the name of a job folder (new_id("vi"))
STOP_GRACE_S = 60       # every stop (cancel, switch, drill, node drain) sends SIGTERM, waits this long, then kills:
                        # a job tool saves the unit in work and exits 0 (stopped, resumable)
STATUS_SECONDS = 5.0
NODE_FACTS = r'''
import json, pathlib, subprocess, sys
images = json.loads(sys.argv[1])
run = lambda *a: subprocess.run(a, capture_output=True, text=True, timeout=25).stdout
mem = {l.split(":")[0]: int(l.split()[1]) / 1048576 for l in open("/proc/meminfo") if l.startswith(("MemAvailable", "MemTotal"))}
gpu = run("nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits").strip().splitlines()[0].split(", ")
containers = {}
exited = run("docker", "ps", "-a", "--filter", "label=spark.job", "--filter", "status=exited", "--format", "{{.Names}}").split()
if run("docker", "ps", "-q", "--filter", "label=spark.job").strip():
    for line in run("docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}} {{.MemUsage}}").splitlines():
        name, cpu, used = line.split()[:3]
        if name.startswith("spk-"):
            unit = {"KiB": 1 / 1048576, "MiB": 1 / 1024, "GiB": 1}.get(used[-3:], 0)
            containers[name] = {"cpus": float(cpu.rstrip("%")) / 100, "mem": float(used[:-3]) * unit}
root, folders = pathlib.Path.home() / "spark/jobs", {}
for d in (root.iterdir() if root.is_dir() else []):         # job folders that still hold more than job.log and receipts
    try:
        if d.is_dir() and any(e.is_dir() or (e.name != "job.log" and e.stat().st_size >= 1 << 20) for e in d.iterdir()):
            log = d / "job.log"
            folders[d.name] = __import__("time").time() - max(d.stat().st_mtime, log.stat().st_mtime if log.exists() else 0)
    except OSError:
        pass
print(json.dumps({"folders": folders, "uuid": gpu[0], "gpu_util": float(gpu[1]), "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
                  "mem_avail": mem["MemAvailable"], "mem_total": mem["MemTotal"], "cores": len(__import__("os").sched_getaffinity(0)),
                  "load": float(open("/proc/loadavg").read().split()[0]), "containers": containers, "exited": exited,
                  "images": {i: (run("docker", "image", "inspect", "-f", "{{.Id}}", i).strip() or None) for i in images}}))
'''


def clear_script(job_id):
    """Remove the workspace of a finished job; job.log and small receipts stay. Idempotent: a kill midway is safe."""
    if not JOB_DIR.fullmatch(job_id):
        raise ValueError(f"not a job folder: {job_id}")
    folder = f'"$HOME/spark/jobs/{job_id}"'        # literal in each command: no shell variable can widen it
    return (f"if [ -d {folder} ]; then find {folder} -mindepth 1 -maxdepth 1 ! -name {JOB_LOG} "
            f"! \\( -type f -size -1024k \\) -exec rm -rf {{}} +; fi\n")


def kept_jobs(state):
    """Job ids whose folders stay: live jobs and each stopped job that a waiting or live copy restarts."""
    jobs = state.get("jobs", {})
    keep = {j["id"] for j in jobs.values() if is_vision(j) and j["status"] != "settled"}
    entries = [*state.get("queue", {}).values(), *(jobs[i]["command"].get("entry") or {} for i in keep)]
    for entry in entries:
        before = entry.get("requeued_from")
        while before and before not in keep:
            keep.add(before)
            before = ((jobs.get(before) or {}).get("command") or {}).get("entry", {}).get("requeued_from")
    return keep


def read_secret_env(paths):
    """KEY=VALUE lines of the lab files. Comments and blank lines are skipped. Later files win."""
    env = {}
    for path in paths or []:
        path = Path(path).expanduser()
        if not path.is_file():
            raise Refused(f"secret env file {path} is missing")
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.removeprefix("export ").partition("=")
            key, value = key.strip(), value.strip()
            if not sep or not ENV_NAME.fullmatch(key):
                raise Refused(f"{path}:{number}: not a KEY=VALUE line")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if "\n" in value or "\0" in value:
                raise Refused(f"{path}:{number}: value has a line break")
            env[key] = value
    return env


def secret_payload(env):
    """Shell text that the container sources from the FIFO."""
    return "".join(f"{key}={shlex.quote(value)}\n" for key, value in env.items())


def entry(cmd, with_secrets, path=""):
    run = (f"export PATH={path}:$PATH; " if path else "") + f"exec bash -lc {shlex.quote(cmd)}"
    if not with_secrets:
        return run
    fifo = f"{SECRET_DIR}/env"
    return (f"mkfifo -m 600 {fifo} || exit 96; set -a; . {fifo}; rc=$?; set +a; rm -f {fifo}; "
            f"[ $rc = 0 ] || exit 97; {run}")


def launch_script(job_id, group, image, cmd, workdir, with_secrets, cpus, gpu=True, env=None, devices="", blender=False):
    """Remote shell text. It contains no secret value. Stdin carries the secret payload."""
    name = f"spk-{job_id}"
    tmpfs = (f'--tmpfs {SECRET_DIR}:rw,noexec,nosuid,size=64k,mode=700,uid="$(id -u)",gid="$(id -g)" '
             if with_secrets else "")
    feed = ""
    if with_secrets:
        wait = (f"i=0; while [ ! -p {SECRET_DIR}/env ] && [ $i -lt 600 ]; do sleep 0.05; i=$((i+1)); done; "
                f"[ -p {SECRET_DIR}/env ] && cat > {SECRET_DIR}/env")
        feed = (f"docker exec -i {name} sh -c {shlex.quote(wait)} "
                f"|| {{ docker rm -f {name} >/dev/null 2>&1; echo 'secret env was not delivered' >&2; exit 71; }}\n")
    plain = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in (env or {}).items())
    return (
        "set -u\n"
        f'J="$HOME/spark/jobs/{job_id}"; G="$HOME/spark/groups/{group}"\n'
        'mkdir -p "$J" "$HOME/spark/data" "$HOME/spark/git" || exit 69\n'
        'GF=""; [ -f "$G/docker.env" ] && GF="-v $G:/spark:ro --env-file $G/docker.env"\n'
        + (BLENDER_MOUNTS if blender else 'BM=""\n') +
        f"docker run -d --init --name {name} --label spark.group={group} --label spark.job={job_id} --label spark.kind={KIND} "
        f"--cpus {max(1, round(cpus))} {'--gpus all ' if gpu else ''}{devices}--ipc host --network host "
        "--ulimit memlock=-1 --ulimit stack=67108864 --pull never "
        '-v /models:/models:ro $GF $BM -v "$HOME/spark/data":/mnt/data -v "$J":/work '
        '-v "$HOME/spark/git":"$HOME/spark/git":ro '
        f'-w {shlex.quote(workdir)} --user "$(id -u):$(id -g)" -e HOME=/work -e MODELS_MIRROR=/mnt/data/models-mirror {plain} '
        f"{tmpfs}{shlex.quote(image)} bash -c {shlex.quote(entry(cmd, with_secrets, BLENDER_PATH if blender else ""))} >/dev/null "
        f"|| {{ docker rm -f {name} >/dev/null 2>&1; exit 70; }}\n"
        f"{feed}"
        f"docker inspect -f '{{{{.Id}}}} {{{{.State.StartedAt}}}}' {name}\n"
    )


class Runner:
    def __init__(self, cluster, controller, ssh=None, git=None, clock=time.time):
        self.cluster, self.controller, self.clock = cluster, controller, clock
        self.ssh = ssh or cluster.ssh
        self.git = git or (lambda repo, *args, **kw: subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                                                     text=True, **kw))
        self.book = Path(cluster.STATE).parent / "router.json"      # learned use per job kind; nodes that are silent

    # ---------------------------------------------------------------- facts
    def notes(self):
        try:
            return json.loads(self.book.read_text())
        except (OSError, ValueError):
            return {"learned": {}, "silent_since": {}}

    def save_notes(self, notes):
        self.book.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.book.with_suffix(".tmp")
        temporary.write_text(json.dumps(notes, indent=1))
        temporary.replace(self.book)

    def job_nodes(self, state):
        owner = state["operations"].get(state["layout"].get("operation_id")) or {}
        if owner.get("outcome") != "ready" or owner.get("policy") != "run" or owner.get("retirement"):
            return {}, owner
        groups = state["layout"]["groups"]
        return {node: group for group in owner.get("spec", {}).get("reservations", {}) for node in groups.get(group, [])}, owner

    def facts(self, nodes, images):
        """One ssh per node, in parallel: identity, free memory, cores, GPU use, images and the use of each job container."""
        script = shlex.join(["python3", "-c", NODE_FACTS, json.dumps(sorted(images))])
        def one(node):
            try:
                reply = self.ssh(node, script, timeout=40)
                return json.loads(reply.stdout) if reply.returncode == 0 else None
            except (subprocess.TimeoutExpired, ValueError):
                return None
        return self.cluster.pmap(one, list(nodes)) if nodes else {}

    def world(self, state, facts):
        """The router's view: job nodes of the applied profile with measured room, and the live jobs with their bookings."""
        reserved, owner = self.job_nodes(state)
        claims = owner.get("claims") or {}
        nodes = []
        for node, group in reserved.items():
            fact = facts.get(node)
            lease = state["leases"].get("gpu:" + fact["uuid"]) if fact else None
            up = bool(fact) and bool(lease) and lease.get("operation_id") == owner["id"] and lease.get("state") == "owned"
            nodes.append({"name": node, "up": up, "holder": (claims.get(group) or {}).get("by"),
                          "gpu_util": fact["gpu_util"] if fact else 0, "mem_avail": fact["mem_avail"] if fact else 0, "mem_total": fact["mem_total"] if fact else 121,
                          "cores": fact["cores"] if fact else 20, "images": [i for i, v in (fact or {}).get("images", {}).items() if v]})
        live = []
        for job in state["jobs"].values():
            if is_vision(job) and job["status"] != "settled":
                node = job["command"]["node"]
                seen = (facts.get(node) or {}).get("containers", {}).get(job["command"]["container"])
                live.append({"id": job["id"], "nodes": [node], "need": need_of(job), "mem_now": seen["mem"] if seen else None})
        return nodes, live

    # ---------------------------------------------------------------- code
    def resolve(self, args):
        """(repo, sha, workspace, patch, mirror name), or Nones when the job has no repository."""
        if not args.repo:
            return None, None, None, "", None
        repo = Path(args.repo).expanduser().resolve()
        sha = self.git(repo, "rev-parse", "--verify", "--end-of-options", args.ref + "^{commit}")
        if sha.returncode:
            raise Refused(f"{repo}: ref {args.ref} is unknown")
        patch = ""
        if args.ref == "HEAD" and self.git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
            if not args.allow_dirty:
                raise Refused(f"{repo} has uncommitted changes; commit them or add --allow-dirty")
            patch = self.git(repo, "diff", "HEAD").stdout
        common = self.git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
        name = Path(common).parent.name if Path(common).name == ".git" else repo.name  # one mirror for every worktree
        workspace = args.workspace or name
        component = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
        if not component.fullmatch(workspace) or workspace in (".", ".."):
            raise Refused("workspace must be one directory name with letters, digits, dots, underscores or hyphens")
        if not component.fullmatch(name) or name in (".", ".."):
            raise Refused("repository directory name must be a safe workspace name")
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha.stdout.strip()):
            raise Refused("git returned an invalid commit identity")
        return repo, sha.stdout.strip(), workspace, patch, name

    def ship(self, node, job_id, repo, sha, workspace, patch, name):
        bare = f"spark/git/{name}.git"
        # A commit that the node already has needs no push: one ssh names it for the job.
        have = self.ssh(node, f'mkdir -p "$HOME/spark/git" && (test -d "$HOME/{bare}" || git init -q --bare "$HOME/{bare}") '
                              f'&& if git -C "$HOME/{bare}" cat-file -e {sha}^{{commit}} 2>/dev/null; then '
                              f'git -C "$HOME/{bare}" update-ref refs/jobs/{job_id} {sha} && echo have; else echo need; fi')
        if have.returncode or have.stdout.strip() not in ("have", "need"):
            raise Refused(f"{node}: {have.stderr.strip()[-200:]}")
        if have.stdout.strip() == "need":
            push = self.git(repo, "push", "-q", f"{node}:{bare}", f"{sha}:refs/jobs/{job_id}")
            if push.returncode:
                raise Refused(f"git push to {node}: {push.stderr.strip()[-200:]}")
        tree = f'"$HOME/spark/jobs/{job_id}/{workspace}"'
        script = (f'mkdir -p "$HOME/spark/jobs/{job_id}" && git clone -q --shared --no-checkout "$HOME/{bare}" {tree} '
                  f"&& git -C {tree} checkout -q --detach {sha}" + (f" && git -C {tree} apply" if patch else ""))
        done = self.ssh(node, script, timeout=600, input=patch or None)
        if done.returncode:
            raise Refused(f"{node}: checkout failed: {done.stderr.strip()[-200:]}")

    def tidy(self, state, facts):
        """Finished jobs do not fill the disks: the output goes to job.log in the job folder, then the container and the
        workspace are removed. A folder that an older router or a lost node left is swept after SWEEP_AGE_S."""
        live = {j["command"]["container"] for j in state["jobs"].values() if is_vision(j) and j["status"] != "settled"}
        keep = kept_jobs(state)
        def one(node):
            fact = facts.get(node) or {}
            names = [n for n in fact.get("exited", []) if n not in live and re.fullmatch(r"spk-[\w.-]+", n)][:TIDY_MAX]
            script = "".join(f'mkdir -p "$HOME/spark/jobs/{n[4:]}" && docker logs {n} > "$HOME/spark/jobs/{n[4:]}/{JOB_LOG}" 2>&1 '
                             f"&& docker rm {n} >/dev/null"
                             + (f" && {clear_script(n[4:])}" if n[4:] not in keep and JOB_DIR.fullmatch(n[4:]) else "\n")
                             for n in names)
            stale = [d for d, age in sorted((fact.get("folders") or {}).items()) if age >= SWEEP_AGE_S and d not in keep
                     and JOB_DIR.fullmatch(d) and "spk-" + d not in fact.get("exited", [])][:TIDY_MAX]
            script += "".join(clear_script(d) for d in stale)
            try:
                return len(names) + len(stale) if script and self.ssh(node, script, timeout=120).returncode == 0 else 0
            except subprocess.TimeoutExpired:
                return 0                                       # the next pass takes the rest
        return self.cluster.pmap(one, [node for node, fact in facts.items() if fact and (fact.get("exited") or fact.get("folders"))])

    # ---------------------------------------------------------------- start
    def start(self, entry, node, fact, need, gang=None, last=True):
        """One container on the node the router chose. Returns the durable job. Secret values are read now."""
        cluster, controller = self.cluster, self.controller
        state = controller.store.read()
        reserved, owner = self.job_nodes(state)
        resource, group = "gpu:" + fact["uuid"], reserved.get(node)
        lease = state["leases"].get(resource)
        if not group or not lease or lease.get("operation_id") != owner["id"] or not fact["images"].get(entry["image"]):
            raise Refused(f"{node} cannot take the job: the profile, the GPU lease or the image changed")
        env = read_secret_env(entry.get("env_files"))
        job_id = new_id("vi")
        patch = Path(entry["patch_file"]).read_text() if entry.get("patch_file") else ""
        if entry.get("repo"):
            self.ship(node, job_id, Path(entry["repo"]), entry["sha"], entry["workspace"], patch, entry["name_repo"])
        plain, devices = {"SPARK_JOB": job_id, "SPARK_NODE": node}, ""
        if gang:
            plain.update(gang["env"][node])
            devices = "--device /dev/infiniband "
        command = {"node": node, "group": group, "image": entry["image"], "image_id": fact["images"][entry["image"]],
                   "cmd": entry["cmd"], "container": f"spk-{job_id}", "by": entry["by"], "commit": entry.get("sha"),
                   "dirty_patch": bool(patch), "workspace": entry.get("workspace"), "out": entry.get("out"),
                   "secret_env": sorted(env), "need": need, "kind": router.kind(entry), "priority": entry.get("priority", 0),
                   "submitted": cluster.now(), "restartable": bool(entry.get("restartable")), "entry": entry,
                   **({"gang": gang["id"], "rank": gang["env"][node]["SPARK_RANK"]} if gang else {}), **(entry.get("extra") or {})}
        job = controller.admit_job(owner["id"], job_id, KIND, [resource], max(1, round(need["mem"])), command,
                                   takes_entry=entry["id"] if last else None)
        controller.job_start_permission(job["id"], job["attempt_id"], job["generation"])
        script = launch_script(job_id, group, entry["image"], entry["cmd"],
                               f"/work/{entry['workspace']}" if entry.get("workspace") else "/work", bool(env),
                               entry["cpus"], entry.get("gpu", True), plain, devices, entry.get("class") == "blender")
        try:
            reply = self.ssh(node, script, timeout=120, input=secret_payload(env) if env else None)
        except subprocess.TimeoutExpired:
            return controller.record_job_blocker(job["id"], job["attempt_id"], job["generation"],
                                                 "container start timed out; state is unknown")
        fields = reply.stdout.split()
        if reply.returncode or len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            return self.absent(job, fact["boot_id"], (reply.stderr.strip() or "docker run failed")[-300:])
        token = {"operation_id": owner["id"], "generation": owner["generation"], "resources": job["resources"]}
        birth = {"kind": "container", "boot_id": fact["boot_id"], "identity": fields[0], "started": fields[1]}
        record = {"attempt_id": job["attempt_id"], "token": token, "state": "running", "birth": birth}
        job = controller.observe_job_start(job["id"], job["attempt_id"], job["generation"], {job["resources"][0]: record})
        return controller.record_job_blocker(job["id"], job["attempt_id"], job["generation"], "") if job.get("blocker") else job

    def absent(self, job, boot_id, reason):
        """Settle a start that left no container. An existing container keeps the job open."""
        name, node = job["command"]["container"], job["command"]["node"]
        seen = self.ssh(node, f"docker inspect -f '{{{{.Id}}}}' {name} 2>&1; echo rc=$?", timeout=30)
        if seen.returncode or "rc=1" not in seen.stdout or "No such" not in seen.stdout:
            return self.controller.record_job_blocker(job["id"], job["attempt_id"], job["generation"],
                                                      f"start failed and the container state is unknown: {reason}")
        record = {"state": "never_created", "attempt_id": job["attempt_id"], "container": name, "boot_id": boot_id,
                  "reason": reason}
        return self.controller.settle_absent_vision_job(job["id"], job["attempt_id"], job["generation"], record)

    # ---------------------------------------------------------------- observe
    def refresh(self, only=None):
        """Settle each running job whose exact container has exited. One ssh per node that has running jobs."""
        state = self.controller.store.read()
        running = [j for j in state["jobs"].values() if is_vision(j) and j["status"] == "running" and (not only or j["id"] == only)]
        fmt = "{{.Id}} {{.State.Status}} {{.State.ExitCode}} {{.State.StartedAt}} {{.State.FinishedAt}}"
        def inspect(node):
            ids = " ".join(j["births"][j["resources"][0]]["identity"] for j in running if j["command"]["node"] == node)
            try:
                return self.ssh(node, f"docker inspect -f '{fmt}' {ids} 2>&1", timeout=30).stdout
            except subprocess.TimeoutExpired:
                return None
        replies = self.cluster.pmap(inspect, sorted({j["command"]["node"] for j in running})) if running else {}
        done = []
        for job in running:
            reply = replies.get(job["command"]["node"])
            if reply is None:
                continue                                       # a silent node: `lost` decides after LOST_S
            resource = job["resources"][0]
            birth = job["births"][resource]
            fields = next((line.split() for line in reply.splitlines() if line.startswith(birth["identity"])), None)
            if fields is None or len(fields) != 5:
                if job.get("blocker") != "container is not visible; termination is unknown":
                    self.controller.record_job_blocker(job["id"], job["attempt_id"], job["generation"],
                                                       "container is not visible; termination is unknown")
                continue
            if fields[1] not in ("exited", "dead") or fields[3] != birth["started"]:
                if job["command"].get("superseded"):           # the node came back
                    if not job["command"].get("gang") and self.controller.dequeue(job["command"]["superseded"], ended=f"its node came back and {job['id']} went on"):   # its copy did not start: the job goes on
                        self.controller.mark_job(job["id"], job["attempt_id"], job["generation"], "superseded", "")
                    else:                                      # its copy runs elsewhere: this one stops
                        self.ssh(job["command"]["node"], f"docker stop -t {STOP_GRACE_S} {birth['identity']}", timeout=STOP_GRACE_S + 60)
                continue
            outcome = {"state": "terminated", "birth": birth, "attempt_id": job["attempt_id"],
                       "exit_code": int(fields[2]), "finished": fields[4]}
            self.controller.settle_job(job["id"], job["attempt_id"], job["generation"], {resource: outcome})
            if outcome["exit_code"] and job["command"].get("gang") and not job["command"].get("superseded"):
                for other in running:                          # one rank failed: the other ranks do not wait for it
                    if other["command"].get("gang") == job["command"]["gang"] and other["id"] != job["id"]:
                        self.controller.record_job_blocker(other["id"], other["attempt_id"], other["generation"],
                            f"rank {job['command'].get('rank')} ({job['id']}) exited {outcome['exit_code']}; this rank was stopped")
                        self.ssh(other["command"]["node"], f"docker stop -t {STOP_GRACE_S} {other['births'][other['resources'][0]]['identity']}", timeout=STOP_GRACE_S + 60)
            if outcome["exit_code"] == 0 and job["command"].get("kind"):
                done.append(job["command"]["kind"])
        if done:
            notes = self.notes()
            for name in done:
                router.finished(notes.setdefault("learned", {}), name)
            self.save_notes(notes)
        return self.controller.store.read()

    def exit_code(self, job):
        outcome = next(iter(job.get("outcomes", {}).values()), {})
        return outcome.get("exit_code")

    # ---------------------------------------------------------------- queue
    @contextlib.contextmanager
    def turn(self):
        """One dispatcher at a time: the timer and each submit share this lock."""
        with (Path(self.cluster.STATE).parent / "dispatch.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def lost(self, state, facts, notes):
        """A job node that is silent for LOST_S is down: its restartable jobs go back to the queue."""
        reserved, _ = self.job_nodes(state)
        now, silent = self.clock(), notes.setdefault("silent_since", {})
        for node in reserved:
            if facts.get(node) is not None:
                silent.pop(node, None)
                continue
            silent.setdefault(node, now)
            if now - silent[node] < router.LOST_S:
                continue
            for job in state["jobs"].values():
                command = job.get("command") if is_vision(job) else None
                if (command and job["status"] == "running" and command["node"] == node and command.get("restartable")
                        and not command.get("superseded")):
                    entry = dict(command["entry"], id=new_id("q"), requeued_from=job["id"], queued_at=self.cluster.now(),
                                 reason=f"{node} was lost; requeued from {job['id']}")
                    self.controller.enqueue(entry)
                    self.controller.mark_job(job["id"], job["attempt_id"], job["generation"], "superseded", entry["id"])
                    for rank in state["jobs"].values():        # the other ranks of its gang stop: the gang starts again as one
                        if (command.get("gang") and is_vision(rank) and rank["id"] != job["id"] and rank["status"] == "running"
                                and rank["command"].get("gang") == command["gang"] and not rank["command"].get("superseded")):
                            self.controller.mark_job(rank["id"], rank["attempt_id"], rank["generation"], "superseded", entry["id"])
                            rank["command"]["superseded"] = entry["id"]

    def dispatch(self):
        """One pass: settle exited jobs, read each job node once, start every waiting job that the router places."""
        with self.turn():
            state = self.refresh()
            waiting = list(state.get("queue", {}).values())
            facts = self.facts(self.cluster.RING, {e["image"] for e in waiting} | {DEFAULT_IMAGE})
            notes = self.notes()
            self.lost(state, facts, notes)
            self.learn(state, facts, notes)
            self.save_notes(notes)
            state = self.controller.store.read()
            waiting = list(state.get("queue", {}).values())
            nodes, live = self.world(state, facts)
            hold = state.get("queue_hold")
            if hold and self.clock() - hold["since"] < HOLD_S:
                starts, reasons = [], {e["id"]: f"held: {hold['reason']}" for e in waiting}
            else:
                starts, reasons = router.plan(nodes, live, waiting, notes["learned"])
            def launch(index):                                 # every start of the pass at the same time
                entry, names = starts[index]
                gang, jobs = None, []
                if len(names) > 1:
                    gang = {"id": new_id("gang"), "env": {n: self.cluster.node_env("s" + "".join(x[-1] for x in sorted(names)), names, n) for n in names}}
                # The entry leaves the queue in the commit that writes its job record (the last one of a gang): a
                # dispatcher that dies in a start (a killed submit, an error that is not caught here) loses no job.
                for node in names:
                    try:
                        jobs.append(self.start(entry, node, facts[node], router.estimate(entry, notes["learned"]), gang, last=node == list(names)[-1]))
                    except Exception as error:  # noqa: BLE001 - every failed start keeps the entry, with the reason
                        text = str(error) if isinstance(error, Refused) else f"{type(error).__name__}: {error}"
                        self.controller.enqueue(dict(entry, reason=f"the start on {node} failed ({text[:200]}); the job waits and starts again by itself"))
                        break
                return jobs
            started = [job for jobs in self.cluster.pmap(launch, range(len(starts))).values() for job in jobs]
            self.tidy(state, facts)                            # after the starts: a start never waits for a cleanup
            for entry in waiting:
                reason = reasons.get(entry["id"])
                if reason and entry.get("reason") != reason and entry["id"] in self.controller.store.read().get("queue", {}):
                    self.controller.enqueue(dict(entry, reason=reason))
            return started, facts

    def learn(self, state, facts, notes):
        """Fold the measured use of each running job into its kind. The GPU share of a node is split over its GPU jobs."""
        learned = notes.setdefault("learned", {})
        for node, fact in facts.items():
            if not fact:
                continue
            jobs = [j for j in state["jobs"].values() if is_vision(j) and j["status"] == "running" and j["command"]["node"] == node
                    and j["command"]["container"] in fact["containers"]]
            sharing = sum(1 for j in jobs if need_of(j)["gpu"]) or 1
            for job in jobs:
                seen = fact["containers"][job["command"]["container"]]
                share = fact["gpu_util"] / sharing if need_of(job)["gpu"] else 0
                if job["command"].get("kind"):
                    router.learn(learned, job["command"]["kind"], seen["mem"], seen["cpus"], share)

    def stop(self, job, grace=STOP_GRACE_S, by=None):
        self.controller.cancel_job(job["id"], job["attempt_id"], job["generation"], by)
        self.ssh(job["command"]["node"], f"docker stop -t {grace} {job['births'][job['resources'][0]]['identity']}", timeout=grace + 60)
        return self.refresh(job["id"])["jobs"][job["id"]]

    def requeue(self, job):
        """Stop a restartable job and put its entry back in the queue. It starts again by itself."""
        entry = dict(job["command"]["entry"], id=new_id("q"), requeued_from=job["id"], queued_at=self.cluster.now())
        entry["stops"] = [*entry.get("stops", []), self.clock()][-8:]          # a drill does not stop it again and again
        settled = self.stop(job, by=f"the router (switch, drill or node drain) at {self.cluster.now()}; it waits again as {entry['id']}")
        if settled["status"] != "settled":
            raise Refused(f"{job['id']} did not stop: {settled.get('blocker')}")
        if job["command"].get("gang") and str(job["command"].get("rank")) != "0":
            return dict(entry, id=f"the entry of rank 0 of {job['command']['gang']}")     # one entry for the gang
        self.controller.enqueue(dict(entry, reason=f"requeued from {job['id']}"))
        return entry


AUDIT = r'''
set -u
docker run --rm -i --network none --entrypoint sh --tmpfs /p:rw,noexec,size=1m,mode=700 \
  -v /var/lib/docker:/d:ro -v /var/lib/containerd:/c:ro IMAGE -c '
cat > /p/values
{ find /d/containers /d/volumes /d/image /d/buildkit -type f 2>/dev/null
  find /c -type f -mtime -DAYS -size -64M 2>/dev/null; } > /p/files
echo "files $(wc -l < /p/files)"
tr "\n" "\0" < /p/files | xargs -0 -r grep -lF -f /p/values 2>/dev/null | sed "s/^/hit /"
echo done'
'''


def audit(cluster, args):
    """Search docker storage on each Spark for the secret values, with no sudo: a container with no network reads
    /var/lib/docker and /var/lib/containerd read-only; the values go through ssh stdin to a tmpfs and are never printed.
    Scope: every container config, log and volume, and every file under containerd that changed in the last --days days
    (the writable layer of each job container). Image layers older than that are not read."""
    values = sorted({v for v in read_secret_env(args.env_file_secret).values() if len(v) >= 12})
    if not values:
        raise Refused("no secret value of 12 characters or more in the named files")
    script = AUDIT.replace("IMAGE", shlex.quote(args.image)).replace("DAYS", str(int(args.days)))
    def one(node):
        try:
            return cluster.ssh(node, script, timeout=900, input="\n".join(values) + "\n")
        except subprocess.TimeoutExpired:
            return None
    hits = 0
    for node, reply in cluster.pmap(one, cluster.RING).items():
        lines = reply.stdout.split("\n") if reply else []
        if not reply or reply.returncode or "done" not in lines:
            print(f"{node}: the search did not finish: {(reply.stderr.strip()[-160:] if reply else 'timeout')}")
            hits += 1
            continue
        found = [line[4:] for line in lines if line.startswith("hit ")]
        hits += len(found)
        print(f"{node}: {len(values)} values, {next(l for l in lines if l.startswith('files '))[6:]} files read, {len(found)} files hold a value")
        for path in found[:20]:
            print(f"  {path}")
    raise SystemExit(1 if hits else 0)


def new_id(prefix):
    return f"{prefix}{dt.datetime.now().strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def is_vision(job):
    """Any container job of this tool, including the records made before the router."""
    return bool(job) and job.get("kind") == KIND and isinstance(job.get("command"), dict) and "attempt_id" in job


def need_of(job):
    """The booking of a live job. A record made before the router has only its declared memory and cores."""
    command = job["command"]
    return command.get("need") or {"mem": float(job.get("mem_gib", 8)), "cpus": float(command.get("cpus", 2)), "gpu": float(router.GPU_GUESS)}


def row(job, runner):
    command = job["command"]
    code = runner.exit_code(job)
    status = job["status"] if job["status"] != "settled" else ("never started" if code is None else f"exit {code}")
    if job.get("cancel_consent") and job["status"] != "settled":
        status += " (cancel asked)"
    observed = job.get("observation", {}).get("state")
    if observed and observed not in ("running", "exited", "dead"):
        status += f" ({observed})"
    ended = f"\n    stopped by {job['cancelled_by']}" if job.get("cancelled_by") else ""
    flags = ("R" if command.get("restartable") else "") + (f"p{command['priority']}" if command.get("priority") else "")
    text = (f"{job['id']:22} {status:14} {command['node']:7} {command['by']:14} {flags:4}"
            f"{command.get('shard') or '':7}{command['cmd'][:70]}")
    return text + ended + (f"\n    blocker: {job['blocker']}" if job.get("blocker") else "")


def queue_row(entry):
    flags = ("R" if entry.get("restartable") else "") + (f"p{entry['priority']}" if entry.get("priority") else "")
    return (f"{entry['id']:22} {'waiting':14} {'-':7} {entry['by']:14} {flags:4}{(entry.get('extra') or {}).get('shard') or '':7}"
            f"{entry['cmd'][:70]}\n    waits: {entry.get('reason', 'for the next dispatch')[:200]}")


def stream(node, cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", node, cmd]).returncode


def logs_command(job, flags="", tail="all"):
    """The output of a job: from its container, or from the file that `tidy` kept when it removed the container."""
    name, kept = job["command"]["container"], f'"$HOME/spark/jobs/{job["id"]}/{JOB_LOG}"'
    lines = "+1" if str(tail) == "all" else shlex.quote(str(tail))
    return (f"if docker inspect {name} >/dev/null 2>&1; then docker logs {flags}{name} 2>&1; "
            f"elif [ -f {kept} ]; then tail -n {lines} {kept}; else echo 'no output is kept for {job['id']}' >&2; exit 1; fi")


def command_text(argv):
    """One argument is shell text. Several arguments are an argv and reach the job unchanged."""
    return argv[0] if len(argv) == 1 else shlex.join(argv)


def build(cluster, runner, args):
    """The queue entries of one submit: one, or one per shard, or one per line of --items."""
    if not args.cmd:
        raise Refused("nothing to run: add -- CMD")
    cmd = command_text(args.cmd)
    read_secret_env(args.env_file_secret)                       # fail now on a bad file
    kind = "io" if args.light else "cpu" if args.cpu_only else "blender" if getattr(args, "blender", False) else "gpu"
    mem, cpus = (CLASSES[kind][0] if args.mem is None else args.mem, CLASSES[kind][1] if args.cpus is None else args.cpus)
    if not 0 <= args.priority <= 9 or not all(math.isfinite(v) and v > 0 for v in (mem, cpus)) or not 1 <= args.nodes <= 4:
        raise Refused("--priority is 0 to 9; --mem and --cpus are above 0; --nodes is 1 to 4")
    limit = router.NODE_MEM - router.floor_gib(router.NODE_MEM) - 4
    if mem > limit or cpus > router.NODE_CORES:
        raise Refused(f"no Spark is that large: at most {limit:.0f} GiB and {router.NODE_CORES} cores for one job")
    repo, sha, workspace, patch, mirror = runner.resolve(args)
    base = {"by": cluster.me(args), "image": args.image, "mem": mem, "cpus": cpus, "gpu": kind in ("gpu", "blender"), "class": kind,
            "priority": args.priority, "name": args.name, "nodes": args.nodes,
            "only": [n if n.startswith("spark") else f"spark{n[-1]}" for n in args.only.split(",")] if args.only else None,
            "repo": str(repo) if repo else None, "sha": sha, "workspace": workspace, "name_repo": mirror, "patch_file": None,
            "env_files": [str(Path(p).expanduser().resolve()) for p in args.env_file_secret or []],
            "restartable": not args.once, "out": args.out, "queued_at": cluster.now()}
    if base["only"] and any(n not in cluster.RING for n in base["only"]):
        raise Refused("--only takes node names: spark1,spark3")
    if patch:
        folder = Path(cluster.STATE).parent / "queue"
        folder.mkdir(parents=True, exist_ok=True)
        base["patch_file"] = str(folder / (new_id("patch") + ".diff"))
        Path(base["patch_file"]).write_text(patch)
    if args.items:
        items = [line.strip() for line in Path(args.items).expanduser().read_text().splitlines() if line.strip()]
        if not items:
            raise Refused(f"{args.items} has no item")
        parent = new_id("map")
        return [dict(base, id=new_id("q") + f"-{k}", cmd=cmd.replace("{item}", shlex.quote(item)) if "{item}" in cmd else f"{cmd} {shlex.quote(item)}",
                     extra={"map": parent, "shard": f"{k}/{len(items)}"}) for k, item in enumerate(items)]
    if args.shards:
        if "{shard}" not in cmd:
            cluster.warn("command has no {shard}: every shard runs the same thing")
        parent = new_id("map")
        return [dict(base, id=new_id("q") + f"-{k}", cmd=cmd.replace("{shard}", str(k)).replace("{nshards}", str(args.shards)),
                     extra={"map": parent, "shard": f"{k}/{args.shards}"}) for k in range(args.shards)]
    return [dict(base, id=new_id("q"), cmd=cmd, extra={})]


def adopt(runner, job, args):
    """The queue entry of a live job that was started before the router. Its repository and secret files are named now."""
    command = job["command"]
    files = [str(Path(p).expanduser().resolve()) for p in args.env_file_secret or []]
    missing = sorted(set(command.get("secret_env") or []) - set(read_secret_env(files)))
    if missing:
        raise Refused(f"{job['id']} uses {', '.join(missing)}: name its files with --env-file-secret")
    repo = mirror = None
    if command.get("commit"):
        if not args.repo:
            raise Refused(f"{job['id']} runs commit {command['commit'][:12]}: name its repository with --repo")
        repo = Path(args.repo).expanduser().resolve()
        if runner.git(repo, "cat-file", "-e", command["commit"] + "^{commit}").returncode:
            raise Refused(f"{repo} does not have commit {command['commit'][:12]}")
        if command.get("dirty_patch"):
            raise Refused(f"{job['id']} ran with uncommitted changes; it cannot start again from the commit")
        common = runner.git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
        mirror = Path(common).parent.name if Path(common).name == ".git" else repo.name
    need = need_of(job)
    return {"id": new_id("q"), "by": command["by"], "image": command["image"], "cmd": command["cmd"], "mem": need["mem"],
            "cpus": need["cpus"], "gpu": True, "priority": 0, "name": command.get("kind"), "nodes": 1, "only": None,
            "repo": str(repo) if repo else None, "sha": command.get("commit"), "workspace": command.get("workspace"),
            "name_repo": mirror, "patch_file": None, "env_files": files, "restartable": True, "out": command.get("out"),
            "extra": {k: command[k] for k in ("map", "shard") if command.get(k)}}


def matches(wanted, ident, extra):
    return not wanted or wanted in (ident, (extra or {}).get("map"), (extra or {}).get("requeued_from"))


def status_seconds(value):
    import argparse
    seconds = float(value)
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 60:
        raise argparse.ArgumentTypeError("--max-seconds must be between 0.1 and 60")
    return seconds


def observe_status(runner, jobs, deadline):
    """Read exact container births without dispatch, controller writes or lease changes."""
    running = [j for j in jobs if j["status"] == "running"]
    fmt = "{{.Id}} {{.State.Status}} {{.State.ExitCode}} {{.State.StartedAt}} {{.State.FinishedAt}}"
    def inspect(node):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, "total status deadline expired before probe"
        ids = [j["births"][j["resources"][0]]["identity"] for j in running if j["command"]["node"] == node]
        # The remote timeout also bounds the observer if the SSH connection is lost.
        command = f"timeout -k 0.1 {max(0.01, remaining - 0.2):.3f}s docker inspect -f {shlex.quote(fmt)} {shlex.join(ids)} 2>&1"
        try:
            reply = runner.ssh(node, command, timeout=remaining)
            rows = {fields[0]: fields for line in reply.stdout.splitlines() if len(fields := line.split()) == 5}
            return rows, None if reply.returncode == 0 else f"container inspect failed (exit {reply.returncode})"
        except subprocess.TimeoutExpired:
            return None, "total status deadline expired during SSH/container inspect"
        except OSError as error:
            return None, f"SSH failed: {error}"
    replies = runner.cluster.pmap(inspect, sorted({j["command"]["node"] for j in running})) if running else {}
    errors = {}
    for job in jobs:
        job["recorded_status"] = job["status"]
        job["observation"] = {"source": "controller record"}
        if job["status"] != "running":
            continue
        node, resource = job["command"]["node"], job["resources"][0]
        birth = job["births"][resource]
        rows, error = replies[node]
        fields = (rows or {}).get(birth["identity"])
        if not fields or fields[3] != birth["started"] or fields[1] not in ("running", "exited", "dead", "paused", "restarting", "created", "removing"):
            error = error or "exact container birth is not visible; current state is unknown"
        else:
            error = None   # Docker can return 1 for a different missing container in this batch.
            if fields[1] in ("exited", "dead"):
                try:
                    code = int(fields[2])
                except ValueError:
                    error = "container inspect returned an invalid exit code"
                else:
                    job["status"] = "settled"
                    job["outcomes"] = {resource: {"state": "terminated", "birth": birth,
                        "attempt_id": job["attempt_id"], "exit_code": code, "finished": fields[4]}}
        if error:
            errors[node] = error
            job["observation"]["error"] = error
        else:
            job["observation"] = {"source": "live container inspect", "state": fields[1]}
    return {"state": "partial" if errors else "current", "errors": errors}


def select(state, wanted):
    """Live or finished jobs and waiting entries of one id: a job, a queue entry, or the map id of a shard set."""
    jobs = [j for j in state["jobs"].values() if is_vision(j) and (wanted in (j["id"], j["command"].get("map"), j["command"].get("gang"))
                                                                    or (j["command"].get("entry") or {}).get("id") == wanted)]
    waiting = [e for e in state.get("queue", {}).values() if wanted in (e["id"], (e.get("extra") or {}).get("map"))]
    if not jobs and not waiting:
        note = state.get("queue_ended", {}).get(wanted)
        if note:
            raise Refused(f"{wanted} of {note['by']} never started: {note['what']} (queued {note.get('queued_at') or 'at an unknown time'}: {str(note['cmd'])[:70]})")
        raise Refused(f"no job or waiting entry {wanted}: this id was never submitted here, or it is older than the queue notes")
    return jobs, waiting


def wait_for(runner, ids, follow=False):
    """Until every id has finished: one dispatch pass each 20 s. Returns the largest exit code."""
    codes, shown = {}, set()
    while True:
        runner.dispatch()
        state, left = runner.controller.store.read(), 0
        for wanted in ids:
            jobs, waiting = select(state, wanted)
            for job in jobs:
                if follow and len(ids) == 1 and len(jobs) == 1 and not waiting and job["id"] not in shown and job.get("births"):
                    shown.add(job["id"])                      # the log of one job, to its end
                    stream(job["command"]["node"], logs_command(job, "-f "))
                    job = runner.refresh(job["id"])["jobs"][job["id"]]
                if job["status"] == "settled" and not job["command"].get("superseded"):
                    codes[job["id"]] = runner.exit_code(job) if runner.exit_code(job) is not None else 1
                left += job["status"] != "settled" and not job["command"].get("superseded")
            left += len(waiting)
        if not left:
            return max(codes.values(), default=0)
        time.sleep(20)


def handle(cluster, controller, args, action):
    deadline = time.monotonic() + getattr(args, "max_seconds", STATUS_SECONDS)
    runner = Runner(cluster, controller)
    if action == "submit":
        state = controller.store.read()
        entries = build(cluster, runner, args)
        controller.enqueue_all(entries)
        hold = state.get("queue_hold")
        if hold and runner.clock() - hold["since"] < HOLD_S:
            print(f"queued: {hold['reason']}; the jobs start by themselves when it ends", flush=True)
        elif not runner.job_nodes(state)[0]:                    # a submit is never refused for the state of the cluster
            print(f"queued: profile {state.get('profile') or 'none'} has no job node now; the jobs start by themselves "
                  "when a profile with job nodes is applied (`spark switch jobs`)", flush=True)
        started, _ = runner.dispatch()
        mine = {entry["id"] for entry in entries}
        for job in started:
            if job["command"]["entry"]["id"] in mine:
                print(row(job, runner), flush=True)
        left = [e for e in controller.store.read().get("queue", {}).values() if e["id"] in mine]
        for entry in left:
            print(queue_row(entry), flush=True)
        handle_id = (entries[0].get("extra") or {}).get("map") or entries[0]["id"]
        if args.wait:
            raise SystemExit(wait_for(runner, [handle_id], follow=True))
        print(f"id {handle_id}: spark job status {handle_id} | spark job logs {handle_id} | spark job wait {handle_id}")
        return
    if action in ("status", "queue"):
        state = controller.store.read()
        read_at = dt.datetime.now(dt.timezone.utc).isoformat()
        wanted = getattr(args, "id", None)
        if wanted:
            jobs, waiting = select(state, wanted)
        else:
            jobs = [j for j in state["jobs"].values() if is_vision(j) and (getattr(args, "all", False) or j["status"] != "settled")]
            waiting = router.by_priority(list(state.get("queue", {}).values()))
        if action == "queue":
            jobs = []
        observation = observe_status(runner, jobs, deadline) if action == "status" else None
        if observation is not None:
            observation.update(revision=state["revision"], read_at=read_at)
            observation["next_command"] = "spark job status" + (" " + shlex.quote(wanted) if wanted else "") + (" --all" if getattr(args, "all", False) else "")
        if getattr(args, "json", False):
            print(json.dumps({"jobs": jobs, "waiting": waiting, **({"status_read": observation} if observation else {})}, indent=2))
        else:
            print("\n".join([row(j, runner) for j in jobs] + [queue_row(e) for e in waiting])
                  or ("nothing waits" if action == "queue" else "no live or waiting jobs (--all shows finished jobs)"))
            if observation:
                print(f"status: {observation['state']}; controller revision {observation['revision']}; {observation['read_at']}")
                for node, error in observation["errors"].items():
                    print(f"{node}: {error}; job state above is recorded; retry: {observation['next_command']}")
        if observation and observation["state"] == "partial":
            raise SystemExit(3)
        return
    if action == "wait":
        raise SystemExit(wait_for(runner, args.ids))
    if action == "set":
        jobs, waiting = select(controller.store.read(), args.id)
        for entry in waiting:
            controller.enqueue(dict(entry, restartable=True))
            print(f"{entry['id']}: restartable")
        for job in jobs:
            if job["status"] == "settled":
                continue
            entry = job["command"].get("entry") or adopt(runner, job, args)
            controller.mark_job(job["id"], job["attempt_id"], job["generation"], "entry", dict(entry, restartable=True))
            controller.mark_job(job["id"], job["attempt_id"], job["generation"], "restartable", True)
            print(f"{job['id']}: restartable")
        return
    jobs, waiting = select(controller.store.read(), args.id)
    if action == "logs":
        if not jobs:
            raise Refused(f"{args.id} has not started: {waiting[0].get('reason', 'it waits')}")
        code = 0
        for job in jobs:
            if len(jobs) > 1:
                print(f"== {job['id']} {job['command'].get('shard') or ''} on {job['command']['node']}", flush=True)
            flags = f"--tail {shlex.quote(str(args.tail))} " + ("-f " if args.follow and len(jobs) == 1 else "")
            code = max(code, stream(job["command"]["node"], logs_command(job, flags, args.tail)))
        runner.refresh()
        raise SystemExit(code)
    if action == "cancel":
        me = cluster.me(args)
        if any(owner != me for owner in [j["command"]["by"] for j in jobs] + [e["by"] for e in waiting]) and not args.force:
            raise Refused(f"{args.id} belongs to another holder; add --force")
        for entry in waiting:
            controller.dequeue(entry["id"], ended=f"cancelled in the queue by {me} at {cluster.now()}")
            print(f"{entry['id']}: removed from the queue")
        open_jobs = 0
        for job in jobs:
            if job["status"] == "running":
                job = runner.stop(job, by=f"{me} (cancel) at {cluster.now()}")
            elif job["status"] == "admitted":
                job = runner.absent(job, "unknown", "cancelled before start")
            print(row(job, runner))
            open_jobs += job["status"] != "settled"
        raise SystemExit(3 if open_jobs else 0)
    raise Refused(f"unsupported job command {action}")


def register(cluster, parent):
    """`spark job ...`: the one way to run work on the Sparks."""
    def run(action):
        def call(args):
            from clustercontrol import Controller, Store
            return handle(cluster, Controller(Store(cluster.STATE)), args, action)
        return call
    jobs = parent.add_parser("job", help="run work on the Sparks; the router chooses the node").add_subparsers(required=True)
    def sub(name, help):
        p = jobs.add_parser(name, help=help)
        p.add_argument("--as", dest="as_", default=__import__("argparse").SUPPRESS)
        p.set_defaults(fn=run(name))
        return p
    p = sub("submit", "queue a job; it starts on the node with the most room, now or when room appears")
    kind = p.add_mutually_exclusive_group()
    kind.add_argument("--gpu", action="store_true", help="class gpu (default): GPU in the container; 8 GiB, 2 cores")
    kind.add_argument("--cpu", "--cpu-only", dest="cpu_only", action="store_true", help="class cpu: no GPU; 4 GiB, 4 cores")
    kind.add_argument("--blender", action="store_true", help="class blender: a gpu job with the node's native Blender on PATH; 16 GiB, 16 cores")
    kind.add_argument("--io", "--light", dest="light", action="store_true", help="class io: download, upload, copy; no GPU; 1 GiB, 1 core")
    p.add_argument("--mem", type=float, help="GiB at the peak, a first guess (default: of the class); measured use corrects it")
    p.add_argument("--cpus", type=float, help="cores, a first guess (default: of the class); also the container's CPU limit")
    p.add_argument("--once", action="store_true", help="the job must not run twice: a switch waits for its end, a lost node does not restart it")
    p.add_argument("--restartable", action="store_true", help="the default; kept for old commands")
    p.add_argument("--shards", type=int, help="N jobs; {shard} and {nshards} in CMD")
    p.add_argument("--items", metavar="FILE", help="one job per line; {item} in CMD, else the line is the last argument")
    p.add_argument("--priority", type=int, default=0, help="0 to 9; higher starts first")
    p.add_argument("--name", help="job kind; jobs of one name share the measured use")
    p.add_argument("--nodes", type=int, default=1, help="one container on each of N ring neighbours (DDP); SPARK_RANK, SPARK_WORLD, MASTER_ADDR are set")
    p.add_argument("--env-file-secret", action="append", metavar="PATH", help="lab KEY=VALUE file; values are never recorded")
    p.add_argument("--repo"); p.add_argument("--ref", default="HEAD"); p.add_argument("--workspace"); p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--image", default=DEFAULT_IMAGE); p.add_argument("--out", help="/mnt/data path the job writes")
    p.add_argument("--only", help="operator use: restrict to these nodes (spark1,spark3)")
    p.add_argument("--wait", action="store_true", help="stay until every job has finished; exit with the largest exit code")
    p.add_argument("cmd", nargs="*")
    p = sub("status", "live and waiting jobs; ID shows one job or shard set"); p.add_argument("id", nargs="?")
    p.add_argument("--all", action="store_true"); p.add_argument("--json", action="store_true")
    p.add_argument("--max-seconds", type=status_seconds, default=STATUS_SECONDS, help="total status read budget, 0.1 to 60 seconds (default: 5)")
    p = sub("queue", "waiting jobs in start order, each with the reason it waits"); p.add_argument("--json", action="store_true")
    p = sub("logs", "job output"); p.add_argument("id"); p.add_argument("-f", dest="follow", action="store_true"); p.add_argument("--tail", default="100")
    p = sub("wait", "stay until the jobs have finished"); p.add_argument("ids", nargs="+")
    p = sub("set", "mark a live or waiting job restartable after its submit"); p.add_argument("id")
    p.add_argument("--restartable", action="store_true", required=True)
    p.add_argument("--repo", help="for a job started before the router: its repository on lab")
    p.add_argument("--env-file-secret", action="append", metavar="PATH", help="for a job started before the router: its secret files")
    p = jobs.add_parser("audit", help="search docker storage on each Spark for the values of secret env files; no sudo; values are never printed")
    p.add_argument("--env-file-secret", action="append", metavar="PATH", required=True); p.add_argument("--days", type=int, default=3)
    p.add_argument("--image", default="ubuntu:24.04"); p.add_argument("--as", dest="as_", default=__import__("argparse").SUPPRESS)
    p.set_defaults(fn=lambda a: audit(cluster, a))
    p = sub("cancel", "stop a job or a shard set, or remove it from the queue"); p.add_argument("id"); p.add_argument("--force", action="store_true")
