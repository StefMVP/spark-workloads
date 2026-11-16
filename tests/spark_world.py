"""A simulated four-Spark cluster for the job runner and the router: nodes, containers and time. No host is contacted."""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/lib"))
from clustercontrol import Controller, Store

RING = ["spark1", "spark2", "spark3", "spark4"]
UUID = {node: f"GPU-{k}1111111-2222-3333-4444-555555555555" for k, node in enumerate(RING, 1)}
IMAGE = "spark-workload:local"


class World:
    """The `cluster` module that visionjob.Runner uses. `use(cmd)` gives the true use of a command."""
    MEM, CORES, OTHER, RAMP, RING = 121.0, 20, 3.0, 60, RING

    def __init__(self, folder, use=None):
        self.STATE = Path(folder) / "cluster.json"
        self.ROOT = Path(folder)
        self.t, self.calls, self.down, self.containers, self.serial = 1_800_000_000.0, [], set(), {}, 0
        self.images = {node: {IMAGE: "sha256:" + "cd" * 32} for node in RING}
        self.use = use or (lambda cmd: {"mem": 2.0, "cpus": 1.0, "gpu": 10.0, "seconds": 90, "code": 0})
        self.run_fails, self.run_times_out, self.kept = False, False, {}      # kept: output files of removed containers
        self.folders = {node: {} for node in RING}     # job id: {"made", "log", "workspace"}: ~/spark/jobs/ID on the node
        self.controller = Controller(Store(self.STATE))
        self.controller.store.commit(self.seed)

    @staticmethod
    def seed(control):
        control["generation"] = 1
        resources = ["gpu:" + UUID[node] for node in RING]
        control["operations"]["op"] = {
            "id": "op", "generation": 1, "resources": resources, "policy": "run", "outcome": "ready", "effects": [],
            "blockers": [], "spec": {"name": "jobs", "reservations": {f"s{k}": {"by": "jobs", "for": "jobs"} for k in (1, 2, 3, 4)}}}
        for resource in resources:
            control["leases"][resource] = {"state": "owned", "operation_id": "op", "generation": 1}
        control["layout"] = {"spec": "1x1x1x1", "groups": {f"s{k}": [f"spark{k}"] for k in (1, 2, 3, 4)},
                             "operation_id": "op", "generation": 1}

    # ---------------------------------------------------------------- the cluster module
    def clock(self): return self.t
    def me(self, args): return args.as_
    def now(self): return f"t{self.t:.0f}"
    def warn(self, text): pass
    def pmap(self, fn, items): return {item: fn(item) for item in items}
    def node_env(self, group, members, node):
        return {"SPARK_RANK": str(members.index(node)), "SPARK_WORLD": str(len(members)), "MASTER_ADDR": members[0]}

    # ---------------------------------------------------------------- simulated hosts
    def running(self, node):
        return [c for c in self.containers.values() if c["node"] == node and c["end"] is None]

    def mem_now(self, c):
        return c["use"]["mem"] * min(1.0, (self.t - c["start"]) / self.RAMP)

    def advance(self, seconds):
        self.t += seconds
        for c in self.containers.values():
            if c["end"] is None and c["node"] not in self.down and self.t - c["start"] >= c["use"]["seconds"]:
                c["end"], c["code"] = c["start"] + c["use"]["seconds"], c["use"]["code"]

    def lose(self, node):
        """Power loss: the node is silent and its containers are gone when it returns."""
        self.down.add(node)
        for c in self.running(node):
            c["end"], c["code"] = self.t, 137

    def silence(self, node):
        """Network loss: the node is silent and its containers keep running."""
        self.down.add(node)

    def facts(self, node, images):
        live = self.running(node)
        return {"uuid": UUID[node], "gpu_util": min(100.0, sum(c["use"]["gpu"] for c in live)), "boot_id": "boot-" + node,
                "mem_avail": self.MEM - self.OTHER - sum(self.mem_now(c) for c in live), "mem_total": self.MEM,
                "cores": self.CORES, "load": sum(c["use"]["cpus"] for c in live),
                "containers": {c["name"]: {"cpus": c["use"]["cpus"], "mem": self.mem_now(c)} for c in live},
                "folders": {i: self.t - max(f["made"], f["log"] or 0) for i, f in self.folders[node].items() if f["workspace"]},
                "exited": [c["name"] for c in self.containers.values() if c["node"] == node and c["end"] is not None and not c.get("removed")],
                "images": {image: self.images[node].get(image) for image in images}}

    def ssh(self, node, cmd, timeout=60, input=None):
        self.calls.append({"node": node, "cmd": cmd, "input": input})
        if node in self.down:
            raise subprocess.TimeoutExpired(cmd, timeout)
        reply = lambda out, code=0, err="": subprocess.CompletedProcess([], code, out, err)
        if cmd.startswith("python3 -c"):
            return reply(json.dumps(self.facts(node, json.loads(shlex.split(cmd)[-1]))))
        if "docker rm spk-" in cmd and "docker logs spk-" in cmd:            # visionjob.Runner.tidy
            for name in re.findall(r"docker rm (spk-\S+)", cmd):
                c = next(c for c in self.containers.values() if c["name"] == name and c["node"] == node)
                assert c["end"] is not None, f"{name} was removed while it runs"
                c["removed"], self.kept[name] = True, f"output of {c['cmd']}"
                self.folders[node].setdefault(name[4:], {"made": self.t, "log": None, "workspace": False})["log"] = self.t
        cleared = re.findall(r'find "\$HOME/spark/jobs/(vi[\w-]+)" -mindepth 1 -maxdepth 1 ! -name job.log', cmd)
        for ident in cleared:                                              # visionjob.clear_script: idempotent
            live = [c for c in self.running(node) if c["name"] == "spk-" + ident]
            assert not live, f"the workspace of {ident} was removed while it runs"
            if ident in self.folders[node]:
                self.folders[node][ident]["workspace"] = False
        if cleared or "docker rm spk-" in cmd:
            return reply("")
        if "docker run -d" in cmd:
            if self.run_times_out:
                raise subprocess.TimeoutExpired(cmd, timeout)
            if self.run_fails:
                return reply("", 70, "docker: bad mount")
            name = re.search(r"--name (spk-\S+)", cmd).group(1)
            inner = shlex.split(shlex.split(cmd[cmd.index("bash -c "):].split("\n")[0])[2].split("exec bash -lc ", 1)[1])[0]
            self.serial += 1
            ident = f"{self.serial:064x}"
            self.folders[node][name[4:]] = {"made": self.t, "log": None, "workspace": True}
            self.containers[ident] = {"name": name, "node": node, "start": self.t, "end": None, "code": None,
                                      "use": self.use(inner), "cmd": inner, "launch": cmd}
            return reply(f"{ident} {self.t:.0f}\n")
        if "echo rc=$?" in cmd:
            return reply("Error: No such object: spk\nrc=1\n")
        if cmd.startswith("timeout -k "):
            cmd = cmd[cmd.index("docker inspect"):]
        if cmd.startswith("docker inspect -f '{{.Id}}"):
            rows = []
            for ident in cmd.split("'")[2].split()[:-1]:
                c = self.containers[ident]
                rows.append(f"{ident} {'running' if c['end'] is None else 'exited'} {c['code'] or 0} {c['start']:.0f} {c['end'] or 0:.0f}")
            return reply("\n".join(rows) + "\n")
        if cmd.startswith("docker stop"):
            c = self.containers[cmd.split()[-1]]
            if c["end"] is None:
                c["end"], c["code"] = self.t, 137
            return reply("")
        return reply("")
