"""`spark switch TARGET`: one foreground command that changes what runs on the four Sparks.

Phases: preflight (no effect) -> unload -> lease -> load -> verify. A failure after the first effect rolls back to
the profile that was applied before, and says why. A second run of the same command is a check, not a change.
`Switch` holds the order and the rollback. `Driver` holds every host and controller fact and effect.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid

from clustercontrol import Refused

PHASES = ("preflight", "unload", "lease", "load", "verify")
MIN_DISK_GB = 10
MAX_SECONDS = 300        # one switch, and so one queue hold: past it the switch fails and the earlier profile returns
POLL_S = 2               # every wait in a switch polls this often: a wait ends at most this long after its cause
STOPS_MAX, STOPS_WINDOW_S = 2, 1800   # a drill does not stop a job that was stopped this often in this time
FACTS = r'''
import json, pathlib, shutil, subprocess, sys
images, unit = json.loads(sys.argv[1]), sys.argv[2]
run = lambda *a: subprocess.run(a, capture_output=True, text=True, timeout=20)
gpu = []
jobs = run("docker", "ps", "-q", "--no-trunc", "--filter", "label=spark.job").stdout.split()
for pid in run("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits").stdout.split():
    try:
        group = pathlib.Path(f"/proc/{pid}/cgroup").read_text()
        gpu.append({"pid": int(pid), "job": any(j in group for j in jobs),
                    "cmd": pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:120]})
    except (OSError, ValueError):
        pass
mem = next(int(l.split()[1]) // 1048576 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
print(json.dumps({"disk_gb": shutil.disk_usage(pathlib.Path.home()).free // 10**9, "mem_gib": mem, "gpu": gpu,
                  "images": {i: run("docker", "image", "inspect", "-f", "{{.Id}}", i).returncode == 0 for i in images},
                  "unit": run("systemctl", "--user", "is-active", unit).stdout.strip(),
                  "unit_known": run("systemctl", "--user", "cat", unit).returncode == 0}))
'''
GPU_JOB = ('python -c "import torch,time;t=time.time();x=torch.randn(4096,4096,device=\'cuda\');'
           'y=(x@x).sum().item();print(\'GPU\',torch.cuda.get_device_name(0),\'torch matmul\',round(time.time()-t,2),\'s\')"')
GPU_PLAIN = "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/GPU /'"


class Failed(Refused):
    """A phase failed. `blockers` are (what, fix) pairs for a preflight refusal."""
    def __init__(self, phase, reason, blockers=()):
        super().__init__(reason)
        self.phase, self.blockers = phase, list(blockers)


class Switch:
    def __init__(self, driver, say=print, fault=None, max_seconds=MAX_SECONDS, clock=time.monotonic):
        self.driver, self.say, self.fault, self.max_seconds, self.clock = driver, say, fault, max_seconds, clock
        self.deadline = clock() + max_seconds
        driver.deadline = self.deadline                  # the waits inside a phase end at the same time

    def phase(self, name, target, action):
        started = time.monotonic()
        self.say(f"[{name}] {target} ...")
        try:
            if self.fault == name:
                raise Refused(f"injected fault in {name}")
            if name != "preflight" and self.clock() >= self.deadline:
                raise Refused(f"the switch passed its maximum time of {self.max_seconds:.0f} s (--max-seconds)")
            result = action()
            if name != "verify" and self.clock() >= self.deadline:
                raise Refused(f"the switch passed its maximum time of {self.max_seconds:.0f} s (--max-seconds) in {name}")
        except Failed:
            raise
        except Exception as error:                      # any failure in a phase leads to the rollback
            raise Failed(name, f"{type(error).__name__}: {error}" if not isinstance(error, Refused) else str(error)) from error
        self.say(f"[{name}] done in {time.monotonic() - started:.1f} s")
        return result

    def forward(self, target):
        driver = self.driver
        blockers = self.phase("preflight", target, lambda: driver.preflight(target))
        if blockers:
            raise Failed("preflight", "nothing was changed", blockers)
        if not driver.applied(target):
            self.phase("unload", target, lambda: driver.unload(target))
            self.phase("lease", target, lambda: driver.lease(target))
        self.phase("load", target, lambda: driver.load(target))
        rows = self.phase("verify", target, lambda: driver.verify(target))
        for row in rows:
            self.say(row["line"])
        if not all(row["ok"] for row in rows):
            raise Failed("verify", "; ".join(row["line"] for row in rows if not row["ok"]))

    def run(self, target):
        """0: applied and verified. 2: blocked before any effect. 1: failed (rolled back when there was an earlier profile)."""
        try:
            return self.attempt(target)
        finally:
            for line in self.driver.resume():
                self.say(line)

    def attempt(self, target):
        started, previous = time.monotonic(), self.driver.current()
        try:
            self.forward(target)
        except Failed as failure:
            if failure.phase == "preflight":
                self.say(f"BLOCKED: {target} was not applied; {failure}")
                for what, fix in failure.blockers:
                    self.say(f"  - {what}\n      fix: {fix}")
                return 2
            self.say(f"FAILED in {failure.phase}: {failure}")
            if not previous or previous == target:
                self.say(f"no earlier profile to return to; the state is: {self.driver.current() or 'no profile'}")
                return 1
            self.say(f"rolling back to {previous} ...")
            try:
                Switch(self.driver, self.say, None, self.max_seconds, self.clock).forward(previous)   # its own time
            except Failed as second:
                self.say(f"ROLLBACK FAILED in {second.phase}: {second}; the state is: {self.driver.current() or 'no profile'}")
                for what, fix in second.blockers:
                    self.say(f"  - {what}\n      fix: {fix}")
                return 1
            self.say(f"rolled back: {previous} is applied and verified; {target} failed in {failure.phase}: {failure}")
            return 1
        self.say(f"OK: {target} is applied and verified in {time.monotonic() - started:.1f} s")
        return 0


class Driver:
    """Real cluster. Targets: lab profiles (labprofile) and the stock four-Spark Studio profile (stockstudio)."""
    def __init__(self, cluster, controller, runtime=None, say=print):
        import clusterprofile
        self.c, self.controller, self.say = cluster, controller, say
        self.runtime = runtime or clusterprofile.Runtime(cluster)
        self.keys = {}                    # one request key per target: a rollback is another request
        self.cancel_once = False          # --cancel-once: stop live --once jobs at once, not wait for their end
        self.deadline = time.monotonic() + MAX_SECONDS

    # ---------------------------------------------------------------- facts
    def spec(self, name):
        import clusterprofile, labprofile, stockstudio
        spec = clusterprofile.load(self.c.ROOT, name)
        if not (labprofile.covers(spec) or stockstudio.covers(spec)):
            raise Refused(f"{name} is not a switch target; use `spark profile apply {name} --yes`")
        return spec

    def owner(self):
        state = self.controller.store.read()
        return state["operations"].get(state["layout"].get("operation_id")), state

    def current(self):
        owner, state = self.owner()
        return state.get("profile") if owner and owner.get("outcome") == "ready" else None

    def applied(self, name):
        """The exact saved profile is the READY owner."""
        from clustercontrol import digest
        owner, _ = self.owner()
        spec = self.spec(name)
        tag = {"lab_profile": "lab-v1"} if spec.get("lab") else {"stock_studio": "stock-studio-v1"}
        return bool(owner) and owner.get("outcome") == "ready" and owner["spec_digest"] == digest(dict(spec, **tag))

    @staticmethod
    def minutes_ago(started):
        import datetime as dt
        try:
            when = dt.datetime.fromisoformat(started[:26].rstrip("Z") + "+00:00")
        except ValueError:
            return "?"
        return round((dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60)

    def host_facts(self, spec):
        import labprofile
        images = sorted(set(spec.get("images", {}).values()))
        script = shlex.join(["python3", "-c", FACTS, json.dumps(images), labprofile.UNIT])
        def one(node):
            try:
                reply = self.c.ssh(node, script, timeout=60)
                return json.loads(reply.stdout) if reply.returncode == 0 else {"error": reply.stderr.strip()[-120:] or "ssh failed"}
            except (subprocess.TimeoutExpired, ValueError) as error:
                return {"error": str(error)[-120:]}
        return self.c.pmap(one, self.c.RING)

    # ---------------------------------------------------------------- preflight
    def preflight(self, name):
        """Every check before any effect. Returns (what blocks, the command that fixes it) pairs."""
        import labprofile, studioidle
        import assetrestore
        spec = self.spec(name)
        blockers = assetrestore.restore(self.c, spec, self.say)   # a missing asset that a peer has is copied back now
        owner, state = self.owner()
        applied = self.applied(name)
        for op in list(state["operations"].values()):   # dead ends of removed commands hold nothing: close them
            if (op.get("outcome") is None and not op.get("resources") and not op.get("effects")
                    and op.get("intent") in ("job", "layout", "claim", "release")):
                self.controller.cancel(op["id"])
                self.say(f"  closed dead-end operation {op['id']} ({op['intent']})")
        for op in list(self.controller.store.read()["operations"].values()):
            if op.get("outcome") is not None or not op.get("resources"):
                continue
            family = op["spec"].get("lab_profile") or op["spec"].get("stock_studio")
            if family and op["spec"].get("name") == name:
                continue                                  # load resumes it
            settled = all(e["state"] in ("verified", "rejected") for e in op.get("effects", []))
            if family and settled:                        # a switch that stopped part way: its settings are absolute
                self.controller.cancel(op["id"])
                self.say(f"  closed unfinished operation {op['id']} ({op['spec'].get('name')})")
            else:
                blockers.append((f"operation {op['id']} ({op['spec'].get('name')}) is unfinished and holds GPUs",
                                 f"spark profile cancel {op['id']} --yes"))
        owner, state = self.owner()
        if owner and owner.get("outcome") == "ready" and not applied:
            if not (owner["spec"].get("lab_profile") or owner["spec"].get("stock_studio")):
                blockers.append((f"profile {owner['spec'].get('name')} is applied and is not a switch target",
                                 f"spark profile cancel {owner['id']} --yes"))
            for job in state["jobs"].values():
                if job.get("status") != "settled" and set(job.get("resources", [])) & set(owner["resources"]):
                    node = (job.get("command") or {}).get("node", "?") if isinstance(job.get("command"), dict) else "?"
                    by = (job.get("command") or {}).get("by", "?") if isinstance(job.get("command"), dict) else "?"
                    if isinstance(job.get("command"), dict) and job["command"].get("restartable") and job["command"].get("entry"):
                        self.say(f"  {node}: restartable job {job['id']} of {by} will stop and start again by itself")
                    else:
                        started = (next(iter(job.get("births", {}).values()), {}) or {}).get("started", "")
                        age = f", started {self.minutes_ago(started)} min ago" if started else ""
                        self.say(f"  {node}: job {job['id']} of {by} runs once (--once){age}: the switch waits for its end "
                                 f"(at most its --max-seconds; new jobs wait in the queue). To stop it now: "
                                 f"spark job cancel {job['id']}, or run the switch with --cancel-once")
        facts = self.host_facts(spec)
        reserved = {node: group for group in spec.get("reservations", {}) for node in spec["groups"][group]}
        for node, fact in facts.items():
            if fact.get("error"):
                blockers.append((f"{node}: not reachable ({fact['error']})", f"ssh {node} true"))
                continue
            if node in reserved:
                image = spec["images"][node]
                if not fact["images"].get(image):
                    blockers.append((f"{node}: image {image} is missing",
                                     f"ssh spark1 'docker save {image} | ssh {node} docker load'"))
                if fact["disk_gb"] < MIN_DISK_GB:
                    blockers.append((f"{node}: {fact['disk_gb']} GB disk free, jobs need {MIN_DISK_GB} GB",
                                     f"spark clean {node} --yes"))
        if not facts.get(labprofile.HEAD, {}).get("error") and not facts[labprofile.HEAD]["unit_known"] and spec.get("studio"):
            blockers.append((f"{labprofile.HEAD}: unit {labprofile.UNIT} is not installed", "install Imagination Studio on spark3"))
        if not applied:
            for fact in studioidle.enforce(self.c, release=False):
                jobs = fact.get("jobs") or {}
                if fact.get("error"):
                    continue
                if jobs.get("running") or jobs.get("queued") or fact.get("busy"):
                    blockers.append((f"{fact['node']}: Studio has {jobs.get('running', 0)} running and {jobs.get('queued', 0)} queued jobs",
                                     "wait for the Studio queue, then run the same command"))
                for row in fact["cuda"]:
                    seen = next((g for g in facts.get(fact["node"], {}).get("gpu", []) if g["pid"] == row["pid"]), {})
                    if not row["kind"].startswith("studio:") and not seen.get("job"):   # a job's process: the job rule decides
                        cmd = seen.get("cmd", "")
                        blockers.append((f"{fact['node']}: GPU process {row['pid']} ({row['kind']}: {cmd}) is not Studio's and not a job",
                                         f"ssh {fact['node']} kill {row['pid']}"))
        return blockers

    # ---------------------------------------------------------------- effects
    def unload(self, name):
        """Requeue restartable jobs, stop resident Studio model servers, then wait until no GPU process is left."""
        import studioidle
        owner, _ = self.owner()
        if not owner or owner.get("outcome") != "ready":
            return
        self.requeue(owner)
        for fact in studioidle.enforce(self.c, release=True):
            self.say("  " + studioidle.line(fact))
        deadline = min(time.monotonic() + 120, self.deadline)
        while True:
            busy = {n: f["gpu"] for n, f in self.host_facts({}).items() if f.get("gpu")}
            if not busy:
                return
            if time.monotonic() >= deadline:
                raise Refused("GPU memory is still held: " + "; ".join(f"{n} pid {g[0]['pid']} {g[0]['cmd']}" for n, g in busy.items()))
            time.sleep(POLL_S)

    def requeue(self, owner):
        """Hold the queue, then stop each restartable job of the owner and put it back in the queue."""
        import visionjob
        runner = visionjob.Runner(self.c, self.controller)
        self.controller.hold_queue("a profile switch runs", time.time())
        live = lambda: [j for j in runner.refresh()["jobs"].values() if visionjob.is_vision(j) and j["status"] != "settled"
                        and set(j["resources"]) & set(owner["resources"])]
        deadline = self.deadline
        while True:                       # restartable jobs keep working while the others run to their end
            fixed = [j for j in live() if j["status"] == "running" and not (j["command"].get("restartable") and j["command"].get("entry"))]
            if not fixed:
                break
            names = ", ".join(f"{j['id']} ({j['command']['by']}, {j['command']['node']})" for j in fixed)
            if self.cancel_once:
                by = f"the switch (--cancel-once) at {self.c.now()}"
                once = {job["id"]: job for job in fixed}
                for line in self.c.pmap(lambda ident: f"{ident} on {once[ident]['command']['node']} (--once) stopped: "
                                                      f"{runner.stop(once[ident], by=by)['status']}", list(once)).values():
                    self.say("  " + line)
                continue
            if time.monotonic() >= deadline:
                raise Refused(f"jobs that run once still run at the maximum time of the switch: {names}; "
                              "mark them (`spark job set ID --restartable`), cancel them, or give a larger --max-seconds")
            self.say(f"  waiting for {len(fixed)} jobs that run once: {names}; to stop them: --cancel-once")
            time.sleep(POLL_S)
        jobs = live()
        def one(job):
            if job["status"] == "admitted":
                return runner.absent(job, "unknown", "switch before start")["id"] + " closed"
            return f"{job['id']} on {job['command']['node']} stopped; waits as {runner.requeue(job)['id']}"
        by_id = {job["id"]: job for job in jobs}
        for line in self.c.pmap(lambda ident: one(by_id[ident]), list(by_id)).values() if jobs else []:
            self.say("  " + line)

    def resume(self):
        """End the queue hold and start what fits. Runs after every switch, also a failed one."""
        import visionjob
        self.controller.hold_queue("", 0)
        try:
            started, _ = visionjob.Runner(self.c, self.controller).dispatch()
        except (Refused, OSError, subprocess.SubprocessError) as error:
            return [f"queue: dispatch failed, the timer tries again: {error}"]
        waiting = len(self.controller.store.read().get("queue", {}))
        return [f"queue: {len(started)} jobs started, {waiting} wait"] if started or waiting else []

    def lease(self, name):
        """Retire the old owner. The GPU leases become free for the target."""
        import labprofile
        owner, _ = self.owner()
        if not owner or owner.get("outcome") != "ready":
            return
        if self.spec(name).get("studio") and owner["spec"].get("lab_profile"):
            labprofile.start_head(self.runtime)   # the stock census and settings need the head
        labprofile.retire(self.controller, owner)

    def load(self, name, minutes=20):
        """Take the leases for the target and wait until it is READY. Progress is printed."""
        import labprofile, stockstudio
        from profilecli import source_identity
        spec, source = self.spec(name), source_identity(self.c.ROOT)
        key = self.keys.setdefault(name, uuid.uuid4().hex)
        deadline = min(time.monotonic() + minutes * 60, self.deadline)
        while True:
            if spec.get("lab"):
                op = labprofile.apply(self.c, self.controller, self.runtime, spec, key, source)
            else:
                if not labprofile.head_active(self.runtime):
                    labprofile.start_head(self.runtime)
                op = stockstudio.apply(self.controller, self.runtime, spec, key, source, wait=30)
            blockers = op.get("blockers") or op.get("current_observation", {}).get("blockers") or []
            if op.get("outcome") == "ready" and not blockers:
                return
            reason = "; ".join(str(b.get("reason")) + (f" {b['missing']}" if b.get("missing") else "") for b in blockers) or op.get("phase")
            if op.get("outcome") not in (None, "ready") or time.monotonic() >= deadline:
                raise Refused(f"{name} did not become ready: {reason}")
            self.say(f"  waiting: {reason}")
            time.sleep(POLL_S)

    # ---------------------------------------------------------------- verify
    def verify(self, name):
        """One line per node: a real GPU container job on a job node, the head's own report for a Studio node."""
        import visionjob
        spec = self.spec(name)
        owner, state = self.owner()
        reserved = {node: group for group in spec.get("reservations", {}) for node in spec["groups"][group]}
        overview = self.runtime.studio() if spec.get("studio") else {}
        workers = {w.get("address"): w for w in overview.get("workers", [])}
        runner = visionjob.Runner(self.c, self.controller)

        def one(node):
            if node in reserved:
                group, role, image = reserved[node], spec["reservations"][reserved[node]]["for"], spec["images"][node]
                live = [j["id"] for j in state["jobs"].values() if visionjob.is_vision(j) and j["status"] != "settled"
                        and j["command"]["node"] == node]
                if live:
                    return {"ok": True, "line": f"{node} {group} {role}: OK, {len(live)} live jobs keep running ({', '.join(live[:3])})"}
                held = (owner.get("claims") or {}).get(group)
                if held:
                    return {"ok": True, "line": f"{node} {group} {role}: OK, held by {held['by']} for '{held['for']}'; no test job"}
                cmd = GPU_JOB if image.startswith("spark-workload") else GPU_PLAIN
                entry = {"id": visionjob.new_id("q"), "by": "switch", "image": image, "cmd": cmd, "mem": 4, "cpus": 2,
                         "gpu": True, "name": "switch verify", "extra": {"purpose": "switch verify"}}
                try:
                    fact = runner.facts([node], {image})[node]
                    if not fact:
                        return {"ok": False, "line": f"{node} {group} {role}: FAILED, the node does not answer"}
                    job = runner.start(entry, node, fact, {"mem": 4.0, "cpus": 2.0, "gpu": 25.0})
                    if job["status"] != "running":
                        return {"ok": False, "line": f"{node} {group} {role}: FAILED, job did not start: {job.get('blocker')}"}
                    out = self.c.ssh(node, f"docker wait {job['command']['container']} >/dev/null; docker logs {job['command']['container']} 2>&1 | grep '^GPU ' | tail -1", timeout=180)
                    job = runner.refresh(job["id"])["jobs"][job["id"]]
                    code = runner.exit_code(job)
                    self.c.ssh(node, f"docker rm {job['command']['container']} >/dev/null 2>&1; rm -rf \"$HOME/spark/jobs/{job['id']}\"", timeout=60)
                    ok = code == 0 and out.stdout.startswith("GPU ")
                    return {"ok": ok, "line": f"{node} {group} {role}: {'OK' if ok else 'FAILED'}, job {job['id']} exit {code}, {out.stdout.strip() or 'no GPU line'} ({image})"}
                except (Refused, subprocess.SubprocessError) as error:
                    return {"ok": False, "line": f"{node} {group} {role}: FAILED, {error}"}
            unit = overview.get("this", {}) if node == "spark3" else workers.get(node, {})
            ok = unit.get("state") in ("ready", "busy") and unit.get("guard") is True and (node == "spark3" or (unit.get("enabled") and unit.get("usable")))
            return {"ok": bool(ok), "line": f"{node} Studio: {'OK' if ok else 'FAILED'}, {unit.get('state')}, version {unit.get('version')}, "
                                             f"{unit.get('memory_gb')} GB free memory" + ("" if ok else f", {unit.get('why') or unit.get('unusable') or 'not usable'}")}
        rows = self.c.pmap(one, self.c.RING)
        return [rows[node] for node in self.c.RING]


def targets(root):
    import clusterprofile, labprofile, stockstudio
    names = []
    for name in clusterprofile.NAMES:
        try:
            spec = clusterprofile.load(root, name)
        except Exception:
            continue
        if labprofile.covers(spec) or stockstudio.covers(spec):
            names.append(name)
    return names


def check_only(cluster, args, say):
    """`--preflight`: the checks of a switch and the asset restore, for any saved profile. No lease or model changes."""
    import assetrestore, clusterprofile
    from clustercontrol import Controller, Store
    started = time.monotonic()
    if args.target in targets(cluster.ROOT):
        blockers = Driver(cluster, Controller(Store(cluster.STATE)), say=say).preflight(args.target)
    else:                                                # a model profile: its assets only
        blockers = assetrestore.restore(cluster, clusterprofile.load(cluster.ROOT, args.target), say, args.instance)
    for what, fix in blockers:
        say(f"  - {what}\n      fix: {fix}")
    say(f"{'BLOCKED' if blockers else 'OK'}: preflight of {args.target}{' ' + args.instance if args.instance else ''} "
        f"in {time.monotonic() - started:.1f} s, {len(blockers)} blockers; no profile was changed")
    raise SystemExit(2 if blockers else 0)


def command(cluster, args):
    from clustercontrol import Controller, Store
    if args.target is None:
        print("targets: " + ", ".join(targets(cluster.ROOT)))
        return
    if args.preflight:
        check_only(cluster, args, lambda text: print(text, flush=True))
    if args.detach:
        unit = "spark-switch-" + time.strftime("%H%M%S")
        done = subprocess.run(["systemd-run", "--user", "--unit=" + unit, "--collect", "--setenv=SPARK_STATE=" + str(cluster.STATE),
                               "--setenv=SPARK_AS=" + cluster.me(args), "--setenv=PATH=" + os.environ.get("PATH", ""),
                               sys.executable, str(cluster.ROOT / "tools/cluster"), "switch", args.target,
                               *(["--cancel-once"] if args.cancel_once else [])],
                              capture_output=True, text=True, timeout=15)
        if done.returncode:
            raise Refused("the detached switch did not start: " + done.stderr.strip()[-200:])
        print(f"started {unit}; follow it with: journalctl --user -u {unit} -f")
        return
    say = lambda text: print(text, flush=True)
    driver = Driver(cluster, Controller(Store(cluster.STATE)), say=say)
    driver.cancel_once = args.cancel_once
    driver.spec(args.target)
    raise SystemExit(Switch(driver, say, os.environ.get("SPARK_SWITCH_FAULT") or None, args.max_seconds).run(args.target))


def stopped_often(state, now):
    """Live jobs that a switch or a lost node already stopped STOPS_MAX times in STOPS_WINDOW_S."""
    import visionjob
    return [f"{job['id']} of {job['command']['by']}" for job in state["jobs"].values()
            if visionjob.is_vision(job) and job["status"] != "settled"
            and sum(1 for at in (job["command"].get("entry") or {}).get("stops", []) if now - at < STOPS_WINDOW_S) >= STOPS_MAX]


def drill(cluster, args):
    """`spark drill PHASE`: a switch with a fault in PHASE. The earlier profile returns and the queue is released by
    the same command, inside --max-seconds."""
    from clustercontrol import Controller, Store
    say = lambda text: print(text, flush=True)
    driver = Driver(cluster, Controller(Store(cluster.STATE)), say=say)
    current = driver.current()
    if current not in targets(cluster.ROOT):
        raise Refused(f"a drill needs an applied switch target to return to; the profile is {current or 'none'}; nothing was changed")
    target = args.target or next(name for name in ("jobs", "jobs-studio") if name != current)
    driver.spec(target)
    often = stopped_often(driver.controller.store.read(), time.time())
    if often:
        raise Refused(f"drill refused, nothing was changed: {', '.join(often[:4])} already stopped {STOPS_MAX} times in "
                      f"{STOPS_WINDOW_S // 60} min; wait, or run the drill when these jobs have ended")
    say(f"drill: fault in {args.phase} of a switch from {current} to {target}; at most {args.max_seconds:.0f} s, then {current} returns")
    raise SystemExit(Switch(driver, say, args.phase, args.max_seconds).run(target))


def register(cluster, parent):
    p = parent.add_parser("switch", help="change what runs on the Sparks: preflight, unload, lease, load, verify; rolls back on failure")
    p.add_argument("target", nargs="?", help="jobs, jobs-studio, jobs-studio-pair or studio-4; no name lists the targets")
    p.add_argument("--detach", action="store_true", help="run as a user unit and return")
    p.add_argument("--preflight", action="store_true", help="checks only, for any saved profile; a missing asset that a peer has is copied back")
    p.add_argument("--instance", help="with --preflight: one instance of a model profile (pair34)")
    p.add_argument("--cancel-once", action="store_true", help="stop live --once jobs at once (they do not restart); default: wait for their end")
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS, help=f"the longest time of the switch and of its queue hold (default {MAX_SECONDS}); past it the earlier profile returns")
    p.add_argument("--as", dest="as_", default=argparse.SUPPRESS)
    p.set_defaults(fn=lambda a: command(cluster, a))
    p = parent.add_parser("drill", help="rollback drill: a switch with a fault in one phase; the earlier profile returns by itself")
    p.add_argument("phase", choices=PHASES[1:])
    p.add_argument("--target", help="the profile the drill switches to (default: the other jobs profile)")
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    p.add_argument("--as", dest="as_", default=argparse.SUPPRESS)
    p.set_defaults(fn=lambda a: drill(cluster, a))
