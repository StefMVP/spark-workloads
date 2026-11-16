"""Utilisation of the four Sparks: one light sample per node each tick, a small time series on lab, a summary.

`spark tick` (a user timer on lab, every 30 s) starts waiting jobs that have room, then takes one sample:
one ssh per node. Samples: state/util/YYYY-MM-DD.jsonl. Summary: `spark util [--hours N]` and util.json.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RING = ("spark1", "spark2", "spark3", "spark4")
REPORT = Path(os.environ.get("SPARK_UTIL_JSON", Path(os.environ.get("SPARK_STATE", Path.home() / ".local/state/spark-workloads/cluster.json")).parent / "util.json"))
IDLE_GPU, IDLE_LOAD, IDLE_FLAG_S = 5, 1.0, 300
HOLD_FLAG_S = 300        # switch.MAX_SECONDS: a queue hold that is older is a fault
UNIT = "spark-tick"


def folder(cluster):
    return Path(cluster.STATE).parent / "util"


def sample(cluster, facts, queued):
    """One row from the facts that the dispatch pass read: no extra ssh."""
    nodes = {node: ({"gpu": fact["gpu_util"], "load": fact["load"], "cores": fact["cores"], "mem_free": int(fact["mem_avail"]),
                     "jobs": len(fact["containers"])} if fact else None) for node, fact in facts.items()}
    row = {"t": int(time.time()), "queued": queued, "nodes": nodes}
    path = folder(cluster) / (dt.date.today().isoformat() + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
    return row


def rows(cluster, hours):
    since, found = time.time() - hours * 3600, []
    for path in sorted(folder(cluster).glob("*.jsonl"))[-(int(hours // 24) + 2):]:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row["t"] >= since:
                found.append(row)
    return found


def idle(fact):
    return fact["gpu"] < IDLE_GPU and fact["load"] < IDLE_LOAD and fact["jobs"] == 0


def summarize(samples, hours):
    """Per node: average GPU and CPU use, idle share, and the idle time now. Flags an idle node while jobs wait."""
    nodes, flags = {}, []
    for node in RING:
        facts = [(r["t"], r["nodes"][node], r["queued"]) for r in samples if r["nodes"].get(node)]
        if not facts:
            nodes[node] = {"samples": 0}
            continue
        streak_start = None
        for t, fact, _ in reversed(facts):
            if not idle(fact):
                break
            streak_start = t
        idle_now = facts[-1][0] - streak_start if streak_start is not None else 0
        waited = sum(1 for t, fact, queued in facts if streak_start is not None and t >= streak_start and queued)
        n = len(facts)
        nodes[node] = {"samples": n, "gpu_avg_pct": round(sum(f["gpu"] for _, f, _ in facts) / n, 1),
                       "cpu_avg_pct": round(100 * sum(f["load"] / f["cores"] for _, f, _ in facts) / n, 1),
                       "idle_share_pct": round(100 * sum(idle(f) for _, f, _ in facts) / n, 1),
                       "jobs_avg": round(sum(f["jobs"] for _, f, _ in facts) / n, 2),
                       "mem_free_gib_now": facts[-1][1]["mem_free"], "jobs_now": facts[-1][1]["jobs"],
                       "gpu_now_pct": facts[-1][1]["gpu"], "idle_now_s": idle_now}
        if idle_now >= IDLE_FLAG_S and waited:
            flags.append(f"{node} is idle for {idle_now // 60} min while jobs wait")
    measured = [n for n in nodes.values() if n["samples"]]
    cluster = {"gpu_avg_pct": round(sum(n["gpu_avg_pct"] for n in measured) / len(RING), 1),
               "cpu_avg_pct": round(sum(n["cpu_avg_pct"] for n in measured) / len(RING), 1),
               "idle_share_pct": round(sum(n["idle_share_pct"] for n in measured) / len(RING), 1),
               "nodes_busy_now": sum(1 for n in measured if not n["idle_now_s"] and (n["jobs_now"] or n["gpu_now_pct"] >= IDLE_GPU)),
               "jobs_now": sum(n["jobs_now"] for n in measured)} if measured else {}
    span = (samples[-1]["t"] - samples[0]["t"]) if len(samples) > 1 else 0
    return {"updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "hours": hours, "span_s": span,
            "samples": len(samples), "queued_now": samples[-1]["queued"] if samples else 0, "cluster": cluster, "nodes": nodes, "flags": flags,
            "idle_rule": f"GPU under {IDLE_GPU} %, load under {IDLE_LOAD} and no job container"}


def write_report(summary):
    if REPORT.parent.parent.is_dir():
        REPORT.parent.mkdir(exist_ok=True)
        temporary = REPORT.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=1) + "\n")
        temporary.replace(REPORT)


def check(cluster, controller, now=None):
    """'' when all is well, else one line: a node idle over 5 min while jobs wait, or a queue hold past its deadline."""
    state, now = controller.store.read(), now or time.time()
    import visionjob
    job_nodes = visionjob.Runner(cluster, controller).job_nodes(state)[0]     # a Studio node is idle by design
    faults = [flag for flag in summarize(rows(cluster, 1), 1)["flags"] if flag.split()[0] in job_nodes]
    hold = state.get("queue_hold")
    if hold and now - hold["since"] > HOLD_FLAG_S:
        faults.insert(0, f"the queue is held for {int(now - hold['since']) // 60} min ({hold['reason']}), over the "
                         f"{HOLD_FLAG_S // 60} min limit, {len(state.get('queue', {}))} jobs wait, the hold ends by itself at 7 min")
    if not state.get("profile") and state.get("queue"):
        faults.insert(0, f"no profile is applied and {len(state['queue'])} jobs wait, run `spark switch jobs`")
    return "; ".join(faults)


ALERT_EVERY_S = 900     # one message per cause in this time


def alert(cluster, line, now=None):
    """Send a fault line: SPARK_ALERT_CMD gets it as its last argument. One message per cause per ALERT_EVERY_S,
    and one message when the faults are over."""
    import shlex
    command, memo, now = os.environ.get("SPARK_ALERT_CMD"), folder(cluster) / "alert.json", now or time.time()
    try:
        sent = json.loads(memo.read_text())
    except (OSError, ValueError):
        sent = {}
    causes = [part.split(" for ")[0].split(" and ")[0] for part in line.split("; ") if part] if line else []
    due = [cause for cause in causes if now - sent.get(cause, 0) >= ALERT_EVERY_S]
    over = not causes and bool(sent)
    if not due and not over:
        return False
    sent = {} if over else {**sent, **{cause: now for cause in due}}
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(json.dumps(sent))
    if command:
        subprocess.run([*shlex.split(command), f"spark: {line}" if line else "spark: the fault is over; the queue runs"],
                       capture_output=True, timeout=30)
    return True


def tick(cluster, controller):
    """Dispatch, then sample. A second tick that overlaps this one does nothing."""
    import visionjob
    lock = (Path(cluster.STATE).parent / "tick.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return
    runner = visionjob.Runner(cluster, controller)
    started, facts = runner.dispatch()
    for job in started:
        print(visionjob.row(job, runner))
    sample(cluster, facts, len(controller.store.read().get("queue", {})))
    write_report(summarize(rows(cluster, 1), 1))
    alert(cluster, check(cluster, controller))


def install(cluster, alert_cmd=None):
    """A user timer on lab: `spark tick` every 30 s, at low priority. `alert_cmd` gets each fault line of `spark util --check`."""
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    from sparkinventory import inventory_path
    quote = lambda value: json.dumps(str(value).replace('%', '%%'))
    environment = {"SPARK_STATE": cluster.STATE, "SPARK_INVENTORY": inventory_path(),
                   "SPARK_AS": "spark-tick", "PATH": os.environ.get('PATH', '')}
    if alert_cmd:
        environment['SPARK_ALERT_CMD'] = alert_cmd
    service = "[Unit]\nDescription=spark tick: start waiting Spark jobs and sample utilisation\n\n[Service]\nType=oneshot\nNice=10\n"
    service += ''.join('Environment=' + quote(key+'='+str(value)) + '\n' for key, value in environment.items())
    service += 'ExecStart=' + quote(sys.executable) + ' ' + quote(cluster.ROOT / 'tools/cluster') + ' tick\nTimeoutStartSec=240\n'
    (units / f"{UNIT}.service").write_text(service)
    (units / f"{UNIT}.timer").write_text(
        "[Unit]\nDescription=spark tick every 30 s\n\n[Timer]\nOnBootSec=60\nOnUnitActiveSec=30\nAccuracySec=5\n\n[Install]\nWantedBy=timers.target\n")
    for argv in (["daemon-reload"], ["enable", "--now", f"{UNIT}.timer"]):
        done = subprocess.run(["systemctl", "--user", *argv], capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"spark: systemctl --user {' '.join(argv)}: {done.stderr.strip()}")
    print(f"{UNIT}.timer is on: `systemctl --user status {UNIT}.timer`; stop it with `systemctl --user disable --now {UNIT}.timer`")


def command(cluster, args):
    from clustercontrol import Controller, Store
    controller = Controller(Store(cluster.STATE))
    if args.cmd_name == "tick":
        return install(cluster, args.alert_cmd) if args.install else tick(cluster, controller)
    if args.check:
        line = check(cluster, controller)
        print(line or "ok: no node is idle while jobs wait and the queue is not held past its limit")
        raise SystemExit(1 if line else 0)
    summary = summarize(rows(cluster, args.hours), args.hours)
    write_report(summary)
    if args.json:
        print(json.dumps(summary, indent=1))
        return
    print(f"last {args.hours:g} h: {summary['samples']} samples over {summary['span_s'] // 60} min; {summary['queued_now']} jobs wait now")
    print(f"{'node':7} {'GPU avg':>8} {'CPU avg':>8} {'idle share':>11} {'jobs avg':>9} {'now: GPU':>9} {'jobs':>5} {'idle for':>9}")
    for node, n in summary["nodes"].items():
        if not n["samples"]:
            print(f"{node:7} no samples")
            continue
        print(f"{node:7} {n['gpu_avg_pct']:7.1f}% {n['cpu_avg_pct']:7.1f}% {n['idle_share_pct']:10.1f}% {n['jobs_avg']:9.2f} "
              f"{n['gpu_now_pct']:8.0f}% {n['jobs_now']:5d} {n['idle_now_s'] // 60:6d} min")
    if summary["cluster"]:
        c = summary["cluster"]
        print(f"{'cluster':7} {c['gpu_avg_pct']:7.1f}% {c['cpu_avg_pct']:7.1f}% {c['idle_share_pct']:10.1f}% {'':9} "
              f"{c['nodes_busy_now']} of 4 nodes busy now, {c['jobs_now']} jobs")
    for flag in summary["flags"]:
        print("FLAG: " + flag)
    if not summary["samples"]:
        print("no samples: turn the sampler on with `spark tick --install`")


def register(cluster, parent):
    p = parent.add_parser("util", help="utilisation per node: GPU, CPU, idle share; flags an idle node while jobs wait")
    p.add_argument("--hours", type=float, default=1); p.add_argument("--json", action="store_true")
    p.add_argument("--check", action="store_true", help="exit 1 with one line: a node idle over 5 min while jobs wait, or the queue held over 5 min")
    p.set_defaults(fn=lambda a: command(cluster, a))
    p = parent.add_parser("tick", help="start waiting jobs that have room and take one utilisation sample (run by a timer)")
    p.add_argument("--install", action="store_true", help="install and start the 30 s user timer on lab")
    p.add_argument("--alert-cmd", help="with --install: a command that gets each new fault line of `spark util --check` as its last argument")
    p.set_defaults(fn=lambda a: command(cluster, a))
