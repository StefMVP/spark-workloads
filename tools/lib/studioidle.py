"""spark idle: no Imagination Studio model server stays resident on a Spark without a job.

Each node is asked through its own Studio app (GET/POST 127.0.0.1:7870/api/models[/release]): the app stops only
idle engines and refuses while a job runs or waits. A node whose app is down has no job; an orphaned Studio ComfyUI
there is stopped with the Studio's own `spark/comfy.sh stop`. Other model processes (an LLM profile's instances,
Blender) are reported, never stopped here. Exit 1 when a node with no job still holds an idle Studio model server.
"""
from __future__ import annotations

import json

NODES = ["spark1", "spark2", "spark3", "spark4"]
PROBE = r'''
import json, os, pathlib, subprocess, sys, urllib.request
release = sys.argv[1] == "1"
def api(path, post=False, timeout=5):
    request = urllib.request.Request("http://127.0.0.1:7870/api/" + path, data=b"" if post else None,
                                     method="POST" if post else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
def kind(pid):
    try:
        args = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return "gone"
    line = " ".join(a.decode(errors="replace") for a in args if a)
    if "tensorfold-studio" in line and "main.py" in line and "--port" in line:
        return "studio:ComfyUI"
    for token, name in (("tensorfold-native", "Bolt"), ("music3-generate", "MiniMax Music 3"),
                        ("drama3-generate", "Drama 3"), ("tensorfold.drama3", "Drama 3")):
        if token in line and "tensorfold-studio" in line:
            return "studio:" + name
    if "YuE2-3B" in line and "--serve" in line:
        return "studio:YuE2"
    return "blender" if os.path.basename(line.split(" ")[0]) == "blender" else "other"
def gpu():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    rows = []
    for line in out.splitlines():
        pid, _, mib = (part.strip() for part in line.partition(","))
        if pid.isdigit():
            rows.append({"pid": int(pid), "mib": int(mib) if mib.isdigit() else None, "kind": kind(pid)})
    return rows
fact = {"app": "down", "jobs": None, "busy": "", "released": None}
try:
    jobs = api("telemetry")["jobs"]
    fact.update(app="up", jobs={"running": jobs["running"], "queued": jobs["queued"]})
    try:
        fact["busy"] = api("models").get("busy") or ""
    except Exception as error:
        fact["busy"] = f"the app has no idle-model route ({type(error).__name__}): update it"
except Exception:
    pass
rows = gpu()
studio = [row for row in rows if row["kind"].startswith("studio:")]
if release and studio:
    if fact["app"] == "up" and not fact["busy"]:
        fact["released"] = api("models/release", post=True, timeout=90)
    elif fact["app"] == "down" and any(row["kind"] == "studio:ComfyUI" for row in studio):
        done = subprocess.run(["bash", str(pathlib.Path.home() / "Imagination-Studio/spark/comfy.sh"), "stop"],
                              capture_output=True, text=True, timeout=90)
        fact["released"] = {"comfy.sh stop": done.stdout.strip() or done.stderr.strip()[-200:]}
    rows = gpu()
fact["cuda"] = rows
print(json.dumps(fact))
'''


def idle_resident(fact: dict) -> bool:
    """A node with no job whose GPU still holds a Studio model server."""
    if fact.get("error"):
        return False
    studio = [row for row in fact["cuda"] if row["kind"].startswith("studio:")]
    jobs = fact.get("jobs") or {}
    in_use = fact.get("app") == "up" and (jobs.get("running") or jobs.get("queued") or fact.get("busy"))
    return bool(studio) and not in_use


def probe(cluster, node: str, release: bool) -> dict:
    import shlex
    result = cluster.ssh(node, shlex.join(["python3", "-c", PROBE, "1" if release else "0"]), timeout=150)
    if result.returncode:
        return {"node": node, "error": (result.stderr or result.stdout).strip()[-300:] or f"exit {result.returncode}"}
    try:
        fact = json.loads(result.stdout)
        fact["cuda"]
    except (ValueError, TypeError, KeyError):
        return {"node": node, "error": "the probe did not answer with its facts"}
    fact["node"] = node
    fact["idle_resident"] = idle_resident(fact)
    return fact


def enforce(cluster, release: bool, nodes=NODES) -> list[dict]:
    found = cluster.pmap(lambda node: probe(cluster, node, release), nodes)
    return [found[node] for node in nodes]


def line(fact: dict) -> str:
    if fact.get("error"):
        return f"{fact['node']}: unreachable ({fact['error']})"
    held = ", ".join(f"{row['kind']} pid {row['pid']} {row['mib']} MiB" for row in fact["cuda"]) or "GPU free"
    jobs = fact.get("jobs") or {}
    state = ("IDLE RESIDENT MODEL" if fact["idle_resident"] else
             f"app {fact['app']}, {jobs.get('running', 0)} running, {jobs.get('queued', 0)} queued"
             + (f", {fact['busy']}" if fact.get("busy") else ""))
    stopped = fact.get("released") or {}
    note = f"; stopped {', '.join(stopped.get('stopped', [])) or stopped}" if stopped else ""
    return f"{fact['node']}: {state}; {held}{note}"


def command(cluster, args) -> None:
    facts = enforce(cluster, args.yes)
    if args.json:
        print(json.dumps(facts, indent=2))
    else:
        for fact in facts:
            print(line(fact))
    if any(fact.get("idle_resident") for fact in facts):
        raise SystemExit(1)


def register(cluster, parent) -> None:
    p = parent.add_parser("idle", help="report idle resident Studio model servers on every Spark; --yes stops them")
    p.add_argument("--yes", action="store_true", help="stop idle model servers through each Studio app (never a job's)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=lambda a: command(cluster, a))
