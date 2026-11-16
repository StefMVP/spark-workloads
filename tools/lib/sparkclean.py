"""`spark clean [NODE...]`: remove what is clearly temporary on the Sparks. A dry run unless `--yes`.

Each class is one rule in CLASSES: where it is, the minimum age by last-modified time and the guard. Only classes that
held real space on the four nodes (measured 2026-11-15) are here; old logs and *.part/*.tmp files held nothing.
Never a class: model weights, profile assets and caches (`spark asset rm`), ~/spark/data, ~/spark/git, ~/spark/groups,
docker images, anything of a live, waiting or stopped-restartable job, anything outside these roots.
"""
from __future__ import annotations

import json
import shlex
import subprocess

import visionjob

CLASSES = {   # name: (where, minimum age in hours, guard)
    "job-workspace": ("~/spark/jobs/ID: everything except job.log and files under 1 MB", visionjob.SWEEP_AGE_S / 3600,
                      "the router has the job as finished, cancelled or unknown; never a live, waiting or stopped-restartable job"),
    "build-cache": ("docker build cache that no image shares", 24, "docker builder prune --filter until: never an image"),
    "pip-uv-cache": ("~/.cache/uv, ~/.cache/pip", 24, "no file in it changed within the age"),
}
CACHES = ("~/.cache/uv", "~/.cache/pip")
OWNERS = {"~/.cache/": "spark asset rm {node} PATH", "~/spark/data": "job data: copies of the lake, written by jobs",
          "~/spark/git": "router repository mirrors", "/models": "spark asset rm {node} PATH",
          "docker images": "profiles and jobs name them; not a cleanup class"}
PROBE = r'''
import json, os, pathlib, shutil, subprocess, sys, time
home, now = pathlib.Path.home(), time.time()
run = lambda *a: subprocess.run(a, capture_output=True, text=True, timeout=600).stdout
du = lambda *p: int(run("du", "-xsk", "--", *p).split("\n")[-2].split()[0]) * 1024 if p else 0
out = {"free": shutil.disk_usage(home).free, "jobs": [], "build": [], "caches": [], "other": []}
root = home / "spark/jobs"
for d in sorted(root.iterdir() if root.is_dir() else []):
    try:
        stale = [str(e) for e in d.iterdir() if e.is_dir() or (e.name != "job.log" and e.stat().st_size >= 1 << 20)]
        if d.is_dir() and stale:
            log = d / "job.log"
            age = now - max(d.stat().st_mtime, log.stat().st_mtime if log.exists() else 0)
            out["jobs"].append({"id": d.name, "age_h": age / 3600, "bytes": int(run("du", "-xsck", "--", *stale).split()[-2]) * 1024})
    except (OSError, IndexError, ValueError):
        pass
for line in run("docker", "buildx", "du", "--filter", "until=24h", "--format", "json").splitlines():
    try:
        item = json.loads(line)
    except ValueError:
        continue
    if item.get("Reclaimable") and not item.get("Shared") and not item.get("InUse"):
        size = item.get("Size", "0B")
        unit = next((u for u in ("kB", "MB", "GB", "TB", "B") if size.endswith(u)), "B")
        when = time.mktime(time.strptime(item["CreatedAt"][:19], "%Y-%m-%d %H:%M:%S"))
        out["build"].append({"bytes": int(float(size[:-len(unit)]) * {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[unit]),
                             "age_h": (now - when) / 3600})
for name in json.loads(sys.argv[1]):
    path = pathlib.Path(os.path.expanduser(name))
    if path.is_dir():
        newest = run("find", str(path), "-xdev", "-printf", "%T@\n")
        newest = max((float(x) for x in newest.split()), default=path.stat().st_mtime)
        out["caches"].append({"path": name, "bytes": du(str(path)), "age_h": (now - newest) / 3600})
for path in [*sorted((home / ".cache").glob("*")), home / "spark/data", home / "spark/git", pathlib.Path("/models")]:
    shown = "~/" + str(path.relative_to(home)) if path.is_relative_to(home) else str(path)
    if path.is_dir() and shown not in json.loads(sys.argv[1]):
        out["other"].append({"path": shown, "bytes": du(str(path))})
images = run("docker", "system", "df", "--format", "{{json .}}").splitlines()
for line in images:
    item = json.loads(line)
    if item.get("Type") == "Images":
        size = item["Size"]
        unit = next((u for u in ("kB", "MB", "GB", "TB", "B") if size.endswith(u)), "B")
        out["other"].append({"path": "docker images", "bytes": int(float(size[:-len(unit)]) * {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[unit])})
print(json.dumps(out))
'''


def node_name(text):
    return text if text.startswith("spark") else f"spark{text[-1]}"


def probe(cluster, nodes):
    script = shlex.join(["python3", "-c", PROBE, json.dumps(list(CACHES))])
    def one(node):
        try:
            reply = cluster.ssh(node, script, timeout=900)
            return json.loads(reply.stdout) if reply.returncode == 0 else {"error": reply.stderr.strip()[-200:]}
        except (subprocess.TimeoutExpired, ValueError) as error:
            return {"error": type(error).__name__}
    return cluster.pmap(one, nodes)


def select(fact, keep, older_than=0, only=None):
    """{class: [items]} that `--yes` removes on one node. `keep`: job ids of live, waiting and stopped-restartable jobs."""
    age = {name: max(rule[1], older_than) for name, rule in CLASSES.items()}
    found = {
        "job-workspace": [j for j in fact["jobs"] if j["age_h"] >= age["job-workspace"] and j["id"] not in keep
                          and visionjob.JOB_DIR.fullmatch(j["id"])],
        "build-cache": [b for b in fact["build"] if b["age_h"] >= age["build-cache"]],
        "pip-uv-cache": [c for c in fact["caches"] if c["age_h"] >= age["pip-uv-cache"] and c["path"] in CACHES],
    }
    return {name: items for name, items in found.items() if not only or name in only}


def remove_script(chosen, older_than=0):
    script = "".join(visionjob.clear_script(j["id"]) for j in chosen.get("job-workspace", []))
    if chosen.get("build-cache"):
        script += f"docker builder prune -f --filter until={int(max(CLASSES['build-cache'][1], older_than))}h >/dev/null\n"
    script += "".join(f"rm -rf {c['path']}\n" for c in chosen.get("pip-uv-cache", []) if c["path"] in CACHES)
    return script


def gb(n):
    return n / 1e9


def report(node, fact, chosen, say=print):
    say(f"{node}: {gb(fact['free']):.1f} GB free")
    say(f"  {'class':14} {'count':>6} {'size':>10} {'oldest':>8}")
    for name, items in chosen.items():
        oldest = f"{max(i['age_h'] for i in items):.0f} h" if items else "-"
        say(f"  {name:14} {len(items):6d} {gb(sum(i['bytes'] for i in items)):7.1f} GB {oldest:>8}")
    say(f"  --yes frees {gb(sum(i['bytes'] for items in chosen.values() for i in items)):.1f} GB")
    say("  not touched (largest 5, with the owner):")
    for item in sorted(fact["other"], key=lambda i: -i["bytes"])[:5]:
        owner = next((v for k, v in OWNERS.items() if item["path"].startswith(k)), "not a cleanup class")
        say(f"    {item['path']:40} {gb(item['bytes']):7.1f} GB  {owner.format(node=node)}")


def command(cluster, args):
    from clustercontrol import Controller, Store
    store = Store(cluster.STATE)
    unknown = [c for c in args.cls or [] if c not in CLASSES]
    if unknown:
        raise SystemExit(f"spark clean: unknown class {', '.join(unknown)}; classes: {', '.join(CLASSES)}")
    nodes = [node_name(n) for n in args.nodes] or list(cluster.RING)
    if any(n not in cluster.RING for n in nodes):
        raise SystemExit(f"spark clean: nodes are {', '.join(cluster.RING)}")
    facts = probe(cluster, nodes)
    lines = []
    for node in nodes:
        fact = facts[node]
        if fact.get("error"):
            lines.append(f"{node}: not reachable ({fact['error']}); nothing was removed")
            continue
        keep = visionjob.kept_jobs(Controller(store).store.read())      # read now: a job that started meanwhile is kept
        chosen = select(fact, keep, args.older_than, args.cls)
        report(node, fact, chosen)
        total = sum(i["bytes"] for items in chosen.values() for i in items)
        if not args.yes:
            lines.append(f"{node}: dry run, --yes frees {gb(total):.1f} GB; {gb(fact['free']):.1f} GB free")
            continue
        script = remove_script(chosen, args.older_than)
        done = cluster.ssh(node, script + 'python3 -c "import shutil,pathlib;print(shutil.disk_usage(pathlib.Path.home()).free)"',
                           timeout=3600) if script else None
        free = int(done.stdout.split()[-1]) if done and done.returncode == 0 and done.stdout.split() else fact["free"]
        failed = f"; removal exited {done.returncode}: {done.stderr.strip()[-120:]}" if done and done.returncode else ""
        lines.append(f"{node}: freed {gb(max(0, free - fact['free'])):.1f} GB; {gb(free):.1f} GB free{failed}")
    for line in lines:
        print(line)


def register(cluster, parent):
    p = parent.add_parser("clean", help="remove finished job workspaces, old docker build cache, pip and uv caches; a dry run unless --yes")
    p.add_argument("nodes", nargs="*", metavar="NODE"); p.add_argument("--yes", action="store_true", help="remove; without it only list")
    p.add_argument("--older-than", type=float, default=0, metavar="HOURS", help="only items older than this (never under the class age)")
    p.add_argument("--class", dest="cls", action="append", metavar="NAME", help=f"only this class: {', '.join(CLASSES)}")
    p.add_argument("--as", dest="as_", default=__import__("argparse").SUPPRESS)
    p.set_defaults(fn=lambda a: command(cluster, a))
