"""Named, cached native-recipe switches. No model downloads or global daemon."""
from __future__ import annotations

import contextlib
import argparse
import copy
import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import clusterroute
import opencodeprofile
import studioassets

from profilespec import NAMES, load
DASH_FIELDS = ("llmPorts", "role", "workerNode", "workerHeadId", "llmMonitoring")
OFFLINE = "sparks-offline/offline"
GPU_PROBE = '''import pathlib, subprocess
proc=pathlib.Path(D['proc'])
home=pathlib.Path(D['home']) if D.get('home') else pathlib.Path.home()
root=home/'.local/opt/tensorfold-studio'
python=(root/'venv/bin/python').resolve()
comfy=(root/'ComfyUI').resolve()
guard_script=(home/'Imagination-Studio/spark/memory_guard.py').resolve()
def details(pid):
    p=proc/str(pid)
    args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
    return args,(p/'cwd').resolve(),(p/'exe').resolve()
guards={}
for process in proc.iterdir():
    if not process.name.isdigit(): continue
    try:
        args,cwd,exe=details(process.name)
        if (exe==python and cwd==comfy and 'watch' in args and '--pid' in args
                and any(pathlib.Path(a).resolve()==guard_script for a in args if a.endswith('memory_guard.py'))):
            guards[int(args[args.index('--pid')+1])]=int(process.name)
    except (OSError,ValueError): pass
out=[]
for pid in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
    try:
        args,cwd,exe=details(pid.strip())
        owned=(exe==python and cwd==comfy and 'main.py' in args and int(pid) in guards)
        known=owned or (not any(pathlib.Path(a).name=='main.py' for a in args)
            and any(token in ' '.join(args) for token in ('tensorfold-native','tf-dsv41-lanes','tf-server','ds4-server','comfy','ComfyUI','worker.py')))
        out.append({'pid':int(pid),'known':known,'owned_studio':owned,'guard_pid':guards.get(int(pid))})
    except FileNotFoundError: pass
    except OSError: out.append({'pid':int(pid),'known':False})
print(json.dumps(out))'''


from clustercontrol import Refused


class Pending(Refused):
    pass




def endpoint(i):
    return i["head"] + ":" + str(i["port"])


def primary(p):
    return next((i for i in p["instances"] if i["id"] == p["default"]), None)


def client_model(p):
    return primary(p)["opencode"] if p["instances"] else OFFLINE


def readiness_seconds(value):
    value = int(value)
    if not 0 <= value <= 280:
        raise argparse.ArgumentTypeError("readiness wait must be between 0 and 280 seconds")
    return value


def refresh_complete(report, after):
    return report["ready"] and all((report["nodes"].get(node, {}).get("checked") or 0) > stamp
                                   for node, stamp in after.items())


def instance_healthy(i, live):
    api = live["apis"].get(endpoint(i), {})
    if api.get("health", {}).get("status") != "ok" or i["model"] not in api.get("models", []):
        return False
    return not i.get("cache_check") or (api.get("metrics", {}).get("tensorfold:prompt_cache_can_keep_state", 0) >= 1
        and api.get("metrics", {}).get("tensorfold:prompt_cache_budget_bytes", 0) > 0)


def selection(path):
    if not path.exists():
        return {"present": False}
    data = json.loads(path.read_text())
    return {"present": "model" in data, "model": data.get("model")}


def atomic(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as out:
        json.dump(data, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.chmod(tmp, mode)
    tmp.replace(path)


def metrics(text):
    result = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition(" ")
            try:
                result[key] = float(value)
            except ValueError:
                pass
    return result


def healthy(p, live):
    return all(instance_healthy(i, live) for i in p["instances"]) and (
        bool(p["instances"]) or studio_ready(p.get("studio"), live.get("studio")))


def source_candidates(profiles, live, runtimes):
    """A partial deployment owns its API only when its native runtime matches."""
    found = []
    for name, profile in profiles.items():
        active = [i for i in profile["instances"] if instance_healthy(i, live)]
        if active and all(runtimes[name][i["id"]] == i["cache"]["runtime"] for i in active):
            found.append(name)
    return found


def studio_settings(overview):
    return {"default": overview["default"], "workers": {w["address"]: w["enabled"] for w in overview["workers"]}}


def studio_worker_ready(worker):
    return (worker.get("state") in ("ready", "busy") and worker.get("guard") is True
            and (worker.get("usable") is True or worker.get("enabled") is False)
            and worker.get("code_matches") is not False
            and (not worker.get("working") or worker.get("job")))


def studio_ready(flags, overview):
    if not flags or not overview or studio_settings(overview) != flags:
        return False
    head = overview.get("this", {})
    enabled = [w for w in overview["workers"] if w["enabled"]]
    return (overview.get("available", 0) >= flags["default"]
            and head.get("state") in ("ready", "busy") and head.get("guard") is True
            and all(studio_worker_ready(w) for w in enabled))


def target_studio(p):
    return p.get("studio")


def dashboard(p):
    fields = {n: {"llmPorts": [8888], "role": "standalone", "workerNode": False,
                  "workerHeadId": None, "llmMonitoring": False}
              for n in ("spark1", "spark2", "spark3", "spark4")}
    for i in p["instances"]:
        for n in i["members"]:
            head = n == i["head"]
            fields[n] = {"llmPorts": [i["port"]] if head else [8888], "role": "head" if head else "worker",
                         "workerNode": not head, "workerHeadId": None if head else i["head"], "llmMonitoring": head}
    return fields








class Runtime:
    def __init__(self, cluster):
        self.c = cluster
        self.root = cluster.ROOT
        self.config = Path.home() / ".config/opencode/opencode.json"
        self.routing = clusterroute.Routing(cluster.STATE.parent)
        self.profiles = {n: load(self.root, n) for n in NAMES}
        self.wait_seconds = 280
        self.stop_policy = {}
        self.continue_worker = False
        self.asset_refresh_after = {}

    def remote(self, node, script, timeout=30, input=None):
        r = self.c.ssh(node, script, timeout=timeout, **({'input':input} if input is not None else {}))
        if r.returncode:
            raise Refused(f"{node}: remote operation failed (exit {r.returncode}); inspect host logs")
        return r.stdout

    def python(self, node, code, data):
        body = "import json\nD=json.loads(" + repr(json.dumps(data)) + ")\n" + code
        return json.loads(self.remote(node, "python3 - <<'SPARK_PROFILE_PY'\n" + body + "\nSPARK_PROFILE_PY"))

    def api(self, node, port, path, body=None, method=None, dash=False):
        strict = path.startswith("/api/control/")
        result = self.python(node, '''import urllib.request, urllib.error, pathlib
headers={}
if D['dash']:
    headers['Authorization']='Bearer '+pathlib.Path.home().joinpath('.config/sparkdash/token').read_text().strip()
data=None if D['body'] is None else json.dumps(D['body']).encode()
if data is not None: headers['Content-Type']='application/json'
r=urllib.request.Request('http://127.0.0.1:'+str(D['port'])+D['path'],data=data,headers=headers,method=D['method'])
try:
    with urllib.request.urlopen(r,timeout=5) as f:
        raw=f.read().decode()
        value={} if D['path']=='/health' and not raw.strip() else (json.loads(raw) if not D['path'].endswith('/metrics') else raw)
        print(json.dumps({'ok':True,'value':value} if D['strict'] else value))
except (ValueError, UnicodeError):
    print(json.dumps({'ok':False,'error':'malformed JSON'} if D['strict'] else None))
except (urllib.error.URLError, OSError, TimeoutError) as error:
    print(json.dumps({'ok':False,'error':type(error).__name__} if D['strict'] else None))
''', {"port": port, "path": path, "body": body, "method": method, "dash": dash, "strict": strict})
        if not strict:
            return result
        if not isinstance(result, dict) or result.get('ok') not in (True, False):
            raise Refused(f"unknown Studio control transport outcome on {node}:{port}")
        if result['ok']:
            return result['value']
        if result.get('error') == 'malformed JSON':
            raise Refused(f"malformed Studio control JSON on {node}:{port}")
        raise Refused(f"Studio control API unavailable on {node}:{port} ({result.get('error', 'unknown')})")

    def studio(self, path="/api/sparks", body=None):
        result = self.api("spark3", 7870, path, body)
        if result is None:
            raise Refused("Studio API unavailable")
        return result

    def puffer_report(self, spec):
        return self.python(spec["node"], '''import hashlib, subprocess
from pathlib import Path
root=Path.home()/'spark/puffer-training-1106/PufferLib'
def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1048576),b''): result.update(chunk)
    return result.hexdigest()
try:
    revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
    image=subprocess.run(['docker','image','inspect',D['image'],'--format','{{.Id}}'],capture_output=True,text=True,timeout=5)
    binaries={name:sha(root/name) for name in D['binaries']}
    verified=(revision.returncode==0 and revision.stdout.strip()==D['source_commit']
              and image.returncode==0 and image.stdout.strip()==D['image_id'] and binaries==D['binaries'])
    print(json.dumps({'state':'verified' if verified else 'invalid','source_commit':revision.stdout.strip(),
        'image_id':image.stdout.strip(),'binaries':binaries,'assurance':'installed source/image/binary identity; GPU execution unproved',
        'checkpoint_action':'none'}))
except (OSError,ValueError,subprocess.SubprocessError) as error:
    print(json.dumps({'state':'unknown','reason':type(error).__name__}))
''', spec)

    def observe(self, p):
        def observe_instance(i):
            def node(n):
                expected = i["cache"]["nodes"].get(n, {})
                code = '''import pathlib, subprocess, os
out={'stamps':{},'paths':[]}
for path in D['stamps']:
    try:
        s=os.stat(path);out['stamps'][path]=[s.st_size,int(s.st_mtime),s.st_ino]
    except FileNotFoundError: pass
for path in D['paths']:
    if pathlib.Path(path).exists():out['paths'].append(path)
r=subprocess.run(['docker','image','inspect',D['image']],text=True,capture_output=True)
if r.returncode==0:
    im=json.loads(r.stdout)[0];out['image']={k:im[k] for k in ('RootFS','Config')}
if D['head']:
    r=subprocess.run(['git','-C',D['recipe'],'rev-parse','HEAD'],text=True,capture_output=True)
    out['recipe_ok']=r.returncode==0 and r.stdout.strip().startswith(D['pin']) and pathlib.Path(D['recipe'],D['start']).is_file()
out['mem_gib']=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1048576
print(json.dumps(out))'''
                return self.python(n, code, {"stamps": expected.get("stamps", {}), "paths": i["paths"].get(n, []),
                    "image": i["image"], "head": n == i["head"], "recipe": i["recipe"], "pin": i["recipe"].split("@")[-1], "start": i["start"].removeprefix("./")})
            marker = i["env"]["STATE_DIR"] + "/prepared"
            prepared = self.remote(i["head"], f"cat {shlex.quote(marker)} 2>/dev/null || true").strip() if i["cache"].get("prepared") else None
            return {"nodes": self.c.pmap(node, i["members"]), "runtime": self.effective(i), "prepared": prepared}
        instances = {i["id"]: observe_instance(i) for i in p["instances"]}
        unique = {endpoint(i): i for profile in self.profiles.values() for i in profile["instances"]}
        apis = self.c.pmap(lambda key: self.model_api(unique[key]), list(unique))
        identities = {(i["head"], i["container"], json.dumps(i["cache"]["runtime"], sort_keys=True)): i
                      for profile in self.profiles.values() for i in profile["instances"]}
        effective = self.c.pmap(lambda key: self.effective(identities[key]), list(identities))
        runtimes = {name: {i["id"]: effective[(i["head"], i["container"], json.dumps(i["cache"]["runtime"], sort_keys=True))]
                          for i in profile["instances"]} for name, profile in self.profiles.items()}
        sources = source_candidates(self.profiles, {"apis": apis}, runtimes)
        studio = self.api("spark3", 7870, "/api/sparks")
        jobs = self.api("spark3", 7870, "/api/jobs") if studio else []
        if studio and jobs is None: raise Refused("Studio jobs unavailable; cannot prove idle")
        if not any(api.get("health") for api in apis.values()):
            sources.extend(name for name, profile in self.profiles.items()
                           if not profile["instances"] and studio_ready(profile.get("studio"), studio))
        unknown = []
        if not p["instances"] and not any(api.get("health") for api in apis.values()) and any(value is not None for values in runtimes.values() for value in values.values()):
            unknown.append("native model container still running; cannot reconcile Studio-only placement")
        for n, result in self.c.pmap(lambda n: self.python(n, GPU_PROBE, {"proc": "/proc", "home": None}), self.c.RING).items():
            for process in result:
                if not process["known"]:
                    unknown.append(f"{n}: unrecognized GPU process")
        return {"instances": instances, "apis": apis, "sources": sources,
                "topologies": {n: v["groups"] for n, v in self.profiles.items()},
                "source_endpoints": {n: [endpoint(i) for i in v["instances"]] for n, v in self.profiles.items()},
                "studio": studio, "studio_jobs": jobs, "unknown": unknown,
                "assets": self.asset_report(p, studio) if p.get("assets") else None}

    def asset_check(self, node, manifest):
        code = (self.root / "tools/lib/studioassets.py").read_text()
        return self.python(node, code + "\nprint(json.dumps(check(D,headers=True)))", manifest)

    def asset_report(self, p, overview):
        spec = p["assets"]
        checks = self.c.pmap(lambda n: self.asset_check(n, spec["files"]), ["spark3", *spec["targets"]])
        unknown=studioassets.content_unknown(spec['files'])
        if unknown:
            for check in checks.values():
                check['ok']=False
                check['content_identity_unknown']=unknown
        workers = {w["address"]: w for w in (overview or {}).get("workers", [])}
        for node in spec["targets"]:
            can = workers.get(node, {}).get("can") or {}
            missing = [key for key in spec["capabilities"] if can.get(key) is not True]
            missing += ["image:" + name for name in spec["image_models"] if name not in can.get("image_models", [])]
            if can.get("loras", 0) < spec["loras"]: missing.append("LoRAs:" + str(spec["loras"]))
            checks[node]["missing_capabilities"] = missing
            checks[node]["ok"] = checks[node]["ok"] and not missing
        states = self.c.pmap(self.studio_features, ["spark3", *spec["targets"]])
        for node, state in states.items():
            missing = checks[node].setdefault("missing_capabilities", [])
            if not state.get("voice", {}).get("available"):
                missing.append("voice: " + (state.get("voice", {}).get("why") or "state unavailable"))
            if not state.get("music", {}).get("available") or state.get("music", {}).get("engine") != "yue2":
                missing.append("music:yue2")
            if node == "spark3":
                missing.extend("image:" + name for name in spec["image_models"] if name not in state.get("image_models", []))
            checks[node]["ok"] = checks[node]["ok"] and not missing
        self.asset_checks = checks
        return {"ready": all(r["ok"] for r in checks.values()), "nodes": {
            n: {"ready": r["ok"], "missing_files": len(r["errors"]), "errors": r["errors"][:5],
                "content_identity_unknown": r.get("content_identity_unknown", []),
                "missing_capabilities": r.get("missing_capabilities", []),
                "checked": workers.get(n, {}).get("checked")} for n, r in checks.items()}}

    def studio_features(self, node):
        return self.python(node, '''import urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:7870/api/state',timeout=5) as response:
        state=json.load(response)
except OSError:
    state={}
out={key:{field:state.get(key,{}).get(field) for field in fields}
     for key,fields in {'voice':['available','why','backend'],'music':['available','engine']}.items()}
out['image_models']=[m['id'] for m in state.get('image_models',[])]
print(json.dumps(out))''', {})



    def effective(self, p):
        return self.python(p["head"], """import subprocess
r=subprocess.run(['docker','inspect',D['container']],capture_output=True,text=True)
out=None
if r.returncode==0 and json.loads(r.stdout)[0]['State']['Running']:
    c=json.loads(r.stdout)[0]['Config'];env=dict(x.split('=',1) for x in c['Env'])
    out={'args':c['Cmd'],'env':{k:env.get(k) for k in D['keys']}}
print(json.dumps(out))""", {"container": p["container"], "keys": list(p["cache"]["runtime"]["env"])})

    def model_api(self, p):
        health = self.api(p["head"], p["port"], "/health")
        models = self.api(p["head"], p["port"], "/v1/models") if health else None
        text = self.api(p["head"], p["port"], "/metrics") if health else None
        return {"health": health or {}, "models": [m["id"] for m in (models or {}).get("data", [])], "metrics": metrics(text or "")}

    def settings(self, live):
        # Read only fields this tool changes; credentials never cross SSH.
        dash = self.python("spark1", '''import pathlib
p=pathlib.Path.home()/'sparkDash/config/sparks.json'
out={s['id']:{k:s.get(k) for k in D} for s in json.loads(p.read_text())['sparks']} if p.exists() else {}
if out and not (pathlib.Path.home()/'.config/sparkdash/token').is_file(): raise RuntimeError('dashboard token unavailable')
print(json.dumps(out))''', DASH_FIELDS)
        return {"opencode": self.client_snapshot(), "dash": dash, "studio": studio_settings(live["studio"]) if live["studio"] else None}

    def client_ids(self):
        return {OFFLINE.split("/")[0]} | {i["opencode"].split("/", 1)[0] for p in self.profiles.values() for i in p["instances"]}

    def client_templates(self):
        path = self.routing.directory / "providers.json"
        return json.loads(path.read_text()) if path.exists() else None

    def provider_entry(self, instance, providers):
        available = dict(self.client_templates() or {}, **providers)
        key = instance["opencode"].split("/", 1)[0]
        return available.get(key) or available.get(instance.get("provider_template"))

    def client_snapshot(self):
        saved = selection(self.config)
        data = json.loads(self.config.read_text()) if self.config.exists() else {}
        ids = self.client_ids()
        saved["providers"] = {k: data.get("providers", {}).get(k) for k in ids}
        saved["routing"] = self.routing.snapshot()
        saved["templates"] = self.client_templates()
        return saved

    def preflight_clients(self, p):
        if not self.config.exists(): return
        opencodeprofile.idle()





    def capacity(self, p):
        for n in {n for i in p["instances"] for n in i["members"]}:
            minimum = max(i["min_free_gib"] for i in p["instances"] if n in i["members"])
            raw = self.remote(n, "awk '/MemAvailable:/{print $2}' /proc/meminfo")
            if int(raw) / 1048576 < minimum:
                raise Refused(f"{n}: need {minimum} GiB free after stop")


    def running(self, p):
        return bool(self.remote(p["head"], f"docker ps -q --filter name=^/{shlex.quote(p['container'])}$").strip())

    def ready(self, p):
        if "instances" in p:
            return all(self.ready(i) for i in p["instances"]) and (
                bool(p["instances"]) or studio_ready(p.get("studio"), self.api("spark3", 7870, "/api/sparks")))
        return instance_healthy(p, {"apis": {endpoint(p): self.model_api(p)}}) and self.effective(p) == p["cache"]["runtime"]
















def command(cluster, args):
    from profilecli import command as execute
    return execute(cluster, args)


def register(cluster, parent):
    parser = parent.add_parser("profile", help="plan, apply or recover a saved workload profile")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list").set_defaults(fn=lambda a: command(cluster, a))
    p = commands.add_parser("apply")
    p.add_argument("name", choices=NAMES); p.add_argument("--yes", action="store_true")
    p.add_argument("--operation-key")
    p.add_argument("--wait", type=readiness_seconds, default=0, metavar="SECONDS", help="opt-in foreground wait, 0..280 seconds (default 0); readiness continues automatically")
    p.set_defaults(no_restart=False)
    p.set_defaults(fn=lambda a: command(cluster, a))
    p = commands.add_parser("status")
    p.add_argument("--operation", help="read one saved operation without probing the cluster")
    p.add_argument("--assets", choices=NAMES, help="check the profile asset contract without changing placement")
    p.add_argument("--checked-after", action="append", default=[], help=argparse.SUPPRESS)
    p.add_argument("--instance"); p.add_argument("--ready", choices=NAMES); p.add_argument("--studio", choices=("ready", "stopped", "head", "services")); p.add_argument("--worker", action="append", default=[])
    p.set_defaults(fn=lambda a: command(cluster, a))
    for name in ("resume", "rollback", "cancel"):
        p = commands.add_parser(name)
        p.add_argument("receipt"); p.add_argument("--yes", action="store_true")
        p.add_argument("--wait", type=readiness_seconds, default=0, metavar="SECONDS")
        p.add_argument("--continue-worker", action="store_true", help=argparse.SUPPRESS)
        p.set_defaults(fn=lambda a: command(cluster, a))
