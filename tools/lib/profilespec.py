"""Validated profile and recipe specifications. No runtime or host effects."""
from pathlib import Path
import json
import re
import yaml
from clustercontrol import Refused, resources

ROOT = Path(__file__).resolve().parents[2]
NAMES = tuple(path.stem for path in sorted((ROOT / "profiles").glob("*.yaml"))
              if any(key in (yaml.safe_load(path.read_text()) or {}) for key in ("switch", "control", "native")))

def load(root, name):
    if name not in NAMES:
        raise Refused(f"unknown profile {name}; choose {', '.join(NAMES)}")
    document = yaml.safe_load((root / "profiles" / f"{name}.yaml").read_text())
    if "native" in document:
        from nativeinstalled import load_spec
        return load_spec(root, name, document["native"])
    if "control" in document:
        return load_control(root, name, document["control"])
    p = document["switch"]
    p = dict(p, name=name)
    if name == "deepseek-1017-mia":
        from nativeinstalled import load_spec
        native = load_spec(root, name, "mia")
        p["installed_native"] = native["installed_native"]
    if p.get("assets"):
        manifest = p["assets"]["manifest"]
        if Path(manifest).name != manifest:
            raise Refused("asset manifest must be a profile asset filename")
        p["assets"]["files"] = json.loads((root / "profiles/assets" / manifest).read_text())
    items = {x["id"]: x for x in yaml.safe_load((root / "queue/queue.yaml").read_text())["items"]}
    seen_nodes, seen_ids, seen_endpoints = set(), set(), set()
    from sparkinventory import load as inventory_load
    inventory = inventory_load()["nodes"] if p["instances"] else []
    for i in p["instances"]:
        item = items[i["item"]]
        defaults = item["recipe"]["env"]
        overrides = i.get("env", {})
        allowed = {"WORKER", "MODEL_DIR", "WORKER_MODEL_DIR", "DFLASH2_DIR", "WORKER_DFLASH2_DIR", "PORT", "HOST", "MEMORY_RESERVE_GIB",
                   "FABRIC_PEER", "NFS_SERVER", "NFS_VOLUME", "HEAD_ENGRAM_DIR", "STATE_DIR", "HF_CACHE"}
        if set(overrides) - allowed:
            raise Refused("instance overrides must be host, path, port or memory settings")
        i["profile_name"] = name
        i["env"] = dict(defaults, **overrides)
        i["start"], i["stop"] = item["recipe"]["start"], item["recipe"]["stop"]
        i["model"], i["port"] = i["env"]["SERVED_NAME"], int(i["env"]["PORT"])
        head = next(n for n in inventory if n["alias"] == i["head"])
        if i["endpoint"] != f"http://{head['lan_ip']}:{i['port']}/v1":
            raise Refused("instance endpoint differs from head and port")
        for rank, member in enumerate(i["members"][1:], 1):
            worker = next(n for n in inventory if n["alias"] == member)
            targets = {worker["alias"], worker["hostname"], worker["lan_ip"], *head.get("ring_ips", {}).get(member, [])}
            key = "WORKER" if rank == 1 else f"WORKER{rank}"
            if i["env"].get(key, "").split("@")[-1] not in targets:
                raise Refused("instance worker differs from members")
        if i["head"] != i["members"][0] or len(i["members"]) != len(item["serve_on"]):
            raise Refused("instance head or rank count differs from queue")
        if seen_nodes.intersection(i["members"]) or i["id"] in seen_ids or i["endpoint"] in seen_endpoints:
            raise Refused("profile instances overlap")
        seen_nodes.update(i["members"]); seen_ids.add(i["id"]); seen_endpoints.add(i["endpoint"])
        if i["members"] not in p["groups"].values():
            raise Refused("instance members must match a profile group")
        if not i["recipe"].endswith("@" + item["recipe"]["commit"][:8]):
            raise Refused("profile recipe differs from queue")
        if item["recipe"].get("run", {}).get("workdir", i["recipe"]) != i["recipe"]:
            raise Refused("profile recipe differs from queue run.workdir")
        for path in [i["recipe"], *(v for values in i["paths"].values() for v in values)]:
            if not path.startswith("/") or "\n" in path:
                raise Refused("profile paths must be absolute")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", k) for k in i["env"]):
            raise Refused("invalid recipe environment key")
        i["cache"] = json.loads((root / "profiles/cache" / i["cache"]).read_text())
        if defaults != i["cache"]["recipe_env"] or overrides != i["cache"].get("overrides", {}):
            raise Refused("queue environment or overrides changed from verified profile; re-verify before switching")
        if set(i["cache"]["nodes"]) != set(i["members"]):
            raise Refused("cache manifest must cover every instance member")
    reserved = set()
    for group, lease in p.get("reservations", {}).items():
        if group not in p["groups"] or set(lease) != {"by", "for"} or not all(isinstance(v, str) and v for v in lease.values()):
            raise Refused("reservation requires a profile group, owner and purpose")
        reserved.update(p["groups"][group])
    flags = p.get("studio") or {}
    studio_nodes = {"spark3"} | {n for n, enabled in flags.get("workers", {}).items() if enabled} if flags else set()
    if reserved & (seen_nodes | studio_nodes):
        raise Refused("reserved nodes overlap Studio or model instances")
    if p.get("lab"):
        from labprofile import validate
        validate(p)
    elif not p["instances"]:
        if (p["default"] is not None or set(flags.get("workers", {})) != {"spark1", "spark2", "spark4"}
                or not all(type(v) is bool for v in flags["workers"].values())
                or flags.get("default") != len(studio_nodes)
                or studio_nodes | reserved != {"spark1", "spark2", "spark3", "spark4"}):
            raise Refused("an empty model profile requires Studio placement and reservations for all four nodes")
    elif p["default"] not in seen_ids:
        raise Refused("profile default instance missing")
    return p


def load_control(root, name, document):
    if not isinstance(document, dict) or document.get("recipe_family") != "spark-ds41f":
        raise Refused("unsupported recipe family")
    resources(document.get("groups"))
    reference = document.get("recipe_spec")
    if not isinstance(reference, str) or Path(reference).is_absolute() or ".." in Path(reference).parts:
        raise Refused("recipe specification must be a relative profile path")
    path = root / "profiles" / reference
    if not path.is_file():
        raise Refused("recipe specification is missing")
    from profile_recipes import spark_ds41f_1030, mia_tp4
    load_recipe = mia_tp4.load if reference == 'recipes/mia-tp4-control/spec.json' else spark_ds41f_1030.load
    recipe = load_recipe(root, json.loads(path.read_text()))
    if recipe.get("family") != "spark-ds41f" or recipe.get("engine") not in ("vllm", "tensorfold-mia-tp4"):
        raise Refused("recipe family/engine differs from leaf specification")
    members = {node for group in document["groups"].values() for node in group}
    if set(recipe.get("members", [])) != members:
        raise Refused("recipe ranks differ from target placement")
    return dict(document, name=name, instances=[], recipe=recipe, default=None)
