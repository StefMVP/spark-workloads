"""Pinned 1030 recipe data and previews. This module grants no execution rights."""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import ipaddress
import json
import math
from pathlib import Path
import re
import tarfile

DEPLOYMENT = "9461a70546fcfdcac9c7d8fbc58a53230510fbc9"
NATIVE = "b6de07aad5cb21f2cf8381036c1a9eaf471b07fc"
NATIVE_TREE = "ea1281f880e7535e3d87725be0156a8e38fc2fb2"
FAMILY = "spark-ds41f"
LOCK_SHA256 = "ced669a340a816a8ac25523ac99a5281459391599451789c677297de4e2bd973"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _object(kind, data):
    return hashlib.sha1(kind.encode() + b" " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("source path is missing or outside the package")
    return path


def _entries(raw):
    result = {}
    while raw:
        header, raw = raw.split(b"\0", 1)
        mode, name = header.split(b" ", 1)
        result[name.decode()] = (mode.decode(), raw[:20].hex())
        raw = raw[20:]
    return result


def verify_sources(root: Path) -> dict:
    """Verify an extracted data package. Never infer identity from a parent Git cwd."""
    root = Path(root)
    lock_bytes = _file(root, "source.lock.json").read_bytes()
    if _sha(lock_bytes) != LOCK_SHA256:
        raise ValueError("source lock differs from approved 1030 lock")
    lock = json.loads(lock_bytes)
    for relative, expected in lock["files"].items():
        if _sha(_file(root, relative).read_bytes()) != expected:
            raise ValueError("source drift: " + relative)
    witness = json.loads((root / "native-witness.json").read_text())
    raw = base64.b64decode(witness["raw_commit"], validate=True)
    if _object("commit", raw) != NATIVE or raw.splitlines()[0] != ("tree " + NATIVE_TREE).encode():
        raise ValueError("native commit/tree witness differs")
    trees = {oid: base64.b64decode(data, validate=True) for oid, data in witness["trees"].items()}
    if any(_object("tree", raw) != oid for oid, raw in trees.items()):
        raise ValueError("native tree object hash differs")
    for path, oid in witness["blobs"].items():
        parent = NATIVE_TREE
        for name in path.split("/"):
            _, parent = _entries(trees[parent])[name]
        data = _file(root / "native", path).read_bytes()
        if parent != oid or _object("blob", data) != oid:
            raise ValueError("native file is outside the pinned tree: " + path)
    return lock


def verify_native_archive(package: Path, archive: Path) -> dict:
    """Hash ALL native archive files and recompute its Git tree, without Git or extraction."""
    lock = verify_sources(package)
    data = Path(archive).read_bytes()
    expected = lock["native"]["archive"]["sha256"]
    if _sha(data) != expected:
        raise ValueError("native archive digest differs")
    nodes = {}
    prefix = None
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = Path(member.name).parts
            if not parts or any(p in ("..", "") for p in parts) or member.name.startswith("/"):
                raise ValueError("unsafe native archive path")
            prefix = prefix or parts[0]
            if parts[0] != prefix:
                raise ValueError("native archive has multiple roots")
            if len(parts) == 1 or member.isdir():
                continue
            path = "/".join(parts[1:])
            if path in nodes:
                raise ValueError("duplicate archive file")
            if member.issym():
                mode, blob = "120000", member.linkname.encode()
            elif member.isfile():
                mode = "100755" if member.mode & 0o111 else "100644"
                blob = source.extractfile(member).read()
            else:
                raise ValueError("unsupported native archive record")
            nodes[path] = (mode, _object("blob", blob))
    def tree(prefix=""):
        entries = {}
        for path, value in nodes.items():
            if not path.startswith(prefix):
                continue
            tail = path[len(prefix):]
            name, _, rest = tail.partition("/")
            entries[name] = ("40000", None) if rest else value
        raw = b""
        for name, (mode, oid) in sorted(entries.items(), key=lambda item: (item[0] + ("/" if item[1][0] == "40000" else "")).encode()):
            oid = oid or tree(prefix + name + "/")
            raw += mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(oid)
        return _object("tree", raw)
    actual = tree()
    if actual != NATIVE_TREE:
        raise ValueError("native archive tree differs: " + actual)
    return {"commit": NATIVE, "tree": actual, "archive_sha256": _sha(data), "files": len(nodes)}


def verify_patchset(package: Path, deployment_root: Path) -> dict:
    """Recompute the upstream digest from actual series/patch bytes in an explicit root."""
    verify_sources(package)
    manifest = json.loads((package / "patchset.json").read_text())
    series = _file(deployment_root, manifest["series_path"]).read_bytes()
    if _sha(series) != manifest["series_sha256"]:
        raise ValueError("patch series differs or wrong deployment root")
    result = hashlib.sha256(series)
    for entry in manifest["patches"]:
        data = _file(deployment_root, entry["path"]).read_bytes()
        if _sha(data) != entry["sha256"]:
            raise ValueError("patch bytes differ")
        result.update(entry["name"].encode() + b"\0" + data)
    if result.hexdigest() != manifest["patchset_sha256"]:
        raise ValueError("patchset differs")
    return {"sha256": result.hexdigest(), "patches": len(manifest["patches"])}


def _pure_module(path, name, package=False):
    if package:
        import sys
        prior = sys.modules.get(name)
        if prior is not None and Path(prior.__file__).resolve() == path.resolve():
            return prior
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    if package:
        import sys
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load(root: Path, document: dict) -> dict:
    """Normalize a locked recipe spec; aliases name placement, never GPU identity."""
    package = Path(root) / "recipes/spark-ds41f-1030"
    if not package.is_dir():
        package = Path(root) / "profiles/recipes/spark-ds41f-1030"
    lock = verify_sources(package)
    spec_path = package / "spec.json"
    frozen = json.loads(spec_path.read_text())
    if document != frozen:
        raise ValueError("recipe document differs from pinned site spec")
    members = document["members"]
    if len(members) != 4 or len(set(members)) != 4:
        raise ValueError("TP4 needs four distinct placement aliases")
    model = json.loads((package / "model-files.json").read_text())
    artifacts = []
    for member in members:
        artifacts.extend([
            {"member": member, "path": document["site"]["home"] + "/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" + model["revision"], "kind": "model", "identity": {"revision": model["revision"], "manifest_sha256": lock["files"]["model-files.json"]}, "assurance": "unknown"},
            {"member": member, "path": "image:spark-ds41f:1030", "kind": "image", "identity": lock["image"], "assurance": "unknown"},
            {"member": member, "path": document["site"]["checkout"] + "/cache", "kind": "derived-cache", "identity": {"source": DEPLOYMENT, "contents": "requires native image preparation"}, "assurance": "unknown"},
            {"member": member, "path": document["site"]["home"] + "/.cache/huggingface/hub/models--msuiche--DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75/snapshots/2715ffd51f59899115853e1118db080536966c44/glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf", "kind": "steering-vector", "identity": {"sha256": "9c66d47701e0ceb42c33906f2d3789c0f1f16b21e7d97185f20f7f3f09fbf5cf"}, "assurance": "unknown"},
        ])
    normalized = {"family": FAMILY, "engine": "vllm", "source": {"url": lock["deployment"]["url"], "commit": DEPLOYMENT, "files": lock["files"]}, "members": list(members), "artifacts": artifacts, "native_coverage": {
        "ingress": {"protocol": "spark-ds41f-control-v1", "implementation": "supported", "source_digest": _sha(Path(__file__).with_name("spark_ds41f_control.py").read_bytes())},
        "members": {"plan_contract": "materialize-member-v1", "implementation": "supported", "source_digest": _sha(Path(__file__).read_bytes())},
        "shared_dispatcher": "pending exact-plan/journal/guard verification",
        "reason": "implementation descriptors do not establish installed image, physical identity or current runtime readiness"}, "spec": copy.deepcopy(document), "package": str(package.resolve()), "site_digest": digest(document["site"])}
    normalized["artifacts"].append(control_artifact(normalized))
    return normalized


def render(recipe: dict, action: str) -> tuple[dict, ...]:
    if action not in ("start", "stop"):
        raise ValueError("unknown recipe action")
    package = Path(recipe["package"])
    lock = verify_sources(package)
    spec = recipe["spec"]
    if spec != json.loads((package / "spec.json").read_text()) or recipe["members"] != spec["members"] or digest(spec["site"]) != recipe["site_digest"]:
        raise ValueError("recipe placement changed")
    cluster = json.loads((package / "resolved.json").read_text())
    fabric = _pure_module(package / "sparknet/sparknet/topology/__init__.py", "_ds41f_1030_sparknet", True)
    nodes = copy.deepcopy(json.loads((package / "deployment/config/examples/nodes-ring4.json").read_text())["nodes"])
    results = []
    for rank in ([1, 2, 3, 0] if action == "start" else [0, 3, 2, 1]):
        node = nodes[rank]
        node["management_ip"] = str(ipaddress.ip_address(spec["site"]["management_ips"][rank]))
        env = fabric.render.node_environment({"nodes": [node]}, node, transport="oneshot-ring4", base=cluster["environment"])
        env["VLLM_HOST_IP"] = node["management_ip"]
        args = ["docker", "run", "-d", "--name", cluster["container"]["name"], *cluster["container"]["docker_run_args"]]
        for key, value in sorted(env.items()):
            args += ["--env", key + "=" + value]
        for source, destination, mode in cluster["container"]["mounts"]:
            args += ["--volume", source.format(**spec["site"]) + ":" + destination + ":" + mode]
        args += [cluster["container"]["image"], *cluster["serve_args"], "--node-rank", str(rank), "--master-addr", spec["site"]["management_ips"][0], "--master-port", cluster["distributed"]["master_port"]]
        if rank:
            args += ["--headless"]
        if action == "stop":
            args = ["docker", "stop", "--time", "60", "<exact-container-id-required>"]
        item = {"member": spec["members"][rank], "rank": rank, "argv": args, "source_path": "deployment/bin/spark", "source_digest": lock["files"]["deployment/bin/spark"], "order": len(results), "preview_only": True}
        item["command_digest"] = digest(item["argv"])
        results.append(item)
    return tuple(results)


def rank_plan(recipe: dict, action: str, resources: list[str]) -> tuple[dict, ...]:
    """Typed preview only. Host dispatch must recompute this and bind op/gen/birth."""
    if len(resources) != 4 or any(not isinstance(r, str) or re.fullmatch(r"gpu:GPU-[A-Za-z0-9-]+", r) is None for r in resources) or resources != sorted(set(resources)):
        raise ValueError("full canonical four-GPU footprint required")
    result = []
    for launch in render(recipe, action):
        kinds = ["create", "guard-start", "start", "guard-start"] if action == "start" else ["stop", "guard-stop"]
        for index, kind in enumerate(kinds):
            item = {**launch, "action": kind, "resources": list(resources), "order": len(result), "execution": "unsupported"}
            if kind == "create":
                item["argv"] = ["docker", "create", *launch["argv"][3:]]
                item["requires"] = ["verified immutable image identity", "created immutable container ID"]
            elif kind == "start":
                item["argv"] = ["docker", "start", "<exact-container-id-required>"]
                item["requires"] = ["created immutable container ID", "startup guard exact identity"]
            elif kind.startswith("guard"):
                phase = "startup" if index == 1 and action == "start" else "steady"
                item["argv"] = []
                item["guard"] = {"phase": phase, "threshold_gib": 5 if phase == "startup" else 3, "interval_seconds": 0.25 if phase == "startup" else 2, "wait_seconds": 120 if phase == "startup" else 0}
                item["requires"] = ["immutable container/process birth guard; legacy name-based script forbidden"]
            else:
                item["requires"] = ["sealed native ingress", "all-rank ownership", "exact existing physical birth"]
            item["command_digest"] = digest(item["argv"])
            result.append(item)
    return tuple(result)


def preflight(recipe: dict, facts: dict) -> dict:
    """Pure observations: missing evidence is Unknown and blocks activation."""
    blockers = []
    for rank, member in enumerate(recipe["members"]):
        f = facts.get(member, {})
        expected = {"gpu_uuid": None, "platform": "NVIDIA DGX Spark", "page_size": 65536, "memory_saver": True, "kernel": "7.0.0-1019-nvidia-64k", "qualified_platform": True, "nvidia_runtime": True, "ring_rank": rank, "ring_verified": True, "image_verified": True, "artifacts_verified": True, "native_control_verified": True, "guard_verified": True}
        for key, wanted in expected.items():
            value = f.get(key)
            if value is None:
                blockers.append(member + ": unknown " + key)
            elif (key == "gpu_uuid" and (not isinstance(value, str) or not value.startswith("GPU-"))) or (key != "gpu_uuid" and (type(value) is not type(wanted) or value != wanted)):
                blockers.append(member + ": invalid " + key)
        os_version = f.get("dgx_os")
        try:
            if not isinstance(os_version, str) or tuple(int(p) for p in os_version.split(".")) < (26, 9, 2):
                raise ValueError()
        except ValueError:
            blockers.append(member + ": unknown or unsupported dgx_os")
        for key, minimum in (("free_disk_bytes", 480 * 1024**3), ("available_memory_gib", 100)):
            if type(f.get(key)) not in (int, float) or not math.isfinite(f[key]) or f[key] < minimum:
                blockers.append(member + ": missing or insufficient " + key)
    uuids = [facts.get(m, {}).get("gpu_uuid") for m in recipe["members"]]
    if len(set(uuids)) != 4:
        blockers.append("GPU identity is missing or aliased")
    return {"ready": not blockers, "blockers": blockers, "activation_queued": False, "authorizes_execution": False}


def benchmark_summary(package: Path) -> dict:
    """Audit published data. Batch intervals are not independent benchmark runs."""
    import statistics
    verify_sources(package)
    root = package / "benchmarks/upstream/manifests/suites"
    output = {}
    for concurrency in (1, 8, 16):
        folder = "2026-10-30-spark-ds41f-1030-tp4" + ("-single-stream" if concurrency == 1 else "")
        record = json.loads((root / folder / ("c" + str(concurrency) + ".json")).read_text())
        workload, summary = record["workload"], record["summary"]
        counts, throughput = summary["requests"], summary["output_tps"]
        if (workload["duration_s"] != 300 or workload["warmup_s"] != 60 or workload["batches"] != 10 or throughput["n"] != 10
                or len(throughput["batches"]) != 10 or any(counts[k] for k in ("failed", "rejected", "foreign", "approximate_token_times"))):
            raise ValueError("published benchmark has errors or incompatible measurement bounds")
        table = record["request_table"]
        columns = table["columns"]
        rows = [dict(zip(columns, row, strict=True)) for row in table["rows"]]
        if len(rows) != counts["sent"] or any(row["error"] is not None or not row["completion_tokens"] for row in rows):
            raise ValueError("empty or failed requests in benchmark")
        output[str(concurrency)] = {"mean_aggregate_tps": throughput["mean"], "median_aggregate_tps": statistics.median(throughput["batches"]), "ci95_half_width": throughput["ci95"], "batches": 10, "independent_measured_windows": 1, "warmup_seconds": 60, "measurement_seconds": 300, "sent": counts["sent"], "failed": 0, "rejected": 0, "tokens_per_request": 256, "raw_label": record["label"], "source_file": str((root / folder / ("c" + str(concurrency) + ".json")).relative_to(package))}
    prefill = json.loads((root / "2026-10-30-spark-ds41f-1030-tp4/prefill.json").read_text())
    if any(row["failed"] or row["tps"]["n"] != 3 for row in prefill["prefill"]):
        raise ValueError("prefill has errors or missing repetitions")
    return {"decode": output, "prefill": [{"size_label": row["size"], "actual_prompt_tokens": row["prompt_tokens"], "mean_tps": row["tps"]["mean"], "repetitions": row["tps"]["n"]} for row in prefill["prefill"]], "provenance": "candidate12 labeled raw files adopted by the published 1030 candidate14 reference; no new candidate14 measurement inferred", "image_medians": "same ten batch samples as means; not separate independent runs", "mia_matched": False, "winner": None}



def materialize_member_plan(recipe, action, member, footprint, *, creation_key=None, image_id=None, container=None, control=None, image_environment=None, phase="steady"):
    """Concrete typed plans. Shared HostJournal must reobserve and authorize them."""
    import re
    from .spark_ds41f_control import birth, resources
    footprint = resources(footprint)
    if action not in ("create", "start", "stop", "guard-start", "guard-stop") or member not in recipe["members"]:
        raise ValueError("unsupported family member action")
    rank = recipe["members"].index(member)
    launch = next(p for p in render(recipe, "start") if p["rank"] == rank)
    image_pattern = r"sha256:[0-9a-f]{64}"
    if action == "create":
        if (not isinstance(image_id, str) or re.fullmatch(image_pattern, image_id) is None
                or not isinstance(creation_key, str) or re.fullmatch(r"[0-9a-f]{64}", creation_key) is None or container is not None):
            raise ValueError("create requires immutable image and creation identity")
        binding = {"image_id": image_id, "creation_key": creation_key,
                   "name": "spark-ds41f-" + creation_key + "-r" + str(rank),
                   "labels": {"local.profile.family": FAMILY, "local.profile.creation": creation_key,
                              "local.profile.plan": digest({"source": DEPLOYMENT, "resources": footprint, "rank": rank, "site": recipe["site_digest"], "image_id": image_id, "creation_key": creation_key}), "local.profile.rank": str(rank)}}
        argv = ["docker", "create", *launch["argv"][3:]]
        argv[argv.index("--name") + 1] = binding["name"]
        reference_image = json.loads((Path(recipe["package"]) / "resolved.json").read_text())["container"]["image"]
        image_at = argv.index(reference_image)
        argv[image_at] = image_id
        additions = []
        for key, value in sorted(binding["labels"].items()): additions += ["--label", key + "=" + value]
        if rank == 0:
            if (not isinstance(control, dict) or set(control) not in ({"config_path"}, {"config_path", "state_dir"}) or not isinstance(control["config_path"], str)
                    or not Path(control["config_path"]).is_absolute() or ".." in Path(control["config_path"]).parts
                    or not isinstance(image_environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in image_environment.items())):
                raise ValueError("head create requires prepared control config and observed image environment")
            if "state_dir" in control and (not isinstance(control["state_dir"], str) or not Path(control["state_dir"]).is_absolute() or ".." in Path(control["state_dir"]).parts or control["state_dir"] == control["config_path"]):
                raise ValueError("controller-owned state directory must be canonical and separate from config")
            # Control overlay is separate from the frozen serving recipe. Preserve
            # the image's existing PYTHONPATH; never guess its installed package path.
            artifact = control_artifact(recipe)
            wrapper_identity = artifact["identity"]["digest"]
            control_root = artifact["path"]
            previous = image_environment.get("PYTHONPATH", "")
            for at, argument in enumerate(argv[:-1]):
                if argument == "--env" and argv[at + 1].startswith("PYTHONPATH="):
                    previous = argv[at + 1].partition("=")[2]
            additions += ["--env", "PYTHONPATH=/opt/profile-control/lib" + (":" + previous if previous else ""),
                          "--env", "SPARK_DS41F_CONTROL_CONFIG=/run/profile-control/config.json",
                          "--volume", control_root + "/lib/profile_recipes:/opt/profile-control/lib/profile_recipes:ro",
                          "--volume", control_root + "/source:/opt/profile-control/source:ro",
                          "--volume", control["config_path"] + ":/run/profile-control/config.json:ro"]
            if "state_dir" in control:
                additions += ["--volume", control["state_dir"] + ":/run/profile-control/state:rw"]
            argv += ["--middleware", "profile_recipes.spark_ds41f_control.NativeIngress"]
            binding["control_overlay"] = {"root": control_root, "identity": wrapper_identity, "config_path": control["config_path"], "preserved_pythonpath": previous}
            if "state_dir" in control:
                binding["control_overlay"]["state_dir"] = control["state_dir"]
        argv[image_at:image_at] = additions
    else:
        keys = {"id", "image_id", "created", "started", "process_birth"}
        if (not isinstance(container, dict) or set(container) != keys or not isinstance(container["id"], str) or re.fullmatch(r"[0-9a-f]{64}", container["id"]) is None
                or not isinstance(container["image_id"], str) or re.fullmatch(image_pattern, container["image_id"]) is None or not isinstance(container["created"], str) or not container["created"]):
            raise ValueError("immutable container object proof required")
        binding = copy.deepcopy(container)
        argv = []
        if action == "start":
            if container["started"] is not None or container["process_birth"] is not None:
                raise ValueError("start requires a created, never-started object; reconcile uncertain start instead")
            argv = ["docker", "start", container["id"]]
        elif not (action in ("guard-start", "guard-stop") and phase == "startup" and container["started"] is None and container["process_birth"] is None):
            physical = birth(container["process_birth"])
            if not isinstance(container["started"], str) or not container["started"]:
                raise ValueError("running container incarnation required")
            argv = ["pidfd_send_signal", physical["identity"], physical["boot_id"], physical["started"], "SIGTERM"] if action == "stop" else []
    result = {"family": FAMILY, "action": action, "member": member, "rank": rank, "resources": footprint,
              "argv": argv, "command_digest": digest(argv), "source_path": "tools/lib/profile_recipes/spark_ds41f_1030.py",
              "source_digest": _sha(Path(__file__).read_bytes()), "binding": binding, "preview_only": True}
    if action.startswith("guard"):
        if phase not in ("startup", "steady"):
            raise ValueError("unsupported memory guard phase")
        result["guard"] = {"phase": phase, "threshold_kib": (5 if phase == "startup" else 3) * 1024**2,
                           "interval_seconds": 0.25 if phase == "startup" else 2, "startup_wait_seconds": 120 if phase == "startup" else 0,
                           "birth": container["process_birth"], "container_id": container["id"], "container_started": container["started"],
                           "on_unknown_memory": "kill-exact-birth", "signal": "SIGKILL"}
    return result


def guard_decision(plan, sample):
    """One memory-guard observation. Caller revalidates its permit and uses pidfd."""
    if plan.get("family") != FAMILY or plan.get("action") != "guard-start" or not isinstance(plan.get("guard"), dict):
        raise ValueError("typed guard-start plan required")
    policy = plan["guard"]
    if (policy.get("phase") not in ("startup", "steady") or policy.get("threshold_kib") != (5 if policy["phase"] == "startup" else 3) * 1024**2
            or policy.get("interval_seconds") != (0.25 if policy["phase"] == "startup" else 2)
            or policy.get("birth") != plan["binding"]["process_birth"] or policy.get("container_id") != plan["binding"]["id"] or policy.get("container_started") != plan["binding"]["started"]):
        raise ValueError("memory guard policy differs from source/binding")
    if policy["birth"] is None:
        if sample.get("container_id") != policy["container_id"] or sample.get("container_started") is not None or sample.get("process_birth") is not None:
            return {"state": "retire", "signal_plan": None, "reason": "created object changed; require journaled start outcome"}
        available = sample.get("available_kib")
        return {"state": "await-start-outcome" if type(available) is int and available >= policy["threshold_kib"] else "refuse-start",
                "signal_plan": None, "reason": "bind only the journaled start outcome; never adopt a discovered PID"}
    if (sample.get("process_birth") != policy["birth"] or sample.get("container_id") != policy["container_id"]
            or sample.get("container_started") != policy["container_started"]):
        return {"state": "retire", "signal_plan": None, "reason": "physical incarnation changed; never adopt replacement"}
    available = sample.get("available_kib")
    kill = type(available) is not int or available < 0 or available < policy["threshold_kib"]
    physical = policy["birth"]
    return {"state": "kill" if kill else "watch", "signal_plan": {"action": "process_stop", "birth": physical,
            "argv": ["pidfd_send_signal", physical["identity"], physical["boot_id"], physical["started"], "SIGKILL"]} if kill else None,
            "reason": "memory observation missing/below fixed source threshold" if kill else "memory above source threshold"}


def control_artifact(recipe):
    """Source bundle reservation, separate from GPU leases; this performs no copy."""
    package = Path(recipe["package"])
    lock = verify_sources(package)
    files = {"source/" + path: value for path, value in lock["files"].items()}
    files["source/source.lock.json"] = LOCK_SHA256
    files["lib/profile_recipes/spark_ds41f_1030.py"] = _sha(Path(__file__).read_bytes())
    files["lib/profile_recipes/spark_ds41f_control.py"] = _sha(Path(__file__).with_name("spark_ds41f_control.py").read_bytes())
    identity = digest(files)
    return {"member": recipe["members"][0], "kind": "control-source", "path": str(Path(recipe["spec"]["site"]["home"]) / ".local/share/profile-control/spark-ds41f" / identity), "identity": {"digest": identity, "files": files}, "assurance": "unknown"}
