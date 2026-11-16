"""Public artifact preparation over the existing disk-only Controller authority."""
from __future__ import annotations

import copy
import base64
import hashlib
import math
from pathlib import Path
import re

from clustercontrol import Controller, Refused, digest
from artifacttransport import Transport, CHUNK
from artifactstorage import RESERVE, direct_route, source_order, nonnegative


def blocker(kind, host, path, reason, machine_id=None):
    return {"kind": kind, "host": host, "path": path, "reason": reason,
            "resource": "artifact:" + machine_id + ":" + path if machine_id and path else "*"}


def node_for(runtime, address):
    found = [alias for alias, item in runtime.c.NODE.items()
             if address in (alias, item.get("hostname"), item.get("lan_ip"))]
    if len(found) != 1:
        raise Refused("host alias is unknown or ambiguous")
    return found[0]


def endpoint(transport, runtime, node):
    found = transport.identity(node)
    if found["hostname"] != runtime.c.NODE[node]["hostname"]:
        raise Refused("observed hostname differs from registered transport binding")
    return found


def path_for(manifest, binding, entry):
    relative = Path(entry["path"])
    home = Path(manifest["home"])
    if relative.is_absolute() or ".." in relative.parts or "\n" in str(relative):
        raise Refused("manifest path is not relative to its declared home")
    model_link = Path(manifest.get("model_link", home / "spark-studio-models"))
    root = Path(manifest.get("model_root", model_link))
    logical = home / relative
    if logical.is_relative_to(model_link):
        root = Path(binding["home"]) / root.relative_to(home) if root.is_relative_to(home) else root
        return str(root / logical.relative_to(model_link))
    return str(Path(binding["home"]) / relative)


def manifest_checks(profile):
    assets = profile.get("assets")
    if not isinstance(assets, dict) or not isinstance(assets.get("files"), dict):
        raise Refused("complete profile asset manifest is missing")
    manifest = assets["files"]
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise Refused("complete profile asset file list is missing")
    failures, checksums = [], []
    for item in manifest["files"]:
        path = item.get("path") if isinstance(item, dict) else None
        full = item.get("sha256") or item.get("hf_etag") if isinstance(item, dict) else None
        kind = item.get("kind") if isinstance(item, dict) else None
        if kind != "file":
            failures.append(blocker("unsupported_artifact_kind", manifest.get("source_host"), path, "only regular-file publication is supported; symlinks are not created"))
        elif not isinstance(full, str) or re.fullmatch(r"[0-9a-f]{64}", full) is None:
            failures.append(blocker("content_identity_unknown", manifest.get("source_host"), path, "full content SHA256 is unknown; size/header facts do not authorize publication"))
        elif not isinstance(path, str) or not path or any("checkpoint" in part.lower() for part in Path(path).parts):
            failures.append(blocker("protected_path", manifest.get("source_host"), path, "checkpoint or invalid paths are protected"))
        else:
            checksums.append((item, full))
    return assets, manifest, checksums, failures


def pending(controller, profile, key, source_identity, failures):
    return [controller.prepare_pending({"profile": profile, "request_key": key}, key, source_identity, failures)]


def _drive(controller, runtime, op, source_identity):
    if op["source_identity"] != source_identity:
        raise Refused("source identity changed; use a new preparation request key")
    if op.get("policy") != "run" or op.get("outcome") == "prepared":
        return op
    controller.block(op["id"], [])
    transport = Transport(runtime)
    for index, item in enumerate(op["spec"]["files"]):
        old = next((effect for effect in op["effects"] if effect["id"] == op["id"] + ":artifact:" + str(index)), None)
        if old and old["state"] == "verified":
            continue
        effect = controller.prepare_issue(op["id"], index)
        try:
            if item['source'].get('kind')=='local_control_source':
                source=item['source'];target=source['target_binding'];binding=target['binding']
                if transport.identity(target['node'])!=binding:raise Refused('control artifact host birth changed')
                body=base64.b64decode(source['content'],validate=True)
                if hashlib.sha256(body).hexdigest()!=item['sha256']:raise Refused('local control source content changed')
                common={'effect_id':effect['id'],'target':item['target'],'sha256':item['sha256'],'usage':item['usage'],
                        'binding':binding,'source_binding':binding}
                evidence=transport.call(target['node'],'stage',**common)
                if evidence.get('state')!='verified':
                    for offset in range(evidence['offset'],len(body),CHUNK):
                        transport.call(target['node'],'append',**common,offset=offset,
                                       chunk=base64.b64encode(body[offset:offset+CHUNK]).decode())
                    evidence=transport.call(target['node'],'publish',**common)
            else:evidence = transport.publish(effect)
            controller.prepare_outcome(op["id"], effect["id"], evidence)
        except (OSError, Refused, ValueError) as error:
            return controller.block(op["id"], [blocker("transfer_unknown", item["source"]["target_binding"]["node"],
                item["target"], str(error), op["spec"]["host_identity"])])
    return controller.view(op["id"])


def resume(controller, runtime, op, source_identity):
    if op["source_identity"] != source_identity:
        raise Refused("source identity changed; use a new preparation request key")
    if op.get("policy") != "run":
        return [op]
    if not op["spec"].get("files"):
        request = op["spec"]["request"]
        return prepare(controller, runtime, request["profile"], request["request_key"], source_identity)
    source = op["spec"]["files"][0]["source"]
    return prepare(controller, runtime, source["profile"], source["request_key"], source_identity)


def prepare(controller: Controller, runtime, profile: dict, request_key: str, source_identity: str) -> list[dict]:
    if not isinstance(request_key, str) or not request_key or not isinstance(source_identity, str) or not source_identity:
        raise Refused("preparation requires an operation key and source identity")
    profile_digest = digest(profile)
    state = controller.store.read()
    prior = state["requests"].get(request_key)
    if prior:
        unresolved = state["operations"][prior["operation_id"]]
        expected = {"request": {"profile": profile, "request_key": request_key}, "files": [], "source_identity": source_identity}
        if prior["digest"] != digest(expected):
            raise Refused("artifact idempotency key or source identity conflicts")
        if unresolved["policy"] != "run":
            return [unresolved]
    existing = [op for op in state["operations"].values() if op.get("intent") == "prepare"
        and op.get("spec", {}).get("files") and isinstance(op["spec"]["files"][0].get("source"), dict)
        and op["spec"]["files"][0]["source"].get("request_key") == request_key]
    for op in existing:
        if op["spec"]["files"][0]["source"].get("profile_digest") != profile_digest or op["source_identity"] != source_identity:
            raise Refused("artifact idempotency key or source identity conflicts")
    if profile.get('recipe_family')=='spark-ds41f':
        return prepare_native(controller,runtime,profile,request_key,source_identity,existing)
    try:
        assets, manifest, checksums, failures = manifest_checks(profile)
    except (Refused, ValueError, KeyError, TypeError) as error:
        return pending(controller, profile, request_key, source_identity, [blocker("content_identity_unknown", None, None, str(error))])
    if failures:
        return pending(controller, profile, request_key, source_identity, failures)
    saved = {op["spec"]["files"][0]["source"].get("placement", op["spec"]["files"][0]["source"]["target_binding"]["node"]): op for op in existing}
    transport = Transport(runtime)
    try:
        targets = [node_for(runtime, node) for node in assets["targets"]]
        if not targets or len(targets) != len(set(targets)):
            raise Refused("target list is missing or ambiguous")
        for entry, _ in checksums:
            ranks = entry.get("targets")
            if ranks is not None and (not isinstance(ranks, list) or not ranks
                    or any(node_for(runtime, rank) not in targets for rank in ranks)):
                raise Refused("per-rank artifact targets must be profile targets")
        targets = [node for node in targets if any(entry.get("targets") is None
                   or node in [node_for(runtime, rank) for rank in entry["targets"]] for entry, _ in checksums)]
        requested = targets + (["library"] if assets.get("storage", {}).get("library") else [])
        if set(saved) == set(requested):
            return [_drive(controller, runtime, saved[node], source_identity) for node in requested]
        policy = assets.get("storage", {})
        reserve = nonnegative(policy.get("reserve_bytes", RESERVE), "reserve")
        image_bytes = nonnegative(policy.get("image_bytes", 0), "image build budget")
        bindings = {}
        def binding_for(alias):
            if alias not in bindings:
                bindings[alias] = endpoint(transport, runtime, alias)
            return bindings[alias]
        library = policy.get("library")
        library_state = {"state": "unconfigured", "reason": "model library UUID or label is not configured"}
        library_source = None
        if library:
            try:
                alias = node_for(runtime, library["node"])
                birth = binding_for(alias)
                root = transport.call(alias, "library", binding=birth, identity=library["identity"])["root"]
                library_source = (alias, birth, root)
                library_state = {"state": "mounted", "node": alias, "root": root, "identity": library["identity"]}
            except (OSError, Refused, ValueError, KeyError, TypeError) as error:
                library_state = {"state": "missing", "reason": str(error)}

        def choose_source(node, entry, sha, target_path, target, target_binding):
            if target.get("usage") == "preexisting_verified":
                if "bytes" in entry and entry["bytes"] != target["size"]:
                    raise Refused("verified target size differs from immutable manifest")
                return {"node": node, "binding": target_binding, "path": target_path, "stat": target["stat"]}
            candidates = []
            for alias in runtime.c.NODE:
                try:
                    birth = binding_for(alias)
                    path = path_for(manifest, birth, entry)
                    located = transport.call(alias, "locate", binding=birth, path=path)
                    if located["present"]:
                        candidates.append({"node": alias, "binding": birth, "path": path, "stat": located["stat"]})
                except (OSError, Refused, ValueError):
                    continue
            if library_source:
                alias, birth, root = library_source
                path = str(Path(root) / "local-ai" / "sha256" / sha)
                located = transport.call(alias, "locate", binding=birth, path=path)
                if located["present"]:
                    candidates.append({"node": alias, "binding": birth, "path": path, "stat": located["stat"], "kind": "library"})
            for candidate in candidates:
                # Reuse only complete same-content, same-host, same-inode observations.
                for operation in state["operations"].values():
                    for effect in operation.get("effects", []):
                        payload = effect.get("payload", {})
                        old_source = payload.get("source", {})
                        transfer = (effect.get("outcome") or {}).get("transfer", {})
                        seconds = transfer.get("seconds")
                        if (effect.get("state") == "verified" and payload.get("sha256") == sha
                                and old_source.get("binding") == candidate["binding"]
                                and old_source.get("path") == candidate["path"]
                                and transfer.get("target_node") == node
                                and old_source.get("target_binding", {}).get("binding") == target_binding
                                and transfer.get("source_stat") == candidate["stat"]
                                and transfer.get("bytes") == candidate["stat"][2]
                                and transfer.get("resumed_bytes") == 0
                                and type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0):
                            candidate["seconds"] = seconds
            for candidate in source_order(runtime.c.NODE, node, candidates):
                try:
                    found = transport.call(candidate["node"], "inspect", binding=candidate["binding"],
                                           path=candidate["path"], sha256=sha, source=True)
                    if found["stat"] != candidate["stat"]:
                        continue
                    if "bytes" in entry and entry["bytes"] != found["size"]:
                        raise Refused("source size differs from immutable manifest")
                    route = direct_route(runtime.c.NODE, candidate["node"], node)
                    route_refusal = None
                    if route:
                        try:
                            transport.call(candidate["node"], "route", binding=candidate["binding"], route=route)
                        except (OSError, Refused, ValueError) as error:
                            route, route_refusal = None, str(error)
                    result = {**candidate, "route": route, "route_refusal": route_refusal}
                    if library_source and Path(candidate["path"]).is_relative_to(library_source[2]):
                        result["source_library"] = {"identity": library["identity"], "root": library_source[2]}
                    return result
                except (OSError, Refused, ValueError):
                    continue
            raise Refused("no verified local, direct peer or declared source; " + library_state["reason"]
                          if "reason" in library_state else "no verified local, direct peer or library source")

        # A configured mounted library receives one durable content-addressed
        # copy. It never changes the paths that a serving engine must load.
        if library_source:
            alias, birth, root = library_source
            if "library" not in saved:
                files, sizes, seen_sha = [], [], set()
                for entry, sha in checksums:
                    if sha in seen_sha:
                        continue
                    seen_sha.add(sha)
                    path = str(Path(root) / "local-ai" / "sha256" / sha)
                    target = transport.call(alias, "inspect", binding=birth, path=path, sha256=sha)
                    source = choose_source(alias, entry, sha, path, target, birth)
                    if target["usage"] != "preexisting_verified":
                        sizes.append(source["stat"][2])
                    files.append({"source": {**source, "placement": "library", "request_key": request_key,
                        "profile_digest": profile_digest, "target_binding": {"node": alias, "binding": birth},
                        "storage": {"reserve": reserve, "image_bytes": 0, "library": library_state,
                        "target_library": {"identity": library["identity"], "root": root}}},
                        "target": path, "sha256": sha, "usage": target["usage"]})
                if sizes:
                    receipt = transport.call(alias, "headroom", binding=birth, path=root, sizes=sizes, reserve=reserve)
                    for item in files:
                        item["source"]["storage"]["headroom"] = [receipt]
                files[0]["source"]["profile"] = copy.deepcopy(profile)
                saved["library"] = controller.prepare_request(birth["machine_id"], files, request_key + ":library", source_identity)
            saved["library"] = _drive(controller, runtime, saved["library"], source_identity)
            if saved["library"].get("outcome") != "prepared":
                return [saved["library"]]

        prepared = []
        seen = {op["spec"]["host_identity"] for key, op in saved.items() if key != "library"}
        for node in targets:
            if node in saved:
                continue
            target_binding = binding_for(node)
            if target_binding["machine_id"] in seen:
                raise Refused("target aliases resolve to the same physical machine")
            seen.add(target_binding["machine_id"])
            files, budgets = [], {}
            for entry, sha in checksums:
                if entry.get("targets") is not None and node not in [node_for(runtime, rank) for rank in entry["targets"]]:
                    continue
                target_path = path_for(manifest, target_binding, entry)
                target = transport.call(node, "inspect", binding=target_binding, path=target_path, sha256=sha)
                source = choose_source(node, entry, sha, target_path, target, target_binding)
                if target["usage"] != "preexisting_verified":
                    disk = transport.call(node, "storage", binding=target_binding, path=target_path)
                    budget = budgets.setdefault(disk["device"], {"path": target_path, "sizes": []})
                    budget["sizes"].append(source["stat"][2])
                files.append({"source": {**source,
                    "target_binding": {"node": node, "binding": target_binding}, "request_key": request_key,
                    "profile_digest": profile_digest, "storage": {"reserve": reserve, "image_bytes": 0,
                    "library": library_state}}, "target": target_path, "sha256": sha, "usage": target["usage"]})
            if image_bytes:
                image_path = policy.get("image_path", "/var/lib/docker")
                disk = transport.call(node, "storage", binding=target_binding, path=image_path)
                budgets.setdefault(disk["device"], {"path": image_path, "sizes": []})["image_bytes"] = image_bytes
            for budget in budgets.values():
                receipt = transport.call(node, "headroom", binding=target_binding, **budget,
                                         reserve=reserve)
                for item in files:
                    item["source"]["storage"].setdefault("headroom", []).append(receipt)
            files[0]["source"]["profile"] = copy.deepcopy(profile)
            prepared.append((node, target_binding, files))
    except (OSError, Refused, ValueError, KeyError, TypeError) as error:
        return ([saved["library"]] if "library" in saved else []) + pending(controller, profile, request_key, source_identity, [blocker("identity_or_content_unknown", None, None, str(error))])
    # Reserve every target before the first publication. No GPU request, resolve,
    # lease or Studio observation is part of this path.
    try:
        for node, binding, files in prepared:
            saved[node] = controller.prepare_request(binding["machine_id"], files, request_key + ":" + node, source_identity)
    except Refused as error:
        return pending(controller, profile, request_key, source_identity, [blocker("target_reserved", None, None, str(error))])
    return ([saved["library"]] if "library" in saved else []) + [_drive(controller, runtime, saved[node], source_identity) for node in targets]


def prepare_native(controller,runtime,profile,key,source_identity,existing):
    """Publish the exact owned control source; model/image facts remain separate."""
    from profile_recipes.spark_ds41f_1030 import control_artifact
    recipe=profile['recipe'];artifact=control_artifact(recipe);transport=Transport(runtime)
    node=artifact['member']
    try:
        if existing:
            rows=[_drive(controller,runtime,op,source_identity) for op in existing]
        else:
            binding=endpoint(transport,runtime,node)
            if binding['home']!=recipe['spec']['site']['home']:
                raise Refused('native target home differs from source-exact site binding')
            files=[]
            for relative,expected in sorted(artifact['identity']['files'].items()):
                path=(Path(__file__).parent/relative.removeprefix('lib/')) if relative.startswith('lib/') else Path(recipe['package'])/relative.removeprefix('source/')
                body=path.read_bytes()
                if hashlib.sha256(body).hexdigest()!=expected:raise Refused('native control artifact source differs from lock')
                target=str(Path(artifact['path'])/relative)
                observed=transport.call(node,'inspect',binding=binding,path=target,sha256=expected)
                files.append({'source':{'kind':'local_control_source','content':base64.b64encode(body).decode(),
                    'binding':binding,'target_binding':{'node':node,'binding':binding},'request_key':key,'profile_digest':digest(profile)},
                    'target':target,'sha256':expected,'usage':observed['usage']})
            files[0]['source']['profile']=copy.deepcopy(profile)
            op=controller.prepare_request(binding['machine_id'],files,key+':control-source',source_identity)
            rows=[_drive(controller,runtime,op,source_identity)]
    except (OSError,Refused,ValueError,KeyError,TypeError) as error:
        return pending(controller,profile,key,source_identity,[blocker('identity_or_content_unknown',node,artifact['path'],str(error))])
    # The recipe pins revisions and launch inputs, not full model-file content
    # identities or present immutable image objects. Copying code proves neither.
    unresolved=pending(controller,profile,key+':runtime-artifacts',source_identity,
        [blocker('content_identity_unknown',member,None,'native model full-content/image availability requires separate observed provenance; control source publication does not prepare model bytes')
         for member in recipe['members']])
    return rows+unresolved
