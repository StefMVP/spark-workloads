"""Public profile commands backed by the single controller aggregate."""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
import time
import uuid

from clustercontrol import Controller, Refused, Store, digest, durable, resources, public_projection
from profileadapter import StudioAdapter
from profileoperation import Driver


def public_json(value,**options):
    return json.dumps(public_projection(value),**options)


def source_identity(root):
    paths = [root / "tools/cluster", *sorted((root / "tools/lib").rglob("*.py")),
             *sorted((root / "profiles/recipes").rglob("source.lock.json")),
             *sorted((root / "profiles/recipes").rglob("spec.json")),
             *sorted((root / "profiles/recipes").rglob("resolved.json"))]
    paths += sorted((root / 'profiles/native-installed').glob('*.json'))
    for lock in sorted((root/'profiles/recipes').rglob('source.lock.json')):
        paths += [lock.parent/relative for relative in json.loads(lock.read_text()).get('files',{})]
    return digest({str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def reference(controller, value):
    state = controller.store.read()
    if value in state["operations"]:
        return value
    path = Path(value)
    if path.is_file():
        projection = json.loads(path.read_text())
        identifier = projection.get("operation_id")
        if identifier in state["operations"]:
            return identifier
        # Retain old receipt boundaries as explicit unverified history.
        raise Refused("legacy receipt is unverified; import and reconcile its exact operation")
    raise Refused("unknown saved operation")


def export(controller, identifier):
    op = controller.public_view(identifier)
    path = controller.store.path.parent / "profile-switch" / (identifier + ".json")
    view = dict(op, operation_id=identifier, status=op["outcome"] or "pending", stage=op["phase"],
                target=op["spec"].get("name"), source=op.get("parent"))
    durable(path, view)
    durable(path.parent / "current.json", {"operation_id": identifier})
    return path


def saved_status(controller, value=None):
    if value and Path(value).is_file():
        projection = json.loads(Path(value).read_text())
        if not projection.get("operation_id") and projection.get("status") in ("queued", "failed", "ready"):
            return dict(projection, receipt=str(Path(value).resolve()), historical=True, evidence="legacy saved receipt",
                        ownership="unverified", freshness="unverified after invocation")
    state = controller.store.read()
    if value:
        identifier = reference(controller, value)
    elif state["operations"]:
        identifier = next(reversed(state["operations"]))
    else:
        return {"status": "unknown", "blockers": [{"reason": "no saved operation evidence"}], "historical": True}
    op = controller.public_view(identifier)
    stamps = [entry.get("observed_at", 0) for entry in op.get("observations", {}).get("authorities", {}).values()]
    # Monotonic time belongs to one boot; source identity and service births are
    # retained. A saved success is historical and never implies a fresh probe.
    return dict(op, status=op["outcome"] or "pending", receipt=str(controller.store.path.parent / "profile-switch" / (identifier + ".json")),
                evidence="saved", freshness="unverified after invocation", historical=True)


def requirements(runtime, spec, bindings, authorities):
    blockers, changes = [], []
    if spec.get("recipe_family"):
        blockers.append({"resource": "*", "reason": "recipe family lacks verified ingress/member execution coverage",
                         "recipe_family": spec["recipe_family"], "source": spec.get("recipe", {}).get("source")})
    if spec["instances"]:
        pins = sorted({instance["recipe"].rsplit("@", 1)[-1] for instance in spec["instances"]})
        blockers.append({"resource": "*", "reason": "unsupported native ingress and member effects", "recipe_pins": pins})
    overview = runtime.api("spark3", 7870, "/api/sparks")
    if not isinstance(overview, dict):
        blockers.append({"resource": "*", "reason": "Studio readiness observation is unknown"})
        overview = {}
    flags = spec.get("studio")
    settings_effect = None
    actual_workers = {entry.get("address"): entry for entry in overview.get("workers", []) if isinstance(entry, dict)}
    if flags:
        head = authorities.get("spark3", {}).get("snapshot", {})
        registered = head.get("worker_bindings")
        if not isinstance(registered, list) or type(head.get("sparks")) is not int:
            blockers.append({"resource": "*", "reason": "identity-bound Studio configuration is unknown"})
            registered = []
        updates = []
        for node, enabled in flags["workers"].items():
            target = bindings.get(node)
            matches = [row for row in registered if target and all(row.get(key) == target.get(key)
                       for key in ("resource_id", "instance_id", "birth"))]
            if len(matches) != 1:
                blockers.append({"resource": "*", "reason": "Studio worker binding is missing or ambiguous", "worker": node})
                continue
            row = matches[0]
            worker = actual_workers.get(row["worker"])
            if row["enabled"] != enabled:
                changes.append(node + " Studio worker placement")
                updates.append(dict(row, enabled=enabled))
            if enabled and row['enabled'] is False:
                # A disabled dispatch slot is deliberately unusable in the
                # coordinator. Observe the still-running peer directly before
                # planning its guarded enable; never infer readiness from off.
                from profileplan import parse_snapshot
                own=runtime.api('spark3',7870,'/api/spark/self')
                direct=runtime.api(node,7870,'/api/spark/self')
                try:
                    head_self=parse_snapshot(own['control'])
                    peer_self=parse_snapshot(direct['control'])
                    known=own['code']
                    if (head_self.self_resource.model_dump()!=bindings['spark3']
                            or peer_self.self_resource.model_dump()!=target
                            or head_self.revision<authorities['spark3']['expected']['revision']
                            or peer_self.revision<authorities[node]['expected']['revision']
                            or not isinstance(known,dict) or not known
                            or not isinstance(direct.get('code'),dict)
                            or any(direct.get('code',{}).get(path)!=value for path,value in known.items())
                            or type(direct.get('active')) is not int or direct['active']<0
                            or direct.get('guard') is not True):
                        raise Refused('disabled Studio peer identity/code/guard is unverified')
                    worker={**(worker or {}),'address':row['worker'],'guard':True,'code_matches':True,
                            'state':'busy' if direct.get('active') else 'ready','usable':True,
                            'can':{key:bool(direct.get(key)) for key in ('fasth3','image','ref2va','ref2va_turbo','ffmpeg')} |
                                  {'loras':len(direct.get('loras') or []),'image_models':direct.get('image_models') or []},
                            'direct_evidence':{'expected':{k:direct['control'][k] for k in ('instance_id','birth','revision')},'code':direct['code']}}
                    overview['workers']=[worker if w.get('address')==row['worker'] else w for w in overview.get('workers',[])]
                except (Refused,KeyError,TypeError,ValueError):
                    worker=None
            if enabled and (not worker or worker.get("guard") is not True or worker.get("code_matches") is not True
                            or worker.get("state") not in ("ready", "busy") or worker.get("usable") is not True):
                blockers.append({"resource": target["resource_id"], "reason": "Studio service readiness or code identity is unverified"})
            if not enabled and any(activity.get("kind") in ("comfy", "resident:comfy") and activity.get("state") != "settled"
                                   for activity in authorities[node]["snapshot"].get("activities", [])):
                changes.append(node + " warm Studio engine")
        if head.get("sparks") != flags["default"]:
            changes.append("Studio default placement")
        if updates or head.get("sparks") != flags["default"]:
            settings_effect = {"kind": "configure_workers", "workers": updates}
            if head.get("sparks") != flags["default"]:
                settings_effect["sparks"] = flags["default"]
    artifacts = runtime.asset_report(spec, overview) if spec.get("assets") and overview else {"ready": not spec.get("assets")}
    if not isinstance(artifacts, dict) or artifacts.get("ready") is not True:
        blockers.append({"resource": "*", "reason": "required artifacts/capabilities are unverified", "evidence": artifacts})
    if spec.get("puffer"):
        puffer = runtime.puffer_report(spec["puffer"])
        if not isinstance(puffer, dict) or puffer.get("state") != "verified":
            blockers.append({"resource": spec["puffer"]["node"], "reason": "Puffer installation readiness is unverified", "evidence": puffer})
        if isinstance(artifacts, dict):
            artifacts["puffer"] = puffer
    return {"blockers": blockers, "changes": changes, "settings_effect": settings_effect,
            "artifacts": {"state": "verified" if not any("artifacts" in b["reason"] for b in blockers) else "unknown", "report": artifacts},
            "capabilities": {"state": "verified" if not blockers else "unknown", "authorities": {peer: entry["expected"] for peer, entry in authorities.items()}},
            "services": {"state": "verified" if not blockers else "unknown", "bindings": bindings, "overview": overview},
            "settings": {"state": "verified" if not changes else "unknown", "studio": flags}}


def retire_native(controller,cluster,operation,retain_barriers=False):
    import clusterprofile
    from nativeadapter import NativeAdapter
    from familytransport import RemoteFamily
    from nativeretire import retire
    identity=source_identity(cluster.ROOT)
    op=controller.begin_retire(operation['id'],identity,retain_barriers=retain_barriers)
    runtime=clusterprofile.Runtime(cluster)
    studio=StudioAdapter(runtime,{node:(node,7870) for node in resources(op['spec']['groups'])},time.monotonic)
    return retire(controller,NativeAdapter(controller,RemoteFamily(controller,runtime)),studio,op,identity)


def stock_apply(cluster, controller, runtime, spec, key, wait):
    import os
    import stockstudio
    from nativeinstalled import start_worker
    worker = bool(os.environ.get('SPARK_NATIVE_WORKER'))
    op = stockstudio.apply(controller, runtime, spec, key, source_identity(cluster.ROOT), 280 if worker else wait)
    path = export(controller, op['id'])
    pending = op.get('outcome') is None and op.get('blockers') and op['blockers'][0]['reason'] == stockstudio.PENDING
    if pending and not worker:
        start_worker(cluster, controller, op, path)
    print(public_json(dict(op, receipt=str(path)), indent=2))
    ready = op.get('outcome') == 'ready' and not op.get('current_observation', {}).get('blockers')
    # The resume unit restarts only while workers are still starting.
    raise SystemExit(0 if ready or (worker and not pending) else 3)


def command(cluster, args):
    """Every profile that reaches ready (apply, rollback, or the background worker's resume) ends with no idle Studio
    model server resident on a Spark (studioidle): they are stopped through each node's Studio app, never a job's."""
    import os
    import sys
    # A cluster object without an ssh transport (the control fixtures) reaches no node to check.
    if args.action not in ("apply", "resume", "rollback") or not getattr(args, "yes", False) or not hasattr(cluster, "ssh"):
        return apply_command(cluster, args)
    try:
        apply_command(cluster, args)
        code = 0
    except SystemExit as done:
        code = done.code
        if code not in (0, None):
            raise
    import studioidle
    facts = studioidle.enforce(cluster, release=True)
    print(public_json({"idle_models": facts}, indent=2), file=sys.stderr)  # stdout stays the operation's one JSON
    left = [fact["node"] for fact in facts if fact.get("idle_resident")]
    if left and not os.environ.get("SPARK_NATIVE_WORKER"):  # the worker reports; it never restarts a ready operation
        print(f"spark: idle Studio model servers stay on {', '.join(left)}; see `spark idle`", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(code or 0)


def apply_command(cluster, args):
    import clusterprofile
    controller = Controller(Store(cluster.STATE))
    action = args.action
    from nativeinstalled import handles, command as native_command
    if handles(cluster, controller, args):
        return native_command(cluster, controller, args)
    if action == "list":
        print("\n".join(clusterprofile.NAMES))
        return
    if action == "migrate":
        if not args.yes:
            raise Refused("add --yes to preserve and import legacy state as unverified")
        controller.store.import_legacy()
        print(public_json({"status": "imported", "authority": str(cluster.STATE), "ownership": "unverified"}))
        return
    if action == "status":
        view = saved_status(controller, getattr(args, "operation", None))
        print(public_json(view, indent=2))
        raise SystemExit(0 if view["status"] == "ready" else 1 if view["status"] == "failed" else 3)
    if action == "prepare":
        spec = clusterprofile.load(cluster.ROOT, args.name)
        if not args.yes:
            print(public_json({"effect":"preview", "target":args.name, "authority":"canonical artifact host/path", "gpu_lease_action":"none"}, indent=2))
            return
        from profileprepare import prepare
        key = getattr(args, "operation_key", None) or uuid.uuid4().hex
        rows = prepare(controller, clusterprofile.Runtime(cluster), spec, key, source_identity(cluster.ROOT))
        receipts = [str(export(controller, row["id"])) for row in rows]
        print(public_json({"operation_key":key,"operations":rows,"receipts":receipts,"gpu_lease_action":"none"},indent=2))
        raise SystemExit(0 if rows and all(row.get("outcome")=="prepared" for row in rows) else 3)
    if action in ("resume", "cancel", "rollback"):
        identifier = reference(controller, args.receipt)
        original = controller.view(identifier)
        if action == "resume" and original['spec'].get('stock_studio') and original.get('outcome') is not None:
            raise Refused("finished stock Studio receipt: run 'spark profile apply " + original['spec']['name'] + " --yes' to check health or apply again")
        if action == "resume" and original['spec'].get('stock_studio') and original.get('outcome') is None:
            spec = {k: v for k, v in original['spec'].items() if k != 'stock_studio'}
            stock_apply(cluster, controller, clusterprofile.Runtime(cluster), spec, uuid.uuid4().hex, args.wait)
        if action == "cancel":
            if not args.yes:
                raise Refused("add --yes to cancel this profile operation")
            if original['spec'].get('recipe_family') and original.get('outcome') not in ('released','cancelled'):
                op=retire_native(controller,cluster,original)
                if op.get('outcome')!='released':
                    print(public_json(dict(op,receipt=str(export(controller,identifier))),indent=2))
                    raise SystemExit(3)
            op = controller.cancel(identifier)
            path = export(controller, identifier)
            print(public_json(dict(op, receipt=str(path)), indent=2))
            raise SystemExit(0 if op["outcome"] == "cancelled" else 3)
        spec = original["spec"]
        if original.get("intent") == "prepare" and "artifact_resources" in original:
            if action != "resume":
                raise Refused("artifact rollback cannot replace or remove published files")
            from profileprepare import resume
            rows = resume(controller, clusterprofile.Runtime(cluster), original, source_identity(cluster.ROOT))
            rows = rows if isinstance(rows, list) else [rows]
            print(public_json({"operations":rows,"receipts":[str(export(controller,row["id"])) for row in rows],"gpu_lease_action":"none"},indent=2))
            raise SystemExit(0 if rows and all(row.get("outcome")=="prepared" for row in rows) else 3)
        if action == "rollback":
            prior = original["allocation_before"]
            if not prior.get("groups"):
                raise Refused("saved parent allocation is unknown; rollback cannot invent it")
            parent_op=controller.store.read()['operations'].get(prior.get('operation_id'))
            if parent_op is None:raise Refused('saved rollback target spec is unknown')
            spec=copy.deepcopy(parent_op['spec'])
            if original['spec'].get('recipe_family'):
                result=retire_native(controller,cluster,original,retain_barriers=True)
                if result.get('outcome')!='retired':
                    print(public_json(dict(result,receipt=str(export(controller,identifier))),indent=2));raise SystemExit(3)
            op = controller.request("restore", spec, original["resources"], uuid.uuid4().hex,
                                    source_identity(cluster.ROOT), parent=identifier)
            identifier = op["id"]
    else:
        spec = clusterprofile.load(cluster.ROOT, args.name)
        if not args.yes:
            print(public_json({"target": args.name, "spec_digest": digest(spec), "resources": list(resources(spec["groups"])),
                              "effect": "preview", "native_coverage": "unsupported" if spec["instances"] else "requires complete Studio protocol"}, indent=2))
            return
        import labprofile
        import stockstudio
        if labprofile.covers(spec):
            op = labprofile.apply(cluster, controller, clusterprofile.Runtime(cluster), spec,
                                  getattr(args, 'operation_key', None) or uuid.uuid4().hex, source_identity(cluster.ROOT))
            print(public_json(dict(op, receipt=str(export(controller, op['id']))), indent=2))
            raise SystemExit(0 if op.get('outcome') == 'ready' and not op.get('current_observation', {}).get('blockers') else 3)
        if stockstudio.covers(spec):
            runtime = clusterprofile.Runtime(cluster)
            labprofile.leave(cluster, controller, runtime)
            if stockstudio.stock(runtime) or any(o['spec'].get('stock_studio') and o.get('outcome') is None
                                                 for o in controller.store.read()['operations'].values()):
                stock_apply(cluster, controller, runtime, spec, getattr(args, 'operation_key', None) or uuid.uuid4().hex, args.wait)
        # Persist intent before observation, including missing/unknown identities.
        state = controller.store.read()
        parent = state["layout"].get("operation_id")
        prior = state["operations"].get(parent)
        if prior and prior.get('spec',{}).get('recipe_family') and prior.get('outcome')=='ready':
            prior=retire_native(controller,cluster,prior,retain_barriers=True)
            if prior.get('outcome')!='retired':
                print(public_json(dict(prior,receipt=str(export(controller,prior['id']))),indent=2));raise SystemExit(3)
        if prior and prior.get("outcome") not in ('ready','retired'):
            parent, prior = None, None
        op = controller.request("prepare" if action == "prepare" else "activate", spec, prior["resources"] if prior else [],
                                getattr(args, "operation_key", None) or uuid.uuid4().hex, source_identity(cluster.ROOT), parent=parent)
        identifier = op["id"]
    if not args.yes:
        print(public_json(controller.view(identifier), indent=2))
        return
    runtime = clusterprofile.Runtime(cluster)
    peers = {node: (node, 7870) for node in resources(spec["groups"])}
    studio = StudioAdapter(runtime, peers, time.monotonic)
    driver = Driver(controller, studio, lambda p, b, a: requirements(runtime, p, b, a), source_identity(cluster.ROOT))
    if spec.get('recipe_family'):
        from nativeadapter import NativeAdapter
        from familytransport import RemoteFamily
        driver.native=NativeAdapter(controller,RemoteFamily(controller,runtime))
    op = driver.drive(identifier)
    path = export(controller, identifier)
    print(public_json(dict(op, receipt=str(path)), indent=2))
    raise SystemExit(0 if op["outcome"] == "ready" and op.get('current_observation',{}).get('state')!='unknown' else 3)
