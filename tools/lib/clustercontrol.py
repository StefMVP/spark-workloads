"""The durable authority for allocation, jobs, operations and effect permission."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, TypeVar
import uuid

SCHEMA = 1
ADAPTER = "native-control-v1"
T = TypeVar("T")

# These fields belong to private recovery state, including nested artifact
# descriptors that can contain an encoded native control configuration.
PRIVATE_FIELDS=frozenset({'native_bootstrap','control_token','content'})

def public_projection(value):
    if isinstance(value,dict):return {key:public_projection(item) for key,item in value.items() if key not in PRIVATE_FIELDS}
    if isinstance(value,(list,tuple)):return [public_projection(item) for item in value]
    return copy.deepcopy(value)


class Refused(RuntimeError):
    pass


class EffectState(StrEnum):
    UNSENT = "unsent"
    UNCERTAIN = "may_have_applied"
    VERIFIED = "verified"
    REJECTED = "rejected"


class EffectKind(StrEnum):
    FENCE_ACQUIRE = "fence_acquire"
    FENCE_SEAL = "fence_seal"
    ARTIFACT_PREPARE = "artifact_prepare"
    HOST_STOP = "host_stop"
    HOST_START = "host_start"
    LAYOUT = "layout"
    SETTINGS = "settings"
    FENCE_RELEASE = "fence_release"
    STUDIO_EFFECT = "studio_effect"
    NATIVE_FENCE = "native_fence"
    NATIVE_RUNTIME = "native_runtime"


DISRUPTIVE = {EffectKind.NATIVE_RUNTIME, EffectKind.HOST_STOP, EffectKind.HOST_START, EffectKind.LAYOUT, EffectKind.SETTINGS}


class FactState(StrEnum):
    MISSING = "missing"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    STALE = "stale"
    ABSENT = "absent"
    PRESENT = "present"


@dataclass(frozen=True)
class LeaseToken:
    operation_id: str
    generation: int
    resources: tuple[str, ...]

    def document(self):
        return {"operation_id": self.operation_id, "generation": self.generation,
                "resources": list(self.resources)}


@dataclass(frozen=True)
class Birth:
    kind: str
    boot_id: str
    identity: str
    started: str

    @classmethod
    def parse(cls, value):
        if (not isinstance(value, dict) or value.get("kind") not in ("process", "container", "unit")
                or any(not isinstance(value.get(k), str) or not value[k] for k in ("boot_id", "identity", "started"))):
            raise Refused("invalid or missing physical birth")
        return cls(**{k: value[k] for k in ("kind", "boot_id", "identity", "started")})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resources(groups):
    if not isinstance(groups, dict):
        raise Refused("invalid placement")
    flat = [node for group in groups.values() for node in group]
    if (any(not isinstance(group, list) or not group for group in groups.values())
            or any(node not in ("spark1", "spark2", "spark3", "spark4") for node in flat)
            or len(flat) != len(set(flat))):
        raise Refused("invalid or overlapping placement")
    return tuple(sorted(flat))


def canonical(resource):
    return isinstance(resource, str) and re.fullmatch(r"gpu:GPU-[A-Za-z0-9-]+", resource) is not None


def artifact_resource(host_identity, path):
    if not isinstance(host_identity, str) or not host_identity or ":" in host_identity:
        raise Refused("artifact host identity is invalid")
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise Refused("artifact path must be canonical and absolute")
    return "artifact:" + host_identity + ":" + str(path)


def empty():
    return {"control_schema": SCHEMA, "revision": 0, "generation": 0,
            "layout": {"spec": None, "groups": {}}, "leases": {}, "jobs": {},
            "operations": {}, "requests": {}, "imports": {}, "artifact_reservations": {}}


ENDED_MAX = 500      # notes on queue entries that left the queue with no job (cancelled)


def validate(value):
    if not isinstance(value, dict) or value.get("control_schema") != SCHEMA:
        raise Refused("unsupported controller schema")
    for key in ("revision", "generation"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise Refused("invalid controller " + key)
    for key in ("layout", "leases", "jobs", "operations", "requests", "imports", "artifact_reservations"):
        if not isinstance(value.get(key), dict):
            raise Refused("invalid controller " + key)
    resources(value["layout"].get("groups"))
    occupied = set()
    for node, lease in value["leases"].items():
        if not canonical(node) or not isinstance(lease, dict):
            raise Refused("invalid resource lease")
        if lease.get("state") == "unverified":
            continue
        if (lease.get("state") != "owned" or not isinstance(lease.get("operation_id"), str)
                or not lease["operation_id"] or type(lease.get("generation")) is not int
                or not 0 < lease["generation"] <= value["generation"]):
            raise Refused("invalid owned resource lease")
        occupied.add(node)
    for key, op in value["operations"].items():
        if not isinstance(op, dict) or op.get("id") != key:
            raise Refused("invalid operation identity")
        if op.get("outcome") == "ready":
            if op.get("blockers") or any(e.get("state") not in (EffectState.VERIFIED, EffectState.REJECTED) for e in op.get("effects", [])):
                raise Refused("READY has unresolved evidence")
    return value


def durable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Store:
    def __init__(self, path):
        self.path = Path(path)

    def _load(self):
        try:
            value = json.loads(self.path.read_text()) if self.path.exists() else empty()
        except (OSError, ValueError) as error:
            raise Refused("invalid controller state; no effects permitted") from error
        if "control_schema" not in value:
            raise Refused("legacy controller state requires explicit import")
        return validate(value)

    def read(self):
        return copy.deepcopy(self._load())

    def commit(self, change: Callable[[dict], T], expected=None) -> T:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = self._load()
            if expected is not None and value["revision"] != expected:
                raise Refused("controller revision changed")
            before = copy.deepcopy(value)
            result = change(value)
            if value != before:
                value["revision"] += 1
                durable(self.path, validate(value))
            return copy.deepcopy(result)

    def import_legacy(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                durable(self.path, empty())
                return
            old = json.loads(self.path.read_text())
            if "control_schema" in old:
                validate(old)
                return
            groups = old.get("layout", {}).get("groups", {})
            resources(groups)
            value = empty()
            value["layout"] = old.get("layout", value["layout"])
            key = digest(old)
            backup = self.path.with_name(self.path.name + ".legacy-" + key + ".json")
            if not backup.exists():
                durable(backup, old)
            value["imports"][key] = {"original": str(backup), "state": old}
            for group, lease in old.get("leases", {}).items():
                if group not in groups:
                    raise Refused("legacy lease has unknown placement")
                # Legacy transport aliases do not establish a GPU UUID. Preserve
                # these claims as blockers instead of manufacturing leases.
                value["imports"][key].setdefault("claims", {})[group] = {
                    "state": "unverified", "legacy": lease, "members": groups[group]}
            for jid, job in old.get("jobs", {}).items():
                value["jobs"][jid] = dict(job, authority="unverified", status="unknown" if job.get("status") == "running" else job.get("status"))
            for path in sorted(self.path.parent.glob("profile-switch/*.json")):
                if not path.is_symlink():
                    try:
                        value["imports"][str(path.resolve())] = {"receipt": json.loads(path.read_text()), "state": "unverified"}
                    except ValueError:
                        value["imports"][str(path.resolve())] = {"state": "invalid"}
            durable(self.path, validate(value))


def token_for(op):
    return LeaseToken(op["id"], op["generation"], tuple(op["resources"]))


def owns(control, token):
    return bool(token.resources) and all(control["leases"].get(node, {}).get("state") == "owned"
               and control["leases"][node]["operation_id"] == token.operation_id
               and control["leases"][node]["generation"] == token.generation for node in token.resources)


def require_owner(control, token):
    if not owns(control, token):
        raise Refused("missing, unverified or stale lease generation")


def return_parent_leases(control, op):
    """An unstarted child gives each transferred resource back to its unchanged parent."""
    parent = op.get("parent_authority") or {}
    original = control["operations"].get(parent.get("operation_id"))
    live = (original is not None and original["generation"] == parent.get("generation")
            and original.get("outcome") not in ("released", "cancelled"))
    for resource in op["resources"]:
        if live and resource in parent.get("resources", []):
            control["leases"][resource] = {"state": "owned", "operation_id": original["id"], "generation": original["generation"]}
        else:
            del control["leases"][resource]


class Controller:
    def __init__(self, store):
        self.store = store

    def reconcile_native(self, hosts):
        """Retain legacy evidence; retire only physically absent imported work."""
        from nativeinstalled import reconcile
        return self.store.commit(lambda state: reconcile(state, hosts))

    def retire_absent_installed(self, identifier, hosts):
        """Release a historical owner only after its exact processes have gone."""
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            from nativeinstalled import validate_hosts
            manifest=op['spec'].get('installed_native')
            members=(manifest or {}).get('members') or None
            validate_hosts({n:hosts[n] for n in members if n in hosts} if members else hosts,*([members] if members else []))
            if not manifest or op.get('outcome')!='ready' or any(e['state']==EffectState.UNCERTAIN for e in op['effects']):
                raise Refused('historical native owner is incomplete')
            if any(j.get('operation_id')==identifier and j.get('status')!='settled' for j in state['jobs'].values()):
                raise Refused('unsettled native jobs retain ownership')
            for n,row in manifest['nodes'].items():
                c=hosts[n]['containers'].get(manifest['name'])
                if (hosts[n]['resource_id']!=row['resource'] or not row.get('id') or not c
                    or c['id']!=row['id'] or (row.get('image') and c.get('image_id')!=row['image'])
                    or 'process_birth' not in c or c['process_birth'] is not None
                    or any(g.get('container')==row['id'] for g in hosts[n].get('gpu',[]))):
                    raise Refused('historical native process is present or unknown')
            op['absence_retirement']={'hosts':copy.deepcopy(hosts),'prior_outcome':op['outcome']}
            for resource in op['resources']:del state['leases'][resource]
            op['outcome']='released';op['policy']='stop_after_inflight';op['phase']='finished'
            if state['layout'].get('operation_id')==identifier:
                state['layout']={'spec':None,'groups':{},'operation_id':None,'generation':op['generation']}
                state['profile']=None
            return op
        return self.store.commit(change)

    def repair_unaccepted_native(self, identifier, retired_id, new_source, proof):
        """Attach an absent retired parent only when no host accepted the effect."""
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            old=state['operations'][retired_id]
            if (op['spec'].get('installed_native',{}).get('backend')!='tensorfold-native'
                or op.get('outcome') is not None or len(op['effects'])!=1
                or old.get('outcome')!='released' or not (old.get('absence_retirement') or old.get('intent')=='retire')):
                raise Refused('native refusal recovery requires a physically retired owner')
            effect=op['effects'][0];payload=effect['payload'];member=payload['member']
            row=old['spec']['installed_native']['nodes'].get(member,{})
            prior=proof.get('old_container',{})
            if (effect['state']!=EffectState.UNCERTAIN or payload.get('action')!='guard-start'
                or proof.get('not_accepted') is not True or proof.get('effect_digest')!=digest(effect)
                or proof.get('member')!=member or proof.get('boot_id')!=payload['boot_id']
                or proof.get('unchanged_binding')!=payload['plan']['binding']
                or proof.get('retired_parent')!=token_for(old).document()
                or not set(op['resources'])<=set(old['resources'])
                or prior.get('id')!=row.get('id') or not row.get('id')
                or prior.get('image_id')!=row.get('image') or 'process_birth' not in prior
                or prior['process_birth'] is not None):
                raise Refused('native refusal lacks exact no-acceptance and old-process absence proof')
            op.setdefault('source_repairs',[]).append({'kind':'unaccepted-retired-parent',
                'old_source':op['source_identity'],'new_source':new_source,'proof':copy.deepcopy(proof)})
            op['source_identity']=new_source;op['parent_authority']=token_for(old).document();op['blockers']=[]
            return op
        return self.store.commit(change)

    def begin_installed_retire(self, identifier, source_identity):
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            if op['spec'].get('installed_native',{}).get('backend')!='tensorfold-native':
                raise Refused('native retirement source/backend differs')
            if op['source_identity']!=source_identity:
                # A completed READY owner may retire under new code: stops still bind exact
                # container objects and process births from fresh evidence.
                if (op.get('intent')=='retire' or op.get('outcome')!='ready'
                        or any(e['state'] not in (EffectState.VERIFIED,EffectState.REJECTED) for e in op['effects'])):
                    raise Refused('native retirement source/backend differs')
                op.setdefault('source_repairs',[]).append({'kind':'ready-retire','old_source':op['source_identity'],'new_source':source_identity})
                op['source_identity']=source_identity
            if op.get('intent')=='retire':return op
            if op.get('outcome')!='ready':raise Refused('native stop requires completed readiness')
            op['completed_receipt']={'outcome':op['outcome'],'readiness':copy.deepcopy(op['readiness'])}
            op.update(intent='retire',outcome=None,phase='retiring',policy='run',blockers=[])
            return op
        return self.store.commit(change)

    def repair_native_permission(self, identifier, new_source, proof):
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            from nativeinstalled import repair_permission
            return repair_permission(op,new_source,proof)
        return self.store.commit(change)

    def repair_expired_native_guard(self, identifier, new_source, proof):
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            from nativeinstalled import repair_expired_guard
            return repair_expired_guard(op,new_source,proof)
        return self.store.commit(change)

    def record_expired_native_guard_refusal(self, identifier):
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            from nativeinstalled import expired_guard_refusal
            op['startup_guard_refusal']=dict(expired_guard_refusal(op),effect_digest=digest(op['effects'][1]))
            return op
        return self.store.commit(change)

    def repair_native_readiness(self, identifier, new_source, evidence):
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            from nativeinstalled import repair_readiness
            return repair_readiness(op,new_source,evidence)
        return self.store.commit(change)

    def cancel_native_unstarted(self, identifier):
        def change(state):
            op=state['operations'][identifier];require_owner(state,token_for(op))
            if not op['spec'].get('installed_native') or op['effects']:
                raise Refused('native cancellation requires exact recovery after effects')
            return_parent_leases(state,op)
            op.update(outcome='cancelled',phase='finished',policy='stop_after_inflight',blockers=[])
            return op
        return self.store.commit(change)

    def prepare_pending(self, spec, key, source_identity, blockers):
        """Record unresolved disk preparation without any GPU or write grant."""
        if not isinstance(spec, dict) or not isinstance(blockers, list) or not blockers:
            raise Refused("unknown preparation needs exact input and blockers")
        body = {"request": copy.deepcopy(spec), "files": [], "source_identity": source_identity}
        def change(control):
            prior = control["requests"].get(key)
            if prior:
                if prior["digest"] != digest(body):
                    raise Refused("artifact idempotency key conflicts")
                return control["operations"][prior["operation_id"]]
            identifier = uuid.uuid4().hex
            control["generation"] += 1
            op = {"id": identifier, "intent": "prepare", "spec": body, "spec_digest": digest(body), "adapter": ADAPTER,
                  "source_identity": source_identity, "generation": control["generation"], "resources": [],
                  "artifact_resources": [], "effects": [], "observations": {}, "blockers": copy.deepcopy(blockers),
                  "policy": "run", "outcome": None, "phase": "waiting"}
            control["operations"][identifier] = op
            control["requests"][key] = {"digest": digest(body), "operation_id": identifier}
            return op
        return self.store.commit(change)

    def native_bootstrap(self,identifier,binding):
        def change(control):
            import secrets
            op=control['operations'][identifier];require_owner(control,token_for(op))
            if not op['spec'].get('recipe_family'):raise Refused('native bootstrap requires an owned recipe operation')
            head=op['spec']['recipe']['members'][0]
            if binding.get('boot_id')!=op['observations']['bindings'][head]['birth']['boot_id']:raise Refused('control host and Studio boot identity differ')
            prior=op.get('native_bootstrap')
            if prior:
                if prior['binding']!=binding:raise Refused('native control host birth changed')
                return prior
            root=Path(binding['home'])/'.local/state/spark/native'/identifier
            result={'config_path':str(root/'config.json'),'state_dir':str(root/'state'),'binding':copy.deepcopy(binding),
                'config':{'instance_id':identifier,'owner':token_for(op).document(),'resources':op['resources'],
                    'control_token':secrets.token_hex(32),'source_package':'/opt/profile-control/source',
                    'state_path':'/run/profile-control/state/admission.json'}}
            op['native_bootstrap']=result
            return result
        return self.store.commit(change)

    def capture_native_preflight(self,identifier,inventory,bindings):
        def change(control):
            op=control['operations'][identifier];require_owner(control,token_for(op))
            if bindings!=op.get('bindings'):raise Refused('native preflight placement differs from owned bindings')
            if 'native_preflight' not in op:op['native_preflight']={'inventory':copy.deepcopy(inventory),'bindings':copy.deepcopy(bindings)}
            return op['native_preflight']
        return self.store.commit(change)

    def prepare_request(self, host_identity, files, key, source_identity, *, identifier=None):
        if not isinstance(files, list) or not files:
            raise Refused("artifact prepare needs a file manifest")
        targets = []
        for entry in files:
            if (not isinstance(entry, dict) or set(entry) != {"source", "target", "sha256", "usage"}
                    or not isinstance(entry["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                    or entry["usage"] not in ("new_path", "unused", "preexisting_verified")
                    or any("checkpoint" in part.lower() for part in Path(entry["target"]).parts)):
                raise Refused("invalid or protected artifact manifest")
            targets.append(artifact_resource(host_identity, entry["target"]))
        if len(targets) != len(set(targets)):
            raise Refused("artifact targets overlap")
        body = {"host_identity": host_identity, "files": files, "source_identity": source_identity}
        identifier = identifier or uuid.uuid4().hex
        def change(control):
            prior = control["requests"].get(key)
            if prior:
                if prior["digest"] != digest(body):
                    raise Refused("artifact idempotency key conflicts")
                return control["operations"][prior["operation_id"]]
            for target in targets:
                for reserved in control["artifact_reservations"]:
                    if target == reserved or target.startswith(reserved + "/") or reserved.startswith(target + "/"):
                        raise Refused("artifact path already reserved")
            control["generation"] += 1
            op = {"id": identifier, "intent": "prepare", "spec": body, "spec_digest": digest(body), "adapter": ADAPTER,
                  "source_identity": source_identity, "generation": control["generation"], "resources": [],
                  "artifact_resources": targets, "effects": [], "observations": {}, "blockers": [], "policy": "run", "outcome": None, "phase": "inspecting"}
            control["operations"][identifier] = op
            control["requests"][key] = {"digest": digest(body), "operation_id": identifier}
            for target in targets:
                control["artifact_reservations"][target] = {"operation_id": identifier, "generation": op["generation"]}
            return op
        return self.store.commit(change)

    def prepare_issue(self, identifier, index):
        def change(control):
            op = control["operations"][identifier]
            if op["intent"] != "prepare" or op["policy"] != "run" or op["outcome"] is not None:
                raise Refused("artifact operation does not permit publication")
            if type(index) is not int or not 0 <= index < len(op["spec"]["files"]):
                raise Refused("invalid artifact member")
            target = op["artifact_resources"][index]
            expected = {"operation_id": identifier, "generation": op["generation"]}
            if control["artifact_reservations"].get(target) != expected:
                raise Refused("artifact write ownership changed")
            effect_id = identifier + ":artifact:" + str(index)
            prior = next((e for e in op["effects"] if e["id"] == effect_id), None)
            if prior:
                return prior
            effect = {"id": effect_id, "kind": EffectKind.ARTIFACT_PREPARE, "state": EffectState.UNCERTAIN,
                      "payload": op["spec"]["files"][index], "resource": target, "token": expected, "outcome": None}
            op["effects"].append(effect)
            return effect
        return self.store.commit(change)

    def prepare_outcome(self, identifier, effect_id, evidence):
        def change(control):
            op = control["operations"][identifier]
            effect = next(e for e in op["effects"] if e["id"] == effect_id)
            if control["artifact_reservations"].get(effect["resource"]) != effect["token"]:
                raise Refused("artifact write ownership changed")
            if (evidence.get("state") != "verified" or evidence.get("path") != effect["payload"]["target"]
                    or evidence.get("sha256") != effect["payload"]["sha256"]):
                raise Refused("artifact publication outcome is unknown")
            effect["state"], effect["outcome"] = EffectState.VERIFIED, evidence
            if len(op["effects"]) == len(op["spec"]["files"]) and all(e["state"] == EffectState.VERIFIED for e in op["effects"]):
                op["phase"], op["outcome"] = "finished", "prepared"
                for resource in op["artifact_resources"]:
                    if control["artifact_reservations"].get(resource) != effect["token"]:
                        raise Refused("artifact ownership changed during finalization")
                    del control["artifact_reservations"][resource]
            return op
        return self.store.commit(change)

    def request(self, intent, spec, footprint, key, source_identity, *, parent=None, identifier=None):
        footprint = sorted(set(footprint))
        if any(not canonical(n) for n in footprint):
            raise Refused("operation requires a known complete resource footprint")
        body = {"intent": intent, "spec": spec, "resources": footprint, "source_identity": source_identity, "parent": parent}
        request_digest = digest(body)
        identifier = identifier or uuid.uuid4().hex

        def change(control):
            prior = control["requests"].get(key)
            if prior:
                if prior["digest"] != request_digest:
                    raise Refused("idempotency key reused with different intent")
                return control["operations"][prior["operation_id"]]
            if parent:
                original = control["operations"].get(parent)
                if not original:
                    raise Refused("unknown parent operation")
                require_owner(control, token_for(original))
                if any(job.get('operation_id') == parent and job.get('kind') == 'puffer-native-v1'
                       and job.get('status') != 'settled'
                       for job in control['jobs'].values()):
                    raise Refused('unsettled Puffer attempt retains its original lease authority')
                if any(e["state"] == EffectState.UNCERTAIN for e in original["effects"]):
                    raise Refused("parent has uncertain effects")
                if original['source_identity']!=source_identity:
                    from nativeinstalled import ready_handoff
                    ready_handoff(original,spec,source_identity)
                from familycontrol import live_guards
                if live_guards(original['effects']):
                    raise Refused('native memory guard generation handoff is unverified; retain the current owner')
            blockers = []
            if not footprint:
                blockers.append({"resource": "*", "reason": "physical resource identity is unknown"})
            if any(record.get("claims") for record in control["imports"].values()):
                blockers.append({"resource": "*", "reason": "legacy ownership needs identity-bound reconciliation"})
            for node in footprint:
                lease = control["leases"].get(node)
                if lease and (not parent or lease.get("operation_id") != parent):
                    blockers.append({"resource": node, "reason": "unverified ownership" if lease.get("state") == "unverified" else "resource held", "owner": lease.get("operation_id")})
            control["generation"] += 1
            op = dict(body, id=identifier, generation=control["generation"], adapter=ADAPTER,
                      spec_digest=digest(spec), effects=[], blockers=blockers, policy="run", outcome=None,
                      allocation_before=copy.deepcopy(control["layout"]), observations={}, phase="waiting" if blockers else "inspecting")
            if parent:op['parent_authority']=token_for(original).document()
            control["operations"][identifier] = op
            control["requests"][key] = {"digest": request_digest, "operation_id": identifier}
            if not blockers:
                for node in footprint:
                    control["leases"][node] = {"state": "owned", "operation_id": identifier, "generation": op["generation"]}
            return op
        return self.store.commit(change)

    def block(self, identifier, blockers):
        def change(control):
            op = control["operations"][identifier]
            op["blockers"] = blockers
            op["phase"] = "waiting"
            return op
        return self.store.commit(change)

    def record_health(self, identifier, observations, blockers):
        """A new health read does not rewrite a completed operation's receipt."""
        def change(control):
            op=control['operations'][identifier]
            if op.get('outcome')!='ready':raise Refused('health history requires a completed operation')
            op['current_observation']={'state':'unknown' if blockers else 'verified',
                                       'blockers':copy.deepcopy(blockers),'observations':copy.deepcopy(observations)}
            return op
        return self.store.commit(change)

    def begin_retire(self,identifier,source_identity,retain_barriers=False):
        def change(control):
            op=control['operations'][identifier]
            require_owner(control,token_for(op))
            if op['source_identity']!=source_identity or not op['spec'].get('recipe_family'):raise Refused('native retirement source/family differs')
            if op.get('retirement'):
                if bool(op.get('retirement_retain'))!=bool(retain_barriers):raise Refused('retirement barrier policy is immutable')
                return op
            if op.get('outcome')=='ready':
                op['completed_receipt']={'outcome':'ready','readiness':copy.deepcopy(op['readiness']),'effects':copy.deepcopy(op['effects'])}
            elif op.get('outcome') is not None:raise Refused('terminal operation cannot begin native retirement')
            op.update(retirement=True,retirement_retain=bool(retain_barriers),outcome=None,phase='retiring',policy='run',blockers=[])
            return op
        return self.store.commit(change)

    def finish_retire(self,identifier):
        def change(control):
            from familycontrol import live_guards
            from profileplan import parse_snapshot
            op=control['operations'][identifier];require_owner(control,token_for(op))
            if not op.get('retirement') or op['blockers'] or any(e['state']==EffectState.UNCERTAIN for e in op['effects']) or live_guards(op['effects']):raise Refused('native retirement still has uncertain effects or live guards')
            starts=[e for e in op['effects'] if e.get('payload',{}).get('family')=='spark-ds41f' and e['payload']['action']=='start' and e['state']==EffectState.VERIFIED]
            for start in starts:
                if not any(e.get('payload',{}).get('action')=='stop' and e['state']==EffectState.VERIFIED
                    and e['token']==start['token'] and e['payload']['plan']['binding']==start['outcome']['physical']['container'] for e in op['effects']):raise Refused('native process termination is unverified')
            entries=op.get('observations',{}).get('authorities',{})
            if not set(op['spec']['recipe']['members'])<=set(entries):raise Refused('native retirement authority coverage is incomplete')
            if op.get('retirement_retain'):
                from profileplan import sealed_authorities
                sealed_authorities(op)
            else:
                for entry in entries.values():
                    if entry.get('state')!='fresh' or any(f.operation_id==identifier for f in parse_snapshot(entry['snapshot']).fences):raise Refused('native retirement admission release is unverified')
            if any(j.get('operation_id')==identifier and j.get('status')!='settled' for j in control['jobs'].values()):raise Refused('unsettled attempts retain native ownership')
            if not op.get('retirement_retain'):
                for resource in op['resources']:del control['leases'][resource]
            op.update(outcome='retired' if op.get('retirement_retain') else 'released',phase='finished')
            return op
        return self.store.commit(change)

    def resolve_resources(self, identifier, bindings):
        def change(control):
            op = control["operations"][identifier]
            if op["resources"]:
                require_owner(control, token_for(op))
                return op
            nodes = resources(op["spec"]["groups"])
            if set(bindings) != set(nodes):
                raise Refused("incomplete physical placement bindings")
            from studio_protocol import ResourceIdentity
            parsed = [ResourceIdentity.model_validate(bindings[n]) for n in nodes]
            footprint = sorted({entry.resource_id for entry in parsed})
            if len(footprint) != len(nodes):
                raise Refused("ambiguous physical placement bindings")
            if any(record.get("claims") for record in control["imports"].values()):
                raise Refused("legacy ownership remains unverified")
            if any(resource in control["leases"] for resource in footprint):
                raise Refused("resource held by another operation")
            op["resources"], op["blockers"] = footprint, []
            op["bindings"] = bindings
            for resource in footprint:
                control["leases"][resource] = {"state": "owned", "operation_id": identifier, "generation": op["generation"]}
            return op
        return self.store.commit(change)

    def admit_job(self, identifier, job_id, kind, placement, memory, command, takes_entry=None):
        """takes_entry: the queue entry that this job record replaces; it leaves the queue in the same commit."""
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            if not placement or not set(placement) <= set(op["resources"]):
                raise Refused("job placement exceeds its owned resource footprint")
            if type(memory) is not int or memory < 0:
                raise Refused("invalid memory reservation")
            if job_id in control["jobs"]:
                raise Refused("job identity already exists")
            if kind == "puffer-native-v1":
                if (op.get("outcome") != "ready" or op["policy"] != "run"
                        or op.get("current_observation", {}).get("state", "verified") != "verified"):
                    raise Refused("Puffer requires current verified READY authority")
                if any(job["status"] != "settled" and set(placement) & set(job["resources"])
                       for job in control["jobs"].values()):
                    raise Refused("Puffer placement overlaps an unsettled job")
            job = {"id": job_id, "kind": kind, "operation_id": identifier, "generation": op["generation"],
                   "resources": sorted(placement), "mem_gib": memory, "command": command,
                   "status": "admitted", "attempt_id": identifier + ":" + job_id, "births": {},
                   "blocker": "native launch capability has not been verified"}
            control["jobs"][job_id] = job
            if takes_entry:
                control.get('queue', {}).pop(takes_entry, None)
            return job
        return self.store.commit(change)

    def settle_job(self, job_id, attempt_id, generation, outcomes):
        def change(control):
            job = control["jobs"][job_id]
            if job.get("attempt_id") != attempt_id or job.get("generation") != generation:
                raise Refused("stale job finalizer")
            # A completion belongs to its immutable attempt, even after a
            # metadata adoption. It records termination and never frees a lease.
            if (not isinstance(outcomes, dict) or set(outcomes) != set(job["resources"])
                    or any(not isinstance(result, dict) or result.get("state") != "terminated"
                           or result.get("birth") != job["births"].get(resource)
                           or result.get("attempt_id") != attempt_id for resource, result in outcomes.items())
                    or not job["births"]):
                raise Refused("job termination is unknown")
            job["status"], job["outcomes"] = "settled", outcomes
            return job
        return self.store.commit(change)

    def observe_job_start(self, job_id, attempt_id, generation, records):
        def change(control):
            job = control["jobs"][job_id]
            if job.get("attempt_id") != attempt_id or job.get("generation") != generation or job["status"] != "admitted":
                raise Refused("job start does not match admitted attempt")
            op = control["operations"][job["operation_id"]]
            require_owner(control, LeaseToken(job['operation_id'], job['generation'], tuple(job['resources'])))
            if not isinstance(records, dict) or set(records) != set(job["resources"]):
                raise Refused("job start footprint is incomplete")
            births = {}
            expected_token = {"operation_id": op["id"], "generation": op["generation"], "resources": job["resources"]}
            for resource, record in records.items():
                if (not isinstance(record, dict) or record.get("attempt_id") != attempt_id
                        or record.get("token") != expected_token or record.get("state") != "running"):
                    raise Refused("job start has unknown authority")
                Birth.parse(record.get("birth"))
                births[resource] = record["birth"]
            job["births"], job["status"], job["start_records"] = births, "running", records
            return job
        return self.store.commit(change)

    def settle_unstarted_job(self, job_id, attempt_id, generation, record):
        """An exact cancelled, never-started container has no ProcessBirth."""
        def change(control):
            job=control['jobs'][job_id]
            if (job.get('attempt_id')!=attempt_id or job.get('generation')!=generation
                    or job.get('kind')!='puffer-native-v1' or job['status']!='admitted' or job['births']
                    or job.get('cancel_consent')!={'attempt_id':attempt_id,'generation':generation}):
                raise Refused('unstarted settlement requires an exact cancelled admitted attempt')
            token={'operation_id':job['operation_id'],'generation':generation,'resources':job['resources']}
            require_owner(control,LeaseToken(job['operation_id'],generation,tuple(job['resources'])))
            container=record.get('container',{})
            if (record.get('state')!='never_started' or record.get('cancelled') is not True
                    or record.get('attempt_id')!=attempt_id or record.get('token')!=token
                    or record.get('command_digest')!=digest(job['command'])
                    or not isinstance(container.get('id'),str) or re.fullmatch('[0-9a-f]{64}',container['id']) is None
                    or container.get('image_id')!=job['command']['image_id'] or not container.get('created')
                    or set(container)!={'id','image_id','created','started','process_birth'}
                    or container['started'] is not None or container['process_birth'] is not None):
                raise Refused('unstarted settlement has unknown object or cancellation proof')
            job['status']='settled';job['blocker']='';job['outcomes']={r:copy.deepcopy(record) for r in job['resources']}
            return job
        return self.store.commit(change)

    def settle_absent_vision_job(self, job_id, attempt_id, generation, record):
        """A vision start that left no container has no birth. The host reported the exact name absent."""
        def change(control):
            job = control['jobs'][job_id]
            if (job.get('attempt_id') != attempt_id or job.get('generation') != generation
                    or job.get('kind') != 'vision' or job['status'] != 'admitted' or job['births']):
                raise Refused('absent settlement requires an exact admitted vision attempt with no birth')
            if (not isinstance(record, dict) or record.get('state') != 'never_created'
                    or record.get('attempt_id') != attempt_id
                    or record.get('container') != job['command'].get('container')):
                raise Refused('absent settlement has unknown object proof')
            job['status'], job['blocker'] = 'settled', ''
            job['outcomes'] = {r: copy.deepcopy(record) for r in job['resources']}
            return job
        return self.store.commit(change)

    def enqueue(self, entry):
        """A job that waits for room. It holds no resource."""
        def change(control):
            if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not entry.get('cmd') or not entry.get('image'):
                raise Refused('invalid queue entry')
            control.setdefault('queue', {})[entry['id']] = copy.deepcopy(entry)
            return entry
        return self.store.commit(change)

    def enqueue_all(self, entries):
        """The entries of one submit in one commit."""
        def change(control):
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not entry.get('cmd') or not entry.get('image'):
                    raise Refused('invalid queue entry')
            for entry in entries:
                control.setdefault('queue', {})[entry['id']] = copy.deepcopy(entry)
            return len(entries)
        return self.store.commit(change)

    def hold_queue(self, reason, now):
        """While a switch runs no waiting job starts. An empty reason ends the hold."""
        def change(control):
            if reason:
                control['queue_hold'] = {'reason': reason, 'since': now}
            else:
                control.pop('queue_hold', None)
        return self.store.commit(change)

    def mark_job(self, job_id, attempt_id, generation, key, value):
        """A router note on a live job (superseded: its copy is in the queue; restartable and its queue entry).
        It changes no lease and no status."""
        def change(control):
            job = control['jobs'][job_id]
            if job.get('attempt_id') != attempt_id or job.get('generation') != generation or key not in ('superseded', 'restartable', 'entry'):
                raise Refused('job note does not match the durable attempt')
            job['command'][key] = value
            return job
        return self.store.commit(change)

    def dequeue(self, entry_id, ended=None):
        """Take an entry out of the queue. A job record takes its place, or `ended` says what happened to it
        (cancelled by whom): the last ENDED_MAX such notes stay in `queue_ended`, so status can name the entry."""
        def change(control):
            entry = control.get('queue', {}).pop(entry_id, None)
            if entry and ended:
                notes = control.setdefault('queue_ended', {})
                notes[entry_id] = {'what': ended, 'by': entry.get('by'), 'cmd': entry.get('cmd'), 'queued_at': entry.get('queued_at')}
                for old in list(notes)[:-ENDED_MAX]:
                    del notes[old]
            return entry
        return self.store.commit(change)

    def claim_group(self, group, by, purpose, force=False):
        """Hold one layout group for a named holder. The claim lives on the operation that owns the layout."""
        def change(control):
            op = control['operations'].get(control['layout'].get('operation_id'))
            if group not in control['layout']['groups'] or not op:
                raise Refused('group is missing from the applied layout')
            require_owner(control, token_for(op))
            if not all(isinstance(v, str) and v for v in (by, purpose)):
                raise Refused('a claim needs a holder and a purpose')
            claims = op.setdefault('claims', {})
            held = claims.get(group)
            if held and held['by'] != by and not force:
                raise Refused(f"{group} is held by {held['by']} for '{held['for']}'")
            claims[group] = {'by': by, 'for': purpose}
            return claims[group]
        return self.store.commit(change)

    def unclaim_group(self, group, by, force=False):
        def change(control):
            op = control['operations'].get(control['layout'].get('operation_id')) or {}
            held = op.get('claims', {}).get(group)
            if held and held['by'] != by and not force:
                raise Refused(f"{group} is held by {held['by']}; add --force to take it")
            if any(j.get('kind') == 'vision' and j['status'] != 'settled' and j['command'].get('group') == group
                   for j in control['jobs'].values()):
                raise Refused(f'{group} has live jobs; cancel them first')
            if held:
                del op['claims'][group]
            return held
        return self.store.commit(change)

    def job_start_permission(self, job_id, attempt_id, generation):
        """Authorize a new send; outcome recording remains valid after cancel."""
        def change(control):
            job = control['jobs'][job_id]
            if (job.get('attempt_id') != attempt_id or job.get('generation') != generation
                    or job.get('status') != 'admitted' or job.get('cancel_consent')):
                raise Refused('job start is cancelled, settled or belongs to another attempt')
            op = control['operations'][job['operation_id']]
            require_owner(control, LeaseToken(job['operation_id'], job['generation'], tuple(job['resources'])))
            if op['policy'] != 'run' or op.get('retirement'):
                raise Refused('profile policy forbids a new job start')
            return job
        return self.store.commit(change)

    def cancel_job(self, job_id, attempt_id, generation, by=None):
        def change(control):
            job = control["jobs"][job_id]
            if job.get("attempt_id") != attempt_id or job.get("generation") != generation:
                raise Refused("cancellation consent does not match current job attempt")
            op = control["operations"][job["operation_id"]]
            require_owner(control, LeaseToken(job['operation_id'], job['generation'], tuple(job['resources'])))
            job["cancel_consent"] = {"attempt_id": attempt_id, "generation": generation}
            if by:
                job["cancelled_by"] = by
            job["blocker"] = "identity-bound physical termination must be verified before release"
            return job
        return self.store.commit(change)

    def record_job_blocker(self, job_id, attempt_id, generation, blocker):
        if not isinstance(blocker, str):
            raise Refused('job blocker must be text')
        def change(control):
            job = control['jobs'][job_id]
            if job.get('attempt_id') != attempt_id or job.get('generation') != generation:
                raise Refused('job blocker does not match the durable attempt')
            job['blocker'] = blocker
            return job
        return self.store.commit(change)

    def release(self, identifier):
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            from profileplan import sealed_authorities
            sealed_authorities(op)
            if op['spec'].get('installed_native'):
                from nativeinstalled import release_permission
                release_permission(op)
            if any(e["state"] == EffectState.UNCERTAIN for e in op["effects"]):
                raise Refused("uncertain effects retain resource ownership")
            if any(j.get("operation_id") == identifier and j.get("status") != "settled" for j in control["jobs"].values()):
                raise Refused("unsettled jobs retain resource ownership")
            for resource in op["resources"]:
                del control["leases"][resource]
            op["phase"], op["outcome"] = "finished", "released"
            if control['layout'].get('operation_id')==identifier:
                control['layout']={'spec':None,'groups':{},'operation_id':None,'generation':op['generation']}
                control['profile']=None
            return op
        return self.store.commit(change)

    def update_evidence(self, identifier, observations, blockers):
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            if op['spec'].get('installed_native') and 'native_origin' not in op:
                from nativeinstalled import origin
                op['native_origin']=origin(observations)
            op["observations"], op["blockers"] = observations, blockers
            return op
        return self.store.commit(change)

    def issue(self, identifier, kind, payload, prerequisites, *, birth=None):
        """Commit MayHaveApplied before any external action can be sent."""
        def change(control):
            try:
                effect_kind = EffectKind(kind)
            except ValueError as error:
                raise Refused("unsupported effect kind") from error
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            if op["policy"] != "run" or op["outcome"] is not None:
                raise Refused("operation does not permit new effects")
            if op["blockers"] or any(e["state"] == EffectState.UNCERTAIN for e in op["effects"]):
                raise Refused("unresolved operation evidence or effect")
            known = {e["id"] for e in op["effects"] if e["state"] == EffectState.VERIFIED}
            if not set(prerequisites) <= known:
                raise Refused("effect prerequisite is not verified")
            from profileplan import permission
            permission(op, effect_kind, payload)
            if effect_kind == EffectKind.HOST_STOP:
                Birth.parse(birth)
            if effect_kind in DISRUPTIVE:
                if any(job.get('status')!='settled' and set(job.get('resources',[]))&set(op['resources']) for job in control['jobs'].values()):raise Refused('unsettled controller attempts retain their execution resources')
                from profileplan import sealed_authorities
                sealed_authorities(op)
            if effect_kind==EffectKind.STUDIO_EFFECT and any(job.get('status')!='settled'
                    and set(job.get('resources',[]))&set(payload['request'].get('resources',op['resources'])) for job in control['jobs'].values()):raise Refused('Studio settings/unload must wait for controller attempts')
            effect = {"id": identifier + ":" + str(len(op["effects"])), "kind": kind, "payload": payload,
                      "token": token_for(op).document(), "birth": birth, "prerequisites": list(prerequisites),
                      "state": EffectState.UNCERTAIN, "outcome": None}
            if effect_kind == EffectKind.HOST_STOP:
                if payload.get("family") == "spark-ds41f":
                    from familycontrol import validate_effect
                    validate_effect(effect, payload["resources"][0])
                else:
                    from nativecontrol import validate_member
                    validate_member(effect, payload["resources"][0])
            op["effects"].append(effect)
            op["phase"] = "executing"
            return effect
        return self.store.commit(change)

    def outcome(self, identifier, effect_id, evidence):
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            effect = next((e for e in op["effects"] if e["id"] == effect_id), None)
            if effect is None:
                raise Refused("unknown effect")
            if (not isinstance(evidence, dict) or evidence.get("effect_id") != effect_id
                    or evidence.get("token") != effect["token"] or evidence.get("verified") is not True):
                raise Refused("unknown or mismatched effect outcome")
            if effect["state"] == EffectState.VERIFIED and effect["outcome"] != evidence:
                raise Refused("effect outcome changed")
            if effect.get('payload',{}).get('family')=='spark-ds41f':
                from familycontrol import validate_outcome
                validate_outcome(effect,evidence)
            effect["state"], effect["outcome"] = EffectState.VERIFIED, evidence
            op["phase"] = "inspecting"
            return op
        return self.store.commit(change)

    def rejected(self, identifier, effect_id, evidence):
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            effect = next(e for e in op["effects"] if e["id"] == effect_id)
            if evidence.get("effect_id") != effect_id or evidence.get("token") != effect["token"] or evidence.get("rejected") is not True:
                raise Refused("effect rejection does not match issued intent")
            from studio_protocol import FenceReply, FenceMutation
            reply = FenceReply.model_validate(evidence.get("wire"))
            request = FenceMutation.model_validate(effect["payload"]["request"])
            if (reply.outcome != "blocked" or reply.operation_id != identifier or reply.generation != op["generation"]
                    or reply.idempotency_key != request.idempotency_key or reply.instance_id != request.expected.instance_id
                    or reply.birth != request.expected.birth):
                raise Refused("effect rejection is not a durable matching refusal")
            if request.effect is not None and not any(entry.effect_id == request.effect.effect_id
                    and entry.operation_id == identifier and entry.generation == op["generation"]
                    and entry.state == "Rejected" and entry.kind == request.effect.kind for entry in reply.effects):
                raise Refused("member effect outcome remains uncertain")
            effect["state"], effect["outcome"] = EffectState.REJECTED, evidence
            return op
        return self.store.commit(change)

    def cancel(self, identifier):
        def change(control):
            op = control["operations"][identifier]
            op["policy"] = "stop_after_inflight"
            op["phase"] = "waiting" if any(e["state"] == EffectState.UNCERTAIN for e in op["effects"]) else "finished"
            if op["phase"] == "finished":
                op["outcome"] = "cancelled"
                stock = (op["spec"].get("stock_studio") or op["spec"].get("lab_profile")) and all(e["state"] in (EffectState.VERIFIED, EffectState.REJECTED) for e in op["effects"])
                if (not op["effects"] or stock) and owns(control, token_for(op)):
                    # Stock Studio effects are settled absolute settings; physical GPU census still guards the next owner.
                    return_parent_leases(control, op)
                    if control["layout"].get("operation_id") == identifier:
                        control["layout"] = {"spec": None, "groups": {}, "operation_id": None, "generation": op["generation"]}
                        control["profile"] = None
            return op
        return self.store.commit(change)

    def publish(self, identifier, evidence):
        """Commit target allocation and verified outcomes; keep Finalizing until release."""
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            if not isinstance(evidence, dict):
                raise Refused("target readiness evidence is invalid")
            from profileplan import matched_authorities, sealed_authorities
            if evidence.get("adoption") is True:
                if op['spec'].get('recipe_family'):raise Refused('native generation adoption requires a verified guard handoff')
                matched_authorities(op)
            else:
                sealed_authorities(op)
            required = ("artifacts", "capabilities", "services", "settings")
            if op['spec'].get('recipe_family') and op.get('observations',{}).get('requirements',{}).get('native',{}).get('state')!='Sealed':
                raise Refused('native ingress must remain Sealed before target commit')
            if (op["policy"] != "run" or op["blockers"] or not isinstance(evidence, dict)
                    or evidence.get("spec_digest") != op["spec_digest"]
                    or any(not isinstance(evidence.get(k), dict) or evidence[k].get("state") != "verified" for k in required)
                    or any(evidence.get(k) != op["observations"].get("requirements", {}).get(k) for k in required)
                    or any(e["state"] not in (EffectState.VERIFIED, EffectState.REJECTED) for e in op["effects"])):
                raise Refused("target readiness evidence is incomplete")
            groups = op["spec"].get("groups", {})
            bindings = evidence.get("bindings", {})
            if (not isinstance(bindings, dict) or set(bindings) != set(resources(groups))
                    or any(not isinstance(v, dict) or not canonical(v.get("resource_id"))
                           or not v.get("instance_id") or not isinstance(v.get("birth"), dict)
                           or set(v["birth"]) != {"boot_id", "service_id"}
                           or not all(v["birth"].values()) for v in bindings.values())
                    or not {v["resource_id"] for v in bindings.values()} <= set(op["resources"])
                    or len({v["resource_id"] for v in bindings.values()}) != len(bindings)):
                raise Refused("target placement exceeds owned resources")
            control["layout"] = {"spec": op["spec"].get("layout"), "groups": groups,
                                 "operation_id": identifier, "generation": op["generation"]}
            control["profile"] = op["spec"].get("name")
            op["phase"], op["readiness"] = "finalizing", evidence
            return op
        return self.store.commit(change)

    def finish(self, identifier):
        def change(control):
            op = control["operations"][identifier]
            require_owner(control, token_for(op))
            if (op["policy"] != "run" or control["layout"].get("operation_id") != identifier or op["blockers"]
                    or not op.get("readiness") or any(e["state"] not in (EffectState.VERIFIED, EffectState.REJECTED) for e in op["effects"])):
                raise Refused("target publication is incomplete")
            if op['spec'].get('installed_native'):
                from nativeinstalled import finish_permission
                finish_permission(op)
                op['phase'], op['outcome'] = 'finished', 'ready'
                return op
            if op['spec'].get('lab_profile'):
                from labprofile import finish_permission as lab_finish
                lab_finish(op)
                op['phase'], op['outcome'] = 'finished', 'ready'
                return op
            if op['spec'].get('stock_studio'):
                from stockstudio import finish_permission as stock_finish
                stock_finish(op)
                op['phase'], op['outcome'] = 'finished', 'ready'
                return op
            from profileplan import parse_snapshot, retained_resources
            if op['spec'].get('recipe_family'):
                native=op.get('observations',{}).get('requirements',{}).get('native')
                if not native or native['state']!='Open' or native['owner'] is not None:
                    raise Refused('native admission release has not been verified')
            retained = retained_resources(op)
            entries = op["observations"].get("authorities", {})
            if not entries or any(entry.get("state") != "fresh" for entry in entries.values()):
                raise Refused("admission release is unknown")
            for entry in entries.values():
                snapshot = parse_snapshot(entry["snapshot"])
                keep = retained & set(entry["resources"])
                owned = [f for f in snapshot.fences if f.operation_id == identifier]
                if keep:
                    if len(owned) != 1 or owned[0].generation != op["generation"] or owned[0].state != "sealed" or set(owned[0].resources) != keep:
                        raise Refused("reserved GPU admission fence is incomplete")
                elif owned:
                    raise Refused("admission fence is still held")
            op["phase"], op["outcome"] = "finished", "ready"
            return op
        return self.store.commit(change)

    def view(self, identifier):
        control = self.store.read()
        if identifier not in control["operations"]:
            raise Refused("unknown operation")
        op = copy.deepcopy(control["operations"][identifier])
        op["authority_revision"] = control["revision"]
        op["ownership_current"] = owns(control, token_for(op))
        op["historical"] = op["outcome"] == "ready"
        op["allocation_current"] = control["layout"].get("operation_id") == identifier
        return op

    def public_view(self,identifier):
        return public_projection(self.view(identifier))
