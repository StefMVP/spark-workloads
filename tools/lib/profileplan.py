"""Pure permission checks over current identity-bound observations."""
from __future__ import annotations

from clustercontrol import Refused, canonical

ACTIVITY_CLASSES = frozenset(("queue", "dispatch", "handed", "bolt_turn", "bolt_request", "bolt_load",
                              "voice", "music", "comfy", "motion", "captions", "train", "local_worker"))


def parse_snapshot(raw):
    from pydantic import ValidationError
    from studio_protocol import Snapshot
    try:
        snapshot = Snapshot.model_validate(raw)
    except (ValidationError, ValueError, TypeError) as error:
        raise Refused("unknown: malformed Studio control snapshot") from error
    binding = snapshot.self_resource
    if (binding is None or binding.instance_id != snapshot.instance_id or binding.birth != snapshot.birth
            or not canonical(binding.resource_id)):
        raise Refused("unknown: physical Studio identity is not verified")
    return snapshot


def binding(snapshot):
    return snapshot.self_resource.model_dump()


COORDINATOR_CLASSES = frozenset(("queue", "dispatch", "handed", "local_worker"))


def coverage(snapshot, affected):
    relevant = [entry for entry in snapshot.coverage if entry.resource in affected or entry.resource == "*"]
    if any(entry.status != "supported" for entry in relevant):
        raise Refused("unsupported Studio admission coverage: " + "; ".join(entry.reason for entry in relevant if entry.status != "supported"))
    covered = {(entry.resource, entry.activity_class) for entry in relevant if entry.status == "supported"}
    missing = {(resource, kind) for resource in affected
               for kind in (ACTIVITY_CLASSES if resource == snapshot.self_resource.resource_id else COORDINATOR_CLASSES)} - covered
    if missing:
        raise Refused("unknown: incomplete Studio admission coverage")
    if snapshot.blockers:
        raise Refused("Studio authority blockers: " + "; ".join(snapshot.blockers))


def sealed_authorities(operation):
    if operation.get('spec',{}).get('installed_native'):
        from nativeinstalled import sealed
        return sealed(operation)
    if operation.get('spec',{}).get('stock_studio'):
        from stockstudio import sealed as stock_sealed
        return stock_sealed(operation)
    if operation.get('spec',{}).get('lab_profile'):
        from labprofile import sealed as lab_sealed
        return lab_sealed(operation)
    evidence = operation.get("observations", {})
    authorities = evidence.get("authorities")
    if not isinstance(authorities, dict) or not authorities:
        raise Refused("owned sealed admission evidence required")
    covered = set()
    for entry in authorities.values():
        if not isinstance(entry, dict) or entry.get("state") != "fresh":
            raise Refused("stale or unknown admission evidence")
        snapshot = parse_snapshot(entry.get("snapshot"))
        expected = entry.get("expected")
        identity = {"instance_id": snapshot.instance_id, "birth": snapshot.birth.model_dump(), "revision": snapshot.revision}
        if identity != expected:
            raise Refused("admission identity, birth or revision changed")
        affected = set(operation["resources"]) & set(entry.get("resources", []))
        if not affected:
            continue
        coverage(snapshot, affected)
        fence = next((f for f in snapshot.fences if f.operation_id == operation["id"]
                      and f.generation == operation["generation"] and f.state == "sealed"
                      and affected <= set(f.resources)), None)
        if fence is None:
            raise Refused("owned sealed fence does not cover effect footprint")
        for activity in snapshot.activities:
            overlaps = "*" in activity.resources or bool(affected & set(activity.resources))
            if overlaps and activity.state not in ("queued", "settled"):
                raise Refused("unsettled affected activity")
        for attempt in snapshot.attempts:
            if ("*" in attempt.resources or affected & set(attempt.resources)) and attempt.state != "settled":
                raise Refused("unsettled affected delivery")
        if snapshot.self_resource.resource_id in affected:
            covered.add(snapshot.self_resource.resource_id)
    if covered != set(operation["resources"]):
        raise Refused("not every affected authority is sealed")
    return authorities


def matched_authorities(operation):
    """Adopt unchanged services without draining their known active work."""
    observations = operation.get("observations", {})
    requirements = observations.get("requirements", {})
    if requirements.get("changes") or requirements.get("blockers"):
        raise Refused("target runtime differs from observed placement")
    if any(effect["kind"] not in ("fence_acquire", "fence_seal", "fence_release") for effect in operation["effects"]):
        raise Refused("adoption contains a lifecycle effect")
    covered = set()
    authorities = observations.get("authorities", {})
    for entry in authorities.values():
        if entry.get("state") != "fresh":
            raise Refused("adoption requires current identity-bound observations")
        snapshot = parse_snapshot(entry["snapshot"])
        identity = {"instance_id": snapshot.instance_id, "birth": snapshot.birth.model_dump(), "revision": snapshot.revision}
        if entry["expected"] != identity:
            raise Refused("adoption authority identity changed")
        affected = set(operation["resources"]) & set(entry["resources"])
        coverage(snapshot, affected)
        if any(activity.state == "uncertain" or "*" in activity.resources for activity in snapshot.activities):
            raise Refused("active workload placement or outcome is unknown")
        if any(attempt.state == "uncertain" or "*" in attempt.resources for attempt in snapshot.attempts):
            raise Refused("delivery placement or outcome is unknown")
        if snapshot.self_resource.resource_id in affected:
            covered.add(snapshot.self_resource.resource_id)
    if covered != set(operation["resources"]):
        raise Refused("target service identities do not cover allocation")
    return authorities


def permission(operation, kind, payload):
    """Validate the next finite effect against observed facts, never a stage."""
    from clustercontrol import EffectKind
    effect_kind = EffectKind(kind)
    if operation.get('spec',{}).get('installed_native') and effect_kind != EffectKind.NATIVE_RUNTIME:
        raise Refused('installed native runtimes require their finite member adapter')
    if operation.get('spec',{}).get('lab_profile'):
        if effect_kind != EffectKind.STUDIO_EFFECT:raise Refused('a lab profile permits only the Studio head stop')
        from labprofile import permission as lab_permission
        lab_permission(operation, payload)
        return
    if operation.get('spec',{}).get('stock_studio'):
        if effect_kind != EffectKind.STUDIO_EFFECT:raise Refused('stock Studio permits only saved Studio settings')
        from stockstudio import permission as stock_permission
        stock_permission(operation, payload)
        return
    if effect_kind == EffectKind.NATIVE_RUNTIME:
        from nativeinstalled import permission as native_permission
        native_permission(operation, payload)
        return
    if operation.get('retirement'):
        cleanup=(effect_kind==EffectKind.HOST_STOP and payload.get('family')=='spark-ds41f' and payload.get('action') in ('stop','guard-stop'))
        ingress=(effect_kind==EffectKind.NATIVE_FENCE and payload.get('request',{}).get('action') in ('acquire','drain'))
        release=effect_kind==EffectKind.FENCE_RELEASE
        if not (cleanup or ingress or release):raise Refused('retirement permits only exact stop, ingress drain and final barrier release')
        if release:
            from familycontrol import retired_members
            if not retired_members(operation):raise Refused('native members/guards must retire before Studio barrier release')
    if effect_kind==EffectKind.NATIVE_FENCE:
        from profile_recipes.spark_ds41f_control import parse_request
        if set(payload)!={'adapter','request'} or payload['adapter']!='spark-ds41f-ingress-v1':raise Refused('invalid native fence effect')
        request=parse_request(payload['request'])
        current=operation.get('observations',{}).get('requirements',{}).get('native')
        if (not current or request['instance_id']!=current['instance_id'] or request['birth']!=current['birth']
            or request['expected_revision']!=current['revision'] or request['operation_id']!=operation['id']
            or request['generation']!=operation['generation'] or request['resources']!=operation['resources']):raise Refused('native fence identity/owner/CAS differs')
        if request['action']=='acquire' and operation.get('retirement'):
            if current['state']!='Open' or current['owner'] is not None:raise Refused('native retirement requires open unowned ingress')
        elif current['owner']!={'operation_id':operation['id'],'generation':operation['generation'],'resources':operation['resources']}:
            raise Refused('native bootstrap owner differs')
        if request['action']=='release' and not operation.get('readiness'):raise Refused('native release precedes target commit')
        if request['action'] not in ('drain','release') and not (request['action']=='acquire' and operation.get('retirement')):raise Refused('native bootstrap already owns ingress')
        sealed_authorities(operation)
        return
    if effect_kind == EffectKind.ARTIFACT_PREPARE:
        raise Refused("artifact preparation requires a verified publication adapter")
    if effect_kind in (EffectKind.FENCE_ACQUIRE, EffectKind.FENCE_SEAL, EffectKind.FENCE_RELEASE, EffectKind.STUDIO_EFFECT):
        if set(payload) != {"authority", "request", "method"}:
            raise Refused("invalid fence effect")
        from studio_protocol import FenceMutation
        request = FenceMutation.model_validate(payload["request"])
        if (request.operation_id != operation["id"] or request.generation != operation["generation"]
                or not set(request.resources) <= set(operation["resources"])):
            raise Refused("fence effect does not match operation")
        entry = operation.get("observations", {}).get("authorities", {}).get(payload["authority"])
        if not entry or entry.get("state") != "fresh" or request.expected.model_dump() != entry.get("expected"):
            raise Refused("fence expected identity differs from observation")
        if effect_kind == EffectKind.FENCE_SEAL:
            snapshot = parse_snapshot(entry["snapshot"])
            if not any(f.operation_id == operation["id"] and f.generation == operation["generation"]
                       and set(request.resources) <= set(f.resources) for f in snapshot.fences):
                raise Refused("seal has no owned draining fence")
        if request.transfer is not None:
            old=request.transfer;parent=operation.get('parent_authority')
            snapshot=parse_snapshot(entry['snapshot'])
            if (effect_kind!=EffectKind.FENCE_ACQUIRE or request.state!='draining' or not parent
                or old.operation_id!=parent['operation_id'] or old.generation!=parent['generation']
                or not set(old.resources)<=set(parent['resources'])
                or not any(f.operation_id==old.operation_id and f.generation==old.generation and f.state=='sealed'
                           and f.resources==old.resources for f in snapshot.fences)
                or any(e.state=='MayHaveApplied' and e.operation_id==old.operation_id for e in snapshot.effects)):
                raise Refused('transfer lacks the exact settled sealed parent grant')
        if effect_kind == EffectKind.FENCE_RELEASE and not operation.get("readiness") and operation.get("policy") == "run" and not operation.get('retirement'):
            raise Refused("cannot reopen admission before target commit")
        if effect_kind == EffectKind.STUDIO_EFFECT:
            effect = request.effect
            if effect is None or request.state not in ("draining", "sealed"):
                raise Refused("Studio member effect requires a typed action and owned fence")
            snapshot = parse_snapshot(entry["snapshot"])
            if not any(f.operation_id == operation["id"] and f.generation == operation["generation"]
                       and set(request.resources) <= set(f.resources) for f in snapshot.fences):
                raise Refused("Studio member effect has no owned complete fence")
            if effect.kind == "unload_comfy":
                if request.state != "draining" or not any(activity.runtime_birth == effect.expected_engine_birth
                        and activity.kind in ("comfy", "resident:comfy") for activity in snapshot.activities):
                    raise Refused("warm engine physical birth is unknown")
            else:
                sealed_authorities(operation)
            required = {worker.resource_id for worker in effect.workers}
            required.add(entry["binding"]["resource_id"])
            if effect.sparks is not None:
                required.update(worker.resource_id for worker in snapshot.worker_bindings)
            if not required <= set(request.resources):
                raise Refused("Studio effect request omits actual resource footprint")
    elif effect_kind != EffectKind.ARTIFACT_PREPARE:
        sealed_authorities(operation)
    if effect_kind in (EffectKind.HOST_START, EffectKind.HOST_STOP, EffectKind.LAYOUT, EffectKind.SETTINGS):
        if isinstance(payload, dict) and payload.get("family") == "spark-ds41f":
            from familycontrol import validate_effect,prerequisite
            planned={"id": operation["id"]+":plan", "kind":kind,"payload":payload,
                "token":{"operation_id":operation["id"],"generation":operation["generation"],"resources":operation["resources"]},
                "birth":payload.get('guard_birth') if payload.get('action')=='guard-stop' else payload.get("plan",{}).get("binding",{}).get("process_birth")}
            validate_effect(planned,operation["resources"][0])
            prerequisite(planned,operation['effects'])
            return
        if not isinstance(payload, dict) or payload.get("adapter") != "native-control-v1":
            raise Refused("effect requires a validated native member plan")
        if not set(payload.get("resources", [])) <= set(operation["resources"]) or not payload.get("resources"):
            raise Refused("native member plan has unknown footprint")
        if payload.get("action") not in ("process_stop", "container_stop"):
            raise Refused("native member effect is unsupported")
        # Native starts remain unsupported until their source-backed adapter can
        # validate every nested member launch and the engine's request ingress.
        if effect_kind == EffectKind.HOST_START:
            raise Refused("native start lacks verified member and ingress coverage")


def retained_resources(operation):
    """Disabled Studio workers stay fenced while another workload owns the GPU."""
    if operation['spec'].get('recipe_family'):return set(operation['resources'])
    workers = (operation["spec"].get("studio") or {}).get("workers", {})
    bindings = operation.get("observations", {}).get("bindings", {})
    return {bindings[node]["resource_id"] for node, enabled in workers.items() if enabled is False and node in bindings}
