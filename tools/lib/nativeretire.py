"""Retire native members under their existing full-footprint owner."""
from __future__ import annotations

import copy

from clustercontrol import ADAPTER, EffectKind, EffectState, Refused, token_for
from profileplan import parse_snapshot, sealed_authorities
from profile_recipes.spark_ds41f_control import parse_snapshot as native_snapshot


def _waiting(controller, identifier, reason, effect=None):
    blocker = {"resource": "*", "reason": str(reason)}
    if effect is not None:
        blocker["effect_id"] = effect["id"]
    return controller.block(identifier, [blocker])


def _physical_rows(op):
    token = token_for(op).document()
    rows = []
    for effect in op["effects"]:
        if effect.get("payload", {}).get("family") != "spark-ds41f" or effect["state"] != EffectState.VERIFIED:
            continue
        if effect["token"] != token or not effect.get("outcome", {}).get("verified"):
            raise Refused("retirement receipt belongs to another resource generation")
        rows.append(effect)
    containers, guards = {}, []
    for effect in rows:
        payload, physical = effect["payload"], effect["outcome"]["physical"]
        if payload["action"] == "start":
            binding = physical["container"]
            prior = containers.get(payload["member"])
            if prior is not None and prior != binding:
                raise Refused("multiple native process incarnations require explicit reconciliation")
            containers[payload["member"]] = binding
        elif payload["action"] == "guard-start":
            guards.append((payload["member"], payload["plan"]["binding"], payload["plan"]["guard"]["phase"], physical["guard_birth"]))
    def stopped(member, binding):
        return any(e["payload"]["action"] == "stop" and e["payload"]["member"] == member
                   and e["payload"]["plan"]["binding"] == binding for e in rows)
    def guard_stopped(member, guard_birth):
        return any(e["payload"]["action"] == "guard-stop" and e["payload"]["member"] == member
                   and e["payload"].get("guard_birth") == guard_birth for e in rows)
    return containers, [(m, b, phase, g) for m, b, phase, g in guards if not guard_stopped(m, g)], stopped


def _observe(controller, studio, op):
    previous = op.get("observations", {}).get("authorities", {})
    if not previous or not set(op["spec"]["recipe"]["members"]) <= set(previous):
        raise Refused("original Studio authority evidence is incomplete")
    fresh = {peer: studio.snapshot(peer, prior) for peer, prior in previous.items()}
    observations = copy.deepcopy(op["observations"])
    observations["authorities"] = fresh
    return controller.update_evidence(op["id"], observations, [])


def retire(controller, native_adapter, studio_adapter, op, source_identity):
    """Drain, stop exact receipts, then release Studio fences. Never transfer owner."""
    identifier = op["id"]
    for _ in range(32):
        op = controller.view(identifier)
        if op.get("source_identity") != source_identity or op.get("adapter") != ADAPTER:
            return _waiting(controller, identifier, "source or adapter changed; native retirement blocked")
        if not op.get("retirement"):
            return _waiting(controller, identifier, "begin_retire must retain the original operation and generation")
        if op.get("outcome") in ("released", "retired"):
            return op
        uncertain = None
        try:
            op = _observe(controller, studio_adapter, op)
            uncertain = next((e for e in op["effects"] if e["state"] == EffectState.UNCERTAIN), None)
            if uncertain:
                if uncertain["kind"] in (EffectKind.HOST_START, EffectKind.HOST_STOP, EffectKind.NATIVE_FENCE):
                    result = native_adapter.execute(uncertain, op["observations"]["authorities"])
                elif uncertain["kind"] in (EffectKind.FENCE_ACQUIRE, EffectKind.FENCE_SEAL, EffectKind.FENCE_RELEASE, EffectKind.STUDIO_EFFECT):
                    result = studio_adapter.execute(uncertain)
                else:
                    return _waiting(controller, identifier, "unknown retirement effect cannot be reconciled", uncertain)
                if result.get("rejected"):
                    controller.rejected(identifier, uncertain["id"], result)
                    return _waiting(controller, identifier, "retirement effect was refused; fresh review required", uncertain)
                controller.outcome(identifier, uncertain["id"], result)
                continue
            containers, guards, stopped = _physical_rows(op)
            recipe = op["spec"]["recipe"]
            members = recipe["members"]
            if any(member not in members for member in containers) or any(g[0] not in members for g in guards):
                raise Refused("native receipt placement differs from the original recipe")
            pending = [(member, binding) for member, binding in containers.items() if not stopped(member, binding)]
            if pending or guards:
                # Every new native effect remains under the old complete Studio
                # fence, even when only a partial cold worker set was started.
                sealed_authorities(op)
                head = members[0]
                if head in containers and not stopped(head, containers[head]):
                    live_guard = next((g for g in reversed(guards) if g[0] == head and g[2] == "steady"), None)
                    if live_guard is not None:
                        physical = next(e["outcome"]["physical"] for e in reversed(op["effects"])
                                        if e["state"] == EffectState.VERIFIED and e.get("payload", {}).get("action") == "guard-start"
                                        and e["outcome"]["physical"].get("guard_birth") == live_guard[3])
                        native_adapter.transport.verify_running(op, head, containers[head], physical)
                    raw = native_adapter.transport.snapshot(op, containers[head])
                    snapshot = native_snapshot(raw, instance_id=op["id"], expected_birth=raw["birth"], footprint=op["resources"])
                    owner = token_for(op).document()
                    if snapshot["state"] == "Open" and snapshot["owner"] is None:
                        result = native_adapter.barrier(op, "acquire", snapshot)
                    elif snapshot["state"] == "Draining" and snapshot["owner"] == owner:
                        result = native_adapter.barrier(op, "drain", snapshot)
                    elif snapshot["state"] == "Sealed" and snapshot["owner"] == owner:
                        result = None
                    else:
                        raise Refused("native ingress ownership or drain outcome is unknown")
                    if result is not None:
                        if result.get("blockers"):
                            return result
                        continue
                # Head first, then workers in reverse start order. Retire every
                # startup/steady guard by its own exact receipt before its member.
                order=members if recipe['engine']=='tensorfold-mia-tp4' else [members[0], *reversed(members[1:])]
                for member in order:
                    guard = next((g for g in guards if g[0] == member), None)
                    if guard is not None:
                        _, binding, phase, guard_birth = guard
                        result = native_adapter.member(op, "guard-stop", member, container=binding, phase=phase, guard_birth=guard_birth)
                        if result.get("blockers"):
                            return result
                        break
                    if member in containers and not stopped(member, containers[member]):
                        result = native_adapter.member(op, "stop", member, container=containers[member])
                        if result.get("blockers"):
                            return result
                        break
                else:
                    raise Refused("native retirement made no exact member progress")
                continue
            if op.get("retirement_retain"):
                sealed_authorities(op)
                return controller.finish_retire(identifier)
            # No member or guard receipt remains live. Only now can original
            # Studio fences reopen; do not touch another operation's fence.
            for peer, entry in op["observations"]["authorities"].items():
                snapshot = parse_snapshot(entry["snapshot"])
                fence = next((f for f in snapshot.fences if f.operation_id == identifier), None)
                if fence is None:
                    continue
                if fence.generation != op["generation"]:
                    raise Refused("original Studio fence generation changed")
                payload = studio_adapter.payload(op, peer, release=True)
                effect = controller.issue(identifier, EffectKind.FENCE_RELEASE, payload,
                                          [e["id"] for e in op["effects"] if e["state"] == EffectState.VERIFIED])
                try:
                    result = studio_adapter.execute(effect)
                    if result.get("rejected"):
                        controller.rejected(identifier, effect["id"], result)
                        return _waiting(controller, identifier, "Studio refused retirement release", effect)
                    controller.outcome(identifier, effect["id"], result)
                except (Refused, OSError, ValueError, KeyError, TypeError) as error:
                    return _waiting(controller, identifier, error, effect)
                break
            else:
                return controller.finish_retire(identifier)
        except (Refused, OSError, ValueError, KeyError, TypeError) as error:
            return _waiting(controller, identifier, error, locals().get("uncertain"))
    return _waiting(controller, identifier, "native retirement step budget reached; resume the same operation")
