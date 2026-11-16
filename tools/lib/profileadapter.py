"""Studio's validated wire contract over the existing native HTTP codec."""
from __future__ import annotations

import copy

from clustercontrol import EffectKind, Refused, digest
from profileplan import binding, parse_snapshot
from studio_protocol import ControlEffect, FenceMutation, FenceReply


class StudioAdapter:
    def __init__(self, runtime, peers, clock, maximum_age=15):
        self.runtime, self.peers, self.clock = runtime, peers, clock
        self.maximum_age = maximum_age

    def snapshot(self, peer, prior=None):
        node, port = self.peers[peer]
        snapshot = parse_snapshot(self.runtime.api(node, port, "/api/control/snapshot"))
        identity = {"instance_id": snapshot.instance_id, "birth": snapshot.birth.model_dump(), "revision": snapshot.revision}
        if prior:
            expected = prior["expected"]
            if (identity["instance_id"] != expected["instance_id"] or identity["birth"] != expected["birth"]
                    or identity["revision"] < expected["revision"]):
                raise Refused("stale or changed Studio authority identity")
        return {"state": "fresh", "snapshot": snapshot.model_dump(by_alias=True), "expected": identity,
                "observed_at": self.clock(), "binding": binding(snapshot),
                "resources": sorted({entry.resource for entry in snapshot.coverage if entry.resource.startswith("gpu:GPU-")} | {snapshot.self_resource.resource_id})}

    def payload(self, operation, peer, state=None, release=False, effect=None, footprint=None,transfer=None):
        entry = operation["observations"]["authorities"][peer]
        if self.clock() - entry["observed_at"] > self.maximum_age:
            raise Refused("stale admission snapshot")
        affected = sorted(set(operation["resources"]) & set(entry["resources"]))
        if release:
            fence = next((row for row in entry["snapshot"]["fences"] if row["operation_id"] == operation["id"]
                          and row["generation"] == operation["generation"]), None)
            if not fence:
                raise Refused("owned fence is unknown")
            affected = fence["resources"]
        if footprint is not None:
            if not set(footprint) <= set(affected) or not footprint:
                raise Refused("fence subset is outside authority footprint")
            affected = sorted(footprint)
        if not affected:
            raise Refused("authority has no verified affected resource")
        request = {"schema": 1, "expected": entry["expected"], "operation_id": operation["id"],
                   "generation": operation["generation"], "resources": affected, "state": state}
        if transfer is not None:request['transfer']=copy.deepcopy(transfer)
        if effect is not None:
            parsed = ControlEffect.model_validate(effect)
            request["effect"] = parsed.model_dump()
            effect_resources = {worker.resource_id for worker in parsed.workers}
            effect_resources.add(entry["binding"]["resource_id"])
            if parsed.sparks is not None:
                effect_resources.update(worker["resource_id"] for worker in entry["snapshot"].get("worker_bindings", []))
            if not effect_resources <= set(operation["resources"]):
                raise Refused("Studio effect footprint exceeds owned resources")
            request["resources"] = sorted(set(affected) | effect_resources)
        request["idempotency_key"] = digest({"request": request, "release": release})
        FenceMutation.model_validate(request)
        return {"authority": peer, "request": request, "method": "DELETE" if release else "PUT"}

    def execute(self, effect):
        payload = effect["payload"]
        request = FenceMutation.model_validate(payload["request"])
        node, port = self.peers[payload["authority"]]
        raw = self.runtime.api(node, port, "/api/control/fences/" + request.operation_id,
                               payload["request"], payload["method"])
        return self.accept_reply(effect, raw)

    def accept_reply(self, effect, raw):
        from pydantic import ValidationError
        try:
            reply = FenceReply.model_validate(raw)
            parse_snapshot(raw_snapshot(reply))
        except (ValidationError, Refused, ValueError, TypeError) as error:
            raise Refused("unknown: malformed or unavailable fence outcome") from error
        request = FenceMutation.model_validate(effect["payload"]["request"])
        expected = request.expected
        if (reply.operation_id != request.operation_id or reply.generation != request.generation
                or reply.idempotency_key != request.idempotency_key
                or reply.instance_id != expected.instance_id or reply.birth != expected.birth
                or reply.revision <= expected.revision):
            raise Refused("unknown: fence reply identity, generation or revision mismatch")
        if reply.outcome == "blocked":
            # A validated durable refusal is different from a lost reply. It
            # permits a new plan after fresh observation, never a blind retry.
            if request.effect is not None and not any(entry.effect_id == request.effect.effect_id
                    and entry.operation_id == request.operation_id and entry.generation == request.generation
                    and entry.kind == request.effect.kind and entry.state == "Rejected"
                    and entry.expected_engine_birth == request.effect.expected_engine_birth for entry in reply.effects):
                raise Refused("Studio member effect may have applied; refusal is not termination evidence")
            return {"effect_id": effect["id"], "token": copy.deepcopy(effect["token"]), "verified": False,
                    "rejected": True, "authority": effect["payload"]["authority"], "wire": reply.model_dump(by_alias=True)}
        outcome = "released" if effect["payload"]["method"] == "DELETE" else request.state
        if reply.outcome != outcome:
            raise Refused("pending: Studio has not acknowledged " + str(outcome))
        if outcome != "released" and not any(f.operation_id == request.operation_id and f.generation == request.generation
                and f.state == request.state and set(request.resources) <= set(f.resources) for f in reply.fences):
            raise Refused("unknown: reply omits complete owned fence")
        if request.transfer is not None:
            old=request.transfer
            successor={'operation_id':request.operation_id,'generation':request.generation,'resources':request.resources}
            fence=next(f for f in reply.fences if f.operation_id==request.operation_id)
            if (fence.protected_resources!=old.resources or any(f.operation_id==old.operation_id for f in reply.fences)
                or not any(row.generation==old.generation and row.resources==old.resources
                           and row.successor is not None and row.successor.model_dump()==successor
                           for row in reply.ownership_history.get(old.operation_id,[]))):
                raise Refused('transfer reply omits exact protected predecessor/successor history')
        if request.effect is not None:
            effect_evidence = next((entry for entry in reply.effects if entry.effect_id == request.effect.effect_id), None)
            if (effect_evidence is None or effect_evidence.operation_id != request.operation_id
                    or effect_evidence.generation != request.generation or effect_evidence.kind != request.effect.kind
                    or effect_evidence.expected_engine_birth != request.effect.expected_engine_birth
                    or set(effect_evidence.resources) != set(request.resources)
                    or effect_evidence.state != "Verified"):
                raise Refused("Studio member effect outcome remains unknown")
        return {"effect_id": effect["id"], "token": copy.deepcopy(effect["token"]), "verified": True,
                "authority": effect["payload"]["authority"], "wire": reply.model_dump(by_alias=True)}


def raw_snapshot(reply):
    result = reply.model_dump(by_alias=True)
    for field in ("operation_id", "generation", "idempotency_key", "outcome"):
        result.pop(field)
    return result
