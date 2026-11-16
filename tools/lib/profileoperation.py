"""One-shot finite profile runner. Durable effects precede every external send."""
from __future__ import annotations

import copy

from clustercontrol import ADAPTER, Controller, EffectKind, EffectState, Refused, digest, resources
from profileplan import matched_authorities, parse_snapshot, retained_resources, sealed_authorities


class Driver:
    def __init__(self, controller: Controller, studio, requirements, source_identity, native=None):
        self.control, self.studio = controller, studio
        self.requirements, self.source_identity = requirements, source_identity
        self.native=native

    def observe(self, operation):
        previous = operation.get("observations", {}).get("authorities", {})
        authorities, bindings, blockers = {}, {}, []
        for node in resources(operation["spec"]["groups"]):
            try:
                entry = self.studio.snapshot(node, previous.get(node))
                authorities[node], bindings[node] = entry, entry["binding"]
            except (Refused, OSError, KeyError, ValueError) as error:
                blockers.append({"resource": node, "reason": str(error)})
        if len({entry["resource_id"] for entry in bindings.values()}) != len(bindings):
            blockers.append({"resource": "*", "reason": "ambiguous physical resource bindings"})
        return authorities, bindings, blockers

    def step(self, identifier):
        op = self.control.view(identifier)
        if op["source_identity"] != self.source_identity or op["adapter"] != ADAPTER:
            if op.get('outcome')=='ready':
                return self.control.record_health(identifier,{},[{'resource':'*','reason':'source or adapter changed; current health unknown'}])
            return self.control.block(identifier, [{"resource": "*", "reason": "source or adapter changed; recovery blocked"}])
        if op.get('outcome')=='ready':
            authorities,bindings,blockers=self.observe(op)
            checks={} if blockers else self.native.inspect(op,bindings,authorities) if op['spec'].get('recipe_family') and self.native else self.requirements(op['spec'],bindings,authorities)
            blockers+=checks.get('blockers',[])
            blockers += [{'resource':'*','reason':change} for change in checks.get('changes',[])]
            return self.control.record_health(identifier,{'authorities':authorities,'bindings':bindings,'requirements':checks},blockers)
        if op.get('retirement'):
            if not self.native:return self.control.block(identifier,[{'resource':'*','reason':'native retirement adapter unavailable'}])
            from nativeretire import retire
            return retire(self.control,self.native,self.studio,op,self.source_identity)
        if op["outcome"] in ("cancelled", "released", "prepared"):
            return op
        uncertain = next((e for e in op["effects"] if e["state"] == EffectState.UNCERTAIN), None)
        if uncertain:
            if self.native and uncertain['kind'] in (EffectKind.HOST_START,EffectKind.HOST_STOP,EffectKind.NATIVE_FENCE):
                try:
                    result=self.control.outcome(identifier,uncertain['id'],self.native.execute(uncertain,op['observations']['authorities']))
                    self.control.block(identifier,[])
                    return self.control.view(identifier)
                except (Refused,OSError,ValueError) as error:return self.control.block(identifier,[{'resource':'*','reason':str(error),'effect_id':uncertain['id']}])
            # The Studio protocol deduplicates this exact key/body. Never create
            # another effect after a lost reply. Host effects require their own
            # identity-bound journal query, not a second physical application.
            if uncertain["kind"] not in (EffectKind.FENCE_ACQUIRE, EffectKind.FENCE_SEAL, EffectKind.FENCE_RELEASE, EffectKind.STUDIO_EFFECT):
                return self.control.block(identifier, [{"resource": "*", "reason": "native effect outcome uncertain", "effect_id": uncertain["id"]}])
            try:
                result = self.studio.execute(uncertain)
                return self.record(identifier, uncertain, result)
            except (Refused, OSError, ValueError) as error:
                return self.control.block(identifier, [{"resource": "*", "reason": str(error), "effect_id": uncertain["id"]}])
        if op["policy"] != "run":
            return self.control.cancel(identifier)
        authorities, bindings, blockers = self.observe(op)
        if blockers:
            return self.control.block(identifier, blockers)
        if not op["resources"]:
            try:
                op = self.control.resolve_resources(identifier, bindings)
            except Refused as error:
                return self.control.block(identifier, [{"resource": "*", "reason": str(error)}])
        elif set(op["resources"]) != {entry["resource_id"] for entry in bindings.values()}:
            return self.control.block(identifier, [{"resource": "*", "reason": "physical placement changed; lease recovery required"}])
        if op['spec'].get('recipe_family') and not self.native:
            return self.control.block(identifier,[{'resource':'*','reason':'native family dispatcher is unavailable'}])
        requirements = self.native.inspect(op,bindings,authorities) if op['spec'].get('recipe_family') else self.requirements(op["spec"], bindings, authorities)
        blockers = list(requirements.get("blockers", []))
        observations = {"authorities": authorities, "requirements": requirements, "bindings": bindings}
        op = self.control.update_evidence(identifier, observations, blockers)
        if blockers:
            return op
        retained = retained_resources(op)
        if self.control.view(identifier)["allocation_current"]:
            if op['spec'].get('recipe_family'):
                result=self.native.step(op)
                if result is not None:return result
            for peer, entry in authorities.items():
                snapshot = parse_snapshot(entry["snapshot"])
                fence = next((f for f in snapshot.fences if f.operation_id == identifier and f.generation == op["generation"]), None)
                if fence:
                    keep = retained & set(entry["resources"])
                    if keep and set(fence.resources) != keep:
                        return self.fence(op, peer, "sealed", EffectKind.FENCE_SEAL, footprint=keep)
                    if not keep:
                        return self.fence(op, peer, None, EffectKind.FENCE_RELEASE)
            return self.control.finish(identifier)
        if op["intent"] == "prepare":
            return self.control.block(identifier, [{"resource": "*", "reason": "artifact publication adapter is not yet supported"}])
        lifecycle = any(e["kind"] not in (EffectKind.FENCE_ACQUIRE, EffectKind.FENCE_SEAL, EffectKind.FENCE_RELEASE) for e in op["effects"])
        if not op['spec'].get('recipe_family') and not requirements.get("changes") and not lifecycle:
            for peer, entry in authorities.items():
                keep = retained & set(entry["resources"])
                if not keep:
                    continue
                snapshot = parse_snapshot(entry["snapshot"])
                fence = next((f for f in snapshot.fences if f.operation_id == identifier and f.generation == op["generation"]
                              and keep <= set(f.resources)), None)
                if not fence:
                    return self.fence(op, peer, "draining", EffectKind.FENCE_ACQUIRE, footprint=keep)
                if fence.state != "sealed":
                    return self.fence(op, peer, "sealed", EffectKind.FENCE_SEAL, footprint=keep)
            try:
                matched_authorities(op)
            except Refused as error:
                return self.control.block(identifier, [{"resource": "*", "reason": str(error)}])
            readiness = {key: requirements.get(key) for key in ("artifacts", "capabilities", "services", "settings")}
            readiness.update(spec_digest=op["spec_digest"], bindings=bindings, adoption=True,
                             authorities={peer: entry["expected"] for peer, entry in authorities.items()})
            return self.control.publish(identifier, readiness)
        for peer, entry in authorities.items():
            affected = set(op["resources"]) & set(entry["resources"])
            if not affected:
                continue
            snapshot = parse_snapshot(entry["snapshot"])
            fence = next((f for f in snapshot.fences if f.operation_id == op["id"] and f.generation == op["generation"]
                          and affected <= set(f.resources)), None)
            if fence is None:
                return self.fence(op, peer, "draining", EffectKind.FENCE_ACQUIRE)
            if fence.state != "sealed":
                flags = op["spec"].get("studio") or {}
                if op['spec'].get('recipe_family') or flags and peer != "spark3" and flags.get("workers", {}).get(peer) is False:
                    warm = next((activity for activity in snapshot.activities if activity.kind in ("comfy", "resident:comfy") and activity.runtime_birth is not None
                                 and activity.state != "settled"), None)
                    if warm:
                        if any(activity.state in ("running", "uncertain") and not activity.kind.startswith("resident:")
                               for activity in snapshot.activities) or any(attempt.state != "settled" for attempt in snapshot.attempts):
                            return self.control.block(identifier, [{"resource": entry["binding"]["resource_id"],
                                "reason": "existing native work must settle before warm unload"}])
                        action = {"effect_id": op["id"] + ":unload:" + peer + ":" + digest(warm.runtime_birth.model_dump())[:16],
                                  "kind": "unload_comfy", "expected_engine_birth": warm.runtime_birth.model_dump()}
                        return self.fence(op, peer, "draining", EffectKind.STUDIO_EFFECT, action)
                return self.fence(op, peer, "sealed", EffectKind.FENCE_SEAL)
        try:
            sealed_authorities(op)
        except Refused as error:
            return self.control.block(identifier, [{"resource": "*", "reason": str(error)}])
        # The supported adoption path preserves already running native services.
        # A required lifecycle change is explicit and cannot fall through to READY.
        if op['spec'].get('recipe_family'):
            result=self.native.step(op)
            if result is not None:return result
        elif requirements.get("changes"):
            action = requirements.get("settings_effect")
            if action:
                action = dict(action, effect_id=op["id"] + ":settings:" + digest(action)[:16])
                if any(e["kind"] == EffectKind.STUDIO_EFFECT and e.get("payload", {}).get("request", {}).get("effect", {}).get("effect_id") == action["effect_id"]
                       and e["state"] == EffectState.VERIFIED for e in op["effects"]):
                    return self.control.block(identifier, [{"resource": "*", "reason": "verified settings outcome differs from current observation"}])
                return self.fence(op, "spark3", "sealed", EffectKind.STUDIO_EFFECT, action)
            return self.control.block(identifier, [{"resource": "*", "reason": "native lifecycle/settings change requires identity-bound member adapter", "changes": requirements["changes"]}])
        readiness = {key: requirements.get(key) for key in ("artifacts", "capabilities", "services", "settings")}
        readiness.update(spec_digest=op["spec_digest"], bindings=bindings,
                         authorities={peer: entry["expected"] for peer, entry in authorities.items()})
        return self.control.publish(identifier, readiness)

    def fence(self, operation, peer, state, kind, member_effect=None, footprint=None):
        transfer=None
        if kind==EffectKind.FENCE_ACQUIRE and operation.get('parent_authority'):
            parent=operation['parent_authority']
            prior=next((f for f in operation['observations']['authorities'][peer]['snapshot']['fences'] if f['operation_id']==parent['operation_id']),None)
            if prior:
                transfer={key:prior[key] for key in ('operation_id','generation','resources')}
        payload = self.studio.payload(operation, peer, state, release=kind == EffectKind.FENCE_RELEASE, effect=member_effect, footprint=footprint,transfer=transfer)
        prerequisites = [e["id"] for e in operation["effects"] if e["state"] == EffectState.VERIFIED]
        effect = self.control.issue(operation["id"], kind, payload, prerequisites)
        try:
            result = self.studio.execute(effect)
            return self.record(operation["id"], effect, result)
        except (Refused, OSError, ValueError) as error:
            return self.control.block(operation["id"], [{"resource": "*", "reason": str(error), "effect_id": effect["id"]}])

    def record(self, identifier, effect, result):
        if result.get("rejected"):
            self.control.rejected(identifier, effect["id"], result)
            return self.control.block(identifier, [{"resource": "*", "reason": "Studio refused current effect; fresh evidence is required", "effect_id": effect["id"]}])
        return self.control.outcome(identifier, effect["id"], result)

    def drive(self, identifier, maximum_steps=32):
        for _ in range(maximum_steps):
            before = self.control.view(identifier)
            after = self.step(identifier)
            if after.get("outcome") is not None or after.get("blockers") or after == before:
                return after
        return self.control.block(identifier, [{"resource": "*", "reason": "local step budget reached; resume operation"}])
