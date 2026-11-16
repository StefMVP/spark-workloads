"""Short-lived native member authority. No daemon and no raw recipe execution."""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess

from clustercontrol import ADAPTER, Birth, EffectKind, EffectState, Refused, canonical, digest, durable


def member_plan(action, resource, birth, target):
    if action not in ("process_stop", "container_stop") or not canonical(resource):
        raise Refused("unsupported native member action")
    parsed = Birth.parse(birth)
    if action == "container_stop":
        import re
        if parsed.kind != "container" or re.fullmatch(r"[0-9a-f]{64}", parsed.identity) is None or target != parsed.identity:
            raise Refused("container stop requires immutable full container ID")
        command = ["docker", "stop", "--time", "30", parsed.identity]
    else:
        if parsed.kind != "process" or not parsed.identity.isdigit() or int(parsed.identity) <= 1 or target != parsed.identity:
            raise Refused("process stop requires a complete process birth")
        command = ["pidfd_send_signal", parsed.identity, parsed.boot_id, parsed.started, "SIGTERM"]
    return {"adapter": ADAPTER, "action": action, "resources": [resource], "target": target,
            "birth": birth, "command": command, "command_digest": digest(command)}


def validate_member(effect, resource):
    if effect.get("kind") != EffectKind.HOST_STOP:
        raise Refused("native member effect kind is unsupported")
    token = effect.get("token")
    if (not isinstance(token, dict) or set(token) != {"operation_id", "generation", "resources"}
            or not isinstance(token["operation_id"], str) or not token["operation_id"]
            or type(token["generation"]) is not int or token["generation"] <= 0
            or not isinstance(token["resources"], list) or token["resources"] != sorted(set(token["resources"]))
            or not token["resources"] or any(not canonical(r) for r in token["resources"])
            or resource not in token["resources"]):
        raise Refused("invalid native resource token")
    payload = effect.get("payload", {})
    expected = member_plan(payload.get("action"), resource, effect.get("birth"), payload.get("target"))
    if payload != expected:
        raise Refused("native command, digest or member footprint differs from validated plan")
    if not isinstance(effect.get("id"), str) or not effect["id"].startswith(token["operation_id"] + ":"):
        raise Refused("native effect identity differs from operation")
    return expected


class NativeExecutor:
    def boot_id(self):
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def observe(self, plan):
        expected = Birth.parse(plan["birth"])
        if expected.boot_id != self.boot_id():
            raise Refused("host reboot invalidated physical birth")
        if expected.kind == "process":
            path = Path("/proc") / expected.identity / "stat"
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                return {"state": "absent", "birth": plan["birth"]}
            if fields[19] != expected.started:
                raise Refused("PID now identifies a different physical birth")
            return {"state": "absent" if fields[0] in ("Z", "X") else "present", "birth": plan["birth"]}
        result = subprocess.run(["docker", "inspect", expected.identity], capture_output=True, text=True, timeout=10)
        if result.returncode:
            # Docker errors do not prove absence or termination.
            raise Refused("container observation is unknown")
        try:
            actual = json.loads(result.stdout)[0]
            if actual["Id"] != expected.identity or actual["State"]["StartedAt"] != expected.started:
                raise Refused("container physical birth changed")
            return {"state": "present" if actual["State"]["Running"] else "absent", "birth": plan["birth"]}
        except (ValueError, TypeError, IndexError, KeyError) as error:
            raise Refused("container observation is invalid") from error

    def apply(self, plan):
        if plan["action"] == "process_stop":
            if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
                raise Refused("pidfd termination is unsupported")
            descriptor = os.pidfd_open(int(plan["target"]), 0)
            try:
                if self.observe(plan)["state"] != "present":
                    return
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                poll = select.poll()
                poll.register(descriptor, select.POLLIN)
                if not poll.poll(30_000):
                    raise Refused("process termination remains uncertain")
            finally:
                os.close(descriptor)
        else:
            result = subprocess.run(plan["command"], capture_output=True, text=True, timeout=40)
            if result.returncode:
                raise Refused("native stop outcome is uncertain")


class HostJournal:
    def __init__(self, path, resource, executor, observe_authority):
        if not canonical(resource):
            raise Refused("host physical resource identity is unknown")
        self.path, self.resource, self.executor = Path(path), resource, executor
        self.observe_authority = observe_authority

    def _load(self):
        if not self.path.exists():
            return {"schema": 1, "resource": self.resource, "owner": None, "effects": {}}
        value = json.loads(self.path.read_text())
        if (value.get("schema") != 1 or value.get("resource") != self.resource
                or not isinstance(value.get("effects"), dict)):
            raise Refused("invalid native host journal")
        return value

    def execute(self, effect, fences):
        plan = validate_member(effect, self.resource)
        # Parse the complete original protocol evidence again at the host
        # boundary. A boolean success flag or a head-only token is insufficient.
        from profileplan import sealed_authorities
        fences = {peer: self.observe_authority(peer, prior) for peer, prior in fences.items()}
        sealed_authorities({"id": effect["token"]["operation_id"], "generation": effect["token"]["generation"],
                            "resources": effect["token"]["resources"], "observations": {"authorities": fences}})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = self._load()
            prior = value["effects"].get(effect["id"])
            if prior:
                if prior["digest"] != digest(effect):
                    raise Refused("native effect identity reused for different command")
                if prior["state"] == EffectState.VERIFIED:
                    return copy.deepcopy(prior["outcome"])
                observed = self.executor.observe(plan)
                if observed.get("state") != "absent" or observed.get("birth") != plan["birth"]:
                    raise Refused("issued native effect remains uncertain; no retry")
            else:
                owner = value["owner"]
                token = effect["token"]
                if owner and owner != token:
                    if (owner["generation"] >= token["generation"]
                            or any(record["state"] != EffectState.VERIFIED for record in value["effects"].values())):
                        raise Refused("stale native generation or unresolved prior owner")
                observed = self.executor.observe(plan)
                if observed.get("state") not in ("present", "absent") or observed.get("birth") != plan["birth"]:
                    raise Refused("unknown native physical birth")
                value["owner"] = copy.deepcopy(token)
                value["effects"][effect["id"]] = {"digest": digest(effect), "state": EffectState.UNCERTAIN, "outcome": None}
                durable(self.path, value)  # HostAccept: before physical application.
                if observed["state"] == "present":
                    self.executor.apply(plan)
                observed = self.executor.observe(plan)
                if observed.get("state") != "absent" or observed.get("birth") != plan["birth"]:
                    raise Refused("native physical termination is unconfirmed")
            outcome = {"effect_id": effect["id"], "token": copy.deepcopy(effect["token"]), "verified": True,
                       "birth": copy.deepcopy(plan["birth"]), "state": "terminated"}
            value["effects"][effect["id"]].update(state=EffectState.VERIFIED, outcome=outcome)
            durable(self.path, value)  # HostOutcome: reply can now be lost safely.
            return outcome
