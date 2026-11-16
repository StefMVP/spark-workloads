"""Admission inside the pinned native HTTP server. No host or shell dispatch."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
import uuid

PREFIX = "/profile-control/v1"
ACTIONS = {"acquire", "drain", "release"}
BIRTH_KEYS = {"kind", "identity", "boot_id", "started"}
REQUEST_KEYS = {"version", "action", "key", "expected_revision", "instance_id", "birth", "operation_id", "generation", "resources", "effect_id"}


class Refused(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resources(value):
    if (not isinstance(value, list) or len(value) != 4 or any(not isinstance(v, str) or not v.startswith("gpu:GPU-") or len(v) < 9 for v in value)
            or value != sorted(set(value))):
        raise Refused("full canonical TP4 footprint required")
    return list(value)


def birth(value):
    if (not isinstance(value, dict) or set(value) != BIRTH_KEYS or value["kind"] != "process"
            or any(not isinstance(v, str) or not v for v in value.values()) or not value["identity"].isdigit()
            or not value["started"].isdigit()):
        raise Refused("exact process birth required")
    return dict(value)


def parse_owner(value, footprint):
    if (not isinstance(value, dict) or set(value) != {"operation_id", "generation", "resources"}
            or not isinstance(value["operation_id"], str) or not value["operation_id"] or len(value["operation_id"]) > 256
            or type(value["generation"]) is not int or value["generation"] <= 0
            or resources(value["resources"]) != resources(footprint)):
        raise Refused("exact native bootstrap owner and footprint required")
    return copy.deepcopy(value)


def parse_request(value):
    if not isinstance(value, dict) or set(value) != REQUEST_KEYS:
        raise Refused("malformed native control request")
    if value["version"] != 1 or type(value["version"]) is not int or not isinstance(value["action"], str) or value["action"] not in ACTIONS:
        raise Refused("unsupported native control request")
    if any(not isinstance(value[k], str) or not value[k] for k in ("key", "instance_id", "operation_id", "effect_id")):
        raise Refused("missing native control identity")
    if type(value["expected_revision"]) is not int or value["expected_revision"] < 0 or type(value["generation"]) is not int or value["generation"] <= 0:
        raise Refused("invalid native control revision/generation")
    birth(value["birth"])
    resources(value["resources"])
    return copy.deepcopy(value)


def parse_snapshot(value, *, instance_id, expected_birth, footprint, owner=None, minimum_revision=0):
    keys = {"version", "instance_id", "birth", "resources", "revision", "state", "owner", "active", "effects"}
    if not isinstance(value, dict) or set(value) != keys or type(value["version"]) is not int or value["version"] != 1:
        raise Refused("malformed native snapshot")
    if value["instance_id"] != instance_id or birth(value["birth"]) != birth(expected_birth) or resources(value["resources"]) != resources(footprint):
        raise Refused("native snapshot identity/birth/footprint differs")
    if type(value["revision"]) is not int or value["revision"] < minimum_revision or value["state"] not in ("Open", "Draining", "Sealed", "Unknown"):
        raise Refused("native snapshot stale or unsupported")
    if value["owner"] is not None and (not isinstance(value["owner"], dict) or set(value["owner"]) != {"operation_id", "generation", "resources"} or resources(value["owner"]["resources"]) != footprint or type(value["owner"]["generation"]) is not int or value["owner"]["generation"] <= 0 or not isinstance(value["owner"]["operation_id"], str)):
        raise Refused("malformed native owner")
    if owner is not None and value["owner"] != owner:
        raise Refused("native snapshot belongs to another generation")
    if not isinstance(value["active"], list) or any(not isinstance(v, str) or not v for v in value["active"]) or len(set(value["active"])) != len(value["active"]) or not isinstance(value["effects"], dict):
        raise Refused("malformed native activity/outcomes")
    for effect_id, effect in value["effects"].items():
        if (not isinstance(effect_id, str) or not isinstance(effect, dict) or set(effect) != {"kind", "state", "owner", "birth"}
                or effect["kind"] not in ("pause", "resume") or effect["state"] not in ("MayHaveApplied", "Verified")
                or not isinstance(effect["owner"], dict) or set(effect["owner"]) != {"operation_id", "generation", "resources"}
                or resources(effect["owner"]["resources"]) != footprint or type(effect["owner"]["generation"]) is not int or effect["owner"]["generation"] <= 0
                or not isinstance(effect["owner"]["operation_id"], str) or not effect["owner"]["operation_id"]):
            raise Refused("malformed native effect outcome")
        birth(effect["birth"])
        if effect["birth"] != expected_birth and value["state"] != "Unknown":
            raise Refused("native effect belongs to an old birth")
    if value["state"] == "Open" and (value["owner"] is not None or any(e["state"] == "MayHaveApplied" for e in value["effects"].values())):
        raise Refused("unproved native open state")
    if value["state"] == "Sealed" and (value["active"] or value["owner"] is None or not any(e["kind"] == "pause" and e["state"] == "Verified" and e["owner"] == value["owner"] for e in value["effects"].values()) or any(e["state"] == "MayHaveApplied" for e in value["effects"].values())):
        raise Refused("unproved native sealed state")
    return copy.deepcopy(value)


def descriptors(instance_id, service_birth, footprint):
    """Pure HTTP contract; caller supplies credentials and exact CAS records."""
    return {"version": 1, "instance_id": instance_id, "birth": birth(service_birth), "resources": resources(footprint),
            "snapshot": {"method": "GET", "path": PREFIX},
            **{a: {"method": "PUT", "path": PREFIX, "action": a} for a in sorted(ACTIONS)},
            "atomics": ["persist ingress Draining before request admission", "persist pause MayHaveApplied", "await AsyncLLM.pause_generation(mode=wait,clear_cache=False)", "EngineCoreProc waits has_work=false then collective synchronize_device", "persist pause Verified and Sealed", "persist resume MayHaveApplied", "await resume_generation", "persist resume Verified and reopen ingress"],
            "host_execution": "unsupported"}


def _durable(path, value):
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("x") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


class Authority:
    """One revisioned admission/effect aggregate; old objects cannot adopt new birth."""
    def __init__(self, path, instance_id, service_birth, footprint, *, owner=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.instance_id, self.birth, self.resources = instance_id, birth(service_birth), resources(footprint)
        bootstrap = parse_owner(owner, self.resources) if owner is not None else None
        self.lock = threading.RLock()
        self.closed = False
        self.lifetime = self.path.with_name(self.path.name + ".lifetime").open("a+")
        try:
            fcntl.flock(self.lifetime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lifetime.close()
            raise Refused("another native server holds the authority lifetime") from error
        with self.lock, self.path.with_suffix(".lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if not self.path.exists():
                _durable(self.path, {"version": 1, "instance_id": instance_id, "birth": self.birth, "resources": self.resources, "revision": 0,
                    "state": "Draining" if bootstrap is not None else "Open", "owner": bootstrap, "active": [], "effects": {}, "keys": {},
                    "history": {bootstrap["operation_id"]: bootstrap["generation"]} if bootstrap is not None else {},
                    "last_released_owner": None, "retiring": None})
            else:
                state = json.loads(self.path.read_text())
                if state["instance_id"] != instance_id or state["resources"] != self.resources:
                    raise Refused("authority home belongs to another resource")
                if state["birth"] != self.birth:
                    # Replacement cannot turn old uncertain execution into an idle proof.
                    state["birth"], state["state"] = self.birth, "Unknown"
                    state["revision"] += 1
                    _durable(self.path, state)

    @contextmanager
    def transaction(self):
        with self.lock, self.path.with_suffix(".lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if self.closed:
                raise Refused("native authority lifetime ended")
            state = json.loads(self.path.read_text())
            if state["birth"] != self.birth or state["instance_id"] != self.instance_id or state["resources"] != self.resources:
                raise Refused("stale native authority object")
            before = copy.deepcopy(state)
            yield state
            if before != state:
                state["revision"] += 1
                _durable(self.path, state)

    def close(self):
        with self.lock:
            self.closed = True
            self.lifetime.close()

    @staticmethod
    def snapshot_of(state):
        return {k: copy.deepcopy(v) for k, v in state.items() if k not in ("keys", "history", "last_released_owner", "retiring")}

    def snapshot(self):
        with self.transaction() as state:
            return self.snapshot_of(state)

    def enter(self, identity):
        with self.transaction() as state:
            if state["state"] != "Open" or state["owner"] is not None:
                raise Refused("native ingress fenced or unknown")
            state["active"].append(identity)

    def leave(self, identity):
        with self.transaction() as state:
            if identity in state["active"]:
                state["active"].remove(identity)

    def begin(self, request):
        req = parse_request(request)
        with self.transaction() as state:
            if req["instance_id"] != self.instance_id or req["birth"] != self.birth or req["resources"] != self.resources:
                raise Refused("native binding differs")
            if state.get("retiring") is not None and req["action"] == "release":
                raise Refused("native retirement is terminal; physical stop must follow drain")
            fingerprint = _digest(req)
            if req["key"] in state["keys"]:
                if state["keys"][req["key"]] != fingerprint:
                    raise Refused("native idempotency key reused")
                return False
            if req["expected_revision"] != state["revision"] or state["state"] == "Unknown":
                raise Refused("native revision stale or authority uncertain")
            owner = {k: req[k] for k in ("operation_id", "generation", "resources")}
            if req["action"] == "acquire":
                if state["owner"] is not None or state["state"] != "Open" or state.get("retiring") is not None:
                    raise Refused("native fence already owned or terminal retirement")
                retiring = req["effect_id"] == req["operation_id"] + ":native:retire:acquire"
                if retiring:
                    verified = lambda kind: any(e["kind"] == kind and e["state"] == "Verified" and e["owner"] == owner and e["birth"] == self.birth for e in state["effects"].values())
                    if (state.get("last_released_owner") != owner or state["history"].get(req["operation_id"]) != req["generation"]
                            or not verified("pause") or not verified("resume")
                            or any(e["state"] == "MayHaveApplied" for e in state["effects"].values())):
                        raise Refused("retirement acquire lacks the exact last verified pause/resume owner")
                    state["retiring"] = {"owner": owner, "birth": self.birth, "effect_id": req["effect_id"]}
                elif req["generation"] <= state["history"].get(req["operation_id"], 0):
                    raise Refused("native fence generation was already used")
                state["owner"], state["state"] = owner, "Draining"
                state["history"][req["operation_id"]] = req["generation"]
            else:
                if state["owner"] != owner:
                    raise Refused("native owner differs")
                if any(e["state"] == "MayHaveApplied" for e in state["effects"].values()):
                    raise Refused("native effect outcome uncertain")
                kind = "pause" if req["action"] == "drain" else "resume"
                if req["effect_id"] in state["effects"]:
                    raise Refused("native effect ID reused")
                if req["action"] == "release" and state["state"] != "Sealed":
                    raise Refused("release requires verified Sealed")
                if req["action"] == "drain" and state["state"] != "Draining":
                    raise Refused("drain requires owned Draining")
                state["effects"][req["effect_id"]] = {"kind": kind, "state": "MayHaveApplied", "owner": owner, "birth": self.birth}
                if req["action"] == "release":
                    state["state"] = "Draining"
            state["keys"][req["key"]] = fingerprint
            return True

    def complete(self, request):
        req = parse_request(request)
        with self.transaction() as state:
            owner = {k: req[k] for k in ("operation_id", "generation", "resources")}
            effect = state["effects"].get(req["effect_id"])
            if (req["birth"] != self.birth or state["owner"] != owner or not effect or effect["owner"] != owner or effect["birth"] != self.birth or state["active"]):
                raise Refused("late or uncertain native completion")
            effect["state"] = "Verified"
            if req["action"] == "drain" and effect["kind"] == "pause":
                state["state"] = "Sealed"
            elif req["action"] == "release" and effect["kind"] == "resume":
                if state.get("retiring") is not None:
                    raise Refused("late resume cannot reopen terminal retirement")
                state["last_released_owner"] = copy.deepcopy(owner)
                state["state"], state["owner"] = "Open", None
            else:
                raise Refused("native effect kind differs")


class Barrier:
    def __init__(self, authority, engine):
        self.authority, self.engine = authority, engine
        self.serial = asyncio.Lock()

    async def apply(self, request):
        req = parse_request(request)
        async with self.serial:
            fresh = self.authority.begin(req)
            if not fresh or req["action"] == "acquire":
                return self.authority.snapshot()
            if req["action"] == "drain":
                # Ingress was closed in a previous aggregate commit. Existing HTTP
                # roots finish; the native engine separately drains hidden work.
                while self.authority.snapshot()["active"]:
                    await asyncio.sleep(0.01)
                await self.engine.pause_generation(mode="wait", clear_cache=False)
            else:
                await self.engine.resume_generation()
            self.authority.complete(req)
            return self.authority.snapshot()


def verify_runtime(engine, native_root, package):
    """Bind the installed constructor/source path, not just a pause-shaped method."""
    from .spark_ds41f_1030 import verify_sources
    lock = verify_sources(Path(package))
    if native_root is None:
        import inspect
        if type(engine).__module__!='vllm.v1.engine.async_llm':raise Refused('native constructor path is unverified')
        native_root=Path(inspect.getfile(type(engine))).resolve().parents[3]
    for relative, expected in lock["files"].items():
        if relative.startswith("native/"):
            actual = Path(native_root) / relative.removeprefix("native/")
            if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
                raise Refused("installed native source differs")
    if type(engine).__module__ != "vllm.v1.engine.async_llm" or type(engine).__name__ != "AsyncLLM" or type(engine.engine_core).__module__ != "vllm.v1.engine.core_client" or type(engine.engine_core).__name__ != "AsyncMPClient":
        raise Refused("native runtime is not pinned AsyncLLM/AsyncMPClient")


def process_birth():
    pid = os.getpid()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"kind": "process", "identity": str(pid), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "started": fields[19]}


class NativeIngress:
    """Existing vLLM --middleware class hook; engine remains app.state.engine_client."""
    def __init__(self, app, *, authority=None, engine=None, token=None):
        self.app, self.barrier = app, None
        if authority is None:
            config = json.loads(Path(os.environ["SPARK_DS41F_CONTROL_CONFIG"]).read_text())
            # A production wrapper never starts Open. The host journal emits this
            # config only under its accepted create operation; the leaf grants none.
            bootstrap = parse_owner(config.get("owner"), config["resources"])
            state_path = Path(config["state_path"])
            if not state_path.is_absolute() or ".." in state_path.parts or not state_path.is_relative_to("/run/profile-control/state"):
                raise Refused("native state must use the controller-owned writable mount")
            authority = Authority(state_path, config["instance_id"], process_birth(), config["resources"], owner=bootstrap)
            self.runtime = (config.get("native_root"), config["source_package"])
            token = config["control_token"]
        else:
            self.runtime = None  # Explicit in-process proof seam, never serialized.
        if not isinstance(token, str) or len(token) < 32:
            raise Refused("native control authentication required")
        self.authority, self.engine, self.token = authority, engine, token

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        async def response(status, data):
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1013})
            body = json.dumps(data).encode()
            await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
        if scope.get("path") == PREFIX:
            if scope["type"] != "http":
                return await response(409, {"error": "native control requires HTTP"})
            headers = dict(scope.get("headers", []))
            supplied = headers.get(b"authorization", b"").decode("latin1")
            if not hmac.compare_digest(supplied, "Bearer " + self.token):
                return await response(401, {"error": "unauthorized"})
            try:
                if self.barrier is None:
                    engine = self.engine or scope["app"].state.engine_client
                    if self.runtime is not None:
                        verify_runtime(engine, *self.runtime)
                    self.barrier = Barrier(self.authority, engine)
                if scope["method"] == "GET":
                    return await response(200, self.authority.snapshot())
                if scope["method"] != "PUT":
                    raise Refused("unsupported native control method")
                data = b""
                while True:
                    event = await receive()
                    if event["type"] != "http.request":
                        raise Refused("native control request disconnected")
                    data += event.get("body", b"")
                    if len(data) > 16384:
                        raise Refused("native control request too large")
                    if not event.get("more_body", False):
                        break
                return await response(200, await self.barrier.apply(json.loads(data)))
            except (ValueError, KeyError, OSError, AttributeError) as error:
                return await response(409, {"error": str(error)})
        if self.barrier is None:
            try:
                engine = self.engine or scope["app"].state.engine_client
                if self.runtime is not None:
                    verify_runtime(engine, *self.runtime)
                self.barrier = Barrier(self.authority, engine)
            except (ValueError, KeyError, OSError, AttributeError):
                return await response(503, {"error": "native runtime identity is unverified"})
        path = scope.get("path", "")
        # Raw lifecycle routes would bypass the durable owner/effect protocol.
        if path in ("/pause", "/resume", "/sleep", "/wake_up", "/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache", "/collective_rpc", "/scale_elastic_ep"):
            return await response(409, {"error": "use owned native control protocol"})
        if scope["type"] == "http" and scope["method"] == "GET" and path in ("/health", "/v1/models", "/version", "/metrics", "/ping"):
            return await self.app(scope, receive, send)
        identity = uuid.uuid4().hex
        try:
            self.authority.enter(identity)
        except Refused:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1013})
            return await response(503, {"error": "native ingress fenced or unknown"})
        try:
            await self.app(scope, receive, send)
        finally:
            self.authority.leave(identity)
