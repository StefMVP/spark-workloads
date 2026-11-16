"""Bounded controller-mediated artifact streams; no GPU or service authority."""
from __future__ import annotations

import base64
import hashlib
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import shlex
import subprocess
import tempfile
import time

from clustercontrol import Refused
from artifactcontrol import file_digest, publish_missing
from artifactstorage import disk_observation, headroom, library_root, validate_route, RESERVE

# Runtime.remote passes its script as one SSH argv element. Keep the encoded
# block and exact helper sources below Linux's per-argument size limit.
CHUNK = 32 * 1024
STREAM_CHUNK = 1024 * 1024
STREAM_TIMEOUT = 1800
REPLY_LIMIT = 64 * 1024


def machine():
    return {"machine_id": Path("/etc/machine-id").read_text().strip(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "hostname": socket.gethostname(), "home": str(Path.home())}


def check_binding(expected):
    if machine() != expected:
        raise Refused("host machine, boot or home identity changed")


def canonical(path):
    found = Path(path)
    if (not found.is_absolute() or ".." in found.parts or str(found.resolve()) != str(found)
            or any("checkpoint" in part.lower() for part in found.parts)):
        raise Refused("protected or noncanonical artifact path")
    return found


def signature(path):
    return stat_signature(path.stat())


def stat_signature(item):
    return [item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns]


def mapped(path):
    """Known open/mapped targets block publication; absence uses new-path proof."""
    name = str(path)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if any(line.endswith(" " + name) or line.endswith(" " + name + " (deleted)")
                   for line in (process / "maps").read_text().splitlines()):
                return True
            for descriptor in (process / "fd").iterdir():
                try:
                    if os.readlink(descriptor).removesuffix(" (deleted)") == name:
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def inspect(path, expected_sha, *, source=False):
    path = canonical(path)
    if path.is_symlink():
        raise Refused("symlink artifact publication is unsupported")
    if not path.exists():
        if source:
            raise Refused("source artifact is missing")
        if mapped(path):
            raise Refused("target path remains open or mapped")
        return {"path": str(path), "usage": "new_path", "proof": "canonical path is absent", "size": 0}
    if not path.is_file():
        raise Refused("artifact is not a regular file")
    before = signature(path)
    observed = file_digest(path)
    if signature(path) != before or observed != expected_sha:
        raise Refused("source content changed or existing target differs; overwrite forbidden")
    return {"path": str(path), "sha256": observed, "stat": before, "size": before[2],
            "usage": "source" if source else "preexisting_verified", "proof": "full content SHA256"}


def stage_paths(target, effect_id):
    key = hashlib.sha256(effect_id.encode()).hexdigest()
    folder = target.parent / (".profile-input-" + key)
    return folder, folder / "owner.json", folder / "content"


def stage_owner(data):
    return {key: data[key] for key in ("effect_id", "binding", "target", "sha256", "source_binding")}


def stage(data):
    target = canonical(data["target"])
    required = data.get("target_library")
    if required and (library_root(required["identity"]) != required["root"]
                     or not target.is_relative_to(required["root"])):
        raise Refused("model library identity or target mount changed")
    if target.exists():
        return inspect(target, data["sha256"])
    if data["usage"] == "preexisting_verified" or mapped(target):
        raise Refused("target usage no longer permits a new publication")
    folder, owner, content = stage_paths(target, data["effect_id"])
    expected = stage_owner(data)
    if not folder.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.mkdir(mode=0o700)
        with owner.open("x") as stream:
            json.dump(expected, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        with content.open("xb") as stream:
            os.fsync(stream.fileno())
    if (folder.is_symlink() or owner.is_symlink() or content.is_symlink()
            or json.loads(owner.read_text()) != expected or not content.is_file()
            or not stat.S_ISREG(content.stat().st_mode)):
        raise Refused("staging ownership is unknown")
    return {"path": str(content), "offset": content.stat().st_size, "stat": signature(content), "usage": data["usage"]}


def host_action(data):
    action = data["action"]
    if action == "identity":
        return machine()
    check_binding(data["binding"])
    if action == "storage":
        return disk_observation(canonical(data["path"]))
    if action == "library":
        return {"root": library_root(data["identity"])}
    if action == "locate":
        path = canonical(data["path"])
        return {"path": str(path), "present": path.is_file(), "stat": signature(path) if path.is_file() else None}
    if action == "route":
        return validate_route(data["route"])
    if action == "headroom":
        return headroom(disk_observation(canonical(data["path"])), data["sizes"],
                        reserve=data.get("reserve", RESERVE), image_bytes=data.get("image_bytes", 0))
    if action == "inspect":
        return inspect(data["path"], data["sha256"], source=data.get("source", False))
    if action == "read":
        path = canonical(data["path"])
        if signature(path) != data["stat"]:
            raise Refused("source file identity changed during transfer")
        with path.open("rb") as stream:
            stream.seek(data["offset"])
            raw = stream.read(CHUNK)
        return {"offset": data["offset"], "chunk": base64.b64encode(raw).decode()}
    found = stage(data)
    if "sha256" in found:
        # Actual publisher verifies same-content replay without changing that file.
        return publish_missing(data["target"], data["target"], data["sha256"], usage="preexisting_verified")
    if action == "stage":
        return found
    path = Path(found["path"])
    if action == "append":
        raw = base64.b64decode(data["chunk"], validate=True)
        if not raw or len(raw) > CHUNK or found["offset"] != data["offset"]:
            raise Refused("staging offset changed; inspect the exact effect before retry")
        with path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return {"offset": path.stat().st_size}
    if action == "publish":
        # The production publisher streams, fsyncs and links without overwrite.
        evidence = publish_missing(path, data["target"], data["sha256"], usage=data["usage"])
        folder, owner, content = stage_paths(Path(data["target"]), data["effect_id"])
        content.unlink()
        owner.unlink()
        folder.rmdir()
        return evidence
    raise Refused("unsupported artifact transport action")


def send_stream(data, output):
    """Emit the exact suffix, then proof of a complete stable source read."""
    check_binding(data["binding"])
    path = canonical(data["path"])
    expected, offset = data["stat"], data["offset"]
    if type(offset) is not int or not 0 <= offset <= expected[2]:
        raise Refused("source offset is invalid")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        if (not stat.S_ISREG(os.fstat(source.fileno()).st_mode)
                or stat_signature(os.fstat(source.fileno())) != expected or signature(path) != expected):
            raise Refused("source file identity changed during transfer")
        digest, position = hashlib.sha256(), 0
        while position < expected[2]:
            block = source.read(min(STREAM_CHUNK, expected[2] - position))
            if not block:
                raise Refused("source transfer is incomplete")
            digest.update(block)
            if position + len(block) > offset:
                output.write(block[max(0, offset - position):])
            position += len(block)
        if (digest.hexdigest() != data["sha256"] or source.read(1)
                or stat_signature(os.fstat(source.fileno())) != expected or signature(path) != expected):
            raise Refused("source content or file identity changed during transfer")
        check_binding(data["binding"])
        footer = {"sha256": digest.hexdigest(), "stat": expected, "binding": data["binding"], "offset": offset}
        output.write(json.dumps(footer, sort_keys=True).encode() + b"\n")
        output.flush()


def receive_stream(data, incoming):
    """Persist a resumable suffix; publish only after the source footer passes."""
    check_binding(data["binding"])
    # Check actual remaining allocation, including the publisher's private copy.
    # The existing exact-path controller reservation stays authoritative.
    size = data["source_stat"][2]
    headroom(disk_observation(data["target"]), [size], reserve=data.get("reserve", RESERVE),
             image_bytes=data.get("image_bytes", 0), partial_bytes=data["offset"])
    found = stage(data)
    if "sha256" in found:
        raise Refused("target changed after stream setup; reconcile publication")
    if found["offset"] != data["offset"] or found["stat"] != data["stage_stat"]:
        raise Refused("staging offset or inode changed; inspect the exact effect before retry")
    path = Path(found["path"])
    size, offset = data["source_stat"][2], data["offset"]
    if type(offset) is not int or not 0 <= offset <= size:
        raise Refused("staged data exceeds the exact source size")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW), "ab") as target:
        fcntl.flock(target.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stat_signature(os.fstat(target.fileno())) != data["stage_stat"]:
            raise Refused("staging file identity changed before transfer")
        try:
            remaining = size - offset
            while remaining:
                block = incoming.read(min(STREAM_CHUNK, remaining))
                if not block:
                    raise Refused("source stream disconnected; partial stage retained")
                target.write(block)
                remaining -= len(block)
            footer = incoming.readline(REPLY_LIMIT + 1)
            if len(footer) > REPLY_LIMIT or not footer.endswith(b"\n"):
                raise Refused("source verification footer is missing or malformed")
            expected = {"sha256": data["sha256"], "stat": data["source_stat"],
                        "binding": data["source_binding"], "offset": offset}
            if json.loads(footer) != expected or incoming.read(1):
                raise Refused("source verification footer differs from the reserved source")
        finally:
            target.flush()
            os.fsync(target.fileno())
        if signature(path) != stat_signature(os.fstat(target.fileno())):
            raise Refused("staging file was replaced during transfer")
        check_binding(data["binding"])
        # Recheck the effect owner after the stream. The unchanged production
        # publisher hashes all bytes and uses exclusive link publication.
        current = stage(data)
        if current.get("path") != str(path) or current.get("stat") != signature(path):
            raise Refused("staging ownership changed during transfer")
        return host_action({**data, "action": "publish"})


class Transport:
    def __init__(self, runtime):
        self.runtime = runtime
        self.module = Path(__file__).read_text()
        self.publisher = Path(__file__).with_name("artifactcontrol.py").read_text()
        self.storage = Path(__file__).with_name("artifactstorage.py").read_text()

    def bootstrap(self):
        # Only an exception shim is supplied remotely; publication policy is the
        # exact repository artifactcontrol source, never a second implementation.
        return ("import json,types,sys\n"
                "m=types.ModuleType('clustercontrol')\n"
                "exec('class Refused(RuntimeError): pass',m.__dict__)\n"
                "sys.modules['clustercontrol']=m\n"
                "p=types.ModuleType('artifactcontrol')\n"
                "sys.modules['artifactcontrol']=p\n"
                "exec(" + repr(self.publisher) + ",p.__dict__)\n"
                "s=types.ModuleType('artifactstorage')\n"
                "sys.modules['artifactstorage']=s\n"
                "exec(" + repr(self.storage) + ",s.__dict__)\n"
                "t={'__file__':'artifacttransport.py'}\n"
                "exec(" + repr(self.module) + ",t)\n")

    def call(self, node, action, **data):
        code = self.bootstrap() + (
            "try:\n result={'ok':True,'value':t['host_action'](D)}\n"
            "except (OSError,ValueError,m.Refused) as error:\n result={'ok':False,'reason':str(error)}\n"
            "print(json.dumps(result))\n")
        payload = json.dumps({"action": action, **data})
        script = "python3 - <<'PROFILE_ARTIFACT_PY'\nimport json\nD=json.loads(" + repr(payload) + ")\n" + code + "\nPROFILE_ARTIFACT_PY"
        try:
            found = json.loads(self.runtime.remote(node, script, timeout=1800))
        except (ValueError, TypeError) as error:
            raise Refused("artifact host response is malformed") from error
        if not isinstance(found, dict):
            raise Refused("artifact host response is unknown")
        if found.get("ok") is not True:
            raise Refused(found.get("reason", "artifact host response is unknown"))
        if not isinstance(found.get("value"), dict):
            raise Refused("artifact host response has no validated record")
        return found["value"]

    def identity(self, node):
        found = self.call(node, "identity")
        if (set(found) != {"machine_id", "boot_id", "hostname", "home"}
                or any(not isinstance(value, str) or not value or "\n" in value for value in found.values())
                or ":" in found["machine_id"] or not Path(found["home"]).is_absolute()):
            raise Refused("host identity is unknown")
        return found

    def publish(self, effect):
        item = effect["payload"]
        source = item["source"]
        target = source["target_binding"]
        if self.identity(source["node"]) != source["binding"] or self.identity(target["node"]) != target["binding"]:
            raise Refused("source or target host birth changed")
        required_source = source.get("source_library")
        if required_source:
            observed = self.call(source["node"], "library", binding=source["binding"], identity=required_source["identity"])
            if observed["root"] != required_source["root"]:
                raise Refused("source model library mount changed")
        data = {"target_library": source.get("storage", {}).get("target_library"), "effect_id": effect["id"], "target": item["target"], "sha256": item["sha256"], "usage": item["usage"],
                "binding": target["binding"], "source_binding": source["binding"]}
        state = self.call(target["node"], "stage", **data)
        if state.get("state") == "verified":
            return state
        offset, staged = state.get("offset"), state.get("stat")
        if (type(offset) is not int or offset < 0 or offset > source["stat"][2]
                or not isinstance(staged, list) or len(staged) != 5
                or any(type(value) is not int or value < 0 for value in staged) or staged[2] != offset):
            raise Refused("staging offset or file identity is malformed")
        send = {"binding": source["binding"], "path": source["path"], "stat": source["stat"],
                "offset": offset, "sha256": item["sha256"]}
        receive = {**data, "offset": offset, "stage_stat": state["stat"], "source_stat": source["stat"],
                   "reserve": source.get("storage", {}).get("reserve", RESERVE),
                   "image_bytes": source.get("storage", {}).get("image_bytes", 0)}
        started = time.monotonic()
        route = source.get("route")
        if source["node"] == target["node"]:
            result = self.direct_stream(source["node"], None, send, receive)
            kind = "local_pipe"
        elif route:
            observed = self.call(source["node"], "route", binding=source["binding"], route=route)
            result = self.direct_stream(source["node"], observed, send, receive)
            kind = "direct_ring"
        else:
            result = self.stream(source["node"], target["node"], send, receive)
            kind = "controller_pipe"
        return {**result, "transfer": {"kind": kind, "source_node": source["node"],
                "target_node": target["node"], "source_stat": source["stat"],
                "bytes": source["stat"][2] - offset, "resumed_bytes": offset,
                "seconds": time.monotonic() - started, "route": route,
                "measurement": "first measurement; no comparator"}}

    def direct_stream(self, source_node, route, send, receive):
        # Payload travels only on the validated cable. The controller gets the
        # bounded JSON receipt. SSH host keys/keys stay under native SSH policy.
        sender = self.stream_script("send", send)
        receiver = self.stream_script("receive", receive)
        command = sender + " | " + (shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                                                route["address"], receiver]) if route else receiver)
        raw = self.runtime.remote(source_node, "bash -o pipefail -c " + shlex.quote(command), timeout=STREAM_TIMEOUT)
        if len(raw.encode()) > REPLY_LIMIT:
            raise Refused("direct ring reply exceeds its bound")
        found = json.loads(raw)
        if (not isinstance(found, dict) or found.get("ok") is not True
                or not isinstance(found.get("value"), dict)):
            raise Refused("direct ring outcome is unknown; partial stage retained")
        value = found["value"]
        if (value.get("state") != "verified" or value.get("path") != receive["target"]
                or value.get("sha256") != receive["sha256"]):
            raise Refused("direct ring reply differs from reserved target")
        return value

    def stream_script(self, action, data):
        invocation = ("t['send_stream'](D,sys.stdout.buffer)" if action == "send" else
                      "print(json.dumps({'ok':True,'value':t['receive_stream'](D,sys.stdin.buffer)}))")
        code = self.bootstrap() + "D=json.loads(" + repr(json.dumps(data)) + ")\n" + invocation + "\n"
        return "python3 -c " + shlex.quote(code)

    def endpoint(self, node, script, **pipes):
        # Match tools/cluster.ssh exactly: inventory alias, native SSH config,
        # host-key checks and existing key selection remain with OpenSSH.
        return subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, script], **pipes)

    def stream(self, source_node, target_node, send, receive):
        processes = []
        with tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as reply:
            try:
                sender = self.endpoint(source_node, self.stream_script("send", send),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
                processes.append(sender)
                receiver = self.endpoint(target_node, self.stream_script("receive", receive),
                                         stdin=sender.stdout, stdout=reply, stderr=errors)
                processes.append(receiver)
                sender.stdout.close()
                receiver_code = receiver.wait(timeout=STREAM_TIMEOUT)
                sender_code = sender.wait(timeout=STREAM_TIMEOUT)
                if receiver_code or sender_code:
                    raise Refused("artifact stream failed; inspect the exact effect before retry")
                reply.seek(0)
                raw = reply.read(REPLY_LIMIT + 1)
                if len(raw) > REPLY_LIMIT:
                    raise Refused("artifact stream reply exceeds its bound")
                try:
                    found = json.loads(raw)
                except (ValueError, TypeError) as error:
                    raise Refused("artifact stream publication reply is malformed") from error
                if not isinstance(found, dict) or found.get("ok") is not True or not isinstance(found.get("value"), dict):
                    raise Refused("artifact stream publication outcome is unknown")
                value = found["value"]
                if (value.get("state") != "verified" or value.get("path") != receive["target"]
                        or value.get("sha256") != receive["sha256"]):
                    raise Refused("artifact stream reply does not match the reserved target")
                return value
            except subprocess.TimeoutExpired as error:
                raise Refused("artifact stream timed out; partial stage retained") from error
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                if processes and processes[0].stdout:
                    processes[0].stdout.close()
