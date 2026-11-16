"""Disk facts and placement policy for the existing artifact preparer."""
from __future__ import annotations

import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess

from clustercontrol import Refused

RESERVE = 100 * 1024**3


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise Refused(name + " must be nonnegative bytes")
    return value


def library_root(identity):
    """Resolve only the configured volume. Never enumerate unrelated volumes."""
    if (not isinstance(identity, dict) or len(identity) != 1
            or next(iter(identity), None) not in ("uuid", "label")
            or not isinstance(next(iter(identity.values()), None), str)
            or not next(iter(identity.values()), "")
            or any(c in next(iter(identity.values())) for c in "\n\x00")):
        raise Refused("model library requires one configured UUID or label")
    key, value = next(iter(identity.items()))
    result = subprocess.run(["findmnt", "--json", "--output", "TARGET", "--source", key.upper() + "=" + value],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise Refused("model library is missing: configured " + key + " is not mounted")
    mounts = json.loads(result.stdout).get("filesystems", [])
    if len(mounts) != 1:
        raise Refused("model library mount is ambiguous")
    root = Path(mounts[0]["target"])
    if not root.is_absolute() or root == Path("/") or root.resolve() != root or not os.path.ismount(root):
        raise Refused("model library is not a distinct canonical mount")
    return str(root)


def disk_observation(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise Refused("storage path is not canonical")
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    mount = ancestor
    while not os.path.ismount(mount):
        mount = mount.parent
    if (str(path).startswith(("/run/media/", "/media/")) and mount == Path("/")):
        raise Refused("model library is missing; media path is on the system disk")
    facts = os.statvfs(ancestor)
    return {"path": str(path), "device": ancestor.stat().st_dev, "mount": str(mount),
            "available": facts.f_bavail * facts.f_frsize, "capacity": facts.f_blocks * facts.f_frsize}


def headroom(observation, sizes, *, reserve=RESERVE, image_bytes=0, partial_bytes=0):
    """Budget final files plus the publisher's largest simultaneous private copy.

    Files already verified are excluded by the caller. Partial bytes are already
    allocated and charged by statvfs; only their remaining bytes are additional.
    Shared/hardlinked cache objects are never charged twice as new allocations.
    """
    sizes = [nonnegative(size, "artifact size") for size in sizes]
    reserve = nonnegative(reserve, "reserve")
    image_bytes = nonnegative(image_bytes, "image build budget")
    partial_bytes = nonnegative(partial_bytes, "partial size")
    if partial_bytes > sum(sizes):
        raise Refused("partial data exceeds manifest size")
    required = sum(sizes) - partial_bytes + max(sizes, default=0) + image_bytes
    available = nonnegative(observation["available"], "available space")
    receipt = {**observation, "required": required, "reserve": reserve, "image_bytes": image_bytes,
               "partial_bytes": partial_bytes, "fits": available >= required + reserve}
    if not receipt["fits"]:
        raise Refused("insufficient disk headroom: available=" + str(available) + " required=" + str(required)
                      + " reserve=" + str(reserve) + " (includes atomic publication and image budget)")
    return receipt


def direct_route(nodes, source, target):
    """Only reciprocal inventory endpoints authorize a direct cable candidate."""
    outgoing = nodes.get(source, {}).get("ring_ips", {}).get(target, [])
    incoming = nodes.get(target, {}).get("ring_ips", {}).get(source, [])
    if not outgoing or not incoming:
        return None
    remote, local = ipaddress.ip_address(outgoing[0]), ipaddress.ip_address(incoming[0])
    if (remote.version != 4 or local.version != 4 or not remote.is_private or not local.is_private
            or remote == local or ipaddress.ip_network(str(remote) + "/24", strict=False)
            != ipaddress.ip_network(str(local) + "/24", strict=False)):
        raise Refused("direct ring endpoints are invalid")
    return {"address": str(remote), "source_address": str(local)}


def validate_route(route):
    result = subprocess.run(["ip", "-json", "route", "get", route["address"]],
                            capture_output=True, text=True, timeout=10, check=True)
    rows = json.loads(result.stdout)
    if (len(rows) != 1 or rows[0].get("gateway") or not rows[0].get("dev")
            or rows[0].get("prefsrc", rows[0].get("src")) != route["source_address"]):
        raise Refused("direct ring route is not an observed on-link endpoint")
    return {**route, "device": rows[0]["dev"]}


def source_order(nodes, target, candidates):
    """Local first; then observed route cost when comparable, else direct peers.

    A cost is attached only to an exact verified source stat by the preparer.
    Link capacity is never a throughput measurement.
    """
    remote = [item for item in candidates if item["node"] != target or item.get("kind") == "library"]
    comparable = bool(remote) and all(isinstance(item.get("seconds"), (int, float))
                                     and math.isfinite(item["seconds"]) and item["seconds"] > 0 for item in remote)
    def rank(item):
        if item["node"] == target and item.get("kind") != "library":
            return (0, 0)
        if comparable:
            return (1, item["seconds"])
        direct = direct_route(nodes, item["node"], target)
        return (1 if direct else 2, item.get("seconds", float("inf")))
    return sorted(candidates, key=rank)
