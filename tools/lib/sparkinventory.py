"""Inventory boundary for the four-node ring."""
import ipaddress
import json
import os
from pathlib import Path
import re

NODES = tuple(f"spark{i}" for i in range(1, 5))
HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def inventory_path():
    return Path(os.environ.get("SPARK_INVENTORY", Path.home() / ".config/spark-workloads/inventory.json")).expanduser()


def load(optional=False):
    path = inventory_path()
    if optional and not path.exists():
        return {"nodes": []}
    try:
        data = json.loads(path.read_text())
        rows = data["nodes"]
        if len(rows) != 4 or {n["alias"] for n in rows} != set(NODES):
            raise ValueError("nodes must contain spark1 through spark4 exactly once")
        if not 0 < float(data["unified_mem_gib"]) <= 1024:
            raise ValueError("unified_mem_gib must be positive and at most 1024")
        for node in rows:
            if not HOST.fullmatch(node["hostname"]):
                raise ValueError("invalid hostname")
            ipaddress.ip_address(node["lan_ip"])
            if float(node["disk_free_gib"]) < 0:
                raise ValueError("disk_free_gib must not be negative")
            for peer, addresses in node.get("ring_ips", {}).items():
                if peer not in NODES or peer == node["alias"] or not addresses:
                    raise ValueError("invalid ring peer")
                for address in addresses:
                    ipaddress.ip_address(address)
        for route in data.get("sync_routes", []):
            if route["from"] not in NODES or route["to_alias"] not in NODES or route["from"] == route["to_alias"]:
                raise ValueError("invalid sync route")
            ipaddress.ip_address(route["to_ip"])
        data["nodes"] = sorted(rows, key=lambda n: NODES.index(n["alias"]))
        return data
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"spark: invalid inventory {path}: {error}; copy and edit examples/inventory.json") from None
