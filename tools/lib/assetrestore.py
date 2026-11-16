"""Missing profile assets: find each file on a peer Spark and copy it back before a load. No model is loaded.

The asset list is the profile cache: each instance names the files of each member node with their sizes.
A file is restored only when a peer has a file of the same path and size and the node keeps `MIN_FREE_GB` after the
copy. The copy runs node to node, then sha256 on both sides must agree. Every other case is a blocker with its fix.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time

MIN_FREE_GB = 50        # a restore never takes a node under this: job nodes write data and job folders
PROBE = r'''
import json, os, shutil, sys
out = {"sizes": {}, "free": {}}
for path in json.loads(sys.argv[1]):
    try:
        out["sizes"][path] = os.stat(path).st_size
    except OSError:
        out["sizes"][path] = None
    folder = os.path.dirname(path)
    while folder != "/" and not os.path.isdir(folder):
        folder = os.path.dirname(folder)
    out["free"][path] = shutil.disk_usage(folder).free
print(json.dumps(out))
'''


def wanted(spec, instance=None):
    """{node: {path: bytes}} from the cache stamps of the profile's instances."""
    found = {}
    for item in spec.get("instances") or []:
        if instance and item["id"] != instance:
            continue
        for node, facts in ((item.get("cache") or {}).get("nodes") or {}).items():
            for path, stamp in (facts.get("stamps") or {}).items():
                found.setdefault(node, {})[path] = int(stamp[0])
    return found


def observe(cluster, paths):
    script = shlex.join(["python3", "-c", PROBE, json.dumps(sorted(paths))])
    def one(node):
        try:
            reply = cluster.ssh(node, script, timeout=60)
            return json.loads(reply.stdout) if reply.returncode == 0 else None
        except (subprocess.TimeoutExpired, ValueError):
            return None
    return cluster.pmap(one, list(cluster.RING))


def copy(cluster, node, peer, path):
    """Node pulls the file from the peer, then both sides hash it. Returns '' or what failed. A bad copy is removed."""
    part, q = shlex.quote(path + ".restore"), shlex.quote(path)
    pull = cluster.ssh(node, f"mkdir -p {shlex.quote(path.rsplit('/', 1)[0])} && rsync -t --inplace -e 'ssh -o BatchMode=yes' "
                             f"{peer}:{q} {part}", timeout=3600)
    if pull.returncode:
        cluster.ssh(node, f"rm -f {part}", timeout=60)
        return f"the copy from {peer} failed ({pull.stderr.strip()[-160:]}); the partial file was removed"
    sums = cluster.pmap(lambda pair: cluster.ssh(pair[0], f"sha256sum {pair[1]} | cut -d' ' -f1", timeout=3600).stdout.strip(),
                        [(peer, q), (node, part)])
    mine, theirs = sums[(node, part)], sums[(peer, q)]
    if len(mine) != 64 or mine != theirs:
        cluster.ssh(node, f"rm -f {part}", timeout=60)
        return f"sha256 of the copy differs from {peer}; the copy was removed"
    moved = cluster.ssh(node, f"mv {part} {q}", timeout=60)
    return "" if moved.returncode == 0 else f"the copy could not be moved in place ({moved.stderr.strip()[-120:]})"


def users(cluster, node, path):
    """Saved profiles whose cache names this file on this node."""
    import clusterprofile
    found = []
    for name in clusterprofile.NAMES:
        try:
            spec = clusterprofile.load(cluster.ROOT, name)
        except Exception:
            continue
        if path in wanted(spec).get(node, {}):
            found.append(name)
    return found


def remove(cluster, node, path, yes, say=print):
    """`spark asset rm NODE PATH`: list the profiles that need the file; refuse when one does and no peer has a copy."""
    from clustercontrol import Refused
    seen = observe(cluster, {path})
    if not seen.get(node) or seen[node]["sizes"].get(path) is None:
        raise Refused(f"{node}: {path} is not a file there; nothing was removed")
    size, needed = seen[node]["sizes"][path], users(cluster, node, path)
    peers = [p for p in cluster.RING if p != node and seen.get(p) and seen[p]["sizes"].get(path) == size]
    say(f"{node}:{path} ({size / 1e9:.1f} GB) is needed by: {', '.join(needed) or 'no saved profile'}; "
        f"copies of the same size: {', '.join(peers) or 'none'}")
    if needed and not peers:
        raise Refused(f"refused, nothing was removed: {', '.join(needed)} need this file and no peer Spark has a copy; "
                      f"copy it first: ssh PEER rsync -t {node}:{shlex.quote(path)} {shlex.quote(path)}")
    if not yes:
        say("nothing was removed: add --yes to remove it")
        return 0
    done = cluster.ssh(node, f"rm -f {shlex.quote(path)}", timeout=120)
    if done.returncode:
        raise Refused(f"{node}: rm failed: {done.stderr.strip()[-160:]}")
    say(f"removed. " + (f"`spark switch --preflight {needed[0]}` copies it back from {peers[0]}" if needed else "No saved profile needs it"))
    return 0


def register(cluster, parent):
    sub = parent.add_parser("asset", help="profile asset files on the Sparks").add_subparsers(required=True)
    p = sub.add_parser("rm", help="remove an asset or cache file; lists the profiles that need it; refused without a copy on a peer")
    p.add_argument("node"); p.add_argument("path"); p.add_argument("--yes", action="store_true")
    p.add_argument("--as", dest="as_", default=__import__("argparse").SUPPRESS)
    p.set_defaults(fn=lambda a: remove(cluster, a.node if a.node.startswith("spark") else f"spark{a.node[-1]}", a.path, a.yes))


def restore(cluster, spec, say=print, instance=None, act=True):
    """Copy back each missing asset that a peer has. Returns (what blocks, the command that fixes it) pairs."""
    want = wanted(spec, instance)
    if not want:
        return []
    seen = observe(cluster, {path for paths in want.values() for path in paths})
    blockers = []
    for node, paths in sorted(want.items()):
        if not seen.get(node):
            blockers.append((f"{node}: not reachable; its assets are unknown", f"ssh {node} true"))
            continue
        for path, size in sorted(paths.items()):
            have, name, gb = seen[node]["sizes"].get(path), path.rsplit("/", 1)[-1], size / 1e9
            if have == size:
                continue
            what = "is missing" if have is None else f"has {have} bytes, not {size}"
            peer = next((p for p in cluster.RING if p != node and seen.get(p) and seen[p]["sizes"].get(path) == size), None)
            if not peer:
                blockers.append((f"{node}: asset {path} {what} and no peer has it; nothing was copied",
                                 f"spark profile prepare {spec.get('name')} --yes"))
                continue
            free, need = seen[node]["free"][path] / 1e9, gb + MIN_FREE_GB
            if have is not None or free < need:
                why = (f"{free:.0f} GB disk free, the copy needs {need:.0f} GB ({gb:.0f} GB file and {MIN_FREE_GB} GB kept free)"
                       if have is None else "a file of another size is in its place and is not overwritten")
                blockers.append((f"{node}: asset {name} ({gb:.1f} GB) {what}; {peer} has it; not restored: {why}",
                                 f"free {max(need - free, 0):.0f} GB on {node}, then run the same command" if have is None
                                 else f"ssh {node} ls -la {shlex.quote(path)}"))
                continue
            if not act:
                say(f"  {node}: asset {name} ({gb:.1f} GB) {what}; {peer} has it; a switch copies it back")
                continue
            started = time.monotonic()
            say(f"  {node}: asset {name} ({gb:.1f} GB) {what}; copying it from {peer} ...")
            failed = copy(cluster, node, peer, path)
            if failed:
                blockers.append((f"{node}: asset {name} was not restored: {failed}", "run the same command"))
            else:
                say(f"  {node}: asset {name} restored from {peer} in {time.monotonic() - started:.0f} s; sha256 equal on both nodes")
    return blockers
