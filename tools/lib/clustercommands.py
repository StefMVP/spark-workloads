"""Native CLI entry points. All intent and job writes use Controller commands."""
from __future__ import annotations

import json

from clustercontrol import Controller, Refused, Store, public_projection


JOBS = {"cmd_cancel"}


def handle(cluster, args, command, kind=None):
    controller = Controller(Store(cluster.STATE))
    state = controller.store.read()
    if command == "cmd_layout_show":
        if getattr(args, "spec", None):
            sizes, explicit = cluster.parse_spec(args.spec)
            print(json.dumps({"options": [explicit] if explicit else cluster.carvings(sizes)}, indent=2))
        else:
            print(json.dumps(public_projection(state["layout"]), indent=2))
        return
    if command == "cmd_layout_set":
        sizes, explicit = cluster.parse_spec(args.spec)
        choices = [explicit] if explicit else cluster.carvings(sizes)
        current = {frozenset(nodes) for nodes in state["layout"]["groups"].values()}
        if any({frozenset(nodes) for nodes in choice} == current for choice in choices):
            print("unchanged: " + "  ".join(f"{g}[{'-'.join(m)}]" for g, m in state["layout"]["groups"].items()))
            return
        raise Refused("the applied profile owns the layout; change it with `spark profile apply NAME --yes`")
    if command == "cmd_status":
        import visionjob
        state = visionjob.Runner(cluster, controller).refresh()
        print(json.dumps(public_projection(status(cluster, state)), indent=2))
        return
    if command == "cmd_claim":
        group = cluster.norm(args.group)
        claim = controller.claim_group(group, cluster.me(args), args.purpose, args.force)
        print(f"{group} held by {claim['by']} for '{claim['for']}'")
        return
    if command == "cmd_release":
        groups = list(state["layout"]["groups"]) if args.group == "all" else [cluster.norm(args.group)]
        for group in groups:
            held = controller.unclaim_group(group, cluster.me(args), args.force)
            print(f"{group}: released" if held else f"{group}: already free")
        return
    if command in JOBS:
        job = state["jobs"].get(args.id)
        if not job or not job.get("attempt_id"):
            raise Refused("job attempt identity is unknown; cancellation cannot use a PID or name")
        if job.get("kind") == "vision":
            raise Refused(f"{args.id} is a router job: `spark job cancel {args.id}`")
        print(json.dumps(controller.cancel_job(job["id"], job["attempt_id"], job["generation"]), indent=2))
        raise SystemExit(3)
    raise Refused(f"unsupported command {command}")


def status(cluster, state):
    """What the controller knows: profile, layout, role and holder of each group, and live jobs."""
    owner = state["operations"].get(state["layout"].get("operation_id")) or {}
    roles, claims = owner.get("spec", {}).get("reservations", {}), owner.get("claims", {})
    groups = {group: {"nodes": nodes, "role": roles.get(group, {}).get("for"), "holder": claims.get(group),
                      "jobs": [j["id"] for j in state["jobs"].values()
                               if j.get("status") != "settled" and isinstance(j.get("command"), dict)
                               and j["command"].get("group") == group]}
              for group, nodes in state["layout"]["groups"].items()}
    return {"profile": state.get("profile"), "layout": state["layout"].get("spec"), "groups": groups,
            "waiting": [{"id": e["id"], "by": e["by"], "cmd": e["cmd"][:80], "reason": e.get("reason")}
                        for e in state.get("queue", {}).values()]}
