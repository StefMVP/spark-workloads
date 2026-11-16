"""Seeded simulation of the Spark job router: random job mixes, a node loss and a profile switch during jobs.

It drives the real visionjob.Runner and router.plan over tests/spark_world.py. Invariants, checked after every pass:
  1. memory is never oversubscribed: on each node the true use stays under the machine, and the bookings of its live
     jobs stay under the machine minus the floor;
  2. no node is idle while a waiting job fits on it;
  3. every restartable job runs to its end with exit 0, also the ones on a lost node or stopped by a switch.
"""
import argparse
import itertools
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_world import IMAGE, RING, World
import router
import visionjob

KINDS = {  # name: declared (mem, cpus, gpu job), true (mem, cpus, gpu %, seconds)
    "harvest": ((24, 4, True), (9.0, 3.0, 30.0, 240)),
    "backfill": ((1, 1, False), (0.6, 0.4, 0.0, 150)),
    "teacher": ((8, 2, True), (14.0, 2.0, 45.0, 300)),       # declared too low: the router must learn it
    "train": ((60, 8, True), (55.0, 8.0, 90.0, 420)),
    "cpu": ((4, 6, False), (3.0, 6.0, 0.0, 200)),
    "index": ((50, 1, False), (46.0, 1.0, 0.0, 260)),        # memory binds first
}


def check(world, runner, label):
    state = world.controller.store.read()
    facts = runner.facts(RING, {IMAGE})
    nodes, live = runner.world(state, facts)
    for node in nodes:
        if not node["up"]:
            continue
        true_use = sum(c["use"]["mem"] for c in world.running(node["name"]))
        assert true_use + world.OTHER <= world.MEM, f"{label}: {node['name']} true memory {true_use:.0f} GiB exceeds the machine"
        booked = sum(job["need"]["mem"] for job in live if node["name"] in job["nodes"])
        limit = world.MEM - router.floor_gib(world.MEM)
        assert booked <= limit + 1e-6, f"{label}: {node['name']} books {booked:.0f} GiB over {limit:.0f}"
    learned, waiting = runner.notes()["learned"], list(state.get("queue", {}).values())
    for node in nodes:
        if state.get("queue_hold") or not node["up"] or any(node["name"] in job["nodes"] for job in live):
            continue
        fits = [e["cmd"] for e in waiting if int(e.get("nodes") or 1) == 1
                and not router.refusal(node, live, e, router.estimate(e, learned))]
        assert not fits, f"{label}: {node['name']} is idle while {fits[0]} fits on it and waits"


def simulate(seed):
    rng = random.Random(seed)
    counter = itertools.count()
    visionjob.new_id = lambda prefix: f"{prefix}{next(counter):06d}"       # wall-clock ids would change the order
    with tempfile.TemporaryDirectory() as folder:
        true = {}
        world = World(folder, use=lambda cmd: dict(zip(("mem", "cpus", "gpu", "seconds"), true[cmd.split()[1]]), code=0))
        runner = visionjob.Runner(world, world.controller, clock=world.clock)
        wanted, serial = set(), 0
        lost_at, back_at = rng.randrange(8, 20), None
        switch_at, restore_at = rng.randrange(22, 34), None
        lost_node, removed = rng.choice(RING), rng.sample(RING, 2)
        for step in range(400):
            if step < 40:
                for _ in range(rng.randrange(0, 5)):
                    kind = rng.choice(list(KINDS))
                    (mem, cpus, gpu), use = KINDS[kind]
                    serial += 1
                    tag = f"{kind}-{serial}"
                    true[tag] = tuple(value * rng.uniform(0.8, 1.0) for value in use[:3]) + (use[3] * rng.uniform(0.5, 1.5),)
                    world.controller.enqueue({"id": f"q{serial}", "by": "sim", "image": IMAGE, "cmd": f"run {tag}", "mem": mem,
                                              "cpus": cpus, "gpu": gpu, "name": kind, "priority": rng.randrange(0, 3),
                                              "restartable": True, "extra": {"tag": tag}})
                    wanted.add(tag)
            if step == lost_at:
                world.lose(lost_node)
                back_at = step + rng.randrange(4, 12)
            if step == back_at:
                world.down.discard(lost_node)
            if step == switch_at:                                # the steps of switch.Driver.requeue, then a smaller pool
                world.controller.hold_queue("a profile switch runs", world.t)
                state = runner.refresh()
                for job in list(state["jobs"].values()):
                    if job["status"] == "running" and job["command"]["node"] not in world.down and not job["command"].get("superseded"):
                        runner.requeue(job)
                def shrink(control):
                    spec = control["operations"]["op"]["spec"]
                    spec["reservations"] = {g: v for g, v in spec["reservations"].items() if f"spark{g[1]}" not in removed}
                world.controller.store.commit(shrink)
                world.controller.hold_queue("", 0)
                restore_at = step + rng.randrange(3, 10)
            if step == restore_at:
                def grow(control):
                    control["operations"]["op"]["spec"]["reservations"] = {f"s{k}": {"by": "jobs", "for": "jobs"} for k in (1, 2, 3, 4)}
                world.controller.store.commit(grow)
            runner.dispatch()
            check(world, runner, f"seed {seed} step {step}")
            world.advance(30)
            state = world.controller.store.read()
            busy = [j for j in state["jobs"].values() if j["status"] != "settled" and not j["command"].get("superseded")]
            if step >= 40 and not state.get("queue") and not busy and step > (restore_at or 0) and step > (back_at or 0):
                break
        state = runner.refresh()
        done = {j["command"]["tag"] for j in state["jobs"].values()
                if j["status"] == "settled" and runner.exit_code(j) == 0 and not j["command"].get("superseded")}
        missing = sorted(wanted - done)
        assert not missing, f"seed {seed}: {len(missing)} of {len(wanted)} restartable jobs did not finish: {missing[:5]}"
        learned = runner.notes()["learned"]
        assert learned.get("teacher", {}).get("mem", 0) > 8, f"seed {seed}: the low teacher guess was not corrected: {learned.get('teacher')}"
        requeued = sum(1 for j in state["jobs"].values() if (j["command"].get("entry") or {}).get("requeued_from"))
        return {"seed": seed, "jobs": len(wanted), "runs": len(state["jobs"]), "restarted": requeued, "steps": step + 1,
                "lost": lost_node, "switched_out": removed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    try:
        print(json.dumps({"event": "done", **simulate(parser.parse_args().seed)}))
    except AssertionError as broken:
        print(json.dumps({"event": "invariant", "detail": str(broken)}))
        sys.exit(1)
