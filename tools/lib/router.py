"""The router: which waiting job starts on which Spark. Pure functions, no host or state access.

Agents never choose a node. They give a command and its needs; `plan` packs the job nodes of the applied profile.

Rules adapted from Imagination Studio:
- `by_priority` is app/queue_order.py: higher priority first, queue order within a level.
- `floor_gib` is spark/memory_guard.py `floor_mb`: a tenth of the machine, never under 8 GB, always stays free.
- `outstanding` is the rule of app/local_gpus.py `ram_reservation`: book peaks, not memory twice. A live job books
  only the growth from what it uses now to its peak, because MemAvailable already excludes what it uses now.
- GPU room is the measured utilisation of the node (Studio's workers report their own memory; here the sampler does).
- A node that stays silent for `LOST_S` (app/sparks.py `LOST`) is down and its restartable work goes back to the queue.
"""
from __future__ import annotations

LOST_S = 60
GPU_GUESS = 25          # percent of one GPU for a GPU job of a kind that was never measured
GPU_MIN = 5
MARGIN = 1.2            # head room on a learned peak
RING = ("spark1", "spark2", "spark3", "spark4")
NODE_MEM, NODE_CORES = 121, 20   # GiB of unified memory and cores of one GB10 node


def by_priority(queued: list) -> list:
    """Higher priority first (0 to 9, none is 0), queue order within a level (the sort is stable)."""
    return sorted(queued, key=lambda job: -int(job.get("priority") or 0))


def floor_gib(total_gib: float) -> float:
    return max(8, total_gib / 10)


def kind(entry: dict) -> str:
    """Jobs of one kind share what was learned: the given name, else the first words of the command.
    A class is part of the kind: an io job never books the memory that a gpu job of the same name used."""
    name = entry.get("name") or " ".join(entry["cmd"].split()[:3])[:60]
    return name if entry.get("class", "gpu") == "gpu" else f"{name} [{entry['class']}]"


def estimate(entry: dict, learned: dict) -> dict:
    """What one job of this entry is booked for: measured use of its kind when there is one, else the declared guess."""
    seen = learned.get(kind(entry)) or {}
    need = {"mem": float(entry["mem"]), "cpus": float(entry["cpus"]), "gpu": float(GPU_GUESS if entry.get("gpu", True) else 0)}
    if seen.get("n"):
        # A measured peak raises the booking at once. It lowers it only after a job of the kind ran to its end.
        measured = max(0.5, seen["mem"] * MARGIN)
        need["mem"] = measured if seen.get("done") else max(need["mem"], measured)
        need["cpus"] = max(0.25, seen["cpus"] * MARGIN)
        if entry.get("gpu", True):
            need["gpu"] = min(100.0, max(GPU_MIN, seen["gpu"] * MARGIN))
    return need


def learn(learned: dict, name: str, mem: float, cpus: float, gpu: float) -> dict:
    """Fold one finished or sampled job into its kind: peak memory, and running means of cores and GPU share."""
    seen = learned.get(name) or {"n": 0, "mem": 0.0, "cpus": 0.0, "gpu": 0.0}
    n = seen["n"] + 1
    weight = 1 / min(n, 10)
    learned[name] = {"n": n, "done": seen.get("done", 0), "mem": round(max(seen["mem"] if n > 1 else 0, mem), 2),
                     "cpus": round(seen["cpus"] + (cpus - seen["cpus"]) * weight, 2),
                     "gpu": round(seen["gpu"] + (gpu - seen["gpu"]) * weight, 1)}
    return learned


def finished(learned: dict, name: str) -> dict:
    """A job of the kind ran to its end with exit 0: its measured peak is the whole peak."""
    if name in learned:
        learned[name]["done"] = learned[name].get("done", 0) + 1
    return learned


def outstanding(job: dict) -> float:
    """Memory a live job may still take: its booked peak minus what it uses now (all of it before the first sample)."""
    used = job.get("mem_now")
    return job["need"]["mem"] if used is None else max(0.0, job["need"]["mem"] - used)


def booked(node: dict, live: list) -> dict:
    mine = [job for job in live if node["name"] in job["nodes"]]
    # GPU: the measured utilisation of the node, plus the booking of each job that is too new to be in it.
    return {"mem_growth": sum(outstanding(job) for job in mine),
            "cpus": sum(job["need"]["cpus"] for job in mine),
            "gpu": node.get("gpu_util", 0.0) + sum(job["need"]["gpu"] for job in mine if job.get("mem_now") is None),
            "jobs": len(mine)}


def refusal(node: dict, live: list, entry: dict, need: dict) -> str:
    """'' when one more job with this need has room on the node now, else why not."""
    if not node.get("up", True):
        return f"{node['name']} is not reachable"
    if node.get("holder") and node["holder"] != entry.get("by"):
        return f"{node['name']} is held by {node['holder']}"
    if entry.get("image") not in node.get("images", ()):
        return f"{node['name']} has no image {entry.get('image')}"
    if entry.get("only") and node["name"] not in entry["only"]:
        return f"{node['name']} is not in the pinned nodes"
    use = booked(node, live)
    free = node["mem_avail"] - floor_gib(node["mem_total"]) - use["mem_growth"]
    if need["mem"] > free:
        return f"{node['name']}: {need['mem']:.1f} GiB needed, {max(free, 0):.1f} GiB free after the floor and live jobs"
    if use["cpus"] + need["cpus"] > node["cores"]:
        return f"{node['name']}: {need['cpus']:.1f} cores needed, {max(node['cores'] - use['cpus'], 0):.1f} free"
    if need["gpu"] and use["gpu"] + need["gpu"] > 100:
        return f"{node['name']}: GPU share {need['gpu']:.0f} % needed, {max(100 - use['gpu'], 0):.0f} % free"
    return ""


def fullness(node: dict, live: list, need: dict) -> float:
    """How full the node is with this job added: the tightest of memory, cores and GPU. Lower is better."""
    use = booked(node, live)
    capacity = node["mem_total"] - floor_gib(node["mem_total"])
    mem = 1 - (node["mem_avail"] - floor_gib(node["mem_total"]) - use["mem_growth"] - need["mem"]) / capacity
    return max(mem, (use["cpus"] + need["cpus"]) / node["cores"], (use["gpu"] + need["gpu"]) / 100)


def neighbours(names: list, count: int) -> list:
    """Every run of `count` ring neighbours inside `names`, in ring order."""
    runs = []
    for start in range(len(RING)):
        run = [RING[(start + i) % len(RING)] for i in range(count)]
        if all(n in names for n in run) and sorted(run) not in [sorted(r) for r in runs]:
            runs.append(run)
    return runs


def plan(nodes: list, live: list, waiting: list, learned: dict) -> tuple[list, dict]:
    """(starts, reasons). starts: (entry, [nodes]) in start order. reasons: entry id -> why it still waits.

    Higher priority first. A job that does not fit never holds back a smaller one behind it, so no node stays idle
    while a waiting job fits. Each start is booked at once, so one pass never oversubscribes a node.
    """
    live, starts, reasons = [dict(job) for job in live], [], {}
    by_name = {node["name"]: node for node in nodes}
    for entry in by_priority(waiting):
        need, width = estimate(entry, learned), int(entry.get("nodes") or 1)
        sets = [[name] for name in by_name] if width == 1 else neighbours(list(by_name), width)
        best, why = None, []
        for names in sets:
            said = [refusal(by_name[name], live, entry, need) for name in names]
            if any(said):
                why += [text for text in said if text]
                continue
            score = (max(fullness(by_name[name], live, need) for name in names),
                     sum(booked(by_name[name], live)["jobs"] for name in names), RING.index(names[0]))
            if best is None or score < best[0]:
                best = (score, names)
        if best is None:
            reasons[entry["id"]] = "; ".join(dict.fromkeys(why))[:300] or "no job node in the applied profile"
            continue
        starts.append((entry, best[1]))
        live.append({"id": entry["id"], "nodes": best[1], "need": need, "mem_now": None})
    return starts, reasons
