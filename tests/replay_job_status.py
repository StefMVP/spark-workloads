"""Isolated status deadline replay; no Spark host or service is contacted.

Use --module to compare an extracted historical visionjob.py with the current file.
The outer deadline kills only this replay's child when the old dispatch lock blocks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_world import World
from test_vision_job import arguments
import visionjob


def child(args):
    with tempfile.TemporaryDirectory() as folder:
        world = World(folder)
        with contextlib.redirect_stdout(io.StringIO()):
            visionjob.handle(world, world.controller, arguments(shards=4), "submit")
        if args.module:
            spec = importlib.util.spec_from_file_location("baseline_status", args.module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = visionjob
        before = world.STATE.read_bytes()
        calls = []
        def slow(node, command, timeout=60, input=None):
            calls.append({"node": node, "timeout": timeout, "command": command})
            return subprocess.run([sys.executable, "-c", "import time; time.sleep(120)"],
                                  capture_output=True, text=True, timeout=timeout)
        if args.case == "slow":
            world.ssh = slow
            world.pmap = lambda fn, items: parallel(fn, list(items))
        lock = (Path(folder) / "dispatch.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX if args.case == "lock" else fcntl.LOCK_UN)
        start = time.monotonic()
        out = io.StringIO()
        code = 0
        try:
            with contextlib.redirect_stdout(out):
                module.handle(world, world.controller, arguments(json=True, max_seconds=args.budget), "status")
        except SystemExit as error:
            code = error.code
        result = json.loads(out.getvalue())
        unchanged = before == world.STATE.read_bytes()
        print(json.dumps({"elapsed_s": time.monotonic() - start, "exit": code,
                          "unchanged": unchanged, "calls": calls, "status_read": result.get("status_read"),
                          "jobs": len(result["jobs"]), "waiting": len(result["waiting"])}))
        if not args.module:
            assert unchanged
            assert len(result["jobs"]) == 4
            assert result["status_read"]["state"] == ("partial" if args.case == "slow" else "current")
            assert code == (3 if args.case == "slow" else 0)
            assert time.monotonic() - start < args.budget + 0.5


def parallel(fn, items):
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return dict(zip(items, pool.map(fn, items)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path)
    p.add_argument("--case", choices=("slow", "lock"), default="slow")
    p.add_argument("--budget", type=float, default=0.2)
    p.add_argument("--outer", type=float, default=3.0)
    p.add_argument("--callers", type=int, default=1)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    if args.child:
        child(args)
        return
    def one(index):
        start = time.monotonic()
        cmd = [sys.executable, __file__, "--child", "--case", args.case, "--budget", str(args.budget)]
        if args.module:
            cmd += ["--module", str(args.module)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=args.outer)
            if process.returncode:
                raise RuntimeError(stderr)
            return {"caller": index, "wall_s": time.monotonic() - start, **json.loads(stdout)}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            if not args.module:
                raise
            return {"caller": index, "wall_s": time.monotonic() - start, "outer_timeout": True}
    print(json.dumps({"module": str(args.module or "current"), "case": args.case,
                      "runs": list(parallel(one, range(args.callers)).values())}))


if __name__ == "__main__":
    main()
