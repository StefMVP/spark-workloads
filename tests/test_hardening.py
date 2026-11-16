"""Faults of the live drills of 2026-11-14, each as a test. No host is contacted."""
import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/lib"))
sys.path.insert(0, str(ROOT / "tests"))
import assetrestore
import labprofile
import switch
import visionjob
from clustercontrol import Refused
from spark_world import IMAGE, World
from test_vision_job import arguments


class Starts(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.world = World(self.dir.name)
        self.runner = visionjob.Runner(self.world, self.world.controller, clock=self.world.clock)

    def submit(self, **changes):
        with contextlib.redirect_stdout(io.StringIO()):
            visionjob.handle(self.world, self.world.controller, arguments(**changes), "submit")
        return self.world.controller.store.read()

    def test_a_shard_set_starts_in_one_parallel_pass_with_one_queue_commit(self):
        before = self.world.controller.store.read()["revision"]
        calls = []
        pmap = self.world.pmap
        self.world.pmap = lambda fn, items: calls.append(len(list(items))) or pmap(fn, items)
        state = self.submit(shards=12, cmd=["python", "job.py", "{shard}"])
        self.assertEqual(sum(j["status"] == "running" for j in state["jobs"].values()), 12)
        self.assertIn(12, calls)                                  # the twelve starts went to pmap together
        self.assertLessEqual(state["revision"] - before, 1 + 12 * 4)   # one enqueue; dequeue, admit, permit, observe per job

    def test_a_start_that_times_out_puts_the_job_back_with_one_line(self):
        self.world.run_times_out = True
        state = self.submit()
        job, = state["jobs"].values()
        self.assertIn("timed out", job["blocker"])                # the container state is unknown: the record stays open
        self.world.run_times_out = False

    def test_a_failed_copy_of_the_code_puts_the_job_back_with_one_line(self):
        def broken(entry, node, fact, need, gang=None, last=True):
            raise subprocess.TimeoutExpired("git clone", 600)
        self.runner.start = broken
        self.world.controller.enqueue({"id": "q1", "by": "t", "image": IMAGE, "cmd": "python job.py", "mem": 1, "cpus": 1})
        self.runner.dispatch()
        entry, = self.world.controller.store.read()["queue"].values()
        self.assertRegex(entry["reason"], r"the start on spark\d failed \(TimeoutExpired.*\); the job waits and starts again by itself")

    def test_finished_containers_are_removed_and_their_output_is_kept(self):
        state = self.submit(shards=3, cmd=["python", "job.py", "{shard}"])
        self.world.advance(30)
        self.runner.dispatch()
        self.assertEqual(self.world.kept, {})                     # a running container is never removed
        self.world.advance(120)
        self.runner.dispatch()                                    # settles the exits, then removes the containers
        self.assertEqual(sorted(self.world.kept), sorted(j["command"]["container"] for j in state["jobs"].values()))
        job = next(iter(state["jobs"].values()))
        text = visionjob.logs_command(job, tail=50)
        self.assertIn(f"docker logs {job['command']['container']}", text)
        self.assertIn(f'tail -n 50 "$HOME/spark/jobs/{job["id"]}/job.log"', text)

    def test_the_container_has_an_init_so_a_stop_does_not_wait_for_the_grace_time(self):
        self.assertIn("docker run -d --init ", visionjob.launch_script("vi1", "s1", IMAGE, "python x.py", "/work", False, 2))

    def test_a_blender_job_gets_the_native_blender_and_only_it_does(self):
        plain = visionjob.launch_script("vi1", "s1", IMAGE, "blender -b", "/work", False, 4)
        self.assertNotIn("/opt/blender", plain)
        text = visionjob.launch_script("vi1", "s1", IMAGE, "blender -b", "/work", False, 4, blender=True)
        self.assertIn('readlink -f "$HOME/.local/bin/blender"', text)
        self.assertIn("$B:/opt/blender:ro", text)
        self.assertIn("$HOME/.cache/cycles:/work/.cache/cycles", text)
        self.assertIn("--gpus all", text)
        self.assertIn("export PATH=/opt/blender:$PATH", text)
        import argparse
        parser = argparse.ArgumentParser(); sub = parser.add_subparsers()
        visionjob.register(type("C", (), {})(), sub)
        parsed = parser.parse_args(["job", "submit", "--blender", "--", "blender", "-b"])
        entry = visionjob.build(type("C", (), {"me": lambda self, a: "t", "RING": (), "now": lambda self: 0, "warn": print, "STATE": "/tmp/x"})(),
                                type("R", (), {"resolve": lambda self, a: (None, None, None, "", None)})(), parsed)[0]
        self.assertEqual((entry["class"], entry["gpu"], entry["mem"], entry["cpus"]), ("blender", True, 16, 16))

    def test_a_shell_loop_reaches_bash_unchanged(self):
        for argv in (["sh", "-c", 'for f in a "b c"; do echo "<$f>"; done'], ['for f in a "b c"; do echo "<$f>"; done']):
            text = visionjob.entry(visionjob.command_text(argv), False)
            out = subprocess.run(["bash", "-c", text], capture_output=True, text=True, timeout=20)
            self.assertEqual((out.returncode, out.stdout), (0, "<a>\n<b c>\n"), out.stderr)


    def test_the_backfill_line_of_1454_runs_as_written_also_through_train_submit(self):
        loop = ("for s in 2913 2396 380; do PYTHONUNBUFFERED=1 python twin/tools/b2lake.py mirror $s "
                "--days 2026-10-21..2026-11-14 --workers 64 & done; wait")
        loader = importlib.machinery.SourceFileLoader("spark_cli_alias", str(ROOT / "tools/cluster"))
        cli = importlib.util.module_from_spec(importlib.util.spec_from_loader("spark_cli_alias", loader))
        loader.exec_module(cli)
        argv = ["spark", "--as", "backfill", "train", "submit", "s3", "--repo", "/r", "--ref", "twin/lab-spine",
                "--env-file-secret", "F1", "--", "sh", "-c", loop]
        with contextlib.redirect_stderr(io.StringIO()):
            cli.alias(argv)                                       # `train submit GROUP` is `job submit`: one code path
        self.assertEqual(argv[3:5] + argv[-4:], ["job", "submit", "--", "sh", "-c", loop])
        with tempfile.TemporaryDirectory() as folder:             # a `python` that prints its arguments
            stub = Path(folder) / "python"
            stub.write_text('#!/bin/sh\necho "$PYTHONUNBUFFERED $2 $3"\n')
            stub.chmod(0o755)
            text = visionjob.entry(visionjob.command_text(argv[argv.index("--") + 1:]), False).replace("bash -lc", "bash -c")
            out = subprocess.run(["bash", "-c", text], capture_output=True, text=True, timeout=20,
                                 env={"PATH": f"{folder}:/usr/bin:/bin"})
        self.assertEqual((out.returncode, sorted(out.stdout.split("\n"))), (0, ["", "1 mirror 2396", "1 mirror 2913", "1 mirror 380"]), out.stderr)


    def test_a_node_that_returns_before_its_jobs_restart_keeps_them_running(self):
        state = self.submit()
        job, = state["jobs"].values()
        node = job["command"]["node"]
        for other in self.world.images:                           # no other node can take the copy
            if other != node:
                self.world.images[other] = {}
        self.world.silence(node)
        self.runner.dispatch()
        self.world.t += 90
        self.runner.dispatch()                                    # lost: the copy waits in the queue
        self.assertEqual(len(self.world.controller.store.read()["queue"]), 1)
        self.world.down.discard(node)
        self.runner.dispatch()
        state = self.world.controller.store.read()
        self.assertEqual((state["queue"], state["jobs"][job["id"]]["status"], state["jobs"][job["id"]]["command"]["superseded"]),
                         ({}, "running", ""))
        self.assertFalse(any(call["cmd"].startswith("docker stop") for call in self.world.calls))
        self.assertEqual(len(state["jobs"]), 1)

    def test_a_gang_is_requeued_once_and_a_failed_rank_stops_the_others(self):
        state = self.submit(nodes=2, mem=8, cpus=2)
        ranks = sorted(state["jobs"].values(), key=lambda j: j["command"]["rank"])
        self.assertEqual(len(ranks), 2)
        self.world.controller.hold_queue("a profile switch runs", self.world.t)
        for job in ranks:
            self.runner.requeue(job)
        self.assertEqual(len(self.world.controller.store.read()["queue"]), 1)       # not one gang per rank
        self.world.controller.hold_queue("", 0)
        started, _ = self.runner.dispatch()
        self.assertEqual(len(started), 2)
        first = next(c for c in self.world.containers.values() if c["name"] == started[0]["command"]["container"])
        first["end"], first["code"] = self.world.t, 1                               # rank 0 fails
        self.runner.dispatch()
        self.world.advance(1)
        state = self.runner.refresh()
        other = state["jobs"][started[1]["id"]]
        self.assertEqual(other["status"], "settled")
        self.assertRegex(other["blocker"], r"rank 0 \(.*\) exited 1; this rank was stopped")

    def test_the_secret_audit_sends_values_only_through_stdin(self):
        env = Path(self.dir.name) / "k.env"
        env.write_text("KEY=abcdefghijklmnop-secret\nSHORT=abc\n")
        self.world.ssh = lambda node, cmd, timeout=60, input=None: (self.world.calls.append({"node": node, "cmd": cmd, "input": input})
                                                                    or subprocess.CompletedProcess([], 0, "files 10\ndone\n", ""))
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as end:
            visionjob.audit(self.world, argparse.Namespace(env_file_secret=[str(env)], image="ubuntu:24.04", days=3))
        self.assertEqual(end.exception.code, 0)
        self.assertEqual({call["input"] for call in self.world.calls}, {"abcdefghijklmnop-secret\n"})
        self.assertFalse(any("abcdefghijklmnop" in call["cmd"] for call in self.world.calls))
        self.assertNotIn("abcdefghijklmnop", out.getvalue())
        self.assertIn("--network none", self.world.calls[0]["cmd"])
        self.assertIn(":ro", self.world.calls[0]["cmd"])


class RollbackKeys(unittest.TestCase):
    """A rollback is a second request: the controller refuses one request key for two profiles."""
    def drive(self, fault):
        seen, state = [], {"profile": "jobs-studio", "used": {}}
        driver = switch.Driver.__new__(switch.Driver)
        driver.keys, driver.say = {}, lambda text: None
        driver.current = lambda: state["profile"]
        driver.applied = lambda name: state["profile"] == name
        driver.preflight = lambda name: []
        driver.unload = lambda name: None
        driver.resume = lambda: []
        driver.verify = lambda name: [{"ok": True, "line": "ok"}]
        def lease(name):
            state["profile"] = None
        def load(name):                                         # the key rule of Controller.request
            key = driver.keys.setdefault(name, f"key-{len(driver.keys)}")
            if state["used"].setdefault(key, name) != name:
                raise Refused("idempotency key reused with different intent")
            state["profile"] = name
        driver.lease, driver.load = lease, load
        lines = []
        code = switch.Switch(driver, lines.append, fault).run("jobs")
        return code, state["profile"], lines

    def test_a_fault_in_each_phase_returns_to_the_earlier_profile(self):
        for fault in ("unload", "lease", "load", "verify"):
            code, profile, lines = self.drive(fault)
            self.assertEqual((code, profile), (1, "jobs-studio"), (fault, lines))
            self.assertTrue(lines[-1].startswith("rolled back: jobs-studio is applied and verified; jobs failed in " + fault))

    def test_the_real_driver_uses_one_key_per_target(self):
        import inspect
        source = inspect.getsource(switch.Driver.load)
        self.assertIn("self.keys.setdefault(name", source)
        self.assertNotIn("self.key,", source)


class Deadline(unittest.TestCase):
    def test_a_switch_past_its_maximum_time_returns_to_the_earlier_profile_and_releases_the_queue(self):
        from test_switch import FakeDriver
        now = [0.0]
        driver = FakeDriver("jobs")
        slow = driver.unload
        def unload(name):
            slow(name)
            if name == "studio-4":
                now[0] += 400                                   # the stop of the jobs takes 400 s
        driver.unload = unload
        lines = []
        code = switch.Switch(driver, lines.append, None, 300, lambda: now[0]).run("studio-4")
        self.assertEqual((code, driver.profile, driver.resumed), (1, "jobs", 1))
        self.assertIn("FAILED in unload: the switch passed its maximum time of 300 s (--max-seconds) in unload", lines)
        self.assertTrue(lines[-1].startswith("rolled back: jobs is applied and verified; studio-4 failed in unload"))

    def test_a_drill_does_not_stop_a_job_that_was_stopped_twice_in_30_minutes(self):
        job = lambda stops: {"id": "vi1", "kind": "vision", "attempt_id": "a", "status": "running",
                             "command": {"by": "detector-climb", "entry": {"stops": stops}}}
        self.assertEqual(switch.stopped_often({"jobs": {"vi1": job([1000, 1500])}}, 2000), ["vi1 of detector-climb"])
        self.assertEqual(switch.stopped_often({"jobs": {"vi1": job([1000])}}, 2000), [])
        self.assertEqual(switch.stopped_often({"jobs": {"vi1": job([1000, 1500])}}, 1000 + 1801), [])

    def test_each_requeue_is_counted_on_the_job(self):
        with tempfile.TemporaryDirectory() as folder:
            world = World(folder)
            runner = visionjob.Runner(world, world.controller, clock=world.clock)
            with contextlib.redirect_stdout(io.StringIO()):
                visionjob.handle(world, world.controller, arguments(), "submit")
            for _ in range(2):
                job = next(j for j in world.controller.store.read()["jobs"].values() if j["status"] == "running")
                runner.requeue(job)
                world.advance(30)
                runner.dispatch()
            job = next(j for j in world.controller.store.read()["jobs"].values() if j["status"] == "running")
            self.assertEqual(len(job["command"]["entry"]["stops"]), 2)


class UtilCheck(unittest.TestCase):
    def test_a_hold_past_the_limit_and_no_profile_are_one_line_and_one_alert(self):
        import sparkutil
        with tempfile.TemporaryDirectory() as folder:
            world = World(folder)
            self.assertEqual(sparkutil.check(world, world.controller, now=world.t), "")
            world.controller.hold_queue("a profile switch runs", world.t)
            self.assertEqual(sparkutil.check(world, world.controller, now=world.t + 200), "")
            line = sparkutil.check(world, world.controller, now=world.t + 400)
            self.assertIn("the queue is held for 6 min (a profile switch runs), over the 5 min limit", line)
            t = world.t                                           # one message per cause per 15 min, one when it is over
            self.assertEqual([sparkutil.alert(world, line, t), sparkutil.alert(world, line, t + 60), sparkutil.alert(world, line, t + 600),
                              sparkutil.alert(world, line, t + 901), sparkutil.alert(world, "", t + 930), sparkutil.alert(world, "", t + 960)],
                             [True, False, False, True, True, False])

    def test_a_studio_node_is_not_flagged_idle(self):
        import sparkutil
        with tempfile.TemporaryDirectory() as folder:
            world = World(folder)
            idle = {"gpu": 0, "load": 0.1, "cores": 20, "mem_free": 100, "jobs": 0}
            (Path(folder) / "util").mkdir()
            import json, time
            rows = [{"t": int(time.time()) - 400 + 30 * k, "queued": 3, "nodes": {n: idle for n in ("spark1", "spark2", "spark3", "spark4")}} for k in range(14)]
            (Path(folder) / "util" / "2026-11-14.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            sparkutil.rows = lambda cluster, hours: rows
            def studio3(control):
                control["operations"]["op"]["spec"]["reservations"].pop("s3")
            world.controller.store.commit(studio3)
            line = sparkutil.check(world, world.controller)
            self.assertIn("spark1 is idle for 6 min while jobs wait", line)
            self.assertNotIn("spark3", line)


class AssetRemoval(unittest.TestCase):
    def test_a_file_that_a_profile_needs_is_not_removed_without_a_copy_on_a_peer(self):
        base = Assets()
        cluster, calls = base.cluster({"spark4": 100 * 10**9}, free=400)
        assetrestore.users = lambda c, node, path: ["deepseek-1017-exl3-2x2"]
        lines = []
        with self.assertRaisesRegex(Refused, "refused, nothing was removed: deepseek-1017-exl3-2x2 need this file and no peer"):
            assetrestore.remove(cluster, "spark4", Assets.PATH, True, lines.append)
        self.assertIn("is needed by: deepseek-1017-exl3-2x2; copies of the same size: none", lines[0])
        self.assertFalse(any(cmd.startswith("rm ") for _, cmd in calls))
        cluster, calls = base.cluster({"spark4": 100 * 10**9, "spark2": 100 * 10**9}, free=400)
        assetrestore.remove(cluster, "spark4", Assets.PATH, False, lines.append)
        self.assertFalse(any(cmd.startswith("rm ") for _, cmd in calls))                # no --yes: a report only
        assetrestore.remove(cluster, "spark4", Assets.PATH, True, lines.append)
        self.assertTrue(any(node == "spark4" and cmd.startswith("rm -f") for node, cmd in calls))
        self.assertIn("copies it back from spark2", lines[-1])


class OnceJobs(unittest.TestCase):
    """The switch of 2026-11-15 waited 270 s for a --once job and then up to 20 s more for its own poll."""
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.world = World(self.dir.name)
        with contextlib.redirect_stdout(io.StringIO()):
            visionjob.handle(self.world, self.world.controller, arguments(once=True, cmd=["python", "kill_test.py"]), "submit")
            visionjob.handle(self.world, self.world.controller, arguments(), "submit")
        self.lines = []
        self.driver = switch.Driver(self.world, self.world.controller, runtime=object(), say=self.lines.append)
        self.owner = self.world.controller.store.read()["operations"]["op"]
        self.slept = []
        def sleep(seconds):                                     # the world moves on while the switch waits
            self.slept.append(seconds)
            self.world.advance(seconds)
        patch = unittest.mock.patch.object(switch.time, "sleep", sleep)
        patch.start()
        self.addCleanup(patch.stop)

    def states(self):
        return sorted((j["command"]["cmd"], j["status"]) for j in self.world.controller.store.read()["jobs"].values())

    def test_cancel_once_stops_the_once_job_without_a_wait(self):
        self.driver.cancel_once = True
        self.driver.requeue(self.owner)
        self.assertEqual(self.slept, [])
        self.assertEqual(self.states(), [("python job.py", "settled"), ("python kill_test.py", "settled")])
        self.assertTrue(any("(--once) stopped: settled" in line for line in self.lines))
        self.assertEqual(len(self.world.controller.store.read()["queue"]), 1)    # only the restartable job waits again

    def test_the_wait_for_a_once_job_ends_within_one_poll_of_its_end(self):
        self.driver.requeue(self.owner)                         # the job ends after 90 s (spark_world)
        self.assertEqual(set(self.slept), {switch.POLL_S})
        self.assertLessEqual(sum(self.slept) - 90, switch.POLL_S)
        self.assertIn("--cancel-once", self.lines[0])


class HeadStop(unittest.TestCase):
    def runtime(self, replies):
        calls = []
        def ssh(node, cmd, timeout=60, input=None):
            calls.append(cmd)
            return subprocess.CompletedProcess([], 0, replies.pop(0), "")
        return types.SimpleNamespace(c=types.SimpleNamespace(ssh=ssh)), calls

    def test_a_head_that_ignores_the_stop_is_killed_after_the_grace_time_and_the_tool_says_so(self):
        runtime, calls = self.runtime(["killed\n", "inactive\n"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(labprofile.stop_head(runtime, grace=10), "killed")
        self.assertIn("-lt 10", calls[0])
        self.assertIn("kill --kill-whom=all --signal=KILL", calls[0])
        self.assertIn("ignored the stop signal for 10 s and was killed", out.getvalue())

    def test_a_head_that_stays_active_is_a_refusal(self):
        runtime, _ = self.runtime(["killed\n", "active\n"])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(Refused, "did not stop"):
            labprofile.stop_head(runtime)


class Assets(unittest.TestCase):
    PATH = "/opt/fixtures/model/rank1.bin"

    def cluster(self, sizes, free):
        calls = []
        def ssh(node, cmd, timeout=60, input=None):
            calls.append((node, cmd))
            if cmd.startswith("python3 -c"):
                import json
                return subprocess.CompletedProcess([], 0, json.dumps({"sizes": {self.PATH: sizes.get(node)}, "free": {self.PATH: free * 10**9}}), "")
            if cmd.startswith("sha256sum"):
                return subprocess.CompletedProcess([], 0, "ab" * 32 + "\n", "")
            return subprocess.CompletedProcess([], 0, "", "")
        cluster = types.SimpleNamespace(RING=["spark1", "spark2", "spark3", "spark4"], ssh=ssh,
                                        pmap=lambda fn, items: {item: fn(item) for item in items})
        return cluster, calls

    SPEC = {"name": "model-2x2", "instances": [{"id": "pair34", "cache": {"nodes": {"spark4": {"stamps": {PATH: [100 * 10**9, 1, 2]}}}}}]}

    def test_a_missing_asset_is_copied_from_the_peer_that_has_it_and_hashed(self):
        cluster, calls = self.cluster({"spark2": 100 * 10**9}, free=400)
        lines = []
        self.assertEqual(assetrestore.restore(cluster, self.SPEC, lines.append), [])
        self.assertTrue(any(node == "spark4" and "rsync" in cmd and "spark2:" in cmd for node, cmd in calls))
        self.assertIn("restored from spark2", lines[-1])

    def test_a_copy_that_would_fill_the_disk_is_a_blocker_with_the_exact_need(self):
        cluster, calls = self.cluster({"spark2": 100 * 10**9}, free=117)
        (what, fix), = assetrestore.restore(cluster, self.SPEC, lambda text: None)
        self.assertIn("not restored: 117 GB disk free, the copy needs 150 GB", what)
        self.assertEqual(fix, "free 33 GB on spark4, then run the same command")
        self.assertFalse(any("rsync" in cmd for _, cmd in calls))

    def test_no_peer_and_a_lab_profile(self):
        cluster, _ = self.cluster({}, free=400)
        (what, fix), = assetrestore.restore(cluster, self.SPEC, lambda text: None)
        self.assertIn("no peer has it; nothing was copied", what)
        self.assertEqual(assetrestore.restore(cluster, {"name": "jobs", "instances": []}, lambda text: None), [])


class Interpreter(unittest.TestCase):
    def test_an_interpreter_without_the_dependencies_is_refused_before_any_effect(self):
        loader = importlib.machinery.SourceFileLoader("spark_cli", str(ROOT / "tools/cluster"))
        cli = importlib.util.module_from_spec(importlib.util.spec_from_loader("spark_cli", loader))
        loader.exec_module(cli)
        cli.require_runtime(find=lambda name: object())
        with self.assertRaises(SystemExit) as refused:
            cli.require_runtime(find=lambda name: None if name == "pydantic" else object())
        self.assertRegex(str(refused.exception), r"lacks pydantic; nothing was changed\. Run `spark")

    def test_the_twin_lab_profiles_verify_spark4_with_the_vision_image(self):
        for name in ("jobs", "jobs-studio", "jobs-studio-pair"):
            self.assertIn("spark4: spark-workload:local", (ROOT / f"profiles/{name}.yaml").read_text())

    def test_a_lab_profile_takes_studio_workers_and_counts_them_in_the_default(self):
        import clusterprofile
        self.assertIn("jobs-studio-pair", switch.targets(ROOT))
        spec = clusterprofile.load(ROOT, "jobs-studio-pair")
        self.assertEqual((spec["studio"]["default"], set(spec["images"])), (2, {"spark1", "spark4"}))
        base = clusterprofile.load(ROOT, "jobs-studio")
        bad = [dict(spec, studio=dict(spec["studio"], default=1)),                       # default must count the worker
               dict(spec, studio={"default": 2, "workers": dict(spec["studio"]["workers"], spark1=True)}),  # worker on a job node
               dict(spec, studio={"default": 2, "workers": {"spark2": True}})]           # every worker is named
        for p in bad:
            with self.assertRaises(Refused):
                labprofile.validate(p)
        labprofile.validate(base)


if __name__ == "__main__":
    unittest.main()
