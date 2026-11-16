"""Router jobs: durable order, secret env, argv, waiting and automatic start, priority, shards, requeue. No host is contacted."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_world import IMAGE, RING, UUID, World
import visionjob
from clustercontrol import Refused

SECRET = "K005veryNotARealKeyValue+/x9"


def arguments(**changes):
    values = dict(as_="harvest", image=IMAGE, repo=None, ref="HEAD", workspace=None, allow_dirty=False, out=None, mem=8, cpus=2,
                  env_file_secret=None, wait=False, light=False, cpu_only=False, gpu=False, priority=0, name=None, nodes=1,
                  only=None, restartable=False, once=False, shards=None, items=None, cmd=["python", "job.py"])
    return argparse.Namespace(**dict(values, **changes))


class RouterJobs(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.world = World(self.dir.name)
        self.controller = self.world.controller
        self.runner = visionjob.Runner(self.world, self.controller, clock=self.world.clock)
        self.env = Path(self.dir.name) / ".env.b2"
        self.env.write_text(f"# key\nB2_READONLY_KEY_ID=id123\nB2_READONLY_APPLICATION_KEY={SECRET}\n")

    def submit(self, **changes):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            visionjob.handle(self.world, self.controller, arguments(**changes), "submit")
        self.said = out.getvalue()
        return self.jobs()

    def jobs(self):
        return list(self.controller.store.read()["jobs"].values())

    def queue(self):
        return list(self.controller.store.read().get("queue", {}).values())

    def test_secret_values_reach_only_stdin(self):
        job, = self.submit(env_file_secret=[str(self.env)])
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["command"]["secret_env"], ["B2_READONLY_APPLICATION_KEY", "B2_READONLY_KEY_ID"])
        for text in (self.world.STATE.read_text(), json.dumps(job), self.said, *(call["cmd"] for call in self.world.calls)):
            self.assertNotIn(SECRET, text)
            self.assertNotIn("id123", text)
        fed = [call for call in self.world.calls if call["input"]]
        self.assertEqual(len(fed), 1)
        self.assertIn(f"B2_READONLY_APPLICATION_KEY={SECRET}\n", fed[0]["input"])
        launch = fed[0]["cmd"]
        self.assertIn("--tmpfs /run/spark-secret", launch)       # RAM only, private to the container
        self.assertIn("rm -f /run/spark-secret/env", launch)     # the FIFO is removed before the job command
        self.assertNotRegex(launch, r"-e '?B2_|--env B2_|--env-file /dev/stdin")

    def test_a_waiting_job_keeps_only_the_secret_file_path(self):
        self.world.images = {node: {} for node in RING}                       # no node can start it now
        self.submit(env_file_secret=[str(self.env)])
        self.assertEqual([e["env_files"] for e in self.queue()], [[str(self.env)]])
        self.assertNotIn(SECRET, self.world.STATE.read_text())

    def test_start_is_observed_and_exit_settles(self):
        job, = self.submit()
        resource = job["resources"][0]
        self.assertEqual(job["births"][resource]["identity"], f"{1:064x}")
        self.world.advance(120)
        job = self.runner.refresh()["jobs"][job["id"]]
        self.assertEqual((job["status"], job["outcomes"][resource]["exit_code"]), ("settled", 0))

    def test_status_observes_exit_without_changing_saved_jobs(self):
        job, = self.submit()
        self.world.advance(120)
        before = self.world.STATE.read_bytes()
        self.world.calls.clear()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            visionjob.handle(self.world, self.controller,
                             arguments(id=job["id"], json=True, max_seconds=1), "status")
        result = json.loads(out.getvalue())
        self.assertEqual(self.world.STATE.read_bytes(), before)
        self.assertEqual(result["jobs"][0]["status"], "settled")
        self.assertEqual(result["status_read"]["state"], "current")
        self.assertTrue(all("docker inspect" in call["cmd"] for call in self.world.calls))

    def test_status_timeout_keeps_recorded_state_and_names_backend(self):
        job, = self.submit()
        self.world.down.add(job["command"]["node"])
        before = self.world.STATE.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as exit:
            visionjob.handle(self.world, self.controller,
                             arguments(id=job["id"], json=True, max_seconds=0.1), "status")
        result = json.loads(out.getvalue())
        self.assertEqual(exit.exception.code, 3)
        self.assertEqual(result["jobs"][0]["status"], "running")
        self.assertEqual(result["status_read"]["state"], "partial")
        self.assertIn(job["command"]["node"], result["status_read"]["errors"])
        self.assertEqual(self.world.STATE.read_bytes(), before)

    def test_status_partial_keeps_healthy_node_and_failed_exit(self):
        self.submit(shards=4)
        jobs = self.jobs()
        failed = jobs[0]
        self.world.advance(120)
        self.world.containers[failed["births"][failed["resources"][0]]["identity"]]["code"] = 7
        silent = jobs[-1]
        self.world.down.add(silent["command"]["node"])
        before = self.world.STATE.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            visionjob.handle(self.world, self.controller, arguments(json=True), "status")
        result = json.loads(out.getvalue())
        seen = {j["id"]: j for j in result["jobs"]}
        self.assertEqual(visionjob.Runner(self.world, self.controller).exit_code(seen[failed["id"]]), 7)
        self.assertEqual(seen[silent["id"]]["status"], "running")
        self.assertEqual(len(result["status_read"]["errors"]), 1)
        self.assertEqual(self.world.STATE.read_bytes(), before)

    def test_status_finished_and_queued_records_need_no_backend(self):
        job, = self.submit()
        self.world.advance(120)
        self.runner.refresh()
        self.controller.hold_queue("test hold", self.world.t)
        self.submit()
        with patch.object(self.world, "ssh", side_effect=AssertionError("unexpected probe")):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                visionjob.handle(self.world, self.controller, arguments(json=True, all=True), "status")
        result = json.loads(out.getvalue())
        self.assertEqual(result["jobs"][0]["id"], job["id"])
        self.assertEqual(len(result["waiting"]), 1)
        self.assertEqual(result["status_read"]["state"], "current")

    def test_status_does_not_claim_current_for_wrong_birth_or_transport_error(self):
        job, = self.submit()
        ident = job["births"][job["resources"][0]]["identity"]
        import subprocess
        for reply in (subprocess.CompletedProcess([], 0, f"{ident} exited 0 wrong-start end\n", ""),
                      subprocess.CompletedProcess([], 255, "", "connection refused")):
            with self.subTest(reply=reply), patch.object(self.world, "ssh", return_value=reply):
                with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                    visionjob.handle(self.world, self.controller, arguments(json=True), "status")
                result = json.loads(out.getvalue())
                self.assertEqual(result["jobs"][0]["status"], "running")
                self.assertEqual(result["jobs"][0]["observation"]["source"], "controller record")

    def test_status_budget_rejects_invalid_values(self):
        for value in ("nan", "inf", "0", "-1", "61"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                visionjob.status_seconds(value)

    def test_status_deadline_with_slow_backends_and_held_dispatch_lock(self):
        import subprocess
        replay = Path(__file__).with_name("replay_job_status.py")
        for case in ("slow", "lock"):
            with self.subTest(case=case):
                reply = subprocess.run([sys.executable, str(replay), "--case", case, "--callers", "2", "--outer", "5"],
                                       capture_output=True, text=True, timeout=15)
                self.assertEqual(reply.returncode, 0, reply.stderr)
                result = json.loads(reply.stdout)
                self.assertEqual(len(result["runs"]), 2)
                self.assertTrue(all(run["unchanged"] for run in result["runs"]))

    def test_failed_start_with_no_container_settles_and_frees_the_node(self):
        self.world.run_fails = True
        job, = self.submit()
        self.assertEqual((job["status"], next(iter(job["outcomes"].values()))["state"]), ("settled", "never_created"))

    def test_an_argv_reaches_the_job_unchanged_and_one_argument_is_shell_text(self):
        loop = "for i in 1 2; do echo $i; done"
        self.submit(cmd=["sh", "-c", loop])
        self.submit(cmd=["echo a && echo b"])
        ran = [c["cmd"] for c in self.world.containers.values()]
        self.assertEqual(ran, [f"sh -c '{loop}'", "echo a && echo b"])

    def test_no_agent_names_a_node_and_shards_use_all_four(self):
        jobs = self.submit(shards=8, cmd=["python", "harvest.py", "--shard", "{shard}/{nshards}"])
        self.assertEqual(sorted({j["command"]["node"] for j in jobs}), RING)
        self.assertEqual(sorted(j["command"]["cmd"] for j in jobs)[0], "python harvest.py --shard '0/8'")
        self.assertFalse(hasattr(arguments(), "group"))

    def test_an_item_list_makes_one_job_per_line(self):
        items = Path(self.dir.name) / "items.txt"
        items.write_text("store 2913\n\nstore 380\n")
        jobs = self.submit(items=str(items), light=True, cmd=["python", "mirror.py"])
        self.assertEqual(sorted(j["command"]["cmd"] for j in jobs), ["python mirror.py 'store 2913'", "python mirror.py 'store 380'"])

    def test_a_job_with_no_room_waits_and_starts_by_itself(self):
        jobs = self.submit(shards=10, mem=50, cpu_only=True, cmd=["python", "index.py", "{shard}"])
        self.assertEqual((len(jobs), len(self.queue())), (8, 2))               # two 50 GiB jobs per node
        self.assertIn("GiB needed", self.queue()[0]["reason"])
        self.world.advance(120)                                                 # the first jobs end
        started, _ = self.runner.dispatch()                                     # what the 30 s timer runs
        self.assertEqual((len(started), self.queue()), (2, []))

    def test_light_jobs_pack_by_cores_without_a_gpu(self):
        jobs = self.submit(shards=40, light=True, cmd=["python", "mirror.py", "{shard}"])
        self.assertEqual(len(jobs), 40)
        self.assertTrue(all("--gpus" not in c["launch"] for c in self.world.containers.values()))
        self.assertEqual({sum(j["command"]["node"] == n for j in jobs) for n in RING}, {10})

    def test_higher_priority_starts_first(self):
        self.world.images = {node: ({IMAGE: "sha256:" + "cd" * 32} if node == "spark1" else {}) for node in RING}
        self.submit(mem=90, cpu_only=True, cmd=["python", "big.py"])
        self.submit(mem=90, cpu_only=True, cmd=["python", "low.py"])
        self.submit(mem=90, cpu_only=True, priority=5, cmd=["python", "high.py"])
        self.world.advance(120)
        started, _ = self.runner.dispatch()
        self.assertEqual([j["command"]["cmd"] for j in started], ["python high.py"])

    def test_a_node_that_the_profile_gives_to_studio_gets_no_job(self):
        def mixed(control):
            del control["operations"]["op"]["spec"]["reservations"]["s3"]
        self.controller.store.commit(mixed)
        jobs = self.submit(shards=9, cmd=["python", "h.py", "{shard}"])
        self.assertEqual(sorted({j["command"]["node"] for j in jobs}), ["spark1", "spark2", "spark4"])

    def test_a_job_over_two_nodes_gets_ring_neighbours(self):
        jobs = self.submit(nodes=2, mem=40, cmd=["python", "ddp.py"])
        nodes = [j["command"]["node"] for j in jobs]
        self.assertEqual(len(jobs), 2)
        self.assertEqual(abs(RING.index(nodes[0]) - RING.index(nodes[1])) in (1, 3), True)
        self.assertEqual(sorted(j["command"]["rank"] for j in jobs), ["0", "1"])
        self.assertEqual(len({j["command"]["gang"] for j in jobs}), 1)

    def test_measured_use_corrects_the_declared_guess(self):
        self.world.use = lambda cmd: {"mem": 30.0, "cpus": 1.0, "gpu": 60.0, "seconds": 300, "code": 0}
        first, = self.submit(name="teacher", mem=4, cmd=["python", "teach.py"])
        self.world.advance(90)
        self.runner.dispatch()                                                  # one sample of the running job
        second, = [j for j in self.submit(name="teacher", mem=4, cmd=["python", "teach.py"]) if j["id"] != first["id"]]
        self.assertGreater(second["command"]["need"]["mem"], 30)
        self.assertNotEqual(second["command"]["node"], first["command"]["node"])    # 60 % + 72 % GPU do not share

    def test_a_restartable_job_is_requeued_and_a_hold_keeps_it_waiting(self):
        job, = self.submit(restartable=True)
        self.controller.hold_queue("a profile switch runs", self.world.t)
        entry = self.runner.requeue(job)
        self.assertEqual((self.jobs()[0]["status"], entry["requeued_from"]), ("settled", job["id"]))
        self.assertEqual(self.runner.dispatch()[0], [])
        self.assertIn("held", self.queue()[0]["reason"])
        self.controller.hold_queue("", 0)
        started, _ = self.runner.dispatch()
        self.assertEqual([j["command"]["cmd"] for j in started], ["python job.py"])

    def test_every_stop_gives_the_job_60_seconds_to_save(self):
        """Cancel and requeue (switch, drill, node drain) send SIGTERM and kill after STOP_GRACE_S, not after 10 or 20 s."""
        job, = self.submit(restartable=True)
        self.runner.requeue(job)
        job = self.runner.dispatch()[0][0]
        self.runner.stop(job)
        stops = [c["cmd"] for c in self.world.calls if c["cmd"].startswith("docker stop")]
        self.assertEqual(visionjob.STOP_GRACE_S, 60)
        self.assertEqual(len(stops), 2)
        self.assertTrue(all(c.startswith("docker stop -t 60 ") for c in stops), stops)

    def test_a_waiting_job_does_not_vanish_when_the_dispatcher_dies_in_a_start(self):
        """2026-11-14: q1114-193331-4c24 left the queue with no job and no log. The entry was removed before the start;
        a dispatcher that died in the start (a killed submit, an error that the start did not catch) lost it."""
        self.controller.hold_queue("hold", self.world.t)
        self.submit()
        self.controller.hold_queue("", 0)
        entry, = self.queue()
        ship, self.runner.start = self.runner.start, lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())   # the process is killed in the start
        with self.assertRaises(KeyboardInterrupt):
            self.runner.dispatch()
        self.assertEqual([e["id"] for e in self.queue()], [entry["id"]])          # still waits
        self.runner.start = lambda *a, **k: (_ for _ in ()).throw(KeyError("fact"))  # an error that is not a refusal
        self.assertEqual(self.runner.dispatch()[0], [])
        self.assertIn("the start on", self.queue()[0]["reason"])
        self.runner.start = ship
        started, _ = self.runner.dispatch()
        self.assertEqual(([j["command"]["entry"]["id"] for j in started], self.queue()), ([entry["id"]], []))   # one job, the entry is gone

    def test_status_says_what_happened_to_a_cancelled_entry_and_who_stopped_a_job(self):
        self.controller.hold_queue("hold", self.world.t)
        self.submit()
        entry, = self.queue()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                visionjob.handle(self.world, self.controller, argparse.Namespace(id=entry["id"], as_="harvest", force=False), "cancel")
        with self.assertRaisesRegex(Refused, f"{entry['id']} of harvest never started: cancelled in the queue by harvest at t"):
            visionjob.select(self.controller.store.read(), entry["id"])
        with self.assertRaisesRegex(Refused, "never submitted here"):
            visionjob.select(self.controller.store.read(), "q-unknown")
        self.controller.hold_queue("", 0)
        job, = self.submit()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                visionjob.handle(self.world, self.controller, argparse.Namespace(id=job["id"], as_="harvest", force=False), "cancel")
        self.assertIn("stopped by harvest (cancel) at t", out.getvalue())

    def test_jobs_are_restartable_unless_once(self):
        self.submit()
        self.submit(once=True, cmd=["python", "pay.py"])
        self.assertEqual({j["command"]["cmd"]: j["command"]["restartable"] for j in self.jobs()},
                         {"python job.py": True, "python pay.py": False})

    def test_a_live_job_is_marked_restartable_after_submit(self):
        job, = self.submit(env_file_secret=[str(self.env)])
        def old_record(control):                                                # a job from before the router
            for key in ("entry", "need", "kind", "restartable"):
                control["jobs"][job["id"]]["command"].pop(key)
        self.controller.store.commit(old_record)
        mark = argparse.Namespace(as_="lead", id=job["id"], restartable=True, repo=None, env_file_secret=None)
        with self.assertRaisesRegex(Refused, "--env-file-secret"):
            visionjob.handle(self.world, self.controller, mark, "set")
        mark.env_file_secret = [str(self.env)]
        with contextlib.redirect_stdout(io.StringIO()):
            visionjob.handle(self.world, self.controller, mark, "set")
        entry = self.runner.requeue(self.jobs()[0])
        self.assertEqual((entry["env_files"], entry["by"], entry["cmd"]), ([str(self.env)], "harvest", "python job.py"))
        self.assertNotIn(SECRET, self.world.STATE.read_text())
        started, _ = self.runner.dispatch()
        self.assertEqual(len(started), 1)

    def test_a_lost_node_gives_its_restartable_jobs_back(self):
        job, = self.submit(restartable=True)
        node = job["command"]["node"]
        self.world.lose(node)
        self.runner.dispatch()
        self.assertEqual(self.queue(), [])                                      # silent for under LOST_S: wait
        self.world.advance(90)
        started, _ = self.runner.dispatch()
        self.assertEqual(len(started), 1)
        self.assertNotEqual(started[0]["command"]["node"], node)

    def folder(self, job):
        return self.world.folders[job["command"]["node"]][job["id"]]

    def test_a_finished_job_loses_its_workspace_and_keeps_its_log(self):
        job, = self.submit()
        self.runner.dispatch()
        self.assertTrue(self.folder(job)["workspace"])                       # live: untouched
        self.world.advance(120)
        self.runner.dispatch()                                               # settles, saves job.log, removes the rest
        self.assertEqual((self.folder(job)["workspace"], self.folder(job)["log"] is not None), (False, True))
        self.assertIn("spk-" + job["id"], self.world.kept)

    def test_a_stopped_restartable_job_keeps_its_folder_while_its_copy_waits(self):
        job, = self.submit(restartable=True)
        self.controller.hold_queue("a profile switch runs", self.world.t)
        self.runner.requeue(job)
        for _ in range(3):
            self.world.advance(visionjob.SWEEP_AGE_S)
            self.controller.hold_queue("a profile switch runs", self.world.t)
            self.runner.dispatch()
        self.assertIn(job["id"], visionjob.kept_jobs(self.controller.store.read()))
        self.assertTrue(self.folder(job)["workspace"])
        self.assertIsNotNone(self.folder(job)["log"])                        # the container is gone, its log is kept
        self.controller.hold_queue("", 0)
        copy, = self.runner.dispatch()[0]
        self.world.advance(120)
        self.runner.dispatch()
        self.world.advance(visionjob.SWEEP_AGE_S)
        self.runner.dispatch()                                               # the copy ended: both folders go
        self.assertEqual((self.folder(job)["workspace"], self.folder(copy)["workspace"]), (False, False))

    def test_a_folder_that_an_older_router_left_is_swept_after_two_hours(self):
        self.world.folders["spark2"]["vi1101-120000-abcd"] = {"made": self.world.t, "log": self.world.t, "workspace": True}
        self.runner.dispatch()
        self.assertTrue(self.world.folders["spark2"]["vi1101-120000-abcd"]["workspace"])
        self.world.advance(visionjob.SWEEP_AGE_S)
        self.runner.dispatch()
        self.assertFalse(self.world.folders["spark2"]["vi1101-120000-abcd"]["workspace"])

    def test_the_removal_keeps_job_log_and_is_safe_to_repeat(self):
        import subprocess
        home = Path(self.dir.name) / "home"
        folder = home / "spark/jobs/vi1115-135452-2c18"
        (folder / "example-project/.git").mkdir(parents=True)
        (folder / ".cache/huggingface").mkdir(parents=True)
        (folder / "job.log").write_text("out")
        (folder / "pid").write_text("1")
        (folder / "big.bin").write_bytes(b"0" * (2 << 20))
        other = home / "spark/jobs/vi1115-135452-ffff/example-project"
        other.mkdir(parents=True)
        script = visionjob.clear_script("vi1115-135452-2c18")
        for _ in range(2):                                                   # a second run (after a kill) changes nothing
            done = subprocess.run(["bash", "-c", script], env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
            self.assertEqual(done.returncode, 0)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ["job.log", "pid"])
        self.assertTrue(other.is_dir())
        with self.assertRaises(ValueError):
            visionjob.clear_script("../data")

    def test_refusals_have_no_effect(self):
        bad = Path(self.dir.name) / "bad.env"
        bad.write_text("export B2=1\nnot a line\n")
        for changes in (dict(env_file_secret=[str(bad)]), dict(cmd=[]), dict(priority=12), dict(nodes=5), dict(mem=200)):
            with self.assertRaises(Refused):
                self.submit(**changes)
        self.assertEqual((self.jobs(), self.queue(), self.world.containers), ([], [], {}))

    def test_a_submit_with_no_job_node_or_under_a_hold_queues_and_says_so(self):
        def studio(control):
            control["operations"]["op"]["spec"]["reservations"] = {}
        self.controller.store.commit(studio)
        self.submit()                                                           # never refused for the cluster state
        self.assertIn("queued: profile", self.said)
        self.assertIn("start by themselves", self.said)
        self.assertEqual((self.jobs(), len(self.queue()), self.world.containers), ([], 1, {}))
        self.controller.store.commit(lambda control: World.seed(control))
        self.controller.hold_queue("a profile switch runs", self.world.t)
        self.submit(cmd=["python", "two.py"])
        self.assertIn("queued: a profile switch runs", self.said)
        self.assertEqual(self.jobs(), [])                                       # a held job is not started, so not killed
        self.controller.hold_queue("", 0)
        self.assertEqual(len(self.runner.dispatch()[0]), 2)

    def test_job_classes_have_their_own_defaults_and_their_own_measured_use(self):
        self.submit(name="harvest", mem=None, cpus=None)
        self.submit(name="harvest", light=True, mem=None, cpus=None, cmd=["python", "copy.py"])
        self.submit(name="harvest", cpu_only=True, mem=None, cpus=None, cmd=["python", "index.py"])
        entries = {j["command"]["cmd"]: j["command"]["entry"] for j in self.jobs()}
        self.assertEqual({cmd: (e["class"], e["mem"], e["cpus"]) for cmd, e in entries.items()},
                         {"python job.py": ("gpu", 8, 2), "python copy.py": ("io", 1, 1), "python index.py": ("cpu", 4, 4)})
        learned = {"harvest": {"n": 5, "done": 1, "mem": 20.0, "cpus": 3.0, "gpu": 40.0}}     # a gpu job of the name used 20 GiB
        import router
        self.assertEqual(router.estimate(entries["python copy.py"], learned)["mem"], 1.0)     # the io job is not booked for it
        self.assertEqual(router.estimate(entries["python job.py"], learned)["mem"], 24.0)


if __name__ == "__main__":
    unittest.main()
