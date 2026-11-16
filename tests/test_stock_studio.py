import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/lib"))
import clusterprofile
import stockstudio
from clustercontrol import Controller, EffectState, Refused, Store

ROOT = Path(__file__).resolve().parents[1]
UUID = {n: "GPU-" + n for n in stockstudio.NODES}
CAN = {"fasth3": True, "image": True, "ref2va": True, "ref2va_turbo": True, "ffmpeg": True, "loras": 22,
       "image_models": ["qwen", "qwen-full", "flux3k-fast", "flux3k-full", "moss-image"]}


class Head:
    """Stock Studio 4.3.1 head: /api/sparks, worker enable, /update and sparks=N only."""
    def __init__(self):
        ready = {"state": "ready", "guard": True, "usable": True, "code_matches": True, "working": False, "job": None}
        self.overview = {"default": 2, "this": {"state": "ready", "guard": True}, "workers": [
            {"address": "spark1", "enabled": False, "state": "stopped"},
            dict(ready, address="spark2", enabled=True, can=CAN),
            {"address": "spark4", "enabled": False, "state": "stopped"}]}
        self.sends, self.lose = [], None

    def worker(self, address):
        return next(w for w in self.overview["workers"] if w["address"] == address)

    def view(self):
        value = copy.deepcopy(self.overview)
        value["available"] = 1 + sum(1 for w in value["workers"] if w["enabled"] and w["state"] in ("ready", "busy"))
        return value

    def post(self, path, body):
        self.sends.append(path)
        if path == "/api/sparks":
            self.overview["default"] = body["sparks"]
        elif path.endswith("/update"):
            self.worker(path.split("/")[-2]).update(state="ready", guard=True, usable=True, code_matches=True, working=False, job=None, can=CAN)
        else:
            worker = self.worker(path.split("/")[-1])
            worker.update({k: body[k] for k in ("enabled", "use") if k in body})
        if self.lose == path:
            self.lose = None
            raise Refused("Studio control transport lost the reply")
        return self.view()


class Runtime:
    def __init__(self, head, gpu=None, control=404):
        self.head, self.gpu, self.control = head, gpu or {}, control
        self.c = self

    def pmap(self, fn, items):
        return {i: fn(i) for i in items}

    def python(self, node, code, data):
        if code == stockstudio.PROBE:
            return {"sparks": 200, "control": self.control}
        if code == stockstudio.HOST:
            return {"uuids": [UUID[node]], "boot_id": "boot-" + node}
        if code == clusterprofile.GPU_PROBE:
            return self.gpu.get(node, [])
        raise AssertionError("unexpected remote program")

    def api(self, node, port, path, body=None, method=None, dash=False):
        assert (node, port) == ("spark3", 7870) and not path.startswith("/api/control/")
        return self.head.view() if body is None else self.head.post(path, body)

    def studio(self):
        return self.head.view()

    def asset_report(self, spec, overview):
        raise AssertionError("stock apply must not run the full asset audit")


class StockStudioTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.control = Controller(Store(Path(folder.name) / "cluster.json"))
        self.spec = clusterprofile.load(ROOT, "studio-4")
        self.head = Head()

    def apply(self, runtime=None, key="k1"):
        return stockstudio.apply(self.control, runtime or Runtime(self.head), self.spec, key, "source-a")

    def test_stock_head_without_control_routes_reaches_four_spark_placement(self):
        self.assertTrue(stockstudio.stock(Runtime(self.head)))
        self.assertFalse(stockstudio.stock(Runtime(self.head, control=200)))
        op = self.apply()
        self.assertEqual(op["outcome"], "ready")
        self.assertEqual([e["payload"]["request"]["action"] for e in op["effects"]],
                         ["enable", "enable", "update", "update", "default"])
        self.assertTrue(clusterprofile.studio_ready(self.spec["studio"], self.head.view()))
        state = self.control.store.read()
        self.assertEqual(state["layout"]["operation_id"], op["id"])
        self.assertEqual(sorted(state["leases"]), sorted("gpu:" + u for u in UUID.values()))

    def test_workers_the_profile_turns_on_take_studio_jobs(self):
        """A "render" left from a Blender session (studio-cli use ALIAS render) does not survive a switch into a
        profile that gives the Studio that Spark (2026-11-15: spark2 came up enabled, use render, and got no job)."""
        for address in ("spark1", "spark2", "spark4"):
            self.head.worker(address)["use"] = "render"
        op = self.apply()
        self.assertEqual(op["outcome"], "ready")
        self.assertEqual({w["address"]: w.get("use") for w in self.head.overview["workers"]},
                         {"spark1": "jobs", "spark2": "jobs", "spark4": "jobs"})
        self.assertIn("/api/sparks/workers/spark2", self.head.sends)  # already on, use set back

    def test_repeat_converges_without_writes(self):
        first = self.apply()
        sends = list(self.head.sends)
        again = self.apply(key="k2")
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(again["current_observation"]["state"], "verified")
        self.assertEqual(self.head.sends, sends)
        self.assertEqual(len(self.control.view(first["id"])["effects"]), 5)

    def test_foreign_gpu_process_refuses_before_any_intent(self):
        runtime = Runtime(self.head, gpu={"spark1": [{"pid": 93309, "known": True, "owned_studio": False}]})
        with self.assertRaisesRegex(Refused, "spark1: GPU process \\[93309\\]"):
            self.apply(runtime)
        self.assertEqual((self.control.store.read()["operations"], self.head.sends), ({}, []))

    def test_live_previous_owner_must_retire_first(self):
        old = self.control.request("activate", {"name": "glm", "groups": {"s14": ["spark1", "spark4"]}},
                                   ["gpu:GPU-spark1", "gpu:GPU-spark4"], "glm", "source-a", identifier="glm-op")
        def ready(state):
            state["operations"]["glm-op"]["outcome"] = "ready"
            state["layout"] = {"spec": "14", "groups": {}, "operation_id": old["id"], "generation": old["generation"]}
        self.control.store.commit(ready)
        with self.assertRaisesRegex(Refused, "glm-op .*must retire first"):
            self.apply()
        self.assertEqual(self.head.sends, [])

    def test_lost_reply_resumes_without_a_second_send(self):
        self.head.lose = "/api/sparks/workers/spark1/update"
        with self.assertRaises(Refused):
            self.apply()
        pending = next(iter(self.control.store.read()["operations"].values()))
        self.assertEqual(pending["effects"][-1]["state"], EffectState.UNCERTAIN)
        op = self.apply(key="k2")
        self.assertEqual(op["outcome"], "ready")
        self.assertEqual(self.head.sends.count("/api/sparks/workers/spark1/update"), 1)

    def test_busy_enabled_worker_keeps_its_job(self):
        self.head.worker("spark2").update(state="busy", working=True, job="clip-7")
        op = self.apply()
        self.assertEqual(op["outcome"], "ready")
        self.assertNotIn("/api/sparks/workers/spark2/update", self.head.sends)
        self.assertEqual(self.head.worker("spark2")["job"], "clip-7")


    def test_drift_on_ready_owner_reapplies_settings_through_child_without_stop(self):
        first = self.apply()
        self.head.worker("spark4")["enabled"] = False
        self.head.overview["default"] = 3
        op = self.apply(key="k2")
        self.assertEqual((op["outcome"], op["parent"]), ("ready", first["id"]))
        self.assertEqual([e["payload"]["request"]["action"] for e in op["effects"]], ["enable", "default"])
        self.assertFalse(any(path.endswith("/stop") for path in self.head.sends))
        self.assertEqual({l["operation_id"] for l in self.control.store.read()["leases"].values()}, {op["id"]})

    def test_starting_workers_leave_one_pending_operation_for_the_resume_unit(self):
        post = self.head.post
        def slow(path, body):
            result = post(path, body)
            if path.endswith("/update"):
                self.head.worker(path.split("/")[-2]).update(state="starting", working=True)
            return result
        self.head.post = slow
        op = self.apply()
        self.assertEqual((op["outcome"], op["blockers"][0]["reason"]), (None, stockstudio.PENDING))
        for w in ("spark1", "spark4"):
            self.head.worker(w).update(state="ready", working=False)
        resumed = self.apply(key="k2")
        self.assertEqual((resumed["id"], resumed["outcome"]), (op["id"], "ready"))
        self.assertEqual(self.head.sends.count("/api/sparks/workers/spark1/update"), 1)


    def test_reserved_or_partial_profiles_never_take_the_stock_route(self):
        self.assertTrue(stockstudio.covers(self.spec))
        self.assertFalse(stockstudio.covers(dict(self.spec, reservations={"s4": {"by": "user", "for": "jobs"}})))
        with self.assertRaisesRegex(Refused, "four-Spark"):
            stockstudio.apply(self.control, Runtime(self.head), dict(self.spec, reservations={"s4": {"by": "user", "for": "jobs"}}), "k", "s")
        self.assertEqual(self.head.sends, [])

    def test_cancel_of_settled_stock_operation_frees_its_leases_and_layout(self):
        op = self.apply()
        self.assertEqual(self.control.cancel(op["id"])["outcome"], "cancelled")
        state = self.control.store.read()
        self.assertEqual((state["leases"], state["layout"]["operation_id"]), ({}, None))
        self.assertFalse(any(path.endswith("/stop") for path in self.head.sends))
        self.assertEqual(self.apply(key="k2")["outcome"], "ready")


    def test_missing_installed_capability_refuses_without_asset_audit(self):
        self.spec["assets"] = {"targets": ["spark2"], "capabilities": ["ref2va"], "image_models": [], "loras": 0}
        self.head.worker("spark2")["can"] = dict(CAN, ref2va=False)
        op = self.apply()
        self.assertIsNone(op["outcome"])
        self.assertEqual(op["blockers"][-1]["missing"], {"spark2": ["ref2va"]})
        self.assertNotEqual(self.control.store.read()["layout"].get("operation_id"), op["id"])


if __name__ == "__main__":
    unittest.main()
