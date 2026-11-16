import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/lib"))
from clustercontrol import Birth, Controller, EffectState, Refused, Store, digest, durable


class ControlStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name) / "control.json")
        self.controller = Controller(self.store)
        self.spec = {"name": "anonymous", "layout": "1", "groups": {"s1": ["spark1"]}}

    def request(self, **options):
        return self.controller.request("activate", self.spec, ["gpu:GPU-fixture-a"], "request-a", "source-a", identifier="operation-a", **options)

    def fence(self, op):
        snapshot = json.loads((Path(__file__).parent / "fixtures/control-protocol-v1.json").read_text())["snapshot"]
        expected = {k: snapshot[k] for k in ("instance_id", "birth", "revision")}
        self.controller.update_evidence(op["id"], {"authorities": {"peer": {"state": "fresh", "snapshot": snapshot, "expected": expected, "resources": op["resources"]}}}, [])
        return {"authority": "peer", "method": "PUT", "request": {"schema": 1, "expected": expected, "operation_id": op["id"], "generation": op["generation"], "idempotency_key": "fence-a", "resources": op["resources"], "state": "draining"}}

    def test_request_replay_and_conflict(self):
        first = self.request()
        revision = self.store.read()["revision"]
        self.assertEqual(first, self.request())
        self.assertEqual(self.store.read()["revision"], revision)
        self.spec["name"] = "different"
        with self.assertRaisesRegex(Refused, "idempotency"):
            self.request()

    def test_foreign_and_same_label_newer_owner_cannot_be_overwritten(self):
        original = self.request()
        other = self.controller.request("activate", self.spec, ["gpu:GPU-fixture-a"], "request-b", "source-a", identifier="operation-b")
        self.assertTrue(other["blockers"])
        with self.assertRaises(Refused):
            self.controller.issue(other["id"], "host_start", {}, [])
        self.assertEqual(self.store.read()["leases"]["gpu:GPU-fixture-a"]["operation_id"], original["id"])

    def test_superseded_operation_cannot_issue_from_its_saved_valid_evidence(self):
        old = self.request()
        payload = self.fence(old)
        newer = self.controller.request("activate", dict(self.spec, name="new metadata"), old["resources"],
                                        "request-b", "source-a", parent=old["id"], identifier="operation-b")
        with self.assertRaises(Refused):
            self.controller.issue(old["id"], "fence_acquire", payload, [])
        self.assertEqual(self.controller.view(old["id"])["effects"], [])
        self.assertEqual(self.store.read()["leases"][old["resources"][0]]["operation_id"], newer["id"])

    def test_unknown_effect_blocks_duplicate_and_rollback(self):
        op = self.request()
        effect = self.controller.issue(op["id"], "fence_acquire", self.fence(op), [])
        restarted = Controller(Store(self.store.path))
        with self.assertRaisesRegex(Refused, "unresolved"):
            restarted.issue(op["id"], "fence_acquire", self.fence(op), [])
        with self.assertRaisesRegex(Refused, "uncertain"):
            restarted.request("restore", self.spec, op["resources"], "rollback-a", "source-a", parent=op["id"])
        self.assertEqual(restarted.view(op["id"])["effects"][0]["id"], effect["id"])

    def test_cancel_retains_uncertain_leases_and_prevents_new_effects(self):
        op = self.request()
        self.controller.issue(op["id"], "fence_acquire", self.fence(op), [])
        cancelled = self.controller.cancel(op["id"])
        self.assertEqual(cancelled["phase"], "waiting")
        self.assertIsNone(cancelled["outcome"])
        self.assertIn("gpu:GPU-fixture-a", self.store.read()["leases"])
        with self.assertRaises(Refused):
            self.controller.issue(op["id"], "host_start", {}, [])

    def ready_parent(self):
        op = self.request()
        def finish(state):
            state["operations"][op["id"]].update(outcome="ready", phase="finished", readiness={})
            state["layout"] = {"spec": "1", "groups": self.spec["groups"], "operation_id": op["id"], "generation": op["generation"]}
        self.store.commit(finish)
        return self.controller.view(op["id"])

    def test_cancel_of_unstarted_child_returns_parent_lease(self):
        parent = self.ready_parent()
        child = self.controller.request("activate", dict(self.spec, name="studio"), parent["resources"], "request-b",
                                        "source-a", parent=parent["id"], identifier="operation-b")
        self.assertFalse(self.controller.view(parent["id"])["ownership_current"])
        self.assertEqual(self.controller.cancel(child["id"])["outcome"], "cancelled")
        self.assertTrue(self.controller.view(parent["id"])["ownership_current"])
        self.assertFalse(self.controller.view(child["id"])["ownership_current"])
        # Repeated cancel converges and does not take the lease back.
        self.controller.cancel(child["id"])
        self.assertEqual(self.store.read()["leases"]["gpu:GPU-fixture-a"],
                         {"state": "owned", "operation_id": parent["id"], "generation": parent["generation"]})

    def test_cancel_of_unstarted_child_with_changed_parent_frees_nothing_foreign(self):
        parent = self.ready_parent()
        child = self.controller.request("activate", dict(self.spec, name="studio"), parent["resources"], "request-b",
                                        "source-a", parent=parent["id"], identifier="operation-b")
        self.store.commit(lambda state: state["operations"][parent["id"]].update(outcome="released"))
        self.controller.cancel(child["id"])
        self.assertNotIn("gpu:GPU-fixture-a", self.store.read()["leases"])

    def test_cancel_after_effect_keeps_child_lease(self):
        parent = self.ready_parent()
        child = self.controller.request("activate", dict(self.spec, name="studio"), parent["resources"], "request-b",
                                        "source-a", parent=parent["id"], identifier="operation-b")
        self.controller.issue(child["id"], "fence_acquire", self.fence(child), [])
        self.controller.cancel(child["id"])
        self.assertEqual(self.store.read()["leases"]["gpu:GPU-fixture-a"]["operation_id"], child["id"])

    def test_ready_native_owner_may_begin_retire_under_new_source_only_once_complete(self):
        self.spec["installed_native"] = {"backend": "tensorfold-native"}
        parent = self.ready_parent()
        op = self.controller.begin_installed_retire(parent["id"], "source-b")
        self.assertEqual((op["intent"], op["source_identity"]), ("retire", "source-b"))
        self.assertEqual(op["source_repairs"][-1]["old_source"], "source-a")
        with self.assertRaisesRegex(Refused, "source/backend differs"):
            self.controller.begin_installed_retire(parent["id"], "source-c")

    def stopped_pair(self, **change):
        import time
        hosts = {}
        for n, gpu in (("spark1", "gpu:GPU-fixture-a"), ("spark4", "gpu:GPU-fixture-b")):
            c = {"id": "id-" + n, "image_id": "img-" + n, "process_birth": None}
            c.update(change.get(n, {}))
            hosts[n] = {"resource_id": gpu, "boot_id": "boot-" + n, "studio_absent": True, "observed_at": time.time(),
                        "containers": {"glm": c}, "gpu": change.get("gpu", {}).get(n, []), "legacy_absent": {}}
        return hosts

    def glm_parent(self):
        self.spec = {"name": "glm", "layout": "14", "groups": {"s14": ["spark1", "spark4"]}, "installed_native": {
            "backend": "tensorfold-native", "name": "glm", "members": ["spark1", "spark4"],
            "nodes": {n: {"id": "id-" + n, "image": "img-" + n, "resource": g}
                      for n, g in (("spark1", "gpu:GPU-fixture-a"), ("spark4", "gpu:GPU-fixture-b"))}}}
        op = self.controller.request("activate", self.spec, ["gpu:GPU-fixture-a", "gpu:GPU-fixture-b"], "glm", "source-a", identifier="glm-op")
        def finish(state):
            state["operations"][op["id"]].update(outcome="ready", phase="finished", readiness={})
            state["layout"] = {"spec": "14", "groups": self.spec["groups"], "operation_id": op["id"], "generation": op["generation"]}
        self.store.commit(finish)
        return op

    def test_externally_stopped_exact_pair_releases_with_absence_evidence(self):
        op = self.glm_parent()
        for bad in ({"spark1": {"id": "replaced"}}, {"spark4": {"image_id": "other"}},
                    {"spark1": {"process_birth": {"identity": "1"}}}, {"gpu": {"spark4": [{"container": "id-spark4"}]}}):
            with self.assertRaises(Refused):
                self.controller.retire_absent_installed(op["id"], self.stopped_pair(**bad))
        self.assertTrue(self.controller.view(op["id"])["ownership_current"])
        released = self.controller.retire_absent_installed(op["id"], self.stopped_pair())
        self.assertEqual(released["outcome"], "released")
        self.assertEqual(released["effects"], [])
        self.assertIn("absence_retirement", released)
        state = self.store.read()
        self.assertEqual((state["leases"], state["layout"]["operation_id"]), ({}, None))
        with self.assertRaises(Refused):
            self.controller.retire_absent_installed(op["id"], self.stopped_pair())

    def test_cancel_of_unstarted_child_keeps_subset_and_pending_parent_leases(self):
        two = ["gpu:GPU-fixture-a", "gpu:GPU-fixture-b"]
        parent = self.controller.request("activate", self.spec, two, "p", "source-a", identifier="parent")
        child = self.controller.request("activate", self.spec, two[:1], "c", "source-a", parent=parent["id"], identifier="child")
        self.controller.cancel(child["id"])
        leases = self.store.read()["leases"]
        self.assertEqual({k: v["operation_id"] for k, v in leases.items()}, {two[0]: "parent", two[1]: "parent"})
        self.assertTrue(self.controller.view(parent["id"])["ownership_current"])

    def test_stop_needs_exact_birth_and_sealed_generation(self):
        op = self.request()
        with self.assertRaises(Refused):
            self.controller.issue(op["id"], "host_stop", {}, [])
        birth = {"kind": "process", "boot_id": "boot-a", "identity": "123", "started": "42"}
        with self.assertRaisesRegex(Refused, "sealed"):
            self.controller.issue(op["id"], "host_stop", {}, [], birth=birth)

    def test_mismatched_outcome_never_settles_an_effect(self):
        op = self.request()
        effect = self.controller.issue(op["id"], "fence_acquire", self.fence(op), [])
        with self.assertRaises(Refused):
            self.controller.outcome(op["id"], effect["id"], {"verified": True})
        self.assertEqual(self.controller.view(op["id"])["effects"][0]["state"], EffectState.UNCERTAIN)

    def test_ready_without_target_binding_or_services_is_rejected(self):
        op = self.request()
        with self.assertRaises(Refused):
            self.controller.publish(op["id"], {"spec_digest": digest(self.spec), "artifacts": True, "capabilities": True, "services": True, "births": True, "settings": True})
        self.assertEqual(self.store.read()["layout"]["groups"], {})

    def test_legacy_import_retains_unverified_claim_and_original(self):
        old = {"layout": self.spec | {"groups": self.spec["groups"]}, "leases": {"s1": {"by": "same-label", "for": "work"}}, "jobs": {}}
        self.store.path.write_text(json.dumps(old))
        with self.assertRaisesRegex(Refused, "legacy"):
            self.store.read()
        self.store.import_legacy()
        first = self.store.read()
        self.store.import_legacy()
        self.assertEqual(first, self.store.read())
        self.assertEqual(first["leases"], {})
        self.assertTrue(self.request()["blockers"])
        backup = next(iter(first["imports"].values()))["original"]
        self.assertEqual(json.loads(Path(backup).read_text()), old)

    def test_transport_alias_cannot_claim_physical_resource(self):
        with self.assertRaisesRegex(Refused, "footprint"):
            self.controller.request("activate", self.spec, ["spark1"], "bad", "source")

    def test_old_completion_after_metadata_adoption_does_not_release_new_lease(self):
        op = self.request()
        gpu = op["resources"][0]
        job = self.controller.admit_job(op["id"], "job-a", "studio", [gpu], 4, "native")
        birth = {"kind": "process", "boot_id": "boot-a", "identity": "123", "started": "42"}
        self.controller.observe_job_start(job["id"], job["attempt_id"], job["generation"], {
            gpu: {"attempt_id": job["attempt_id"], "token": {"operation_id": op["id"], "generation": op["generation"], "resources": op["resources"]}, "state": "running", "birth": birth}})
        adopted = self.controller.request("activate", self.spec, op["resources"], "adopt", "source-a", parent=op["id"], identifier="adopted")
        result = self.controller.settle_job(job["id"], job["attempt_id"], job["generation"], {
            gpu: {"state": "terminated", "attempt_id": job["attempt_id"], "birth": birth}})
        self.assertEqual(result["status"], "settled")
        self.assertEqual(self.store.read()["leases"][gpu]["operation_id"], adopted["id"])
        self.assertEqual(self.controller.view(adopted["id"])["effects"], [])


if __name__ == "__main__":
    unittest.main()

class PendingArtifactTest(unittest.TestCase):
    def test_unknown_preparation_has_no_gpu_or_write_grant(self):
        with tempfile.TemporaryDirectory() as folder:
            controller=Controller(Store(Path(folder)/'state.json'))
            op=controller.prepare_pending({'profile':'studio-4'},'unknown','source',[{'kind':'content_identity_unknown','resource':'*','reason':'no full SHA'}])
            self.assertEqual(controller.prepare_pending({'profile':'studio-4'},'unknown','source',[{'reason':'same input'}]),op)
            self.assertEqual(controller.store.read()['leases'],{})
            self.assertEqual(controller.store.read()['artifact_reservations'],{})
            with self.assertRaises(Refused):controller.prepare_issue(op['id'],0)
            with self.assertRaises(Refused):controller.prepare_pending({'profile':'different'},'unknown','source',[{'reason':'changed'}])
