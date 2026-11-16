import copy
import hashlib
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/lib"))
from artifactcontrol import publish_missing
from clustercontrol import Controller, Refused, Store
from nativecontrol import HostJournal, NativeExecutor, member_plan, validate_member
from profileplan import parse_snapshot, sealed_authorities
from control_fixtures import authority

GPU = "gpu:GPU-test-card-0001"


class PhysicalIO:
    def __init__(self, birth):
        self.birth, self.present, self.applied, self.lose_reply = birth, True, 0, False

    def observe(self, plan):
        if plan["birth"] != self.birth:
            raise Refused("physical birth changed")
        return {"state": "present" if self.present else "absent", "birth": self.birth}

    def apply(self, plan):
        self.applied += 1
        self.present = False
        if self.lose_reply:
            raise OSError("applied; reply lost")


class NativeEffectTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.birth = {"kind": "process", "boot_id": "boot-a", "identity": "123", "started": "42"}
        self.io = PhysicalIO(self.birth)
        self.fences = {"peer": authority()}
        self.effect = {"id": "operation-a:0", "kind": "host_stop", "token": {"operation_id": "operation-a", "generation": 1, "resources": [GPU]},
                       "birth": self.birth, "payload": member_plan("process_stop", GPU, self.birth, "123")}
        self.host = HostJournal(Path(self.temporary.name) / "host.json", GPU, self.io, lambda peer, prior: copy.deepcopy(self.fences[peer]))

    def test_applied_lost_reply_reconciles_without_second_application(self):
        self.io.lose_reply = True
        with self.assertRaises(OSError):
            self.host.execute(self.effect, self.fences)
        result = self.host.execute(self.effect, self.fences)
        self.assertEqual(result["state"], "terminated")
        self.assertEqual(self.io.applied, 1)
        self.assertEqual(self.host.execute(self.effect, self.fences), result)

    def test_reused_pid_and_wrong_command_digest_cannot_stop(self):
        self.io.birth = dict(self.birth, started="newer")
        with self.assertRaises(Refused):
            self.host.execute(self.effect, self.fences)
        self.assertEqual(self.io.applied, 0)
        forged = copy.deepcopy(self.effect)
        forged["payload"]["command"].append("arbitrary")
        with self.assertRaisesRegex(Refused, "command"):
            validate_member(forged, GPU)

    def test_host_reobserves_admission_and_requires_every_class(self):
        self.fences["peer"]["snapshot"]["coverage"].pop()
        with self.assertRaisesRegex(Refused, "coverage"):
            self.host.execute(self.effect, self.fences)
        self.assertEqual(self.io.applied, 0)

    def test_arbitrary_kind_or_head_only_token_cannot_authorize_member(self):
        forged = copy.deepcopy(self.effect)
        forged["kind"] = "innocent-looking-stop"
        with self.assertRaises(Refused):
            validate_member(forged, GPU)
        forged = copy.deepcopy(self.effect)
        forged["token"]["resources"] = ["gpu:GPU-test-other-0001"]
        with self.assertRaises(Refused):
            validate_member(forged, GPU)

    def test_stale_unknown_and_malformed_snapshots_block(self):
        for raw in (None, {}, [], {"error": "truncated"}):
            with self.assertRaises(Refused):
                parse_snapshot(raw)
        self.fences["peer"]["snapshot"]["birth"]["service_id"] = "replacement"
        with self.assertRaises(Refused):
            self.host.execute(self.effect, self.fences)

    def test_real_pidfd_terminates_only_the_named_birth(self):
        with subprocess.Popen([sys.executable, "-c", "import signal;signal.pause()"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as child:
            fields = (Path("/proc") / str(child.pid) / "stat").read_text().rsplit(")", 1)[1].split()
            birth = {"kind": "process", "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                     "identity": str(child.pid), "started": fields[19]}
            effect = dict(self.effect, birth=birth, payload=member_plan("process_stop", GPU, birth, str(child.pid)))
            host = HostJournal(Path(self.temporary.name) / "physical.json", GPU, NativeExecutor(), lambda peer, prior: self.fences[peer])
            outcome = host.execute(effect, self.fences)
            self.assertEqual(outcome["birth"], birth)
            self.assertEqual(outcome["state"], "terminated")
            child.wait(timeout=2)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.write_bytes(b"anonymous model fixture")
        self.target = self.root / "new-asset"
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_prepare_does_not_transfer_active_gpu_lease(self):
        controller = Controller(Store(self.root / "control.json"))
        controller.request("activate", {"name": "live", "groups": {"s1": ["spark1"]}}, [GPU], "live", "source", identifier="live")
        before = controller.store.read()["leases"]
        op = controller.prepare_request("host-a", [{"source": str(self.source), "target": str(self.target), "sha256": self.sha, "usage": "new_path"}], "artifact-a", "source", identifier="artifact-a")
        effect = controller.prepare_issue(op["id"], 0)
        evidence = publish_missing(**{k: effect["payload"][k] for k in ("source", "target")}, expected_sha256=self.sha, usage="new_path")
        self.assertEqual(controller.prepare_outcome(op["id"], effect["id"], evidence)["outcome"], "prepared")
        self.assertEqual(controller.store.read()["leases"], before)

    def test_checkpoint_existing_wrong_content_and_unknown_usage_are_preserved(self):
        self.target.write_bytes(b"keep")
        with self.assertRaises(Refused):
            publish_missing(self.source, self.target, self.sha, usage="unused")
        self.assertEqual(self.target.read_bytes(), b"keep")
        with self.assertRaises(Refused):
            publish_missing(self.source, self.root / "checkpoints/last.bin", self.sha, usage="new_path")
        with self.assertRaises(Refused):
            publish_missing(self.source, self.root / "missing", self.sha, usage="unknown")

    def test_publication_replay_verifies_same_content_without_overwrite(self):
        first = publish_missing(self.source, self.target, self.sha, usage="new_path")
        inode = self.target.stat().st_ino
        second = publish_missing(self.source, self.target, self.sha, usage="new_path")
        self.assertEqual(second["publication"], "preexisting")
        self.assertEqual(self.target.stat().st_ino, inode)
        self.assertEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
