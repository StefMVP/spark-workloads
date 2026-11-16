"""`spark switch` order, idempotence and rollback. A failure is injected in each phase. No host is contacted."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/lib"))
import switch
from clustercontrol import Refused


class FakeDriver:
    """Tracks the applied profile the way the controller does: lease retires the owner, load makes the target READY."""
    def __init__(self, applied="studio-4"):
        self.profile, self.resident, self.calls = applied, {applied} if applied else set(), []
        self.blockers, self.broken, self.verify_fails = {}, set(), set()

    def current(self): return self.profile
    def applied(self, name): return self.profile == name

    def preflight(self, name):
        self.calls.append(("preflight", name))
        return self.blockers.get(name, [])

    def unload(self, name):
        self.calls.append(("unload", name))
        self.resident.clear()

    def lease(self, name):
        self.calls.append(("lease", name))
        self.profile = None

    def load(self, name):
        self.calls.append(("load", name))
        if name in self.broken:
            raise Refused(f"{name} did not become ready: workers did not start")
        self.profile, self.resident = name, {name}

    def resume(self):
        self.resumed = getattr(self, "resumed", 0) + 1
        return []

    def verify(self, name):
        self.calls.append(("verify", name))
        return [{"ok": not (name in self.verify_fails and node == "spark2"),
                 "line": f"{node}: {'FAILED' if name in self.verify_fails and node == 'spark2' else 'OK'}"}
                for node in ("spark1", "spark2", "spark3", "spark4")]


class SwitchOrder(unittest.TestCase):
    def run_switch(self, driver, target, fault=None):
        lines = []
        code = switch.Switch(driver, lines.append, fault).run(target)
        return code, lines

    def test_first_run_changes_and_second_run_only_checks(self):
        driver = FakeDriver()
        code, lines = self.run_switch(driver, "jobs")
        self.assertEqual((code, driver.profile), (0, "jobs"))
        self.assertEqual([c[0] for c in driver.calls], list(switch.PHASES))
        self.assertEqual(sum(line.startswith("spark") for line in lines), 4)       # one final line per node
        driver.calls.clear()
        code, _ = self.run_switch(driver, "jobs")
        self.assertEqual((code, driver.profile), (0, "jobs"))
        self.assertEqual([c[0] for c in driver.calls], ["preflight", "load", "verify"])   # no unload, no lease

    def test_preflight_blocker_changes_nothing_and_prints_the_fix(self):
        driver = FakeDriver()
        driver.blockers["jobs"] = [("spark2: image spark-workload:local is missing", "ssh spark1 'docker save ...'")]
        code, lines = self.run_switch(driver, "jobs")
        self.assertEqual((code, driver.profile, driver.calls), (2, "studio-4", [("preflight", "jobs")]))
        self.assertTrue(any("fix: ssh spark1" in line for line in lines))

    def test_failure_in_each_phase_rolls_back_to_the_earlier_profile(self):
        for phase in ("unload", "lease", "load", "verify"):
            with self.subTest(phase=phase):
                driver = FakeDriver()
                code, lines = self.run_switch(driver, "jobs", fault=phase)
                self.assertEqual((code, driver.profile, driver.resident), (1, "studio-4", {"studio-4"}))
                self.assertTrue(any(line.startswith(f"FAILED in {phase}: injected fault") for line in lines))
                self.assertTrue(lines[-1].startswith("rolled back: studio-4 is applied and verified"))
                self.assertEqual(driver.calls[-1], ("verify", "studio-4"))           # the old profile is verified again

    def test_real_load_and_verify_failures_roll_back(self):
        for field in ("broken", "verify_fails"):
            with self.subTest(field=field):
                driver = FakeDriver()
                getattr(driver, field).add("jobs")
                code, lines = self.run_switch(driver, "jobs")
                self.assertEqual((code, driver.profile), (1, "studio-4"))
                self.assertIn("rolled back", lines[-1])

    def test_failed_rollback_and_no_earlier_profile_are_reported(self):
        driver = FakeDriver()
        driver.broken.update({"jobs", "studio-4"})
        code, lines = self.run_switch(driver, "jobs")
        self.assertEqual((code, driver.profile), (1, None))
        self.assertTrue(lines[-1].startswith("ROLLBACK FAILED in load"))
        driver = FakeDriver(applied=None)
        driver.broken.add("jobs")
        code, lines = self.run_switch(driver, "jobs")
        self.assertEqual(code, 1)
        self.assertIn("no earlier profile", lines[-1])


if __name__ == "__main__":
    unittest.main()
