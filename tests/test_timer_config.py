"""The user timer keeps the install interpreter and inventory selection."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/lib'))
import sparkutil


class TimerConfig(unittest.TestCase):
    def test_timer_keeps_the_selected_inventory_and_python(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            cluster = SimpleNamespace(STATE=home/'state with space/cluster.json', ROOT=home/'install with space')
            inventory = str(home/'config with space/inventory.json')
            with patch.object(Path, 'home', return_value=home), patch.dict(os.environ, SPARK_INVENTORY=inventory), \
                 patch.object(sparkutil.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
                sparkutil.install(cluster)
            service = (home/'.config/systemd/user/spark-tick.service').read_text()
            self.assertIn('SPARK_INVENTORY='+inventory, service)
            self.assertIn(sys.executable, service)
            self.assertIn('Environment="SPARK_STATE=', service)


if __name__ == '__main__':
    unittest.main()
