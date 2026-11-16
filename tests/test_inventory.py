import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/lib'))
import sparkinventory


class Inventory(unittest.TestCase):
    def test_example_and_invalid_ssh_targets(self):
        example = json.loads((Path(__file__).resolve().parents[1]/'examples/inventory.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'inventory.json'
            with patch.dict(os.environ, SPARK_INVENTORY=str(path)):
                path.write_text(json.dumps(example))
                self.assertEqual([n['alias'] for n in sparkinventory.load()['nodes']], list(sparkinventory.NODES))
                for field, value in (('alias', '-oProxyCommand=bad'), ('hostname', 'a;command'), ('lan_ip', '$(bad)')):
                    bad = copy.deepcopy(example)
                    bad['nodes'][0][field] = value
                    path.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(SystemExit):
                        sparkinventory.load()


if __name__ == '__main__':
    unittest.main()
