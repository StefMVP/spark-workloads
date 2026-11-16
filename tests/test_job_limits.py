"""Invalid resource requests stop before queue or remote effects."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_world import World
from test_vision_job import arguments
from clustercontrol import Refused
import visionjob


class JobLimits(unittest.TestCase):
    def test_nonpositive_and_nonfinite_resources_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            world = World(folder)
            runner = visionjob.Runner(world, world.controller, clock=world.clock)
            for key in ('mem', 'cpus'):
                for value in (0, -1, float('nan'), float('inf')):
                    with self.subTest(key=key, value=value), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(Refused):
                        visionjob.build(world, runner, arguments(**{key: value}))
            self.assertEqual(world.controller.store.read().get('queue', {}), {})
            self.assertEqual(world.calls, [])


if __name__ == '__main__':
    unittest.main()
