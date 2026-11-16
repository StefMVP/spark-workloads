import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/lib'))
from clustercontrol import Refused
from visionjob import Runner


class RepositoryPaths(unittest.TestCase):
    def test_workspace_cannot_escape_the_job_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder) / 'repo'
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=Example', '-c',
                            'user.email=example@example.invalid', 'commit', '-q', '--allow-empty', '-m', 'Fixture'], check=True)
            runner = Runner(argparse.Namespace(STATE=Path(folder)/'state.json', ssh=None), None)
            for workspace in ('../escape', '/absolute', 'nested/path', '$(touch marker)', 'bad"name'):
                with self.subTest(workspace=workspace), self.assertRaisesRegex(Refused, 'workspace'):
                    runner.resolve(argparse.Namespace(repo=repo, ref='HEAD', allow_dirty=False, workspace=workspace))


if __name__ == '__main__':
    unittest.main()
