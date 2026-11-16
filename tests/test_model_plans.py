"""Download plans preserve arguments at both shell boundaries. No host is contacted."""
import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader('model_cli', str(ROOT / 'tools/models'))
models = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(models)


class DownloadPlans(unittest.TestCase):
    def test_execution_uses_argv_and_quotes_remote_paths(self):
        inventory = {'nodes': [{'alias': 'spark1'}], 'sync_routes': []}
        args = argparse.Namespace(repo='example/model', revision=None, include=None, to='/models/a;touch /tmp/INJECT',
                                  host='spark1', keep_free=1, no_sync=True, yes=True)
        with tempfile.TemporaryDirectory() as temp, patch.object(models, 'PLAN_DIR', Path(temp)), \
             patch.object(models, 'hosts', return_value=inventory), \
             patch.object(models, 'repo_files', return_value=(SimpleNamespace(sha='a'*40, gated=False, tags=[]), [('a.safetensors', 10)])), \
             patch.object(models, 'live_free_bytes', return_value=100*models.GIB), \
             patch.object(models.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            models.cmd_pull(args)
        self.assertTrue(run.called)
        for call in run.call_args_list:
            self.assertIsInstance(call.args[0], list)
            self.assertFalse(call.kwargs.get('shell', False))
            self.assertIn("'/models/a;touch /tmp/INJECT/example/model'", call.args[0][-1])


if __name__ == '__main__':
    unittest.main()
