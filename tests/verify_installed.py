"""Exercise an installed CLI with a temporary home and local lifecycle fixtures."""
import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    root = prefix/'lib/spark-workloads'
    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        sentinel = temp/'bin'
        sentinel.mkdir()
        ssh = sentinel/'ssh'
        ssh.write_text('#!/bin/sh\nprintf "unexpected SSH call\\n" >&2\nexit 91\n')
        ssh.chmod(0o755)
        inventory = root/'examples/inventory.json'
        env = dict(os.environ, HOME=str(temp/'home'), SPARK_INVENTORY=str(inventory), SPARK_STATE=str(temp/'state/cluster.json'),
                   PATH=str(sentinel)+':'+os.environ.get('PATH', ''))
        for command in ([str(prefix/'bin/spark'), '--help'], [str(prefix/'bin/spark'), 'profile', 'list'],
                        [str(prefix/'bin/spark'), 'profile', 'apply', 'jobs-studio-pair'],
                        [str(prefix/'bin/models'), 'fit', 'example/model', '--size-gib', '140', '--offline']):
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            assert 'unexpected SSH' not in result.stderr
            print(json.dumps({'command': command[1:], 'status': 'passed', 'output': result.stdout.strip()}))
        assert not (temp/'state/cluster.json').exists(), 'previews wrote controller state'
        sys.path.insert(0, str(root/'tools/lib'))
        loader = importlib.machinery.SourceFileLoader('installed_cluster', str(root/'tools/cluster'))
        cli = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
        sys.modules[loader.name] = cli
        with patch.dict(os.environ, env):
            loader.exec_module(cli)
        import visionjob
        assert Path(visionjob.__file__).resolve().is_relative_to(root), visionjob.__file__
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from spark_world import World
        world = World(temp/'world')
        for name in ('STATE', 'ssh', 'pmap', 'node_env', 'now', 'me', 'warn'):
            setattr(cli, name, getattr(world, name))
        def invoke(*argv):
            output = io.StringIO()
            with patch.object(sys, 'argv', ['spark', '--as', 'fixture', *argv]), contextlib.redirect_stdout(output), \
                 patch.object(visionjob.time, 'time', side_effect=world.clock):
                cli.main()
            return output.getvalue()
        invoke('job', 'submit', '--cpu-only', '--image', 'spark-workload:local', '--', 'echo', 'fixture')
        jobs = list(world.controller.store.read()['jobs'].values())
        assert len(jobs) == 1 and jobs[0]['status'] == 'running', jobs
        job = jobs[0]
        assert job['command']['entry']['class'] == 'cpu'
        print(json.dumps({'entry': 'installed spark job submit', 'status': 'running', 'node': job['command']['node']}))
        world.advance(200)
        before = world.STATE.read_bytes()
        observed = json.loads(invoke('job', 'status', job['id'], '--json'))
        assert observed['jobs'][0]['status'] == 'settled', observed
        assert world.STATE.read_bytes() == before, 'job status changed durable state'
        import sparkutil
        with patch.object(sparkutil, 'REPORT', temp/'util.json'), patch.object(sparkutil, 'alert'):
            invoke('tick')
        settled = world.controller.store.read()['jobs'][job['id']]
        assert settled['status'] == 'settled', settled
        print(json.dumps({'entry': 'installed spark job status and tick', 'status': 'settled', 'physical_backend': 'local fixture'}))


if __name__ == '__main__':
    main()
