"""Profile-owned HAProxy configuration and rootless container lifecycle."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

IMAGE = 'docker.io/library/haproxy@sha256:5378234d19457d645ed6d5a0ac994ef196614f9a2cabda8f6ccd6ac8815f9098'
URL = 'http://127.0.0.1:11435/v1'
NAME = 'spark-profile-heads'
RECOVERY_UNIT = 'spark-profile-heads-recovery-1106.service'


def process_alive(pid):
    if not pid:
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def config(profile, port=11435, stats_port=11436):
    health_check = '  http-check send meth GET uri /health ver HTTP/1.1 hdr Host localhost'
    generation = hashlib.sha256(json.dumps({
        'heads': [(i['id'], i['endpoint']) for i in profile['instances']],
        'health_check': health_check,
    }).encode()).hexdigest()[:12]
    lines = [
        'global', '  log stdout format raw local0',
        '  stats socket /tmp/profile-heads.sock mode 600 expose-fd listeners',
        'defaults', '  mode http', '  log global',
        '  log-format "%b/%s session=%[var(txn.session)] status=%ST bytes=%B termination=%ts"',
        '  timeout connect 3s', '  timeout client 1h', '  timeout server 1h',
        '  timeout http-request 30s', '  retries 0', '  option abortonclose',
        f'frontend model_{generation}', f'  bind 127.0.0.1:{port}',
        '  http-request set-var(txn.session) req.hdr(x-opencode-session-id),sha2(256),hex',
        '  default_backend heads',
        'backend heads', '  balance hdr(x-opencode-session-id)', '  hash-type consistent',
        '  option httpchk', health_check, '  http-check expect status 200',
        '  default-server inter 1s fall 2 rise 1',
    ]
    for instance in profile['instances']:
        address = urlsplit(instance['endpoint'])
        if address.scheme != 'http' or address.path != '/v1' or address.query or address.fragment or address.username:
            raise ValueError('routing requires an HTTP /v1 endpoint with an IP address')
        host = ipaddress.ip_address(address.hostname)
        name = instance['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('invalid routing instance ID')
        # Explicit IDs keep the hash ring stable when another head leaves the profile.
        identity = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big') % 2147483646 + 1
        target = f'[{host}]' if host.version == 6 else str(host)
        lines.append(f'  server {name} {target}:{address.port} id {identity} check')
    lines += ['listen status', f'  bind 127.0.0.1:{stats_port}', '  stats enable', '  stats uri /']
    return '\n'.join(lines) + '\n'


class Routing:
    def __init__(self, state_dir, *, name=NAME):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('invalid profile routing container name')
        self.directory = Path(state_dir).resolve() / 'profile-routing'
        self.path = self.directory / 'haproxy.cfg'
        self.cidfile = self.directory / 'container.cid'
        self.name = name
        identity = hashlib.sha256(f'{self.directory}:{name}'.encode()).hexdigest()[:20]
        self.unit = f'local-ai-profile-routing-{identity}.service'
        self.description = f'local-ai profile routing {self.directory} ({name})'

    def run(self, *args, check=True):
        result = subprocess.run(['podman', *args], text=True, capture_output=True, timeout=30)
        if check and result.returncode:
            raise RuntimeError(f'profile routing: podman {args[0]} failed; inspect podman logs {self.name}')
        return result

    def inspect(self, identifier=None):
        identifier = identifier or self.name
        result = self.run('inspect', identifier, check=False)
        if result.returncode:
            if self.run('container', 'exists', identifier, check=False).returncode == 1:
                return None
            raise RuntimeError('profile routing: cannot inspect the container')
        data = json.loads(result.stdout)[0]
        if data.get('Config', {}).get('Labels', {}).get('local-ai.profile-routing') != str(self.directory):
            raise RuntimeError('profile routing container name belongs to another state directory')
        return data

    def running(self):
        return self.check_running(self.inspect())

    def check_running(self, data):
        if data is None or not data['State']['Running']:
            return False
        for key in ('Pid', 'ConmonPid'):
            pid = data['State'].get(key, 0)
            if not process_alive(pid):
                raise RuntimeError('profile routing: stale running state; proxy and collector must both be alive')
        return True

    def systemctl(self, *args):
        result = subprocess.run(['systemctl', '--user', *args], text=True, capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError('profile routing: user manager operation failed; inspect systemctl --user status')
        return result.stdout

    def manager(self):
        raw = self.systemctl('show', self.unit, '-p', 'LoadState', '-p', 'Description', '-p', 'Transient')
        state = dict(line.split('=', 1) for line in raw.splitlines())
        if state.get('LoadState') != 'not-found' and (
                state.get('Description') != self.description or state.get('Transient') != 'yes'):
            raise RuntimeError('profile routing: user unit belongs to another owner')
        return state['LoadState'] != 'not-found'

    def recovery_group(self, data):
        # Preserve the repaired live route. Retire its unit only after an explicit stop.
        if self.name != NAME:
            return None
        pid = data['State'].get('ConmonPid', 0) if data else 0
        if not pid:
            return None
        group = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::', 1)[-1]
        if group.endswith('/' + RECOVERY_UNIT):
            return Path('/sys/fs/cgroup') / group.lstrip('/')
        return None

    def retire_recovery(self, group):
        if group is None:
            return
        raw = self.systemctl('show', RECOVERY_UNIT, '-p', 'ControlGroup', '-p', 'Transient')
        state = dict(line.split('=', 1) for line in raw.splitlines())
        if state.get('Transient') != 'yes' or group != Path('/sys/fs/cgroup') / state.get('ControlGroup', '').lstrip('/'):
            raise RuntimeError('profile routing: recovery unit ownership changed; leave it intact')
        if any(path.read_text().strip() for path in group.rglob('cgroup.procs')):
            raise RuntimeError('profile routing: recovery unit still has processes; leave it intact')
        self.systemctl('stop', RECOVERY_UNIT)

    def stop_managed(self):
        identifier = self.cidfile.read_text().strip() if self.cidfile.exists() else None
        if identifier is not None and not re.fullmatch(r'[0-9a-f]{64}', identifier):
            raise RuntimeError('profile routing: invalid managed container ID; leave it intact')
        data = self.inspect(identifier) if identifier else None
        pids = [data['State'].get(key, 0) for key in ('Pid', 'ConmonPid')] if data else []
        if self.manager():
            self.systemctl('stop', self.unit)
        if ((identifier and self.inspect(identifier) is not None) or self.manager()
                or any(process_alive(pid) for pid in pids)):
            raise RuntimeError('profile routing: managed cleanup incomplete; retain the transaction for recovery')

    def start(self, mount):
        podman = shutil.which('podman')
        if not podman:
            raise RuntimeError('profile routing: podman is unavailable')
        self.cidfile.unlink(missing_ok=True)
        cidfile = json.dumps(str(self.cidfile).replace('%', '%%'))
        try:
            result = subprocess.run([
                'systemd-run', '--user', '--unit', self.unit, '--description', self.description,
                '--collect', '--service-type=oneshot', '--remain-after-exit', '--expand-environment=no',
                '--property=TimeoutStartSec=30', '--property=TimeoutStopSec=45',
                f'--property=ExecStop=:{podman} stop --ignore --time 30 --cidfile {cidfile}',
                f'--property=ExecStopPost=:{podman} rm --force --ignore --cidfile {cidfile}',
                podman, 'run', '-d', '--pull=never', '--name', self.name, '--cidfile', str(self.cidfile),
                '--label', f'local-ai.profile-routing={self.directory}',
                '--network=host', '--restart=unless-stopped', '-v', mount, IMAGE,
                'haproxy', '-W', '-db', '-f', '/usr/local/etc/haproxy/haproxy.cfg'],
                text=True, capture_output=True, timeout=40)
            if result.returncode:
                raise RuntimeError('profile routing: managed container creation failed; inspect the user unit journal')
        except BaseException:
            # Stop the service before returning a partial creation to the transaction.
            self.stop_managed()
            raise

    def preflight(self):
        self.run("image", "inspect", IMAGE)
        self.running()
        self.manager()

    def snapshot(self):
        return {'config': self.path.read_text() if self.path.exists() else None, 'running': self.running()}

    def restore(self, saved):
        self.apply(saved['config'] if saved['running'] else None)
        if not saved['running'] and saved['config'] is not None:
            candidate = self.directory / 'candidate.cfg'
            candidate.write_text(saved['config'])
            candidate.replace(self.path)

    def apply(self, text):
        data = self.inspect()
        running = self.check_running(data)
        managed = self.manager()
        recovery = self.recovery_group(data) if running else None
        self.directory.mkdir(parents=True, exist_ok=True)
        if text is None:
            if managed:
                self.stop_managed()
            elif running:
                self.run('stop', '--time', '30', data['Id'])
            self.retire_recovery(recovery)
            self.path.unlink(missing_ok=True)
            return
        changed = not self.path.exists() or self.path.read_text() != text
        candidate = self.directory / 'candidate.cfg'
        candidate.write_text(text)
        mount = f'{self.directory}:/usr/local/etc/haproxy:ro'
        # Network access and image pulls do not occur during profile transactions.
        self.run('run', '--rm', '--pull=never', '--network=none', '-v', mount, IMAGE,
                 'haproxy', '-c', '-f', '/usr/local/etc/haproxy/candidate.cfg')
        if changed:
            candidate.replace(self.path)
        else:
            candidate.unlink()
        names = re.findall(r'^  server ([^ ]+)', text, re.MULTILINE)
        stats_port = re.search(r'listen status\n  bind 127.0.0.1:(\d+)', text).group(1)
        generation = re.search(r'frontend (model_[^\n]+)', text).group(1)
        if running:
            # Disk can be ahead of the worker after a failed signal or CLI exit.
            if changed or not ready(names, stats_port, generation, require_up=False):
                self.run('kill', '--signal', 'USR2', self.name)
        else:
            if managed:
                self.stop_managed()
            elif data is not None:
                self.run('rm', data['Id'])
            self.start(mount)
        try:
            result = subprocess.run(['agent-tool', 'wait', '--timeout', '15', '--interval', '0.2', '--',
                                     sys.executable, str(Path(__file__).resolve()), json.dumps(names), stats_port, generation],
                                    capture_output=True, text=True, timeout=20)
            if result.returncode:
                raise RuntimeError('profile routing did not reach the expected ready membership; resume settings')
        except BaseException:
            if not running:
                self.stop_managed()
            raise


def ready(names, port=11436, generation=None, *, require_up=True):
    import csv
    import io
    import urllib.request
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/;csv', timeout=2) as response:
            rows = list(csv.DictReader(io.StringIO(response.read().decode())))
        heads = {row['svname']: row['status'] for row in rows if row['# pxname'] == 'heads' and row['svname'] != 'BACKEND'}
        return (set(heads) == set(names) and (not require_up or all(status == 'UP' for status in heads.values()))
                and (generation is None or any(row['# pxname'] == generation for row in rows)))
    except (OSError, ValueError, KeyError):
        return False


if __name__ == '__main__':
    raise SystemExit(0 if ready(json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3]) else 3)
