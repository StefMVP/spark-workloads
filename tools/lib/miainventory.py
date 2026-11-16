"""Read-only Mia installation checks. Unknown provenance never authorizes effects."""
import hashlib
import json
import os
from pathlib import Path
import re

from clustercontrol import Refused
from profile_recipes import mia_tp4


def inventory(recipe, member, operation, io, command):
    lock = mia_tp4.verify_sources(recipe['package'])
    root = Path(recipe['package'])
    receipt = json.loads((root / 'image.json').read_text())
    image_id = receipt.get('image_id')
    if receipt.get('state') != 'verified' or not isinstance(image_id, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
        raise Refused('Mia guarded image has no immutable build receipt')
    if not re.fullmatch(r'[0-9a-f]{64}', receipt.get('native_binary_sha256') or ''):
        raise Refused('Mia native executable build identity is unknown')
    raw = json.loads(command(['docker', 'image', 'inspect', image_id]))
    expected = {'local.profile.mia.control': 'v1', 'local.profile.mia.recipe': mia_tp4.RECIPE,
                'local.profile.mia.engine': mia_tp4.NATIVE, 'local.profile.mia.source': lock['build_digest'],
                'local.profile.mia.base': receipt.get('base_image_id')}
    if len(raw) != 1 or raw[0].get('Id') != image_id or any(raw[0]['Config'].get('Labels', {}).get(k) != v for k, v in expected.items()):
        raise Refused('Mia image differs from its exact guarded build')
    resource = io.resource()
    owned = {e['outcome']['physical']['container']['id'] for e in operation['effects']
             if e.get('outcome') and e.get('payload', {}).get('action') in ('create', 'start')}
    for pid in command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).split():
        cgroup = (Path('/proc') / pid / 'cgroup').read_text()
        if not any(identifier in cgroup for identifier in owned):
            raise Refused('foreign GPU work blocks Mia member lifecycle')
    missing = []
    artifacts = json.loads((root / 'artifact-files.json').read_text())
    rows = artifacts['members'][member]
    for item in rows:
        path = Path(item['path'])
        if item.get('sha256') is None and item.get('git_blob') is None:
            missing.append(item['path'] + ': full content identity unknown')
            continue
        if not path.is_file() or path.stat().st_size != item['bytes']:
            missing.append(item['path'] + ': missing or changed')
            continue
        h = hashlib.sha256() if item.get('sha256') else hashlib.sha1(b'blob ' + str(item['bytes']).encode() + b'\0')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
        if h.hexdigest() != (item.get('sha256') or item['git_blob']): missing.append(item['path'] + ': content changed')
    # Mounts are finite, read-only and already installed. This path never creates
    # an NFS volume, a graph, a cache file or a preparation watcher.
    for volume, expected_options in artifacts['volumes'].get(member, {}).items():
        found = json.loads(command(['docker', 'volume', 'inspect', volume]))
        if len(found) != 1 or found[0].get('Driver') != 'local' or found[0].get('Options') != expected_options:
            missing.append(volume + ': NFS provenance changed')
    if not artifacts.get('complete'): missing.append('Mia consumed-file manifest is incomplete')
    addresses = json.loads(command(['ip', '-j', 'address', 'show']))
    observed = {x.get('local') for item in addresses for x in item.get('addr_info', [])}
    topology = artifacts['topology'][member]
    ring = set(topology['addresses']) <= observed
    for hca in topology['hcas']:
        port = Path('/sys/class/infiniband') / hca / 'ports/1'
        try:
            ring = ring and (port / 'state').read_text().strip() == '4: ACTIVE' and (port / 'link_layer').read_text().strip() == 'Ethernet'
            ring = ring and (port / 'gid_attrs/types/3').read_text().strip() == 'RoCE v2'
        except OSError: ring = False
    mem = Path('/proc/meminfo').read_text()
    available = next((int(line.split()[1]) / 1024**2 for line in mem.splitlines() if line.startswith('MemAvailable:')), None)
    return {'resource': resource, 'image_id': image_id, 'image_environment': dict(x.split('=', 1) for x in raw[0]['Config'].get('Env', []) if '=' in x),
            'facts': {'gpu_uuid': resource.removeprefix('gpu:'), 'image_verified': True, 'artifacts_verified': not missing,
                      'artifact_blockers': missing, 'ring_verified': ring, 'ring_rank': recipe['members'].index(member),
                      'nvidia_runtime': 'nvidia' in json.loads(command(['docker', 'info', '--format', '{{json .Runtimes}}'])),
                      'guard_verified': True, 'available_memory_gib': available}}
