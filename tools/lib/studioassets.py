"""Check cached Studio assets and copy missing files from the existing head."""
import hashlib
import json
import os
import stat
import re
from pathlib import Path


def check(manifest, home=None, headers=False):
    home = Path.home() if home is None else Path(home)
    errors = []
    for entry in manifest['files']:
        path = home / entry['path']
        try:
            info = path.lstat()
            if entry['kind'] == 'symlink':
                if not path.is_symlink() or os.readlink(path) != entry['target']:
                    raise ValueError('different symlink target')
                if not path.is_file():
                    raise ValueError('required resolved file missing')
                if path.resolve() != (home / entry['required_resolved_path']).resolve():
                    raise ValueError('different resolved file')
                if path.stat().st_size != entry['required_resolved_bytes']:
                    raise ValueError('different resolved file size')
            else:
                if not stat.S_ISREG(info.st_mode) or info.st_size != entry['bytes']:
                    raise ValueError('different file type or size')
                if stat.S_IMODE(info.st_mode) != int(entry['mode'], 8):
                    raise ValueError('different file mode')
                if headers and entry.get('first64k_sha256'):
                    with path.open('rb') as stream:
                        digest = hashlib.sha256(stream.read(65536)).hexdigest()
                    if digest != entry['first64k_sha256']:
                        raise ValueError('different first 64 KiB SHA-256')
                expected=entry.get('sha256') or entry.get('hf_etag')
                if headers and (not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None):
                    raise ValueError('content_identity_unknown: full expected SHA256 is absent')
                if expected:
                    result=hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
                    if result.hexdigest() != expected:
                        raise ValueError('different full-file SHA-256')
        except (OSError, ValueError) as error:
            errors.append({'path': entry['path'], 'error': str(error)})
    return {'ok': not errors, 'entries': len(manifest['files']), 'errors': errors}

def content_unknown(manifest):
    files={row['path']:row for row in manifest['files']}
    missing=[]
    for row in files.values():
        identity=files.get(row.get('required_resolved_path'),{}) if row['kind']=='symlink' else row
        value=identity.get('sha256') or identity.get('hf_etag')
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:missing.append(row['path'])
    return missing


def owner_busy(home=None):
    home = Path.home() if home is None else Path(home)
    directory = home / '.cache/studio-asset-sync-1106'
    for name in ('live-owner', 'spark1', 'spark2', 'spark1-drama3', 'spark2-drama3'):
        path = directory / (name + '.json')
        if path.exists() and json.loads(path.read_text()).get('complete') is not True:
            return True
    return False


def sync_commands(node, inventory, home=None):
    """Use the verified fabric path. Keep existing files and directory links."""
    import shlex
    if home is None:
        raise ValueError('asset synchronization requires an explicit remote home')
    nodes = {n['alias']: n for n in inventory}
    peer = nodes['spark3']['ring_ips']['spark2'][0]
    address = peer if node == 'spark2' else nodes['spark2']['ring_ips']['spark1'][0]
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', 'HostName=' + address, '-o', 'HostKeyAlias=' + nodes[node]['lan_ip']]
    if node == 'spark1':
        proxy = ['ssh', '-o', 'BatchMode=yes', '-o', 'HostName=' + peer,
                 '-o', 'HostKeyAlias=' + nodes['spark2']['lan_ip'], '-W', '%h:%p', 'spark2']
        ssh += ['-o', 'ProxyCommand=' + shlex.join(proxy)]
    return [['rsync', '-a', '--ignore-existing', '--partial-dir=.studio-sync-1106',
             '--checksum-choice=xxh128', '--exclude=*.lock', '--exclude=*.incomplete',
             '-e', shlex.join(ssh), home + '/' + relative + '/', node + ':' + home + '/' + relative + '/']
            for relative in ('ImaginationStudio/loras', 'spark-studio-models', 'models/drama-3')]
