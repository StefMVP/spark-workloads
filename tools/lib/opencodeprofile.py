"""Refresh native OpenCode v2 config only when its sessions are idle."""
from __future__ import annotations

import json
import subprocess
import sys


class PartialReloadError(RuntimeError):
    """Native location rebuild can fail after applying the managed config."""


def command(*args):
    result = subprocess.run(['opencode', *args], text=True, capture_output=True, timeout=30)
    if result.returncode:
        diagnostic = result.stdout + result.stderr
        if args == ('reload',) and all(marker in diagnostic for marker in (
                'ServiceUnavailableError', 'TypeError: o is not a function', 'ServerProcess.start')):
            raise PartialReloadError(
                f'native OpenCode reload exited {result.returncode}: '
                'ServiceUnavailableError / ServerProcess.start / TypeError: o is not a function')
        raise RuntimeError('native OpenCode command failed; inspect opencode service status')
    return json.loads(result.stdout) if args[0] == 'api' else None


def idle():
    active = command('api', 'GET', '/api/session/active')
    if not isinstance(active, dict) or not isinstance(active.get('data'), dict):
        raise RuntimeError('cannot establish native OpenCode session state')
    if active['data']:
        raise RuntimeError('OpenCode has active sessions; resume profile settings after they finish')


def matches(path, data, managed=None):
    documents = command('api', 'GET', '/api/config')
    source = next((d.get('info', {}) for d in documents if d.get('path') == str(path)), None)
    if source is None:
        return False
    model = data.get('model')
    if isinstance(model, str):
        provider, name = model.split('/', 1)
        model = {'providerID': provider, 'model': name}
    if source.get('model') != model:
        return False
    for name, agent in data.get('agents', {}).items():
        expected = agent.get('model')
        if isinstance(expected, str):
            provider, selected = expected.split('/', 1)
            expected = {'providerID': provider, 'model': selected}
        if source.get('agents', {}).get(name, {}).get('model') != expected:
            return False
    # Native config resolves credentials. Check only profile-owned fields.
    def fields(provider):
        if provider is None:
            return None
        return {key: provider.get(key) for key in ('name', 'package', 'models', 'body')} | {
            'baseURL': provider.get('settings', {}).get('baseURL')}
    providers = data.get('providers', {})
    return all(fields(source.get('providers', {}).get(key)) == fields(providers.get(key))
               for key in (managed if managed is not None else providers))


def verify_model(data):
    model = data.get('model')
    if model is None:
        return
    if isinstance(model, str):
        provider, name = model.split('/', 1)
    else:
        provider, name = model['providerID'], model['model']
    expected = data.get('providers', {}).get(provider)
    models = command('api', 'GET', '/api/model')['data']
    selected = next((m for m in models if m['providerID'] == provider and m['id'] == name), None)
    if (selected is None or (expected is not None and (
            selected.get('package') != expected.get('package')
            or selected.get('settings', {}).get('baseURL') != expected.get('settings', {}).get('baseURL')
            or selected.get('name') != expected.get('models', {}).get(name, {}).get('name', selected.get('name'))))):
        raise RuntimeError('OpenCode resolved model differs from profile settings')


def refresh(path, data, managed=None):
    reload_error = None
    if not matches(path, data, managed):
        idle()
        try:
            command('reload')
        except PartialReloadError as error:
            reload_error = error
            print(f'warning: {error}; checking effective managed config and resolved model', file=sys.stderr)
        if not matches(path, data, managed):
            raise RuntimeError('OpenCode loaded config differs after native reload; resume profile settings') from reload_error
    verify_model(data)
    if reload_error:
        print('native OpenCode recovery: effective managed config and resolved model verified; no reload retry', file=sys.stderr)
