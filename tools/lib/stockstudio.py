"""Stock Imagination Studio placement for a head without the optional /api/control routes.

Facts: controller leases, an owned-process GPU census on every Spark and the stock head API.
Effects: worker enable (with use jobs), worker /update and the default Spark count, each journaled first.
"""
from __future__ import annotations

import time

from clustercontrol import EffectKind, EffectState, Refused, digest, token_for

ADAPTER = 'stock-studio-v1'
NODES = ('spark1', 'spark2', 'spark3', 'spark4')
PENDING = 'Studio workers are starting; the resume unit continues this operation'
PROBE = '''import urllib.request, urllib.error
def code(path):
    try:
        return getattr(urllib.request.urlopen('http://127.0.0.1:7870'+path,timeout=5),'status',200)
    except urllib.error.HTTPError as e: return e.code
    except OSError: return None
print(json.dumps({'sparks':code('/api/sparks'),'control':code('/api/control/snapshot')}))'''
HOST = '''import subprocess, pathlib
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
print(json.dumps({'uuids':uuids,'boot_id':boot}))'''


def covers(spec):
    """Only a Studio-only profile on all four Sparks, with no reservation, uses this route."""
    flags = spec.get('studio') or {}
    nodes = {'spark3'} | {n for n, on in flags.get('workers', {}).items() if on}
    return (bool(flags) and not spec.get('instances') and not spec.get('recipe_family') and not spec.get('puffer')
            and not spec.get('reservations') and nodes == set(NODES))


def stock(runtime):
    """Only a running head that answers HTTP 404 for the control route is stock Studio."""
    return runtime.python('spark3', PROBE, {}) == {'sparks': 200, 'control': 404}


def census(runtime):
    from clusterprofile import GPU_PROBE
    hosts = runtime.c.pmap(lambda n: runtime.python(n, HOST, {}), NODES)
    gpus = runtime.c.pmap(lambda n: runtime.python(n, GPU_PROBE, {'proc': '/proc', 'home': None}), NODES)
    out = {}
    for n in NODES:
        foreign = [p['pid'] for p in gpus[n] if not p.get('owned_studio')]
        if foreign:
            raise Refused(f'{n}: GPU process {foreign} is not a guarded Studio process; its owner must stop it first')
        if len(hosts[n]['uuids']) != 1:
            raise Refused(f'{n}: GPU identity is ambiguous')
        out[n] = {'resource_id': 'gpu:' + hosts[n]['uuids'][0], 'boot_id': hosts[n]['boot_id'],
                  'studio_gpu': sorted(p['pid'] for p in gpus[n]), 'observed_at': time.time()}
    return out


def bindings(hosts):
    # Census evidence only: GPU UUID and host boot. No process birth is claimed.
    return {n: {'resource_id': h['resource_id'], 'instance_id': 'gpu-census:' + n,
                'birth': {'boot_id': h['boot_id'], 'service_id': 'gpu-census:' + h['boot_id']}} for n, h in hosts.items()}


def plan(flags, overview):
    """Next stock setting that differs from the saved profile, in a safe order."""
    from clusterprofile import studio_worker_ready
    workers = {w.get('address'): w for w in overview.get('workers', [])}
    for node, enabled in flags['workers'].items():
        w = workers.get(node)
        if w is None:
            raise Refused(f'{node} is not registered on the Studio head')
        # A Spark the profile gives the Studio takes its jobs: a "render" left from a Blender session is set back.
        if w.get('enabled') != enabled or (enabled and w.get('use') == 'render'):
            if not enabled and (w.get('job') or w.get('working')):
                raise Refused(f'{node} has Studio work in progress; it stays enabled')
            return {'action': 'enable', 'worker': node, 'enabled': enabled}
    for node, enabled in flags['workers'].items():
        w = workers[node]
        if enabled and not studio_worker_ready(w) and not w.get('working'):
            return {'action': 'update', 'worker': node}
    if overview.get('default') != flags['default']:
        return {'action': 'default', 'sparks': flags['default']}
    return None


def applied(request, overview):
    workers = {w.get('address'): w for w in overview.get('workers', [])}
    if request['action'] == 'enable':
        w = workers.get(request['worker'], {})
        return w.get('enabled') == request['enabled'] and not (request['enabled'] and w.get('use') == 'render')
    if request['action'] == 'update':
        from clusterprofile import studio_worker_ready
        w = workers.get(request['worker'], {})
        return bool(w.get('working')) or studio_worker_ready(w)
    return overview.get('default') == request['sparks']


def send(runtime, request):
    path, body = {'enable': (f"/api/sparks/workers/{request.get('worker')}", {'enabled': request.get('enabled'), **({'use': 'jobs'} if request.get('enabled') else {})}),
                  'update': (f"/api/sparks/workers/{request.get('worker')}/update", {}),
                  'default': ('/api/sparks', {'sparks': request.get('sparks')})}[request['action']]
    result = runtime.api('spark3', 7870, path, body)
    if not isinstance(result, dict):
        raise Refused('Studio head refused or lost the stock request; rerun the same apply')
    return result


def permission(op, payload):
    """Only settings named by the saved profile, on owned resources."""
    flags = op['spec']['studio']
    request = payload.get('request', {})
    if set(payload) != {'adapter', 'request'} or payload['adapter'] != ADAPTER:
        raise Refused('invalid stock Studio effect')
    valid = ((request.get('action') == 'enable' and set(request) == {'action', 'worker', 'enabled'}
              and flags['workers'].get(request['worker']) == request['enabled'])
             or (request.get('action') == 'update' and set(request) == {'action', 'worker'}
                 and flags['workers'].get(request['worker']) is True)
             or (request.get('action') == 'default' and request == {'action': 'default', 'sparks': flags['default']}))
    if not valid:
        raise Refused('stock Studio effect differs from the saved profile')
    sealed(op)


def sealed(op):
    hosts = op.get('observations', {}).get('stock')
    if not isinstance(hosts, dict) or set(hosts) != set(NODES):
        raise Refused('stock Studio GPU census is missing')
    if {h['resource_id'] for h in hosts.values()} != set(op['resources']):
        raise Refused('stock Studio GPU identity changed; lease recovery required')
    return hosts


def finish_permission(op):
    sealed(op)
    checks = op['observations'].get('requirements', {})
    if any(checks.get(k, {}).get('state') != 'verified' for k in ('artifacts', 'capabilities', 'services', 'settings')):
        raise Refused('stock Studio readiness is not verified')
    if checks != {k: op['readiness'][k] for k in checks}:
        raise Refused('stock Studio readiness evidence changed')


def installed(spec, overview):
    """Capabilities each asset target reports through /api/sparks. Not a file or weight audit."""
    assets = spec.get('assets')
    if not assets:
        return {}
    workers = {w.get('address'): w for w in overview.get('workers', [])}
    missing = {}
    for node in assets['targets']:
        can = workers.get(node, {}).get('can') or {}
        gaps = [key for key in assets['capabilities'] if can.get(key) is not True]
        gaps += ['image:' + name for name in assets['image_models'] if name not in can.get('image_models', [])]
        if can.get('loras', 0) < assets['loras']:
            gaps.append('LoRAs:' + str(assets['loras']))
        if gaps:
            missing[node] = gaps
    return missing


def requirements(runtime, spec, overview):
    from clusterprofile import studio_ready, studio_settings
    ready = studio_ready(spec['studio'], overview)
    missing = installed(spec, overview) if ready else {}
    blockers = [] if ready else [{'resource': '*', 'reason': PENDING}]
    if missing:
        blockers.append({'resource': '*', 'reason': 'installed Studio capability is missing', 'missing': missing})
    state = lambda ok: 'verified' if ok else 'unknown'
    # Full file and hash audit stays explicit: spark profile status --assets <profile>.
    return blockers, {'artifacts': {'state': state(ready and not missing), 'evidence': 'installed-capability',
                                    'missing': missing, 'weight_integrity': 'not checked'},
                      'capabilities': {'state': state(ready), 'adapter': ADAPTER},
                      'services': {'state': state(ready), 'available': overview.get('available')},
                      'settings': {'state': state(ready), 'studio': studio_settings(overview)}}


def step(controller, runtime, identifier):
    op = controller.view(identifier)
    uncertain = next((e for e in op['effects'] if e['state'] == EffectState.UNCERTAIN), None)
    if uncertain:
        request = uncertain['payload']['request']
        overview = runtime.studio()
        if not applied(request, overview):
            if not applied(request, send(runtime, request)):  # absolute settings; Studio serializes worker actions
                raise Refused('Studio head did not apply the resent setting')
        controller.outcome(identifier, uncertain['id'], {'effect_id': uncertain['id'], 'token': uncertain['token'],
                                                         'verified': True, 'request': request})
        return controller.view(identifier), True
    hosts = census(runtime)
    if {h['resource_id'] for h in hosts.values()} != set(op['resources']):
        return controller.block(identifier, [{'resource': '*', 'reason': 'stock Studio GPU identity changed; lease recovery required'}]), False
    overview = runtime.studio()
    request = plan(op['spec']['studio'], overview)
    if request is not None:
        controller.update_evidence(identifier, {'stock': hosts, 'bindings': bindings(hosts)}, [])
        effect = controller.issue(identifier, EffectKind.STUDIO_EFFECT, {'adapter': ADAPTER, 'request': request}, [])
        send(runtime, request)
        controller.outcome(identifier, effect['id'], {'effect_id': effect['id'], 'token': effect['token'],
                                                      'verified': True, 'request': request})
        return controller.view(identifier), True
    blockers, checks = requirements(runtime, op['spec'], overview)
    op = controller.update_evidence(identifier, {'stock': hosts, 'bindings': bindings(hosts), 'requirements': checks}, blockers)
    if blockers:
        return op, False
    evidence = dict(checks, spec_digest=op['spec_digest'], bindings=bindings(hosts))
    controller.publish(identifier, evidence)
    return controller.finish(identifier), False


def apply(controller, runtime, spec, key, source, wait=0):
    """Resume one pending stock operation, converge a ready one, or start a new one."""
    if not covers(spec):
        raise Refused('stock Studio route covers only a Studio-only four-Spark profile')
    spec = dict(spec, stock_studio=ADAPTER)
    state = controller.store.read()
    owner = state['operations'].get(state['layout'].get('operation_id'))
    pending = [o for o in state['operations'].values() if o['spec'].get('stock_studio') and o.get('outcome') is None]
    if pending:
        if len(pending) != 1 or pending[0]['spec_digest'] != digest(spec):
            raise Refused('another stock Studio operation needs cancel first: ' + pending[0]['id'])
        identifier = pending[0]['id']
        if not controller.view(identifier)['ownership_current']:
            return pending[0]
        if pending[0]['blockers']:
            controller.block(identifier, [])
    elif owner and owner.get('outcome') == 'ready' and owner['spec_digest'] == digest(spec):
        hosts = census(runtime)
        if plan(spec['studio'], runtime.studio()) is None:
            blockers, checks = requirements(runtime, spec, runtime.studio())
            return controller.record_health(owner['id'], {'stock': hosts, 'bindings': bindings(hosts), 'requirements': checks}, blockers)
        # Drifted settings: a child takes the same leases and re-applies them; Studio keeps running.
        op = controller.request('activate', spec, owner['resources'], key, source, parent=owner['id'])
        if op['blockers']:
            return op
        identifier = op['id']
    elif owner and owner.get('outcome') not in (None, 'released', 'cancelled'):
        raise Refused(f"current owner {owner['id']} ({owner['spec'].get('name')}) must retire first")
    else:
        hosts = census(runtime)
        op = controller.request('activate', spec, [h['resource_id'] for h in hosts.values()], key, source)
        if op['blockers']:
            return op
        identifier = op['id']
    deadline = time.monotonic() + wait
    for _ in range(16):
        op, progressed = step(controller, runtime, identifier)
        if op.get('outcome') or op['phase'] == 'waiting' and op['blockers'] and op['blockers'][0]['reason'] != PENDING:
            return op
        if not progressed:
            if time.monotonic() >= deadline:
                return op
            time.sleep(10)
    return controller.view(identifier)
