"""Lab profile route: Sparks reserved for container jobs, with Imagination Studio off or its head on spark3 (and workers).

Facts: controller leases, the GPU census of stockstudio, the Studio head unit on spark3 and the stock head API.
Effects, each journaled first: stop or start the Studio head user unit; stock worker settings when the profile
keeps Studio on spark3. Workers get work only from the head, so a stopped head leaves no Studio work on any Spark.
"""
from __future__ import annotations

import time

from clustercontrol import EffectKind, EffectState, Refused, digest

ADAPTER = 'lab-v1'
HEAD, UNIT = 'spark3', 'imagination-studio.service'
NODES = ('spark1', 'spark2', 'spark3', 'spark4')
STOP_GRACE_S = 10


def covers(spec):
    return spec.get('lab') is True


def validate(p):
    """Four single-node groups. Each node is a job node, or spark3 is the Studio head and the other Studio nodes are
    its workers. The Studio default counts the head and each worker."""
    groups, flags = p.get('groups', {}), p.get('studio') or {}
    reserved = {node for group in p.get('reservations', {}) for node in groups.get(group, [])}
    workers = flags.get('workers') if flags else {}
    studio = {HEAD} | {n for n, on in (workers or {}).items() if on is True} if flags else set()
    if (p.get('instances') or p.get('assets') or p.get('puffer') or p.get('default') is not None
            or sorted(map(tuple, groups.values())) != [(n,) for n in NODES]
            or reserved | studio != set(NODES) or reserved & studio
            or (flags and (not isinstance(workers, dict) or set(workers) != set(NODES) - {HEAD}
                           or not all(type(v) is bool for v in workers.values()) or flags.get('default') != len(studio)))
            or set(p.get('images', {})) != reserved or not all(isinstance(v, str) and v for v in p.get('images', {}).values())):
        raise Refused('a lab profile requires four single-node groups, each a job node with an image, '
                      'or a Studio node (the head on spark3, workers on others) counted in the Studio default')


def head_active(runtime):
    reply = runtime.c.ssh(HEAD, f'systemctl --user is-active {UNIT}', timeout=30)
    state = reply.stdout.strip()
    if state not in ('active', 'activating', 'inactive', 'failed', 'deactivating'):
        raise Refused(f'{HEAD}: the Studio head unit state is unknown ({state or reply.stderr.strip()[-80:]})')
    return state not in ('inactive', 'failed')


def studio_idle(cluster):
    """Refuse while Studio has work. Idle model servers are released through each node's Studio app."""
    import studioidle
    for fact in studioidle.enforce(cluster, release=True):
        jobs = fact.get('jobs') or {}
        if fact.get('error'):
            raise Refused(f"{fact['node']}: Studio state is unknown: {fact['error']}")
        if jobs.get('running') or jobs.get('queued') or fact.get('busy'):
            raise Refused(f"{fact['node']}: Studio has work in progress; the profile stays")
        if any(row['kind'].startswith('studio:') for row in fact['cuda']):
            raise Refused(f"{fact['node']}: a Studio model server stays on the GPU; see `spark idle`")


def live_jobs(control, op):
    return [j['id'] for j in control['jobs'].values()
            if j.get('status') != 'settled' and set(j.get('resources', [])) & set(op['resources'])]


def head_request(action):
    return {'action': action, 'node': HEAD, 'unit': UNIT}


def permission(op, payload):
    flags, request = op['spec'].get('studio'), payload.get('request')
    if set(payload) != {'adapter', 'request'} or payload['adapter'] != ADAPTER:
        raise Refused('invalid lab profile effect')
    if request != head_request('start-head' if flags else 'stop-head'):
        if not flags:
            raise Refused('a lab profile with no Studio permits only the Studio head stop')
        from stockstudio import ADAPTER as STOCK, permission as stock_permission
        stock_permission(op, {'adapter': STOCK, 'request': request})
    sealed(op)


def sealed(op):
    from stockstudio import sealed as census_sealed
    return census_sealed(op)


def finish_permission(op):
    from stockstudio import finish_permission as checks
    checks(op)


def wait_head(runtime, seconds=120):
    import stockstudio
    deadline = time.monotonic() + seconds
    while not stockstudio.stock(runtime):
        if time.monotonic() >= deadline:
            raise Refused(f'the Studio head did not answer in {seconds} s; rerun the same command')
        time.sleep(3)


def start_head(runtime):
    started = runtime.c.ssh(HEAD, f'systemctl --user start {UNIT}', timeout=120)
    if started.returncode:
        raise Refused(f'{HEAD}: the Studio head did not start: {started.stderr.strip()[-200:]}')
    wait_head(runtime)


def stop_head(runtime, grace=STOP_GRACE_S):
    """Stop the Studio head. The preflight proved that Studio has no work, so a head that ignores the stop signal
    is killed after `grace` seconds and not after the 90 s stop timeout of its unit."""
    script = (f'systemctl --user stop --no-block {UNIT}; i=0; '
              f'while [ "$(systemctl --user is-active {UNIT})" != inactive ] && [ "$(systemctl --user is-active {UNIT})" != failed ] '
              f'&& [ $i -lt {grace} ]; do sleep 1; i=$((i+1)); done; '
              f'case "$(systemctl --user is-active {UNIT})" in inactive|failed) echo stopped;; '
              f'*) systemctl --user kill --kill-whom=all --signal=KILL {UNIT}; sleep 1; echo killed;; esac')
    reply = runtime.c.ssh(HEAD, script, timeout=grace + 60)
    if reply.stdout.strip() == 'killed':
        print(f'  {HEAD}: the Studio head ignored the stop signal for {grace} s and was killed', flush=True)
    if head_active(runtime):
        raise Refused('the Studio head did not stop; rerun the same command')
    return reply.stdout.strip()


def requirements(runtime, spec):
    flags, active = spec.get('studio'), head_active(runtime)
    reason, settings = '', None
    if flags:
        from clusterprofile import studio_ready, studio_settings
        overview = runtime.studio() if active else None
        settings = studio_settings(overview) if overview else None
        if not active or not studio_ready(flags, overview):      # the head, and each enabled worker, ready
            reason = 'the Studio head on spark3 is not ready with the workers of the profile'
    elif active:
        reason = 'the Studio head is running'
    state = 'unknown' if reason else 'verified'
    return ([{'resource': '*', 'reason': reason}] if reason else [],
            {'artifacts': {'state': state, 'evidence': 'none required'},
             'capabilities': {'state': state, 'adapter': ADAPTER},
             'services': {'state': state, 'studio_head': 'active' if active else 'inactive'},
             'settings': {'state': state, 'studio': settings}})


def next_request(runtime, spec):
    """The next setting that differs from the profile, or None."""
    flags, active = spec.get('studio'), head_active(runtime)
    if not flags:
        return head_request('stop-head') if active else None
    if not active:
        return head_request('start-head')
    import stockstudio
    return stockstudio.plan(flags, runtime.studio())


def send(runtime, request):
    if request['action'] == 'stop-head':
        stop_head(runtime)
    elif request['action'] == 'start-head':
        start_head(runtime)
    else:
        import stockstudio
        if not stockstudio.applied(request, stockstudio.send(runtime, request)):
            raise Refused('the Studio head did not apply the setting; rerun the same command')


def step(controller, runtime, identifier):
    """Apply each differing setting, then publish and finish. Safe to run again after any stop."""
    from stockstudio import bindings, census
    for _ in range(12):
        op = controller.view(identifier)
        hosts = census(runtime)
        if {h['resource_id'] for h in hosts.values()} != set(op['resources']):
            return controller.block(identifier, [{'resource': '*', 'reason': 'GPU identity changed; lease recovery required'}])
        effect = next((e for e in op['effects'] if e['state'] == EffectState.UNCERTAIN), None)
        request = effect['payload']['request'] if effect else next_request(runtime, op['spec'])
        if request is None:
            break
        if effect is None:
            controller.update_evidence(identifier, {'stock': hosts, 'bindings': bindings(hosts)}, [])
            effect = controller.issue(identifier, EffectKind.STUDIO_EFFECT, {'adapter': ADAPTER, 'request': request}, [])
        send(runtime, request)
        controller.outcome(identifier, effect['id'], {'effect_id': effect['id'], 'token': effect['token'],
                                                      'verified': True, 'request': request})
    blockers, checks = requirements(runtime, op['spec'])
    op = controller.update_evidence(identifier, {'stock': hosts, 'bindings': bindings(hosts), 'requirements': checks}, blockers)
    if blockers:
        return op
    controller.publish(identifier, dict(checks, spec_digest=op['spec_digest'], bindings=bindings(hosts)))
    return controller.finish(identifier)


def retire(controller, owner):
    """Give up a READY owner that has no live job. Its leases become free for the next profile."""
    if live_jobs(controller.store.read(), owner):
        raise Refused(f"{owner['spec'].get('name')} has live jobs; wait for them or cancel them (`spark job status`)")
    done = controller.cancel(owner['id'])
    if done.get('outcome') != 'cancelled' or controller.store.read()['layout'].get('operation_id') == owner['id']:
        raise Refused(f"current owner {owner['id']} ({owner['spec'].get('name')}) did not retire")


def apply(cluster, controller, runtime, spec, key, source):
    """Resume a pending lab operation, check a ready one, or take over from a ready Studio or lab profile."""
    from stockstudio import bindings, census
    spec = dict(spec, lab_profile=ADAPTER)
    state = controller.store.read()
    owner = state['operations'].get(state['layout'].get('operation_id'))
    pending = [o for o in state['operations'].values() if o['spec'].get('lab_profile') and o.get('outcome') is None]
    if pending:
        if len(pending) != 1 or pending[0]['spec_digest'] != digest(spec):
            raise Refused('another lab operation needs cancel first: ' + pending[0]['id'])
        if pending[0]['blockers']:
            controller.block(pending[0]['id'], [])
        return step(controller, runtime, pending[0]['id'])
    if owner and owner.get('outcome') == 'ready' and owner['spec_digest'] == digest(spec):
        hosts = census_with_jobs(runtime)
        blockers, checks = requirements(runtime, spec)
        return controller.record_health(owner['id'], {'stock': hosts, 'bindings': bindings(hosts), 'requirements': checks}, blockers)
    if owner and owner.get('outcome') not in (None, 'released', 'cancelled'):
        if owner.get('outcome') != 'ready' or not (owner['spec'].get('stock_studio') or owner['spec'].get('lab_profile')):
            raise Refused(f"current owner {owner['id']} ({owner['spec'].get('name')}) must retire first")
        studio_idle(cluster)
        retire(controller, owner)
    hosts = census(runtime)
    op = controller.request('activate', spec, [h['resource_id'] for h in hosts.values()], key, source)
    if op['blockers']:
        return op
    return step(controller, runtime, op['id'])


def census_with_jobs(runtime):
    """GPU identity of each node. Job containers use the GPU under a READY lab profile, so processes are not judged."""
    from stockstudio import HOST
    hosts = runtime.c.pmap(lambda n: runtime.python(n, HOST, {}), NODES)
    if any(len(h['uuids']) != 1 for h in hosts.values()):
        raise Refused('GPU identity is ambiguous')
    return {n: {'resource_id': 'gpu:' + h['uuids'][0], 'boot_id': h['boot_id'], 'observed_at': time.time()}
            for n, h in hosts.items()}


def leave(cluster, controller, runtime):
    """Before a Studio profile: start the Studio head, then retire a READY lab owner."""
    state = controller.store.read()
    owner = state['operations'].get(state['layout'].get('operation_id'))
    if not owner or not owner['spec'].get('lab_profile') or owner.get('outcome') != 'ready':
        return False
    if live_jobs(state, owner):
        raise Refused(f"{owner['spec'].get('name')} has live jobs; wait for them or cancel them (`spark job status`)")
    start_head(runtime)
    retire(controller, owner)
    return True
