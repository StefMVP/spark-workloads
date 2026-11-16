"""One native Puffer attempt under an existing mixed-profile GPU reservation."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from clustercontrol import Birth, Refused, digest, durable


FAMILY = 'puffer-native-v1'
SOURCE = 'ea7f989371270d2948f96728bda8aec5e7634a4c'


def _argv(spec, attempt):
    run = 'profile-' + digest(attempt)
    return ['timeout', '120', './puffer-' + spec['environment'], 'train', '--base.run_id=' + run,
            '--base.checkpoint_dir=/work/results/checkpoints', '--base.log_dir=/work/results/logs',
            '--base.async=0', '--base.seed=' + str(spec['seed']), '--train.gpus=1',
            '--train.total_timesteps=' + str(spec['steps']), '--vec.total_agents=4096',
            '--vec.num_buffers=1', '--vec.num_threads=8', '--train.horizon=32', '--train.minibatch_size=65536']


def _validate_plan(job):
    plan = job['command']
    expected_keys = {'node', 'resources', 'source_commit', 'image_id', 'binary_sha256', 'environment',
                     'steps', 'seed', 'mem_gib', 'family', 'attempt_id', 'binding', 'helper_sha256',
                     'token', 'output', 'argv'}
    if (not isinstance(plan, dict) or set(plan) != expected_keys or plan['family'] != FAMILY
            or plan['node'] != 'spark4' or plan['source_commit'] != SOURCE
            or plan['attempt_id'] != job['operation_id'] + ':' + job['id']
            or plan['attempt_id'] != job['attempt_id'] or plan['token'] != _token(job)
            or len(job['resources']) != 1 or plan['resources'] != job['resources']
            or not re.fullmatch(r'gpu:GPU-[A-Za-z0-9-]+', job['resources'][0])
            or plan['environment'] not in ('breakout', 'enduro')
            or type(plan['steps']) is not int or plan['steps'] < 524288 or plan['steps'] % 131072
            or type(plan['seed']) is not int or not 0 <= plan['seed'] < 2**31
            or type(plan['mem_gib']) is not int or not 1 <= plan['mem_gib'] <= 128
            or plan['mem_gib'] != job['mem_gib']
            or not re.fullmatch(r'sha256:[a-f0-9]{64}', plan['image_id'])
            or any(not re.fullmatch(r'[a-f0-9]{64}', plan[key]) for key in ('binary_sha256', 'helper_sha256'))):
        raise Refused('Puffer job command identity differs from its admitted single-GPU attempt')
    binding = plan['binding']
    if (not isinstance(binding, dict) or set(binding) != {'home', 'machine_id', 'boot_id', 'resource_id'}
            or any(not isinstance(v, str) or not v or '\n' in v for v in binding.values())
            or binding['resource_id'] != job['resources'][0] or not Path(binding['home']).is_absolute()
            or str(Path(binding['home'])) != binding['home'] or '..' in Path(binding['home']).parts):
        raise Refused('Puffer host binding is unknown')
    output = str(Path(binding['home']) / 'spark/puffer-training-1106/results/profile-jobs' / ('profile-' + digest(job['attempt_id'])))
    if plan['argv'] != _argv(plan, job['attempt_id']) or plan['output'] != output:
        raise Refused('Puffer command or checkpoint path differs from the supported native helper')
    return plan


def _command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise Refused('Puffer physical command outcome is unknown')
    return result.stdout


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


class PhysicalIO:
    def binding(self):
        ids = _command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).split()
        if len(ids) != 1 or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', ids[0]):
            raise Refused('Puffer GPU identity is unknown or ambiguous')
        return {'resource_id': 'gpu:' + ids[0], 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'machine_id': Path('/etc/machine-id').read_text().strip(), 'home': str(Path.home().resolve())}

    def probe(self, spec):
        binding = self.binding()
        root = Path(binding['home']) / 'spark/puffer-training-1106'
        source = root / 'PufferLib'
        binary = source / ('puffer-' + spec['environment'])
        if source.resolve() != source or binary.resolve() != binary or not binary.is_file():
            raise Refused('Puffer source or binary path is absent or redirected')
        revision = _command(['git', '-C', str(source), 'rev-parse', 'HEAD']).strip()
        image = json.loads(_command(['docker', 'image', 'inspect', spec['image_id']]))
        if (revision != spec['source_commit'] or _sha(binary) != spec['binary_sha256']
                or len(image) != 1 or image[0]['Id'] != spec['image_id']):
            raise Refused('Puffer source/image/full binary SHA differs from the profile pin')
        return binding

    def inspect(self, identifier):
        raw = json.loads(_command(['docker', 'inspect', identifier]))
        if len(raw) != 1 or raw[0]['Id'] != identifier:
            raise Refused('Puffer immutable container identity changed')
        return raw[0]

    def find(self, label):
        ids = _command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'label=spark.puffer.attempt=' + label]).split()
        if len(ids) > 1 or any(re.fullmatch(r'[0-9a-f]{64}', item) is None for item in ids):
            raise Refused('Puffer attempt has ambiguous container identity')
        return ids

    def process(self, pid):
        try:
            fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            return None
        if fields[0] in ('Z', 'X'):
            return None
        return {'kind': 'process', 'boot_id': self.binding()['boot_id'], 'identity': str(pid), 'started': fields[19]}

    def create(self, plan, output):
        root = Path(plan['binding']['home']) / 'spark/puffer-training-1106'
        argv = ['docker', 'create', '--gpus', 'device=' + plan['resources'][0].removeprefix('gpu:'),
                '--dns', '172.17.0.1', '--user', '1000:1000', '-e', 'HOME=/tmp',
                '-e', 'PUFFER_WORLD=1', '-e', 'PUFFER_RANK=0', '--entrypoint', 'bash',
                '--label', 'spark.puffer.attempt=' + plan['attempt_id'],
                '--label', 'spark.puffer.command=' + digest(plan),
                '-v', str(root / 'PufferLib') + ':/work/PufferLib:ro',
                '-v', str(output) + ':/work/results:rw', '-w', '/work/PufferLib',
                plan['image_id'], '-lc', __import__('shlex').join(plan['argv'])]
        identifier = _command(argv).strip()
        if not re.fullmatch(r'[a-f0-9]{64}', identifier):
            raise Refused('Puffer creation reply lacks an immutable container ID')
        return identifier

    def launch(self, identifier):
        _command(['docker', 'start', identifier])

    def terminate(self, birth):
        from nativecontrol import NativeExecutor, member_plan
        NativeExecutor().apply(member_plan('process_stop', self.binding()['resource_id'], birth, birth['identity']))

    def absent(self, birth):
        from nativecontrol import NativeExecutor, member_plan
        return NativeExecutor().observe(member_plan('process_stop', self.binding()['resource_id'], birth, birth['identity']))['state'] == 'absent'


class Journal:
    """Persist acceptance before every physical launch; uncertain launch is never repeated."""
    def __init__(self, root, io):
        self.root, self.io = Path(root), io

    def execute(self, action, job):
        plan = _validate_plan(job)
        if action not in ('start', 'observe', 'cancel'):
            raise Refused('unsupported Puffer host action')
        if self.io.binding() != plan['binding']:
            raise Refused('Puffer host or GPU birth changed')
        key = digest(job['attempt_id'])
        if self.root.resolve() != self.root:
            raise Refused('Puffer host journal path is redirected')
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (key + '.json')
        with (self.root / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            row = json.loads(path.read_text()) if path.exists() else None
            if row and row['plan'] != plan:
                raise Refused('Puffer host attempt conflicts with the admitted command')
            consent = {'attempt_id': job['attempt_id'], 'generation': job['generation']}
            cancelling = job.get('cancel_consent') == consent
            if action == 'cancel' and not cancelling:
                raise Refused('Puffer cancellation requires exact Controller.cancel_job consent')
            if cancelling:
                if row is None:
                    row = {'plan': copy.deepcopy(plan), 'phase': 'cancelled', 'container': None, 'birth': None}
                row['cancelled'] = True
                durable(path, row)
            if row and row['phase'] == 'cancelled':
                raise Refused('Puffer cancelled before host acceptance; no launch is permitted')
            if row is None:
                if action != 'start':
                    raise Refused('Puffer host acceptance is unknown; no launch on resume')
                self.io.probe(plan)
                # A stable private directory is new for this attempt. No old
                # checkpoints are mounted writable, removed, or overwritten.
                output = Path(plan['output'])
                if output.exists() or output.is_symlink() or output.resolve() != output:
                    raise Refused('Puffer output already exists; checkpoint overwrite forbidden')
                output.mkdir(parents=True)
                durable(output / 'owner.json', {'attempt_id': job['attempt_id'], 'command_digest': digest(plan)})
                row = {'plan': copy.deepcopy(plan), 'phase': 'create_uncertain', 'container': None, 'birth': None}
                durable(path, row)
                identifier = self.io.create(plan, output)
                row['container'] = self._object(identifier, plan)
                row['phase'] = 'created'
                durable(path, row)
            if row['phase'] == 'create_uncertain':
                ids = self.io.find(job['attempt_id'])
                if len(ids) != 1:
                    raise Refused('Puffer creation may have applied; no second create')
                row['container'] = self._object(ids[0], plan)
                row['phase'] = 'created'
                durable(path, row)
            if row['phase'] == 'created':
                if row.get('cancelled'):
                    raw=self._same_object(row['container'],plan)
                    state=raw['State']
                    if (state.get('Running') is not False or state.get('Pid') != 0
                            or not str(state.get('StartedAt','')).startswith('0001-')):
                        raise Refused('Puffer created object may have started; retain reservation')
                    return {'state':'never_started','attempt_id':job['attempt_id'],'token':_token(job),
                            'cancelled':True,'command_digest':digest(plan),
                            'container':{**row['container'],'started':None,'process_birth':None}}
                if job['status'] != 'admitted':
                    raise Refused('a running Puffer attempt cannot return to prelaunch')
                self.io.probe(plan)
                row['phase'] = 'start_uncertain'
                durable(path, row)
                self.io.launch(row['container']['id'])
            raw = self._same_object(row['container'], plan)
            state = raw['State']
            started = state.get('StartedAt')
            if row['birth'] is None:
                if not state['Running'] or not started or started.startswith('0001-'):
                    raise Refused('Puffer launch outcome is uncertain; no second start')
                birth = self.io.process(state['Pid'])
                Birth.parse(birth)
                if birth['boot_id'] != plan['binding']['boot_id']:
                    raise Refused('Puffer process belongs to another boot')
                row.update(birth=birth, started=started, phase='running')
                durable(path, row)
            if started != row['started']:
                raise Refused('Puffer container was restarted; physical birth changed')
            if state['Running']:
                if self.io.process(state['Pid']) != row['birth']:
                    raise Refused('Puffer PID now denotes another process birth')
                if row.get('cancelled'):
                    row['phase'] = 'stop_uncertain'
                    durable(path, row)
                    self.io.terminate(row['birth'])
                    raw = self._same_object(row['container'], plan)
                    state = raw['State']
                if state['Running']:
                    return self._result(row, job, 'running')
            if not self.io.absent(row['birth']):
                raise Refused('Puffer physical termination remains unknown')
            row['phase'] = 'terminated'
            row['exit_code'] = state.get('ExitCode')
            durable(path, row)
            return self._result(row, job, 'terminated')

    def _object(self, identifier, plan):
        raw = self.io.inspect(identifier)
        labels = raw['Config'].get('Labels', {})
        if (raw['Image'] != plan['image_id'] or not raw.get('Created')
                or labels.get('spark.puffer.attempt') != plan['attempt_id']
                or labels.get('spark.puffer.command') != digest(plan)):
            raise Refused('Puffer container labels, image or creation birth differ')
        return {'id': identifier, 'created': raw['Created'], 'image_id': raw['Image']}

    def _same_object(self, expected, plan):
        if self._object(expected['id'], plan) != expected:
            raise Refused('Puffer immutable container birth changed')
        return self.io.inspect(expected['id'])

    @staticmethod
    def _result(row, job, state):
        return {'state': state, 'attempt_id': job['attempt_id'], 'token': _token(job), 'birth': row['birth'],
                'container': {**row['container'], 'started': row['started'], 'process_birth': row['birth']},
                'checkpoint_root': row['plan']['output'], 'exit_code': row.get('exit_code')}


def _token(job):
    return {'operation_id': job['operation_id'], 'generation': job['generation'], 'resources': job['resources']}


def host_action(data):
    io = PhysicalIO()
    if data['action'] == 'probe':
        return io.probe(data['spec'])
    plan = data['job']['command']
    root = Path(plan['binding']['home']) / '.local/state/spark/puffer'
    return Journal(root, io).execute(data['action'], data['job'])


class Transport:
    def __init__(self, runtime):
        self.runtime = runtime
        folder = Path(__file__).parent
        self.sources = {name: (folder / (name + '.py')).read_text()
                        for name in ('clustercontrol', 'nativecontrol', 'puffercontrol')}

    def call(self, node, action, **arguments):
        payload = json.dumps({'action': action, **arguments})
        code = 'import json,sys,types\nD=json.loads(' + repr(payload) + ')\n'
        for name, source in self.sources.items():
            code += ('m=types.ModuleType(' + repr(name) + ')\nm.__file__=' + repr(name + '.py')
                     + '\nsys.modules[' + repr(name) + ']=m\nexec(' + repr(source) + ',m.__dict__)\n')
        code += ("try:\n result={'ok':True,'value':m.host_action(D)}\n"
                 "except (OSError,ValueError,KeyError,sys.modules['clustercontrol'].Refused) as error:\n result={'ok':False,'reason':str(error)}\n"
                 "print(json.dumps(result))\n")
        script = "python3 - <<'PUFFER_CONTROL_PY'\n" + code + '\nPUFFER_CONTROL_PY'
        raw = json.loads(self.runtime.remote(node, script, timeout=180))
        if not isinstance(raw, dict) or raw.get('ok') is not True:
            raise Refused(raw.get('reason', 'Puffer transport result is unknown') if isinstance(raw, dict) else 'Puffer transport result is unknown')
        return raw['value']


def _current(controller, value):
    from clustercontrol import require_owner, token_for
    op = controller.view(value['id'])
    if op['generation'] != value['generation'] or op['spec_digest'] != value['spec_digest']:
        raise Refused('Puffer profile receipt is stale')
    require_owner(controller.store.read(), token_for(op))
    if (op.get('outcome') != 'ready' or op.get('policy') != 'run'
            or op.get('current_observation', {}).get('state', 'verified') != 'verified'):
        raise Refused('Puffer requires a currently READY owned mixed profile')
    return op


def _spec(op, request):
    allowed = {'job_id', 'environment', 'steps', 'seed', 'mem_gib'}
    if (not isinstance(request, dict) or set(request) - allowed or not isinstance(request.get('job_id'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', request['job_id'])):
        raise Refused('invalid Puffer job request')
    environment = request.get('environment')
    steps, seed, memory = request.get('steps', 524288), request.get('seed', 73), request.get('mem_gib', 32)
    if (environment not in ('breakout', 'enduro') or type(steps) is not int or steps < 524288 or steps % 131072
            or type(seed) is not int or not 0 <= seed < 2**31 or type(memory) is not int or not 1 <= memory <= 128):
        raise Refused('unsupported Puffer environment, steps, seed or memory reservation')
    puffer = op['spec'].get('puffer', {})
    node = puffer.get('node')
    groups = op['spec'].get('groups', {})
    reserved = any(nodes == [node] and op['spec'].get('reservations', {}).get(group, {}).get('by') == 'puffer-gx10'
                   for group, nodes in groups.items())
    resource = op.get('bindings', {}).get(node, {}).get('resource_id')
    binary = puffer.get('binaries', {}).get('puffer-' + environment)
    if (node != 'spark4' or not reserved or resource not in op['resources']
            or puffer.get('source_commit') != SOURCE or not re.fullmatch(r'sha256:[0-9a-f]{64}', puffer.get('image_id', ''))
            or not isinstance(binary, str) or not re.fullmatch(r'[0-9a-f]{64}', binary)):
        raise Refused('Puffer requires exact single-GPU4 reservation and source/image/full binary pins')
    return {'node': node, 'resources': [resource], 'source_commit': SOURCE, 'image_id': puffer['image_id'],
            'binary_sha256': binary, 'environment': environment, 'steps': steps, 'seed': seed, 'mem_gib': memory}


def _barriers(runtime, op, resource):
    from profileadapter import StudioAdapter
    from profileplan import parse_snapshot, sealed_authorities
    previous = op.get('observations', {}).get('authorities', {})
    if not {'spark3', 'spark4'} <= set(previous):
        raise Refused('Puffer needs retained coordinator and GPU4 authority evidence')
    adapter = StudioAdapter(runtime, {n: (n, 7870) for n in ('spark3', 'spark4')}, time.monotonic)
    fresh = {n: adapter.snapshot(n, previous[n]) for n in ('spark3', 'spark4')}
    worker = parse_snapshot(fresh['spark4']['snapshot'])
    head = parse_snapshot(fresh['spark3']['snapshot'])
    if head.self_resource.model_dump() != op['bindings']['spark3']:
        raise Refused('Puffer coordinator physical binding changed')
    expected = op['bindings']['spark4']
    if worker.self_resource.model_dump() != expected:
        raise Refused('Puffer GPU4 alias no longer binds the registered instance/birth')
    rows = [r for r in head.worker_bindings if r.worker == 'spark4']
    if (len(rows) != 1 or rows[0].resource_id != resource or rows[0].instance_id != worker.instance_id
            or rows[0].birth != worker.birth or rows[0].enabled):
        raise Refused('Puffer coordinator binding is ambiguous, changed or still enabled')
    open_resources = {op['bindings'][n]['resource_id'] for n in ('spark1', 'spark2', 'spark3')}
    if any('*' in f.resources or open_resources & set(f.resources) for f in head.fences):
        raise Refused('Puffer must preserve open admission on enabled Studio resources')
    for node in ('spark1', 'spark2'):
        expected = op['bindings'][node]
        rows = [r for r in head.worker_bindings if r.worker == node]
        if (len(rows) != 1 or not rows[0].enabled or any(rows[0].model_dump()[k] != expected[k]
                for k in ('resource_id', 'instance_id', 'birth'))):
            raise Refused('enabled Studio worker binding changed')
    sealed_authorities({'id': op['id'], 'generation': op['generation'], 'resources': [resource],
                        'observations': {'authorities': fresh}})
    return fresh


def _save_blocker(controller, job, error):
    return controller.record_job_blocker(job['id'], job['attempt_id'], job['generation'], str(error))


def _observe(controller, runtime, job, action):
    current = controller.store.read()['jobs'].get(job['id'])
    if (not current or current['attempt_id'] != job['attempt_id'] or current['generation'] != job['generation']
            or current['command'] != job['command'] or current['kind'] != FAMILY):
        raise Refused('Puffer job receipt differs from its durable attempt')
    if current['command'].get('helper_sha256') != _sha(__file__):
        raise Refused('Puffer helper source changed; original source is required for resume')
    if current['status'] == 'settled':
        return current
    if action == 'cancel' and current.get('cancel_consent') != {'attempt_id': current['attempt_id'], 'generation': current['generation']}:
        raise Refused('Puffer cancellation requires exact Controller.cancel_job consent')
    op = controller.view(current['operation_id'])
    cancelled = current.get('cancel_consent') == {'attempt_id': current['attempt_id'], 'generation': current['generation']}
    if action == 'cancel' or cancelled:
        from clustercontrol import require_owner, token_for
        require_owner(controller.store.read(), token_for(op))
        _barriers(runtime, op, current['resources'][0])
    elif current['status'] == 'admitted':
        _current(controller, op)
        _barriers(runtime, op, current['resources'][0])
        controller.job_start_permission(current['id'], current['attempt_id'], current['generation'])
    try:
        result = Transport(runtime).call(current['command']['node'], action, job=current)
        if (result['state'] not in ('running', 'terminated', 'never_started') or result['token'] != _token(current)
                or result['attempt_id'] != current['attempt_id']):
            raise Refused('Puffer host reply has unknown ownership')
        if result['state']=='never_started':
            return controller.settle_unstarted_job(current['id'],current['attempt_id'],current['generation'],result)
        if current['status'] == 'admitted':
            records = {current['resources'][0]: {**result, 'state': 'running'}}
            current = controller.observe_job_start(current['id'], current['attempt_id'], current['generation'], records)
        elif current['births'] != {current['resources'][0]: result['birth']}:
            raise Refused('Puffer host reply changed physical process birth')
        if result['state'] == 'terminated':
            current = controller.settle_job(current['id'], current['attempt_id'], current['generation'], {current['resources'][0]: result})
        return _save_blocker(controller, current, '')
    except (OSError, ValueError, KeyError, Refused) as error:
        return _save_blocker(controller, current, error)


def start(controller, runtime, profileop, requestdict):
    op = _current(controller, profileop)
    spec = _spec(op, requestdict)
    existing = controller.store.read()['jobs'].get(requestdict['job_id'])
    if existing:
        if (existing['operation_id'] != op['id'] or any(existing['command'].get(k) != v for k, v in spec.items())):
            raise Refused('Puffer job identity conflicts with another request')
        return resume(controller, runtime, existing)
    _barriers(runtime, op, spec['resources'][0])
    transport = Transport(runtime)
    binding = transport.call(spec['node'], 'probe', spec=spec)
    if (binding['resource_id'] != spec['resources'][0]
            or binding['boot_id'] != op['bindings']['spark4']['birth']['boot_id']):
        raise Refused('Puffer host probe differs from Studio GPU4 binding/birth')
    attempt = op['id'] + ':' + requestdict['job_id']
    run = 'profile-' + digest(attempt)
    plan = {**spec, 'family': FAMILY, 'attempt_id': attempt, 'binding': binding, 'helper_sha256': _sha(__file__),
            'token': {'operation_id': op['id'], 'generation': op['generation'], 'resources': spec['resources']},
            'output': str(Path(binding['home']) / 'spark/puffer-training-1106/results/profile-jobs' / run),
            'argv': _argv(spec, attempt)}
    job = controller.admit_job(op['id'], requestdict['job_id'], FAMILY, spec['resources'], spec['mem_gib'], plan)
    return _observe(controller, runtime, job, 'start')


def resume(controller, runtime, job):
    return _observe(controller, runtime, job, 'observe')


def cancel(controller, runtime, job):
    return _observe(controller, runtime, job, 'cancel')
