"""Finite RDMA TP4 plans for the pinned Mia engine. No host effects."""
import copy
import hashlib
import json
from pathlib import Path
import re

from .spark_ds41f_1030 import digest, guard_decision
from .spark_ds41f_control import birth, resources

FAMILY = 'spark-ds41f'  # Shared native member journal and guard contract.
ENGINE = 'tensorfold-mia-tp4'
RECIPE = '8ffa67d53ef15bab997399135f85e8b0910f30be'
NATIVE = '59402086381a42c74d53053443224eec2790b4c5'
LOCK_SHA256 = '3dd833a72547488dcaa637c28c50b196e9579f01aa2a7fccf63dc4647c6e9cc9'
START_ORDER = ['spark4', 'spark3', 'spark2', 'spark1']
STOP_ORDER = ['spark1', 'spark2', 'spark3', 'spark4']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(package):
    package = Path(package)
    if sha(package / 'source.lock.json') != LOCK_SHA256:
        raise ValueError('Mia source lock differs')
    lock = json.loads((package / 'source.lock.json').read_text())
    for name, expected in lock['runtime_files'].items():
        if Path(name).name != name or sha(package / name) != expected:
            raise ValueError('Mia runtime source differs: ' + name)
    return lock


def load(root, document):
    package = Path(root) / 'recipes/mia-tp4-control'
    if not package.is_dir(): package = Path(root) / 'profiles/recipes/mia-tp4-control'
    lock = verify_sources(package)
    if document != json.loads((package / 'spec.json').read_text()):
        raise ValueError('Mia serving spec differs from pinned source')
    return {'family': FAMILY, 'engine': ENGINE, 'members': document['members'], 'spec': copy.deepcopy(document),
            'source': {'url': lock['recipe']['repository'], 'commit': RECIPE, 'lock_sha256': LOCK_SHA256},
            'package': str(package.resolve()), 'site_digest': digest(document['site']),
            'artifacts': json.loads((package / 'cache-files.json').read_text()),
            'native_coverage': {'ingress': {'implementation': 'supported', 'protocol': 'mia-tp4-control-v1'},
                                'members': {'implementation': 'supported', 'plan_contract': 'materialize-member-v1'},
                                'limits': 'exact RDMA TP4 only; installed image and complete artifacts require observed provenance'}}


def render(recipe, action):
    verify_sources(recipe['package'])
    expected = load(Path(recipe['package']).parent.parent, recipe['spec'])
    if {k:v for k,v in recipe.items() if k != 'package'} != {k:v for k,v in expected.items() if k != 'package'}:
        raise ValueError('Mia recipe or site differs')
    if action != 'start': raise ValueError('Mia compound scripts are never executable plans')
    return copy.deepcopy(json.loads((Path(recipe['package']) / 'launch.json').read_text())['rows'])


def materialize_member_plan(recipe, action, member, footprint, *, creation_key=None, image_id=None,
                            container=None, control=None, image_environment=None, phase='steady'):
    footprint = resources(footprint)
    if action not in ('create', 'start', 'stop', 'guard-start', 'guard-stop') or member not in recipe['members']:
        raise ValueError('unsupported Mia member action')
    rank = recipe['members'].index(member)
    launch = next(row for row in render(recipe, 'start') if row['rank'] == rank)
    if action == 'create':
        if (not isinstance(image_id, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id)
                or not isinstance(creation_key, str) or not re.fullmatch(r'[0-9a-f]{64}', creation_key) or container is not None):
            raise ValueError('Mia create requires immutable image and creation identity')
        binding = {'image_id': image_id, 'creation_key': creation_key, 'name': 'mia-tp4-' + creation_key + '-r' + str(rank),
                   'labels': {'local.profile.family': FAMILY, 'local.profile.creation': creation_key,
                              'local.profile.plan': digest({'source': LOCK_SHA256, 'resources': footprint, 'rank': rank,
                                                            'image_id': image_id, 'creation_key': creation_key}),
                              'local.profile.rank': str(rank)}}
        argv = ['docker', 'create', *launch['argv'][3:]]
        argv[argv.index('--name') + 1] = binding['name']
        at = argv.index('tensorfold-dsv41:native-6b3711e64978')
        argv[at] = image_id
        additions = []
        for key, value in sorted(binding['labels'].items()): additions += ['--label', key + '=' + value]
        # Discard the inherited NVIDIA shell entrypoint. The journaled init is
        # the exact native rank, including the API process whose birth is checked.
        additions += ['--entrypoint', '/opt/tensorfold/native/bin/tensorfold-native' if rank == 0 else '/opt/tensorfold/bin/tf-dsv41-lanes']
        del argv[at + 1]
        if rank == 0:
            if (not isinstance(control, dict) or set(control) != {'config_path', 'state_dir'}
                    or any(not isinstance(value, str) or not Path(value).is_absolute() or '..' in Path(value).parts for value in control.values())
                    or not isinstance(image_environment, dict)):
                raise ValueError('Mia head requires the prepared exact owner configuration')
            additions += ['--env', 'MIA_PROFILE_CONTROL_CONFIG=/run/profile-control/config.json',
                          '--volume', control['config_path'] + ':/run/profile-control/config.json:ro']
            binding['control_overlay'] = {'config_path': control['config_path'], 'native_source': LOCK_SHA256}
        argv[at:at] = additions
    else:
        if (not isinstance(container, dict) or set(container) != {'id', 'image_id', 'created', 'started', 'process_birth'}
                or not re.fullmatch(r'[0-9a-f]{64}', container.get('id', ''))
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', container.get('image_id', '')) or not container.get('created')):
            raise ValueError('immutable Mia container object required')
        binding = copy.deepcopy(container); argv = []
        if action == 'start':
            if container['started'] is not None or container['process_birth'] is not None:
                raise ValueError('Mia start needs a never-started object; reconcile the same effect')
            argv = ['docker', 'start', container['id']]
        elif not (action.startswith('guard') and phase == 'startup' and container['started'] is None and container['process_birth'] is None):
            physical = birth(container['process_birth'])
            if not isinstance(container['started'], str) or not container['started']:
                raise ValueError('exact running Mia birth required')
            if action == 'stop': argv = ['pidfd_send_signal', physical['identity'], physical['boot_id'], physical['started'], 'SIGTERM']
    plan = {'family': FAMILY, 'action': action, 'member': member, 'rank': rank, 'resources': footprint,
            'argv': argv, 'command_digest': digest(argv), 'source_path': 'tools/lib/profile_recipes/mia_tp4.py',
            'source_digest': sha(__file__), 'binding': binding, 'preview_only': True}
    if action.startswith('guard'):
        if phase not in ('startup', 'steady'): raise ValueError('unsupported Mia guard phase')
        plan['guard'] = {'phase': phase, 'threshold_kib': (5 if phase == 'startup' else 3) * 1024**2,
                         'interval_seconds': .25 if phase == 'startup' else 2, 'startup_wait_seconds': 120 if phase == 'startup' else 0,
                         'birth': container['process_birth'], 'container_id': container['id'], 'container_started': container['started'],
                         'on_unknown_memory': 'kill-exact-birth', 'signal': 'SIGKILL'}
    return plan


def preflight(recipe, facts):
    verify_sources(recipe['package'])
    blockers = []
    for rank, member in enumerate(recipe['members']):
        row = facts.get(member, {})
        for key in ('image_verified', 'artifacts_verified', 'ring_verified', 'nvidia_runtime', 'guard_verified'):
            if row.get(key) is not True: blockers.append(member + ': Mia ' + key + ' is unknown or false')
        blockers.extend(member + ': ' + reason for reason in row.get('artifact_blockers', []))
        if row.get('ring_rank') != rank: blockers.append(member + ': Mia physical ring rank differs')
        memory = row.get('available_memory_gib')
        if type(memory) not in (int, float) or memory < 100: blockers.append(member + ': Mia needs 100 GiB startup memory')
    return {'state': 'verified' if not blockers else 'unknown', 'blockers': blockers}
