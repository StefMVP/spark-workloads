"""Native installed-runtime host boundary. No Studio service is required."""


def rpc(data):
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import signal
    import subprocess
    import tempfile
    import time
    import urllib.request

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def configuration_matches(raw, expected):
        config = {k:raw[k] for k in ('Config','HostConfig')}
        if digest(config) == expected: return True
        if config['HostConfig'].get('Dns') != []: return False
        # Docker can serialize inherited DNS as [] after a daemon restart.
        config['HostConfig'] = dict(config['HostConfig'], Dns=None)
        return digest(config) == expected

    def run(argv, timeout=30):
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError('native command failed: ' + argv[0])
        return r.stdout

    def durable(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
            fd = os.open(path.parent, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally:
            Path(temp).unlink(missing_ok=True)

    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def birth(pid):
        try: fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError: return None
        if fields[0] in ('Z', 'X'): return None
        return {'kind':'process', 'identity':str(pid), 'boot_id':boot, 'started':fields[19]}

    def inspect(identifier):
        rows = json.loads(run(['docker', 'inspect', identifier]))
        if len(rows) != 1 or rows[0]['Id'] != identifier:
            raise RuntimeError('immutable native object differs')
        c = rows[0]; state = c['State']
        return {'id':c['Id'], 'image_id':c['Image'], 'created':c['Created'],
                'started':state['StartedAt'] if state['Running'] else None,
                'process_birth':birth(state['Pid']) if state['Running'] else None}, c

    node = data['member']
    manifests = data['manifests']
    uuid = run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).strip()
    resource = 'gpu:' + uuid
    if any(m['nodes'][node]['resource'] != resource for m in manifests.values() if node in m['nodes']):
        raise RuntimeError('native host GPU differs from registered physical placement')

    def census():
        containers = {}
        for m in manifests.values():
            if node not in m['nodes']:continue
            row = m['nodes'][node]
            if 'id' in row:
                current, raw = inspect(row['id'])
                if current['image_id'] != row['image'] or not configuration_matches(raw, row['config_digest']):
                    raise RuntimeError('original native image or configuration changed')
                containers[m['name']] = current
        for name, current in data.get('owned', {}).items():
            found, raw = inspect(current['id'])
            if found['image_id'] != manifests[name]['nodes'][node]['image']:
                raise RuntimeError('owned native image changed')
            containers[name] = found
        allowed = {c['id'] for c in containers.values() if c['process_birth']}
        gpu = []
        for pid in run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).split():
            group = (Path('/proc') / pid / 'cgroup').read_text()
            if not any(cid in group for cid in allowed):
                raise RuntimeError('foreign or unknown GPU owner blocks native switch')
            gpu.append({'birth':birth(pid), 'container':next(cid for cid in allowed if cid in group)})
        # An absent control socket alone is not proof that Studio is stopped.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit(): continue
            try: args = (proc / 'cmdline').read_bytes().split(b'\0')
            except FileNotFoundError: continue
            executable=Path(os.fsdecode(args[0])).name if args else ''
            if executable.startswith(('python','uvicorn')) and any(Path(os.fsdecode(a)).name in ('main.py','StudioServer.py','app.py') or a in (b'server.app:app',b'studio.server:app') for a in args[1:] if a):
                raise RuntimeError('Studio or unknown Python GPU producer blocks native switch')
        import socket
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex(('127.0.0.1', 7870)) == 0:
                raise RuntimeError('running Studio requires its admission authority')
        missing = {}
        for key, pid in data.get('legacy_pids', {}).items():
            if not str(pid).isdigit() or birth(pid) is not None:
                raise RuntimeError('legacy job process is present or unknown')
            missing[key] = {'pid':str(pid), 'absent':True, 'boot_id':boot}
        available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) / 1024**2
        return {'resource_id':resource, 'boot_id':boot, 'containers':containers, 'gpu':gpu,
                'studio_absent':True, 'legacy_absent':missing, 'memory_gib':available, 'observed_at':time.time()}

    def idle(manifest):
        if node != manifest['head']: return
        url = 'http://127.0.0.1:' + str(manifest['port'])
        with urllib.request.urlopen(url + '/metrics', timeout=5) as reply:
            text = reply.read().decode()
        prefix = 'vllm:' if manifest['port'] == 8000 else 'tensorfold:'
        wanted = ['num_requests_running','num_requests_waiting'] if prefix == 'vllm:' else ['requests_running','requests_waiting']
        for key in wanted:
            values = [float(line.split()[-1]) for line in text.splitlines() if line.split('{')[0].split(' ')[0] == prefix+key]
            if not values or any(v != 0 for v in values):
                raise RuntimeError('native requests are busy or unknown: ' + key)
        # No keep-alive client may enqueue work after the idle observation.
        sockets = run(['ss','-Htn','state','established','sport','=',':'+str(manifest['port'])])
        if sockets.strip(): raise RuntimeError('native client connection is still active')

    action = data['action']
    if action == 'observe':
        for m in manifests.values():
            if node not in m['nodes']:continue
            row=m['nodes'][node]
            for name,expected in row.get('volumes',{}).items():
                volume=json.loads(run(['docker','volume','inspect',name]))
                if len(volume)!=1 or volume[0]['Driver']!='local' or volume[0].get('Options')!=expected:
                    raise RuntimeError('native NFS volume differs from saved read-only source')
            images=json.loads(run(['docker','image','inspect',row['image']]))
            if len(images)!=1 or images[0]['Id']!=row['image']:raise RuntimeError('native image is unavailable')
            if 'create' in row:
                argv=row['create']
                for i,a in enumerate(argv):
                    if a!='-v':continue
                    source=argv[i+1].split(':',1)[0]
                    if source.startswith('/'):
                        if not Path(source).exists():raise RuntimeError('native mounted asset is absent: '+source)
                    else:run(['docker','volume','inspect',source])
        return census()
    if action == 'idle':
        idle(manifests[data['target']]); return {'idle':True}
    effect = data['effect']; payload = effect['payload']; plan = payload['plan']
    if payload['member'] != node or payload['resource'] != resource or payload['boot_id'] != boot:
        raise RuntimeError('native effect physical placement changed')
    journal = Path.home() / '.local/state/spark/native-installed.json'
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(journal.read_text()) if journal.exists() else {'owner':None,'effects':{}}
        owner = state['owner']; token = effect['token']
        if action=='unaccepted':
            retired=data['retired'];old=retired['manifest']['nodes'][node]
            if owner!=retired['token'] or effect['id'] in state['effects']:
                raise RuntimeError('native refusal has an accepted effect or different host owner')
            current,_=inspect(plan['binding']['id'])
            if current!=plan['binding']:raise RuntimeError('native refusal target incarnation changed')
            prior,raw=inspect(old['id'])
            if (prior['process_birth'] is not None or prior['image_id']!=old['image']
                or not configuration_matches(raw, old['config_digest'])):
                raise RuntimeError('retired native owner is present or changed')
            return {'not_accepted':True,'effect_digest':digest(effect),'member':node,'boot_id':boot,
                    'retired_parent':owner,'old_container':prior,'unchanged_binding':current}
        if owner not in (None, token, data.get('parent')):
            raise RuntimeError('native host journal has a foreign owner')
        prior = state['effects'].get(effect['id'])
        if prior and prior['digest'] != digest(effect): raise RuntimeError('native effect identity reused')
        if prior and prior['state']=='Rejected' and action not in ('expired-guard-recovery','permission-recovery'):
            if prior['outcome'].get('reason')=='startup guard refused before acceptance':return prior['outcome']
            raise RuntimeError('native effect was rejected; no second send')
        if prior and prior['state'] == 'Verified': return prior['outcome']
        if action=='expired-guard-recovery':
            refusal=data['refusal']
            if (owner!=token or payload['action']!='start'
                or data.get('old_source')!='ff69b8f1916e83e88354e3bbba920db78cd1444e413ad0df36e5a2bb18b2d752'
                or refusal.get('source')!=data['old_source'] or refusal.get('reason')!='native startup guard is absent'
                or not prior or prior['state'] not in ('MayHaveApplied','Rejected')):
                raise RuntimeError('not the exact legacy startup guard refusal')
            proof=dict(refusal,effect_digest=digest(effect),member=node,boot_id=boot,
                       unchanged_binding=plan['binding'],rejected=True)
            if prior['state']=='Rejected':
                if prior['outcome']!=proof:raise RuntimeError('native rejection proof differs')
                return proof
            guard=state['effects'].get(refusal['guard_id'],{})
            gp=guard.get('effect',{}).get('payload',{})
            gb=refusal['guard_birth']
            if (guard.get('state')!='Verified' or guard.get('effect',{}).get('token')!=token
                or gp.get('action')!='guard-start' or gp.get('plan',{}).get('binding')!=plan['binding']
                or gp['plan'].get('guard',{}).get('phase')!='startup'
                or any(gp.get(k)!=payload.get(k) for k in ('adapter','target','member','resource','boot_id'))
                or guard.get('outcome',{}).get('guard_birth')!=gb or gb.get('boot_id')!=boot
                or birth(gb['identity']) is not None):
                raise RuntimeError('native startup guard is live, changed or unknown')
            receipt=Path(str(journal)+'.'+digest(refusal['guard_id'])+'.guard.json')
            if (not receipt.exists() or json.loads(receipt.read_text())!=
                {'effect_digest':guard['digest'],'token':token,'birth':gb}):
                raise RuntimeError('native startup guard receipt differs')
            census()
            current,raw=inspect(plan['binding']['id'])
            stopped={'StartedAt':'2026-11-12T19:08:53.64699539Z','FinishedAt':'2026-11-12T20:41:16.092404181Z'}
            if (current!=plan['binding'] or current['process_birth'] is not None
                or refusal.get('stopped_state')!=stopped
                or any(raw['State'].get(k)!=v for k,v in stopped.items())
                or raw['State'].get('Restarting') is not False or raw['State'].get('Pid')!=0):
                raise RuntimeError('native refused start binding changed')
            # The old source's exact reply attests its pre-Docker branch. The
            # lock fences old RPCs; this tombstone prevents every later replay.
            prior.update(state='Rejected',outcome=proof)
            guard['retired_by']=effect['id']
            durable(journal,state)
            return proof
        if action=='permission-recovery':
            if prior and prior['state']=='Rejected':return prior['outcome']
            if (os.getuid()!=0 or journal.stat().st_uid!=1000 or not prior or prior['state']!='MayHaveApplied'
                or payload['action']!='stop'
                or data.get('old_source')!='68c3fac2ab7811151fd8c42b414b2fc79ef52ad8ba73c088f527342251c41ca1'):
                raise RuntimeError('not the exact unprivileged native stop failure')
            current,_=inspect(plan['binding']['id'])
            if current!=plan['binding'] or (Path('/proc')/current['process_birth']['identity']).stat().st_uid!=0:
                raise RuntimeError('permission failure process incarnation differs')
            census()
            proof={'reason':'unprivileged-pidfd-permission-denied','effect_digest':digest(effect),'old_source':data['old_source'],
                   'journal_uid':1000,'target_uid':0,'original_executor_capabilities':0,'unchanged_binding':current}
            prior.update(state='Rejected',outcome=proof);durable(journal,state)
            return proof
        census()
        kind = payload['action']
        if kind not in ('create','start','stop','guard-start'):raise RuntimeError('unsupported native effect')
        target = manifests[payload['target']]
        if kind=='create' and plan!={'argv':target['nodes'][node]['create']}:raise RuntimeError('native create plan changed')
        def observed():
            if kind == 'create':
                result = run(['docker','ps','-aq','--no-trunc','--filter','label=local.profile.effect='+effect['id']]).split()
                if not result: return None
                if len(result) != 1: raise RuntimeError('native create identity is ambiguous')
                current, raw = inspect(result[0])
                if current['image_id'] != target['nodes'][node]['image']: raise RuntimeError('native created image differs')
                return {'container':current}
            current, raw = inspect(plan['binding']['id'])
            if any(current[k] != plan['binding'][k] for k in ('id','image_id','created')):
                raise RuntimeError('native container object differs')
            if kind == 'stop':
                if raw['State']['StartedAt'] != plan['binding']['started']:
                    raise RuntimeError('native stop incarnation changed')
                if not current['process_birth']: return {'container':current,'terminated_birth':plan['binding']['process_birth']}
                if current != plan['binding']: raise RuntimeError('native process birth changed')
            if kind == 'start' and current['process_birth']: return {'container':current}
            if kind == 'guard-start':
                receipt = Path(str(journal)+'.'+digest(effect['id'])+'.guard.json')
                if receipt.exists():
                    found = json.loads(receipt.read_text())
                    if found['effect_digest'] != digest(effect) or birth(found['birth']['identity']) != found['birth']:
                        raise RuntimeError('native memory guard birth changed')
                    return {'container':plan['binding'],'guard_birth':found['birth'],'policy':plan['guard']}
            return None
        found = observed()
        if prior and found is None: raise RuntimeError('native effect remains uncertain; no second send')
        if not prior:
            if found is not None: raise RuntimeError('native physical outcome has no prior host acceptance')
            if kind == 'stop':
                idle(target)
            if kind in ('create','start'):
                if census()['gpu']: raise RuntimeError('GPU must be empty before native member start')
                if census()['memory_gib'] < 100: raise RuntimeError('native startup needs 100 GiB available')
            if kind=='start':
                guards = [r for r in state['effects'].values() if r.get('state')=='Verified' and r.get('outcome') and not r.get('retired_by')
                          and r['effect']['token']==token and r['effect']['payload']['action']=='guard-start'
                          and r['effect']['payload']['plan']['binding']==plan['binding']
                          and r['effect']['payload']['plan']['guard']['phase']=='startup']
                if len(guards)!=1 or birth(guards[0]['outcome']['guard_birth']['identity'])!=guards[0]['outcome']['guard_birth']:
                    if (len(guards)==1 and birth(guards[0]['outcome']['guard_birth']['identity']) is None
                        and target.get('backend')=='tensorfold-native'):
                        guard=guards[0]
                        proof={'source':data['old_source'],'reason':'startup guard refused before acceptance',
                               'guard_id':guard['effect']['id'],'guard_birth':guard['outcome']['guard_birth'],
                               'effect_digest':digest(effect),'member':node,'boot_id':boot,
                               'unchanged_binding':plan['binding'],'rejected':True}
                        state['owner']=token
                        state['effects'][effect['id']]={'digest':digest(effect),'effect':effect,'state':'Rejected','outcome':proof}
                        guard['retired_by']=effect['id']
                        durable(journal,state)
                        return proof
                    raise RuntimeError('native startup guard is absent')
            state['owner'] = token
            state['effects'][effect['id']] = {'digest':digest(effect),'effect':effect,'state':'MayHaveApplied','outcome':None}
            durable(journal, state)
            if kind == 'create':
                argv = list(target['nodes'][node]['create'])
                index = next(i for i,a in enumerate(argv) if a.startswith('sha256:'))
                argv[index:index] = ['--label','local.profile.effect='+effect['id']]
                run(argv)
            elif kind == 'start':
                run(['docker','start',plan['binding']['id']])
            elif kind == 'stop':
                b = plan['binding']['process_birth']
                fd = os.pidfd_open(int(b['identity']))
                try:
                    if birth(b['identity']) != b: raise RuntimeError('native stop birth changed')
                    if not (payload['target']=='deepseek-1017-mia' and node!=target['head']):
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                    import select
                    p = select.poll(); p.register(fd, select.POLLIN)
                    if not p.poll(60000): raise RuntimeError('native termination pending; no escalation')
                finally: os.close(fd)
            elif kind == 'guard-start':
                grant = Path(str(journal)+'.'+digest(effect['id'])+'.grant.json')
                durable(grant, {'effect':effect,'journal':str(journal)})
                subprocess.Popen([os.sys.executable,'-c',data['guard_program'],str(grant)],start_new_session=True,
                                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                receipt = Path(str(journal)+'.'+digest(effect['id'])+'.guard.json')
                until=time.monotonic()+3
                while not receipt.exists() and time.monotonic()<until: time.sleep(.01)
            else: raise RuntimeError('unsupported native member action')
            found = observed()
        if found is None: raise RuntimeError('native physical outcome remains uncertain')
        state['effects'][effect['id']].update(state='Verified',outcome=found)
        durable(journal,state)
        return found
