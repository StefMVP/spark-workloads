"""Finite swaps between observed installed native runtimes, under Controller leases."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from clustercontrol import Birth, EffectKind, EffectState, Refused, digest, require_owner, resources, token_for

NODES = ('spark1', 'spark2', 'spark3', 'spark4')
UNPRIVILEGED_SOURCE = '68c3fac2ab7811151fd8c42b414b2fc79ef52ad8ba73c088f527342251c41ca1'
EMPTY_HEALTH_SOURCE = '0195c9bb3fe4538781bfbb5842340792ba27e3317068dca6cb5bb9074e44cb75'
EXPIRED_GUARD_SOURCE = 'ff69b8f1916e83e88354e3bbba920db78cd1444e413ad0df36e5a2bb18b2d752'
FILES = {'mia':'deepseek-1017-mia', 'owen1107':'deepseek-1017-owen-1107', 'glm1104':'glm-flash-aligned-14'}
DAILY_DEFAULTS = {'thinking':{'type':'enabled'},'reasoning_effort':'medium','temperature':0,'top_p':1}


def ready_handoff(original,spec,new_source):
    """Transfer a finite installed runtime only from its exact fresh READY birth."""
    old=copy.deepcopy(original['spec']);new=copy.deepcopy(spec)
    if (old.get('name') not in FILES.values() or new.get('name') not in FILES.values()
        or original.get('outcome')!='ready'
        or any(e['state'] not in (EffectState.VERIFIED,EffectState.REJECTED) for e in original['effects'])):
        raise Refused('parent source identity changed; retain old ownership for reconciliation')
    current=original.get('current_observation',{})
    ev=current.get('observations',{})
    manifests=ev.get('manifests',{})
    prior=copy.deepcopy(old.get('installed_native',{}));observed=copy.deepcopy(manifests.get(old['name'],{}))
    prior.pop('client_defaults',None);observed.pop('client_defaults',None)
    if (not prior or prior!=observed or new.get('installed_native')!=manifests.get(new['name'])
        or current.get('state')!='verified' or ev.get('source_identity')!=new_source or ev.get('endpoint_ready') is not True):
        raise Refused('native handoff requires fresh exact READY evidence')
    candidate=copy.deepcopy(original);candidate['observations']=ev
    hosts=sealed(candidate)
    for n,h in hosts.items():
        c=h['containers'].get(old['name'])
        if not c or not c.get('process_birth') or c!=original['observations']['native_hosts'][n]['containers'].get(old['name']):
            raise Refused('native handoff engine birth changed')


def settings_plan(manifest):
    plan={'endpoint':manifest.get('endpoint','http://127.0.0.1:'+str(manifest['client_port'])+'/v1'),'model':manifest['model']}
    if manifest.get('client_defaults'):plan['body']=manifest['client_defaults']
    return plan


def load_spec(root, name, key):
    if FILES.get(key) != name:
        raise Refused('unknown installed native runtime')
    manifest = json.loads((root / 'profiles/native-installed' / (key+'.json')).read_text())
    members = ['spark1','spark4'] if key == 'glm1104' else list(NODES)
    if key=='glm1104' and (manifest.get('backend')!='tensorfold-native' or manifest.get('endpoint')!='http://'+next(n['lan_ip'] for n in __import__('sparkinventory').load()['nodes'] if n['alias']=='spark1')+':8020/v1' or manifest.get('model')!='GLM-5.3-Flash' or manifest.get('start_order')!=['spark4','spark1']):
        raise Refused('invalid aligned GLM endpoint or native roles')
    if manifest['schema'] != 1 or manifest['members'] != members or set(manifest['nodes']) != set(members) or manifest['name'] != name:
        raise Refused('invalid installed native manifest')
    return {'name':name, 'layout':'14' if key == 'glm1104' else '4', 'groups':{'s14' if key == 'glm1104' else 's1234':members}, 'default':None,
            'instances':[], 'installed_native':manifest}


def handles(cluster, controller, args):
    if args.action == 'apply':
        from profilespec import load
        return bool(load(cluster.ROOT, args.name).get('installed_native'))
    if args.action in ('resume','rollback','cancel'):
        from profilecli import reference
        return bool(controller.view(reference(controller,args.receipt))['spec'].get('installed_native'))
    return False


def validate_hosts(hosts, members=NODES):
    if set(hosts) != set(members): raise Refused('native physical placement is incomplete')
    if len({h['resource_id'] for h in hosts.values()}) != len(members): raise Refused('native GPU bindings are ambiguous')
    for h in hosts.values():
        if h.get('studio_absent') is not True or not h.get('boot_id'):
            raise Refused('native host ownership is unknown')
        if not 0 <= time.time()-h.get('observed_at',0) < 30:
            raise Refused('native host evidence is stale')


def reconcile(state, hosts):
    validate_hosts(hosts)
    # Only legacy process records with an explicit node/PID can be retired.
    for jid, job in state['jobs'].items():
        if job.get('status') in ('settled','done','failed','cancelled'): continue
        if job.get('authority') != 'unverified' or not job.get('nodes'):
            raise Refused('active or unknown controller job blocks native reconciliation')
        for member, value in job['nodes'].items():
            if member not in hosts or not isinstance(value,str) or not value.startswith('pid:'):
                raise Refused('legacy job physical identity is unknown')
            found = hosts[member].get('legacy_absent',{}).get(jid)
            if not found or found != {'pid':value[4:],'absent':True,'boot_id':hosts[member]['boot_id']}:
                raise Refused('legacy job has no physical absence proof')
    for record in state['imports'].values():
        claims = record.get('claims',{})
        if not claims: continue
        old = record.get('state',{})
        if not isinstance(old,dict) or old.get('profile') != 'studio-three-puffer-gx10':
            raise Refused('unknown imported owner blocks native reconciliation')
        for claim in claims.values():
            if not claim.get('members') or not set(claim['members']) <= set(hosts):
                raise Refused('legacy claim placement is unknown')
        # All GPUs must be assigned to the exact known replacement, or empty.
        for h in hosts.values():
            original = h['containers'].get('deepseek-1017-owen-1107')
            if any(not original or gpu['container'] != original['id'] for gpu in h['gpu']):
                raise Refused('unknown replacement cannot reconcile legacy ownership')
        record['retired_claims'] = copy.deepcopy(claims)
        record['reconciliation'] = {'kind':'physical-absence', 'hosts':copy.deepcopy(hosts)}
        record['claims'] = {}
    for jid, job in state['jobs'].items():
        if job.get('status') not in ('settled','done','failed','cancelled'):
            job['prior_status'] = job['status']; job['status'] = 'settled'
            job['reconciliation'] = {n:hosts[n]['legacy_absent'][jid] for n in job['nodes']}
    return {'reconciled':True}


def origin(evidence):
    hosts=evidence.get('native_hosts',{});validate_hosts(hosts,tuple(hosts))
    active=[name for name,m in evidence['manifests'].items()
            if m['head'] in hosts and hosts[m['head']]['containers'].get(name,{}).get('process_birth')]
    if len(active)>1:raise Refused('native source runtime is ambiguous')
    return active[0] if active else None


def sealed(op):
    evidence = op.get('observations',{})
    hosts = evidence.get('native_hosts',{})
    validate_hosts(hosts,op['spec']['installed_native']['members'])
    if evidence.get('client_idle') is not True or evidence.get('native_idle') is not True:
        raise Refused('native client or engine work is busy or unknown')
    claim = evidence.get('claim',{})
    if not claim.get('agent') or claim.get('owner') != 'pane:'+os.environ.get('HC_TERMINAL_ID','') or claim.get('expires',0) <= time.time():
        raise Refused('native atomic backend claim is missing or foreign')
    if {h['resource_id'] for h in hosts.values()} != set(op['resources']):
        raise Refused('native observations differ from owned GPUs')
    for n,h in hosts.items():
        if h['resource_id'] != op['spec']['installed_native']['nodes'][n]['resource']:
            raise Refused('native manifest physical placement changed')
    return hosts


def permission(op, payload):
    if not op['spec'].get('installed_native'): raise Refused('native effect requires an installed runtime spec')
    hosts = sealed(op)
    if set(payload) != {'adapter','target','member','resource','boot_id','action','plan'} or payload['adapter'] != 'installed-native-v1':
        raise Refused('invalid installed native effect')
    n = payload['member']; h = hosts.get(n)
    if not h or payload['resource'] != h['resource_id'] or payload['boot_id'] != h['boot_id']:
        raise Refused('native effect host identity changed')
    manifests = op['observations']['manifests']
    if payload['target'] not in manifests: raise Refused('unknown native runtime effect target')
    if op.get('intent')=='retire' and payload['action'] not in ('stop','offline-settings'):
        raise Refused('native retirement permits only offline settings and exact stops')
    if payload['action'] not in ('stop','create','start','guard-start','settings','offline-settings'):
        raise Refused('unsupported installed native action')
    target = manifests[payload['target']]
    plan = payload['plan']
    if payload['action']=='offline-settings':
        if (op.get('intent')!='retire' or target.get('backend')!='tensorfold-native'
            or payload['target']!=op['spec']['name'] or plan!={'model':'sparks-offline/offline'}):
            raise Refused('offline settings require owned GLM retirement')
        return
    if payload['action'] == 'settings':
        if payload['target'] != op['spec']['name'] or plan != settings_plan(target):
            raise Refused('native settings differ from requested endpoint')
        return
    if payload['action'] == 'create':
        if payload['target'] != op['spec']['name'] or 'create' not in target['nodes'][n] or plan != {'argv':target['nodes'][n]['create']}:
            raise Refused('native create differs from finite source plan')
    else:
        b = plan['binding']
        current = h['containers'].get(payload['target'])
        if current != b: raise Refused('native effect lacks exact observed container incarnation')
        if payload['action'] == 'stop':
            Birth.parse(b['process_birth'])
        elif payload['target'] != op['spec']['name']:
            raise Refused('native start differs from requested target')
        if payload['action'] == 'guard-start':
            from profile_recipes.spark_ds41f_1030 import guard_decision
            guard_decision(plan,{'available_kib':10*1024**2,'container_id':b['id'],
                                 'container_started':b['started'],'process_birth':b['process_birth']})
    if payload['action'] in ('create','start') and h['gpu']:
        raise Refused('foreign or prior GPU work must retire before native start')


def release_permission(op):
    if op['spec']['installed_native'].get('backend')!='tensorfold-native':return
    hosts=sealed(op)
    if op.get('intent')!='retire' or not any(e['state']==EffectState.VERIFIED and e['payload'].get('action')=='offline-settings' for e in op['effects']):
        raise Refused('GLM release requires owned retirement and verified offline client settings')
    for n in op['spec']['installed_native']['members']:
        c=hosts[n]['containers'].get(op['spec']['name'])
        if not c or 'process_birth' not in c or c['process_birth'] is not None:
            raise Refused('GLM release requires every selected process absent')


def finish_permission(op):
    sealed(op)
    checks = op['observations'].get('requirements',{})
    if checks.get('services',{}).get('ready') is not True or checks.get('settings',{}).get('resolved') is not True:
        raise Refused('native readiness and client routing are not verified')
    if checks != {k:op['readiness'][k] for k in checks}:
        raise Refused('native readiness evidence changed')


class Transport:
    def __init__(self, controller, runtime, manifests):
        self.control, self.runtime, self.manifests = controller, runtime, manifests

    def call(self, member, action, *, op=None, legacy_pids=None, target=None, effect=None, retired=None, refusal=None):
        owned = {}
        operations = list(self.control.store.read()['operations'].values())
        for saved in operations:
            for e in saved['effects']:
                p = e.get('payload',{})
                if p.get('adapter') == 'installed-native-v1' and p.get('target') in self.manifests and p['member'] == member and e.get('outcome'):
                    physical = e['outcome'].get('physical',{})
                    if 'container' in physical and p['action'] in ('create','start'):
                        owned[p['target']] = physical['container']
        data = {'member':member,'action':action,'manifests':self.manifests,'owned':owned,'legacy_pids':legacy_pids or {},'target':target}
        if retired is not None:data['retired']=retired
        if refusal is not None:data['refusal']=refusal
        if effect:
            from nativefamilyio import guard_program
            data.update(effect=effect, parent=op.get('parent_authority'), guard_program=guard_program(), old_source=op['source_identity'])
        code = Path(__file__).with_name('nativeinstalledio.py').read_text()+'''
try:
    value={'ok':True,'value':rpc(D)}
except (RuntimeError,OSError,ValueError,KeyError) as error:
    value={'ok':False,'reason':str(error)}
print(json.dumps(value))
'''
        script='import json\nD=json.loads('+repr(json.dumps(data))+')\n'+code
        result = json.loads(self.runtime.remote(member,'sudo -n env HOME="$HOME" python3 -',input=script,timeout=90))
        if not isinstance(result,dict) or result.get('ok') is not True:
            raise Refused(member+': '+(result.get('reason','native outcome unknown') if isinstance(result,dict) else 'native outcome unknown'))
        return result['value']

    def observe(self, state, op=None):
        def node(n):
            pids = {jid:j['nodes'][n][4:] for jid,j in state['jobs'].items()
                    if j.get('authority')=='unverified' and j.get('status') not in ('settled','done','failed','cancelled')
                    and n in j.get('nodes',{}) and j['nodes'][n].startswith('pid:')}
            return self.call(n,'observe',op=op,legacy_pids=pids)
        members=sorted({n for m in self.manifests.values() for n in m['members']})
        return self.runtime.c.pmap(node,members)


def claim():
    try:
        result = subprocess.run(['offload','leases','--json'],text=True,capture_output=True,timeout=20)
    except OSError as error:raise Refused('native backend claim tool is unavailable') from error
    if result.returncode: raise Refused('native backend claim cannot be observed')
    rows = json.loads(result.stdout)
    backend = next((r for r in rows if r['name']=='tesseract-spark'),{})
    claims = backend.get('claims',[])
    if len(claims)!=1 or claims[0]['owner'] != 'pane:'+os.environ.get('HC_TERMINAL_ID',''):
        raise Refused('native backend claim is foreign or missing')
    return claims[0]


def guard_plan(binding, phase):
    return {'family':'spark-ds41f','action':'guard-start','binding':binding,
            'guard':{'phase':phase,'threshold_kib':(5 if phase=='startup' else 3)*1024**2,
                     'interval_seconds':.25 if phase=='startup' else 2,'startup_wait_seconds':120 if phase=='startup' else 0,
                     'birth':binding['process_birth'],'container_id':binding['id'],'container_started':binding['started'],
                     'on_unknown_memory':'kill-exact-birth','signal':'SIGKILL'}}


class Driver:
    def __init__(self, controller, runtime, manifests):
        self.control,self.runtime,self.manifests=controller,runtime,manifests
        self.transport=Transport(controller,runtime,manifests)
        self.claimed=None
        self.claimed_at=0

    def evidence(self, op):
        import opencodeprofile
        opencodeprofile.idle()
        if self.claimed is None or time.monotonic()-self.claimed_at>30:
            self.claimed=claim();self.claimed_at=time.monotonic()
        claimed=self.claimed
        hosts=self.transport.observe(self.control.store.read(),op)
        for name,m in self.manifests.items():
            c=hosts[m['head']]['containers'].get(name)
            if (name != op['spec']['name'] or op.get('intent')=='retire') and c and c['process_birth']: self.transport.call(m['head'],'idle',op=op,target=name)
        bindings={n:{'resource_id':h['resource_id'],'instance_id':'native-host:'+n,
                     'birth':{'boot_id':h['boot_id'],'service_id':'native-host:'+h['boot_id']}} for n,h in hosts.items()}
        return {'native_hosts':hosts,'bindings':bindings,'native_idle':True,'client_idle':True,'claim':claimed,'manifests':self.manifests}

    def issue(self,op,ev,n,action,target,binding=None,phase=None):
        m=self.manifests[target]
        plan={'model':'sparks-offline/offline'} if action=='offline-settings' else settings_plan(m) if action=='settings' else {'argv':m['nodes'][n]['create']} if action=='create' else guard_plan(binding,phase) if action=='guard-start' else {'binding':binding}
        payload={'adapter':'installed-native-v1','target':target,'member':n,'resource':ev['native_hosts'][n]['resource_id'],
                 'boot_id':ev['native_hosts'][n]['boot_id'],'action':action,'plan':plan}
        effect=self.control.issue(op['id'],EffectKind.NATIVE_RUNTIME,payload,[e['id'] for e in op['effects'] if e['state']==EffectState.VERIFIED])
        return self.execute(op,effect)

    def execute(self,op,effect):
        if effect['payload']['action']=='offline-settings':
            self.offline(); found={'resolved':True}
        elif effect['payload']['action']=='settings':
            self.settings(self.manifests[effect['payload']['target']]); found={'resolved':True}
        else:
            found=self.transport.call(effect['payload']['member'],'execute',op=op,effect=effect)
        if found.get('rejected') and found.get('reason')=='startup guard refused before acceptance':
            return self.control.repair_expired_native_guard(op['id'],op['source_identity'],found)
        return self.control.outcome(op['id'],effect['id'],{'effect_id':effect['id'],'token':effect['token'],'verified':True,'physical':found})

    def step(self,identifier):
        from profilecli import source_identity
        op=self.control.view(identifier)
        if op.get('outcome') in ('released','cancelled'):raise Refused('retired native operation cannot restart a runtime')
        if op['source_identity'] != source_identity(self.runtime.root): raise Refused('native source changed; retain receipts for recovery')
        uncertain=next((e for e in op['effects'] if e['state']==EffectState.UNCERTAIN),None)
        if uncertain:
            result=self.execute(op,uncertain);self.control.block(identifier,[]);return result
        ev=self.evidence(op)
        op=self.control.update_evidence(identifier,ev,[])
        target=op['spec']['name'];manifest=self.manifests[target]
        if op.get('intent')=='retire':
            for n in [manifest['head'],*[n for n in manifest['members'] if n!=manifest['head']]]:
                c=ev['native_hosts'][n]['containers'].get(target)
                if c and c['process_birth']:return self.issue(op,ev,n,'stop',target,c)
            return self.control.release(op['id'])
        # Retire the exact old head first. Worker shutdown follows head drain.
        for name,m in self.manifests.items():
            if name==target:continue
            for n in (m['head'],*[n for n in NODES if n!=m['head']]):
                c=ev['native_hosts'][n]['containers'].get(name)
                if c and c['process_birth']:return self.issue(op,ev,n,'stop',name,c)
        for n in manifest['start_order']:
            c=ev['native_hosts'][n]['containers'].get(target)
            if not c:return self.issue(op,ev,n,'create',target)
            rows=[e for e in op['effects'] if e['state']==EffectState.VERIFIED and e['payload'].get('member')==n]
            if not c['process_birth']:
                retired_guards={r['guard_id'] for r in op.get('source_repairs',[]) if r.get('kind')=='expired-startup-guard'}
                startup=next((e for e in rows if e['id'] not in retired_guards and e['payload']['action']=='guard-start' and e['payload']['plan']['guard']['phase']=='startup'),None)
                if startup is None:return self.issue(op,ev,n,'guard-start',target,c,'startup')
                return self.issue(op,ev,n,'start',target,c)
            steady=next((e for e in rows if e['payload']['action']=='guard-start' and e['payload']['plan']['guard']['phase']=='steady' and e['payload']['plan']['binding']==c),None)
            if steady is None:return self.issue(op,ev,n,'guard-start',target,c,'steady')
        return self.ready(op,ev,manifest)

    def ready(self,op,ev,manifest):
        # Native health/model probes precede any client selection.
        head=manifest['head'];port=manifest['port']
        health=self.runtime.api(head,port,'/health');models=self.runtime.api(head,port,'/v1/models')
        if health is None or not isinstance(models,dict) or manifest['model'] not in [m['id'] for m in models.get('data',[])]:
            return self.control.block(op['id'],[{'resource':'*','reason':'native model load pending; resume this operation'}])
        if not any(e['state']==EffectState.VERIFIED and e['payload'].get('action')=='settings' for e in op['effects']):
            return self.issue(op,ev,head,'settings',manifest['name'])
        self.settings(manifest)
        ev=self.evidence(op)
        checks={'artifacts':{'state':'verified','images':{n:manifest['nodes'][n]['image'] for n in manifest['members']}},
                'capabilities':{'state':'verified','adapter':'installed-native-v1'},
                'services':{'state':'verified','ready':True,'health':health,'models':models},
                'settings':{'state':'verified','resolved':True,'endpoint':settings_plan(manifest)['endpoint']}}
        ev['requirements']=checks
        op=self.control.update_evidence(op['id'],ev,[])
        evidence=dict(checks,spec_digest=op['spec_digest'],bindings=ev['bindings'])
        if not self.control.view(op['id'])['allocation_current']:self.control.publish(op['id'],evidence)
        return self.control.finish(op['id'])

    def settings(self,manifest):
        import opencodeprofile
        # Existing native endpoint forwards are kept. No gateway/router is started.
        url=settings_plan(manifest)['endpoint']
        import urllib.request
        try:
            with urllib.request.urlopen(url+'/models',timeout=5) as reply:models=json.load(reply)
        except OSError:
            if manifest['name']!='deepseek-1017-mia':raise Refused('existing native Owen tunnel is unavailable')
            unit='spark-native-mia-tunnel'
            result=subprocess.run(['systemd-run','--user','--unit='+unit,'--collect','--property=Restart=on-failure',
                '/usr/bin/ssh','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30',
                '-L','127.0.0.1:'+str(manifest['client_port'])+':127.0.0.1:'+str(manifest['port']),manifest['head']],capture_output=True,text=True,timeout=15)
            if result.returncode:raise Refused('native MIA tunnel could not start')
            result=subprocess.run(['agent-tool','wait','--timeout','10','--interval','.2','--','curl','--fail','--silent',url+'/models'],capture_output=True,text=True,timeout=15)
            if result.returncode:raise Refused('native MIA tunnel is not ready')
            models=json.loads(result.stdout)
        if manifest['model'] not in [m['id'] for m in models.get('data',[])]: raise Refused('native client tunnel points to another model')
        path=self.runtime.config
        data=json.loads(path.read_text())
        providers=data.get('providers',{})
        selected=None
        for key,p in providers.items():
            if p.get('settings',{}).get('baseURL')==url and manifest['model'] in p.get('models',{}):selected=key+'/'+manifest['model']
        if selected is None:
            if manifest['name']!='deepseek-1017-mia':raise Refused('native client provider for this endpoint is not configured')
            key='sparks-native-mia'
            providers[key]={'package':'@opencode/ai/providers/openai-compatible','settings':{'baseURL':url},'models':{manifest['model']:{'name':manifest['model'],'limit':{'context':139264,'output':4096},'capabilities':{'tools':True,'input':['text'],'output':['text']}}}}
            data['providers']=providers;selected=key+'/'+manifest['model']
        before=copy.deepcopy(data)
        provider=providers[selected.split('/',1)[0]]
        provider.setdefault('package','@opencode/ai/providers/openai-compatible')
        if manifest.get('client_defaults'):
            provider['body']=dict(provider.get('body',{}),**manifest['client_defaults'])
        data['model']=selected
        if data!=before:
            opencodeprofile.idle();data['model']=selected
            from clusterprofile import atomic
            atomic(path.resolve(),data)
        opencodeprofile.refresh(path,data)


def run_command(cluster, controller, args):
    import clusterprofile
    from profilecli import export, reference, source_identity, public_json
    from profilespec import load
    runtime=clusterprofile.Runtime(cluster)
    manifests={v:load_spec(cluster.ROOT,v,k)['installed_native'] for k,v in FILES.items()}
    requested=load(cluster.ROOT,args.name) if args.action=='apply' else controller.view(reference(controller,args.receipt))['spec']
    if requested.get('installed_native',{}).get('backend')=='tensorfold-native':
        from tensorfoldnative import Driver as TensorFoldDriver
        manifests={requested['name']:requested['installed_native']}
        driver=TensorFoldDriver(controller,runtime,manifests)
    else:
        manifests={k:v for k,v in manifests.items() if v.get('backend')!='tensorfold-native'}
        driver=Driver(controller,runtime,manifests)
    if args.action=='apply':
        spec=load(cluster.ROOT,args.name)
        if not args.yes:
            print(public_json({'target':args.name,'spec_digest':digest(spec),'adapter':'installed-native-v1','effect':'preview'},indent=2));return
        claim()
        hosts=driver.transport.observe(controller.store.read())
        if getattr(args,'no_restart',False):
            raise Refused('installed native apply needs lifecycle checks; --no-restart refuses before effects')
        if requested.get('installed_native',{}).get('backend')!='tensorfold-native':controller.reconcile_native(hosts)
        state=controller.store.read();parent=state['layout'].get('operation_id')
        prior=state['operations'].get(parent)
        if prior and prior.get('outcome') in ('released','cancelled'):parent,prior=None,None
        if prior and requested.get('installed_native',{}).get('backend')=='tensorfold-native' and prior['spec'].get('installed_native',{}).get('backend')!='tensorfold-native':
            audit=Transport(controller,runtime,{v:load_spec(cluster.ROOT,v,k)['installed_native'] for k,v in FILES.items()}).observe(state)
            controller.retire_absent_installed(parent,audit)
            parent,prior=None,None
        if prior and not prior['spec'].get('installed_native'):raise Refused('current controller owner is not an installed native runtime')
        if prior and prior.get('outcome')=='ready' and prior['source_identity']!=source_identity(cluster.ROOT):
            ev=driver.evidence(prior)
            m=manifests[prior['spec']['name']]
            driver.transport.call(m['head'],'idle',op=prior,target=m['name'])
            models=runtime.api(m['head'],m['port'],'/v1/models')
            ev['endpoint_ready']=runtime.api(m['head'],m['port'],'/health') is not None and isinstance(models,dict) and m['model'] in [v['id'] for v in models.get('data',[])]
            ev['source_identity']=source_identity(cluster.ROOT)
            controller.record_health(prior['id'],ev,[])
        if prior and prior['spec_digest']==digest(spec) and prior.get('outcome')=='ready' and prior['source_identity']==source_identity(cluster.ROOT):
            ev=driver.evidence(prior)
            driver.settings(manifests[args.name])
            head=manifests[args.name]['head'];port=manifests[args.name]['port']
            models=runtime.api(head,port,'/v1/models')
            if not models or manifests[args.name]['model'] not in [m['id'] for m in models.get('data',[])]:
                raise Refused('completed native runtime is no longer ready')
            controller.record_health(prior['id'],ev,[])
            print(public_json(dict(controller.view(prior['id']),receipt=str(export(controller,prior['id']))),indent=2));return
        # A pending exact operation is resumed; never replace uncertain effects.
        pending=[o for o in state['operations'].values() if o['spec'].get('installed_native') and o.get('outcome') is None]
        if pending:
            if len(pending)!=1 or pending[0]['spec_digest']!=digest(spec):raise Refused('another native operation requires cancel or rollback')
            identifier=pending[0]['id']
        else:
            op=controller.request('activate',spec,sorted(h['resource_id'] for h in hosts.values()),getattr(args,'operation_key',None) or uuid.uuid4().hex,source_identity(cluster.ROOT),parent=parent if prior else None)
            identifier=op['id']
    else:
        identifier=reference(controller,args.receipt)
        if not args.yes:raise Refused('add --yes for native recovery')
        original=controller.view(identifier)
        if original.get('outcome') in ('released','cancelled'):raise Refused('retired native receipt cannot restart a runtime; apply a saved profile explicitly')
        if requested.get('installed_native',{}).get('backend')=='tensorfold-native' and args.action in ('cancel','rollback'):
            if args.action=='rollback':raise Refused('GLM rollback cannot restore a historical model; use explicit saved profile apply')
            if original.get('outcome')=='ready' and absent(driver.transport.observe(controller.store.read()),original):
                # Exact objects were stopped outside this tool: record observed absence, never a stop effect.
                driver.offline()
                op=controller.retire_absent_installed(identifier,driver.transport.observe(controller.store.read()))
                print(public_json(dict(op,receipt=str(export(controller,identifier))),indent=2));return
            controller.begin_installed_retire(identifier,source_identity(cluster.ROOT))
        if original['source_identity']==EMPTY_HEALTH_SOURCE and original.get('outcome') is None:
            ev=driver.evidence(original)
            manifest=manifests[original['spec']['name']]
            driver.transport.call(manifest['head'],'idle',op=original,target=manifest['name'])
            ev['health']=runtime.api(manifest['head'],manifest['port'],'/health')
            ev['models']=runtime.api(manifest['head'],manifest['port'],'/v1/models')
            controller.repair_native_readiness(identifier,source_identity(cluster.ROOT),ev)
            original=controller.view(identifier)
        if original['source_identity']==UNPRIVILEGED_SOURCE and original['source_identity']!=source_identity(cluster.ROOT):
            validate_permission_failure(original)
            effect=original['effects'][0]
            proof=driver.transport.call(effect['payload']['member'],'permission-recovery',op=original,effect=effect)
            controller.repair_native_permission(identifier,source_identity(cluster.ROOT),proof)
        if args.action in ('cancel','rollback') and requested.get('installed_native',{}).get('backend')!='tensorfold-native':
            original=controller.view(identifier)
            if not original['effects']:
                op=controller.cancel_native_unstarted(identifier)
                print(public_json(dict(op,receipt=str(export(controller,identifier))),indent=2));return
            if any(e['state']==EffectState.UNCERTAIN for e in original['effects']):
                driver.step(identifier);original=controller.view(identifier)
            target=original.get('native_origin')
            if target not in manifests:raise Refused('saved native original is unknown; retain the exact operation')
            spec=load(cluster.ROOT,target)
            op=controller.request('restore',spec,original['resources'],uuid.uuid4().hex,source_identity(cluster.ROOT),parent=identifier)
            controller.cancel(identifier)
            identifier=op['id']
    try:
        for _ in range(40):
            op=driver.step(identifier)
            if op.get('outcome') in ('ready','released') or op.get('blockers'):break
    except (RuntimeError,OSError,ValueError) as error:
        op=controller.block(identifier,[{'resource':'*','reason':str(error)}])
    path=export(controller,identifier)
    if op.get('blockers')==[{'resource':'*','reason':'native model load pending; resume this operation'}] and not os.environ.get('SPARK_NATIVE_WORKER'):
        start_worker(cluster,controller,op,path)
    print(public_json(dict(op,receipt=str(path)),indent=2))
    raise SystemExit(exit_status(op,bool(os.environ.get('SPARK_NATIVE_WORKER'))))


def absent(hosts,op):
    m=op['spec']['installed_native']
    return all(hosts.get(n,{}).get('containers',{}).get(m['name'],{}).get('process_birth',0) is None for n in m['members'])


def exit_status(op,worker=False):
    pending=op.get('blockers')==[{'resource':'*','reason':'native model load pending; resume this operation'}]
    return 0 if op.get('outcome') in ('ready','released') or (worker and not pending) else 3


def command(cluster, controller, args):
    import fcntl
    path=controller.store.path.with_suffix('.native.lock')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Refused('another native profile command is in progress')
        return run_command(cluster,controller,args)


def start_worker(cluster,controller,op,path):
    unit='spark-native-profile-'+op['id']
    result=subprocess.run(['systemd-run','--user','--unit='+unit,'--collect',
        '--property=Restart=on-failure','--property=RestartSec=5','--property=TimeoutStartSec=900',
        '--setenv=HC_TERMINAL_ID='+os.environ.get('HC_TERMINAL_ID',''),
        '--setenv=SPARK_STATE='+str(controller.store.path),'--setenv=SPARK_NATIVE_WORKER=1',
        sys.executable,str(cluster.ROOT/'tools/cluster'),'profile','resume',str(path),'--yes'],capture_output=True,text=True,timeout=15)
    if result.returncode:raise Refused('native worker could not start; resume saved operation explicitly')


def repair_readiness(op,new_source,evidence):
    """Recover only the observed empty-health wait, without replaying host effects."""
    name='deepseek-1017-owen-1107'
    actions=[e['payload'].get('action') for e in op['effects']]
    if (op['source_identity']!=EMPTY_HEALTH_SOURCE or op['spec']['name']!=name
        or op.get('outcome') is not None or len(actions)!=13
        or actions.count('start')!=4 or actions.count('guard-start')!=8 or actions.count('stop')!=1
        or any(e['state']!=EffectState.VERIFIED for e in op['effects'])
        or op.get('blockers') not in ([{'resource':'*','reason':'native model load pending; resume this operation'}],
                                     [{'resource':'*','reason':'native source changed; retain receipts for recovery'}])):
        raise Refused('native repair is not the source-pinned empty-health wait')
    candidate=copy.deepcopy(op);candidate['observations']=evidence
    hosts=sealed(candidate)
    if evidence.get('health') is None or op['spec']['installed_native']['model'] not in [m['id'] for m in (evidence.get('models') or {}).get('data',[])]:
        raise Refused('native readiness recovery requires real health and model evidence')
    for n,h in hosts.items():
        current=h['containers'].get(name)
        if not current or not current.get('process_birth') or current!=op['observations']['native_hosts'][n]['containers'].get(name):
            raise Refused('native readiness recovery container birth changed')
    op.setdefault('source_repairs',[]).append({'kind':'empty-health-readiness','old_source':op['source_identity'],
        'new_source':new_source,'proof':copy.deepcopy(evidence)})
    op['source_identity']=new_source;op['blockers']=[]
    return op


def validate_permission_failure(op):
    if (op['source_identity']!=UNPRIVILEGED_SOURCE or len(op['effects'])!=1
        or op['effects'][0]['state']!=EffectState.UNCERTAIN
        or op['effects'][0]['payload'].get('adapter')!='installed-native-v1'
        or op['effects'][0]['payload'].get('action')!='stop'
        or op.get('blockers')!=[{'resource':'*','reason':op['effects'][0]['payload']['member']+': [Errno 1] Operation not permitted'}]):
        raise Refused('native repair is not the source-pinned unsent permission failure')


def expired_guard_refusal(op):
    """The legacy reply is evidence only at this exact source and effect prefix."""
    if (op.get('source_identity')!=EXPIRED_GUARD_SOURCE or op.get('outcome') is not None
        or op.get('intent')!='activate' or op['spec'].get('name')!='glm-flash-aligned-14'
        or op['spec'].get('installed_native',{}).get('backend')!='tensorfold-native'
        or len(op['effects'])!=2):
        raise Refused('not the source-pinned expired startup guard refusal')
    guard,effect=op['effects'];p=effect['payload'];g=guard['payload']
    token=token_for(op).document()
    if (guard['state']!=EffectState.VERIFIED or effect['state']!=EffectState.UNCERTAIN
        or guard['token']!=token or effect['token']!=token
        or p.get('action')!='start' or g.get('action')!='guard-start'
        or any(p.get(k)!=g.get(k) for k in ('adapter','target','member','resource','boot_id'))
        or p.get('adapter')!='installed-native-v1' or p.get('target')!=op['spec']['name']
        or g['plan']!=guard_plan(p['plan']['binding'],'startup')
        or p['plan']['binding'].get('process_birth') is not None or p['plan']['binding'].get('started') is not None
        or effect['prerequisites']!=[guard['id']]
        or not guard.get('outcome',{}).get('verified')
        or guard['outcome'].get('token')!=token):
        raise Refused('expired guard lacks the exact pre-send refusal reply')
    refusal={'source':EXPIRED_GUARD_SOURCE,'reason':'native startup guard is absent','guard_id':guard['id'],
            'guard_birth':guard['outcome']['physical']['guard_birth'],
            'stopped_state':{'StartedAt':'2026-11-12T19:08:53.64699539Z','FinishedAt':'2026-11-12T20:41:16.092404181Z'}}
    if (op.get('blockers')!=[{'resource':'*','reason':p['member']+': native startup guard is absent'}]
        and op.get('startup_guard_refusal')!=dict(refusal,effect_digest=digest(effect))):
        raise Refused('expired guard lacks the exact pre-send refusal reply')
    return refusal


def repair_expired_guard(op,new_source,proof):
    if proof.get('reason')=='native startup guard is absent':
        refusal=expired_guard_refusal(op)
        guard,effect=op['effects']
    else:
        if (op['source_identity']!=new_source or op.get('outcome') is not None or op.get('intent')!='activate'
            or op['spec'].get('installed_native',{}).get('backend')!='tensorfold-native'):
            raise Refused('guard rejection source or operation differs')
        effect=op['effects'][-1]
        guard=next((e for e in op['effects'] if e['id']==proof.get('guard_id')),None)
        token=token_for(op).document()
        if (not guard or effect['state']!=EffectState.UNCERTAIN or guard['state']!=EffectState.VERIFIED
            or effect['token']!=token or guard['token']!=token or guard['id'] not in effect['prerequisites']
            or effect['payload'].get('action')!='start' or guard['payload'].get('action')!='guard-start'
            or any(effect['payload'].get(k)!=guard['payload'].get(k) for k in ('adapter','target','member','resource','boot_id'))
            or guard['payload']['plan']!=guard_plan(effect['payload']['plan']['binding'],'startup')):
            raise Refused('guard rejection effect prefix differs')
        refusal={'source':new_source,'reason':'startup guard refused before acceptance','guard_id':guard['id'],
                 'guard_birth':guard['outcome']['physical']['guard_birth']}
    p=effect['payload']
    expected=dict(refusal,effect_digest=digest(effect),member=p['member'],boot_id=p['boot_id'],
                  unchanged_binding=p['plan']['binding'],rejected=True)
    if proof!=expected:raise Refused('expired guard recovery proof differs')
    effect['state']=EffectState.REJECTED
    effect['outcome']={'effect_id':effect['id'],'token':effect['token'],'rejected':True,'physical':copy.deepcopy(proof)}
    op.setdefault('source_repairs',[]).append({'kind':'expired-startup-guard','guard_id':guard['id'],
        'old_source':op['source_identity'],'new_source':new_source,'proof':copy.deepcopy(proof)})
    op['source_identity']=new_source;op['blockers']=[]
    return op


def repair_permission(op,new_source,proof):
    validate_permission_failure(op)
    effect=op['effects'][0]
    if (proof.get('reason')!='unprivileged-pidfd-permission-denied' or proof.get('effect_digest')!=digest(effect)
        or proof.get('old_source')!=UNPRIVILEGED_SOURCE or proof.get('journal_uid')!=1000
        or proof.get('target_uid')!=0 or proof.get('original_executor_capabilities')!=0
        or proof.get('unchanged_binding')!=effect['payload']['plan']['binding']):
        raise Refused('native permission failure lacks exact no-send evidence')
    effect['state']=EffectState.REJECTED
    effect['outcome']={'effect_id':effect['id'],'token':effect['token'],'rejected':True,'physical':copy.deepcopy(proof)}
    op.setdefault('source_repairs',[]).append({'old_source':op['source_identity'],'new_source':new_source,'proof':copy.deepcopy(proof)})
    op['source_identity']=new_source;op['blockers']=[]
    return op
