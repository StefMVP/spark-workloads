"""Short-lived Docker member IO. Every command comes from a rederived leaf plan."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from clustercontrol import Birth,Refused,digest,durable
from nativecontrol import NativeExecutor,member_plan
from profile_recipes.spark_ds41f_1030 import guard_decision


def command(argv):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    if result.returncode:raise Refused('native command outcome unknown')
    return result.stdout

def process_birth(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    except (OSError,ValueError) as error:raise Refused('native process birth unavailable') from error
    if fields[0] in ('Z','X'):return None
    return {'kind':'process','identity':str(pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started':fields[19]}

def inspect(identifier):
    raw=json.loads(command(['docker','inspect',identifier]))
    if len(raw)!=1 or raw[0].get('Id')!=identifier:raise Refused('immutable container identity changed')
    item=raw[0];state=item['State']
    return {'id':item['Id'],'image_id':item['Image'],'created':item['Created'],
            'started':state['StartedAt'] if state['Running'] else None,
            'process_birth':process_birth(state['Pid']) if state['Running'] else None},item

class DockerMemberIO:
    def boot(self):return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def resource(self):
        uuids=command(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).strip().splitlines()
        if len(uuids)!=1 or not re.fullmatch(r'GPU-[A-Za-z0-9-]+',uuids[0]):raise Refused('host GPU binding unknown or ambiguous')
        return 'gpu:'+uuids[0]
    def preflight(self,plan,payload,effect,journal):
        if self.resource() not in plan['resources']:raise Refused('native host is outside full resource grant')
        if plan['action'] in ('create','start'):
            state=json.loads(Path(journal).read_text()) if Path(journal).exists() else {'effects':{}}
            owned={r['outcome']['physical']['container']['id'] for r in state['effects'].values()
                   if r.get('outcome') and r['effect']['token']==effect['token'] and r['effect']['payload']['action'] in ('create','start')}
            for pid in command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).split():
                cgroup=(Path('/proc')/pid/'cgroup').read_text()
                if not any(identifier in cgroup for identifier in owned):raise Refused('unknown GPU process blocks native member effect')
        if plan['action']=='create':
            image=json.loads(command(['docker','image','inspect',plan['binding']['image_id']]))
            if len(image)!=1 or image[0]['Id']!=plan['binding']['image_id']:raise Refused('immutable image changed')
            wanted=payload['parameters'].get('image_environment')
            observed=dict(row.split('=',1) for row in image[0]['Config'].get('Env',[]) if '=' in row)
            if wanted is not None and wanted!=observed:raise Refused('image environment differs from observed immutable image')
            overlay=plan['binding'].get('control_overlay')
            if overlay and payload['recipe'].get('engine')=='tensorfold-mia-tp4':
                from profile_recipes import mia_tp4
                mia_tp4.verify_sources(Path(self.recipe_context(payload)['package']))
                if not Path(overlay['config_path']).is_file():raise Refused('prepared Mia owner configuration absent')
            elif overlay:
                from profile_recipes.spark_ds41f_1030 import control_artifact
                artifact=control_artifact(self.recipe_context(payload) if hasattr(self,'recipe_context') else payload['recipe'])
                root=Path(artifact['path'])
                for relative,expected in artifact['identity']['files'].items():
                    import hashlib
                    if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:raise Refused('prepared native control source differs')
                if not Path(overlay['config_path']).is_file():raise Refused('prepared native control configuration absent')
        if plan['action']=='start':
            self.start_guard(plan,effect,journal)
    def start_guard(self,plan,effect,journal):
        from profile_recipes.spark_ds41f_1030 import guard_decision
        state=json.loads(Path(journal).read_text())
        matches=[row for row in state['effects'].values() if row.get('outcome')
                 and row['effect']['token']==effect['token'] and row['effect']['payload']['action']=='guard-start'
                 and row['effect']['payload']['plan']['binding']==plan['binding']
                 and row['effect']['payload']['plan']['guard']['phase']=='startup']
        if len(matches)!=1:raise Refused('exact startup guard registration unknown')
        row=matches[0];birth=row['outcome']['physical']['guard_birth']
        if process_birth(int(birth['identity']))!=birth:raise Refused('startup guard exited or physical birth changed')
        elapsed=time.monotonic()-int(birth['started'])/os.sysconf('SC_CLK_TCK')
        if not 0<=elapsed<120:raise Refused('startup guard registration expired')
        try:available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
        except (OSError,StopIteration,ValueError):available=None
        sample={'available_kib':available,'container_id':plan['binding']['id'],'container_started':None,'process_birth':None}
        if guard_decision(row['effect']['payload']['plan'],sample)['state']!='await-start-outcome':raise Refused('startup guard refuses current memory observation')
    def observe(self,plan,effect,journal):
        action=plan['action'];binding=plan['binding']
        if action=='create':
            args=['docker','ps','-aq','--no-trunc']
            for key,value in sorted(binding['labels'].items()):args+=['--filter','label='+key+'='+value]
            ids=command(args).split()
            if not ids:return {'state':'unsent'}
            if len(ids)!=1 or not re.fullmatch(r'[a-f0-9]{64}',ids[0]):raise Refused('created object mapping is ambiguous')
            current,raw=inspect(ids[0])
            if current['image_id']!=binding['image_id'] or not current['created'] or any(raw['Config']['Labels'].get(k)!=v for k,v in binding['labels'].items()):raise Refused('created object labels/image/birth differ')
            return {'state':'verified','container':current}
        current,raw=inspect(binding['id'])
        if any(current[k]!=binding[k] for k in ('id','image_id','created')):raise Refused('container object birth/image differs')
        if action=='start':
            if current['started'] and current['process_birth']:return {'state':'verified','container':current}
            if raw['State'].get('StartedAt') not in (None,'','0001-01-01T00:00:00Z'):raise Refused('previously started object cannot be restarted')
            return {'state':'unsent'}
        if action=='stop':
            physical=NativeExecutor().observe(member_plan('process_stop',plan['resources'][0],binding['process_birth'],binding['process_birth']['identity']))
            if physical['state']=='absent':return {'state':'verified','container':binding,'terminated_birth':binding['process_birth']}
            if current['started']!=binding['started'] or current['process_birth']!=binding['process_birth']:raise Refused('running container incarnation changed')
            return {'state':'unsent'}
        receipt=Path(str(journal)+'.'+digest(effect['id'])+'.guard.json')
        if action=='guard-start':
            if receipt.exists():
                found=json.loads(receipt.read_text())
                if found['effect_digest']!=digest(effect) or found['token']!=effect['token']:raise Refused('guard launch identity changed')
                if process_birth(int(found['birth']['identity']))!=found['birth']:raise Refused('guard physical birth expired')
                return {'state':'verified','guard_birth':found['birth'],'container':binding,'policy':plan['guard']}
            return {'state':'unsent'}
        state=json.loads(Path(journal).read_text())
        candidates=[row for row in state['effects'].values() if row['effect']['payload']['plan']['action']=='guard-start'
                    and row['effect']['token']==effect['token'] and row['effect']['payload']['plan']['binding']['id']==binding['id']
                    and row.get('outcome') and row['outcome']['physical'].get('guard_birth')]
        births=[row['outcome']['physical']['guard_birth'] for row in candidates
                if row['outcome']['physical']['guard_birth']==effect['payload']['guard_birth']]
        if len(births)!=1:raise Refused('guard process identity unknown or ambiguous')
        alive=[b for b in births if process_birth(int(b['identity']))==b]
        return {'state':'unsent' if alive else 'verified','guard_births':alive,'container':binding,'terminated_guard_birth':births[0]}
    def apply(self,plan,effect,journal):
        action=plan['action']
        if action in ('create','start'):
            if action=='start':self.start_guard(plan,effect,journal)
            command(plan['argv']);return
        if action=='stop':
            birth=plan['binding']['process_birth'];NativeExecutor().apply(member_plan('process_stop',plan['resources'][0],birth,birth['identity']));return
        if action=='guard-start':
            grant=Path(str(journal)+'.'+digest(effect['id'])+'.grant.json')
            durable(grant,{'effect':effect,'journal':str(journal)})
            subprocess.Popen([sys.executable,'-c',guard_program(),str(grant)],start_new_session=True,
                             stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            receipt=Path(str(journal)+'.'+digest(effect['id'])+'.guard.json')
            deadline=time.monotonic()+3
            while not receipt.exists():
                if time.monotonic()>deadline:raise Refused('guard registration reply uncertain')
                time.sleep(.01)
            return
        observation=self.observe(plan,effect,journal)
        for birth in observation['guard_births']:
            NativeExecutor().apply(member_plan('process_stop',plan['resources'][0],birth,birth['identity']))


def watch(grant_path):
    grant=json.loads(Path(grant_path).read_text());effect=grant['effect'];plan=effect['payload']['plan'];journal=Path(grant['journal'])
    receipt=Path(str(journal)+'.'+digest(effect['id'])+'.guard.json')
    durable(receipt,{'effect_digest':digest(effect),'token':effect['token'],'birth':process_birth(os.getpid())})
    start=time.monotonic()
    while True:
        state=json.loads(journal.read_text())
        if state['owner']!=effect['token']:return
        current,_=inspect(plan['binding']['id'])
        try:available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
        except (OSError,StopIteration,ValueError):available=None
        decision=guard_decision(plan,{'available_kib':available,'process_birth':current['process_birth'],
                                      'container_id':current['id'],'container_started':current['started']})
        if decision['state'] in ('retire','refuse-start'):return
        if decision['state']=='await-start-outcome' and time.monotonic()-start>=120:return
        if decision['state']=='kill':
            birth=plan['binding']['process_birth'];fd=os.pidfd_open(int(birth['identity']))
            try:
                if process_birth(int(birth['identity']))!=birth or json.loads(journal.read_text())['owner']!=effect['token']:return
                signal.pidfd_send_signal(fd,signal.SIGKILL)
            finally:os.close(fd)
            return
        time.sleep(plan['guard']['interval_seconds'])

def guard_program():
    """Run the exact reviewed guard functions without a transient source path."""
    import inspect as source
    from profile_recipes.spark_ds41f_1030 import guard_decision
    code='import json,os,sys,re,signal,subprocess,tempfile,time,hashlib\nfrom pathlib import Path\nFAMILY="spark-ds41f"\n'
    for function in (digest,durable,command,process_birth,inspect,guard_decision,watch):code+=source.getsource(function)+'\n'
    code+='watch(sys.argv[1])\n'
    return code

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='guard':raise SystemExit('only a journal-bound guard invocation is supported')
    watch(sys.argv[2])


def hash_file(path):
    import hashlib
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def inventory(recipe,member,op):
    """Read physical/source facts. A missing fact stays Unknown."""
    if recipe.get('engine')=='tensorfold-mia-tp4':
        from miainventory import inventory as mia_inventory
        return mia_inventory(recipe,member,op,DockerMemberIO(),command)
    from profile_recipes.spark_ds41f_1030 import verify_sources,control_artifact,render
    import hashlib
    import shutil
    lock=verify_sources(Path(recipe['package']));resource=DockerMemberIO().resource()
    image=json.loads(command(['docker','image','inspect',lock['image']['reference_tag']]))
    if len(image)!=1:raise Refused('native image identity ambiguous')
    image=image[0];resolved=json.loads((Path(recipe['package'])/'resolved.json').read_text())
    image_verified=image['Id']==lock['image']['reference_local_image_id'] and all(image['Config'].get('Labels',{}).get(k)==v for k,v in resolved['container']['expected_labels'].items())
    ready=True
    for artifact in recipe['artifacts']:
        if artifact['member']!=member:continue
        path=Path(artifact['path'])
        if artifact['kind']=='model':
            manifest=json.loads((Path(recipe['package'])/'model-files.json').read_text())
            for entry in manifest['files']:
                target=path/entry['path']
                if not target.is_file() or target.stat().st_size!=entry['bytes']:ready=False;break
                if 'sha256' in entry:
                    if hash_file(target)!=entry['sha256']:ready=False;break
                else:
                    raw=target.read_bytes()
                    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['git_blob']:ready=False;break
        elif artifact['kind']=='steering-vector':ready=ready and path.is_file() and hash_file(path)==artifact['identity']['sha256']
        elif artifact['kind']=='derived-cache':ready=ready and path.is_dir() and not path.is_symlink()
        elif artifact['kind']=='control-source':
            ready=ready and all((path/relative).is_file() and hash_file(path/relative)==sha for relative,sha in artifact['identity']['files'].items())
    def text(path):
        try:return Path(path).read_text().strip()
        except OSError:return None
    release=text('/etc/dgx-release') or '';version=next((line.partition('=')[2].strip('"') for line in release.splitlines() if line.startswith(('DGX_VERSION=','NVIDIA_DGX_VERSION='))),None)
    platform=text('/sys/class/dmi/id/product_name')
    saver=[p for p in Path('/sys/module/nvidia_uvm/parameters').glob('*leaf*')]
    memory_saver=any(text(p) in ('Y','1') for p in saver) if saver else None
    mem=text('/proc/meminfo') or '';available=next((int(line.split()[1])/1024**2 for line in mem.splitlines() if line.startswith('MemAvailable:')),None)
    launch=next(row for row in render(recipe,'start') if row['member']==member)
    env=dict(launch['argv'][i+1].split('=',1) for i,arg in enumerate(launch['argv'][:-1]) if arg=='--env')
    addresses=json.loads(command(['ip','-j','address','show']))
    seen={address.get('local') for row in addresses for address in row.get('addr_info',[])}
    import ipaddress
    node=json.loads((Path(recipe['package'])/'deployment/config/examples/nodes-ring4.json').read_text())['nodes'][recipe['members'].index(member)]
    ring=env.get('VLLM_HOST_IP') in seen
    for hca,subnet in node['roce_subnets'].items():
        port=Path('/sys/class/infiniband')/hca/'ports/1';index=str(node['roce_gid_index'])
        gid=text(port/'gids'/index)
        try:address=ipaddress.IPv6Address(gid).ipv4_mapped if gid else None
        except ValueError:address=None
        ring=ring and text(port/'state')=='4: ACTIVE' and text(port/'link_layer')=='Ethernet' and text(port/'gid_attrs/types'/index)=='RoCE v2' and address is not None and address in ipaddress.ip_network(subnet)
    # Every GPU user must map to an exact controller-created container cgroup.
    owned={e['outcome']['physical']['container']['id'] for e in op['effects'] if e.get('outcome') and e.get('payload',{}).get('action') in ('create','start')}
    for pid in command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).split():
        cgroup=text('/proc/'+pid+'/cgroup')
        if not cgroup or not any(identifier in cgroup for identifier in owned):raise Refused('unknown GPU process blocks native member lifecycle')
    facts={'gpu_uuid':resource.removeprefix('gpu:'),'platform':platform,'qualified_platform':platform=='NVIDIA DGX Spark',
           'page_size':os.sysconf('SC_PAGE_SIZE'),'kernel':os.uname().release,'dgx_os':version,'memory_saver':memory_saver,
           'nvidia_runtime':'nvidia' in json.loads(command(['docker','info','--format','{{json .Runtimes}}'])),
           'ring_rank':recipe['members'].index(member),'ring_verified':ring,'image_verified':image_verified,
           'artifacts_verified':ready,'native_control_verified':ready,'guard_verified':True,
           'free_disk_bytes':shutil.disk_usage(recipe['spec']['site']['home']).free,'available_memory_gib':available}
    return {'resource':resource,'facts':facts,'image_id':image['Id'],
            'image_environment':dict(row.split('=',1) for row in image['Config'].get('Env',[]) if '=' in row)}


def host_rpc(data,root):
    import copy
    import urllib.request
    from familycontrol import FamilyJournal
    from profileadapter import StudioAdapter
    op=data['operation'];recipe=copy.deepcopy(op['spec']['recipe']);recipe['package']=str(root/'recipe')
    args=data['arguments'];action=data['action'];member=data['member']
    class IO(DockerMemberIO):
        def recipe_context(self,payload):return recipe
    io=IO()
    if action=='inventory':return inventory(recipe,member,op)
    def native_request(body=None):
        config=json.loads(Path(op['native_bootstrap']['config_path']).read_text())
        if config['owner']!={'operation_id':op['id'],'generation':op['generation'],'resources':op['resources']}:raise Refused('native control configuration owner changed')
        port=8888 if recipe.get('engine')=='tensorfold-mia-tp4' else 8000
        request=urllib.request.Request('http://127.0.0.1:'+str(port)+'/profile-control/v1',data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'Bearer '+config['control_token'],'Content-Type':'application/json'},method='GET' if body is None else 'PUT')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        with opener.open(request,timeout=10) as reply:snapshot=json.load(reply)
        # The API PID can be in the container namespace. Bind its NSpid and
        # start ticks through this exact container, rather than equating PIDs.
        head=args.get('container') or op['observations']['requirements']['containers'][recipe['members'][0]]
        current,_=inspect(head['id'])
        if current!=head:raise Refused('native head container process incarnation changed')
        if recipe.get('engine')=='tensorfold-mia-tp4' and (snapshot.get('birth',{}).get('identity')!='1'
                or any(snapshot.get('birth',{}).get(key)!=head['process_birth'][key] for key in ('boot_id','started'))):
            raise Refused('Mia API is not the exact journaled container init birth')
        matched=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                if head['id'] not in (proc/'cgroup').read_text():continue
                status=(proc/'status').read_text();nspid=next(line.split()[1:] for line in status.splitlines() if line.startswith('NSpid:'))
                born=process_birth(int(proc.name))
                if born and snapshot['birth']['identity']==nspid[-1] and all(snapshot['birth'][key]==born[key] for key in ('boot_id','started')):matched.append(born)
            except (OSError,StopIteration):continue
        if len(matched)!=1:raise Refused('native API process binding is unknown or ambiguous')
        from profile_recipes.spark_ds41f_control import parse_snapshot
        return parse_snapshot(snapshot,instance_id=op['id'],expected_birth=snapshot['birth'],footprint=op['resources'])
    if action=='snapshot':return native_request()
    if action=='mutate':return native_request(args['effect']['payload']['request'])
    if action=='verify_running':
        current,_=inspect(args['container']['id'])
        if current!=args['container']:raise Refused('native running container changed')
        guard=args['guard']
        if process_birth(int(guard['guard_birth']['identity']))!=guard['guard_birth']:raise Refused('running memory guard has expired')
        journal=Path.home()/'.local/state/spark'/('native-family-'+digest(io.resource())+'.json')
        state=json.loads(journal.read_text())
        if state['owner']!={'operation_id':op['id'],'generation':op['generation'],'resources':op['resources']}:raise Refused('running guard generation differs')
        return {'verified':True,'container':current,'guard_birth':guard['guard_birth']}
    if action!='execute':raise Refused('unsupported native boundary action')
    class Runtime:
        def api(self,node,port,path,body=None,method=None):
            if port != 7870 or path != '/api/control/snapshot' or body is not None or method not in (None, 'GET'):
                raise Refused('native peer observation permits only the Studio snapshot')
            key = data.get('peer_credentials', {}).get(node)
            if not isinstance(key, str) or not key:
                raise Refused('Studio peer authentication is unavailable')
            request = urllib.request.Request('http://'+data['peers'][node]+':'+str(port)+path,
                                             headers={'X-Studio-Key':key},method='GET')
            # Never send a private peer key to a proxy or redirect target.
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self,*args,**kwargs):return None
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(request,timeout=5) as response:return json.load(response)
    adapter=StudioAdapter(Runtime(),{node:(node,7870) for node in data['peers']},time.monotonic)
    journal=Path.home()/'.local/state/spark'/('native-family-'+digest(io.resource())+'.json')
    return FamilyJournal(journal,io.resource(),io,lambda node,prior:adapter.snapshot(node,prior)).execute(args['effect'],args['authorities'])
