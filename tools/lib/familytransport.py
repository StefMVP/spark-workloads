"""Existing Runtime.remote transport for the reviewed native boundary sources."""
import base64
import copy
import hashlib
import io
import json
import re
from pathlib import Path
import tarfile
from clustercontrol import Refused,digest,token_for
from artifacttransport import Transport,CHUNK

ROOT=Path(__file__).resolve().parents[2]
class RemoteFamily:
    def __init__(self,controller,runtime):
        self.control,self.runtime=controller,runtime
    def call(self,member,operation,action,**arguments):
        # A transient source projection runs the real boundary implementation.
        # No installed CLI, host checkout or compound upstream script is invoked.
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w:gz') as tar:
            for path in sorted((ROOT/'tools/lib').rglob('*.py')):
                tar.add(path,arcname='lib/'+str(path.relative_to(ROOT/'tools/lib')),recursive=False)
            for path in sorted(Path(operation['spec']['recipe']['package']).rglob('*')):
                if operation['spec']['recipe'].get('engine')=='tensorfold-mia-tp4' and path.name.endswith('.tar.gz'):continue
                if path.is_file():tar.add(path,arcname='recipe/'+str(path.relative_to(operation['spec']['recipe']['package'])),recursive=False)
        data={'operation':operation,'action':action,'member':member,'arguments':arguments,
              'peers':{node:self.runtime.c.NODE[node]['lan_ip'] for node in operation['spec']['recipe']['members']}}
        if action == 'execute':
            # Read the running peer's existing access key through authenticated
            # host transport. It is transient transport state, never an effect,
            # receipt, configuration artifact or saved operation field.
            data['peer_credentials'] = {}
            for peer in arguments['authorities']:
                shared = self.runtime.api(peer, 7870, '/api/key/share')
                key = shared.get('key') if isinstance(shared, dict) else None
                if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', key):
                    raise Refused('Studio peer authentication is unavailable')
                data['peer_credentials'][peer] = key
        blob=archive.getvalue()
        script="import json\nBLOB="+repr(base64.b64encode(blob).decode())+"\nCHECKSUM="+repr(hashlib.sha256(blob).hexdigest())+"\nDATA=json.loads("+repr(json.dumps(data))+")\n"+'''import base64,hashlib,io,json,pathlib,sys,tarfile,tempfile
blob=base64.b64decode(BLOB)
if hashlib.sha256(blob).hexdigest()!=CHECKSUM:raise RuntimeError('source projection changed')
with tempfile.TemporaryDirectory(prefix='profile-native-') as folder:
 root=pathlib.Path(folder)
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:archive.extractall(root,filter='data')
 sys.path.insert(0,str(root/'lib'))
 from nativefamilyio import host_rpc
 try:result={'ok':True,'value':host_rpc(DATA,root)}
 except (RuntimeError,ValueError,OSError,KeyError) as error:result={'ok':False,'reason':str(error)}
 print(json.dumps(result))
'''
        reply=json.loads(self.runtime.remote(member,'python3 -',input=script,timeout=1800))
        if not isinstance(reply,dict) or reply.get('ok') is not True:raise Refused(reply.get('reason','native host response unknown') if isinstance(reply,dict) else 'native host response unknown')
        return reply['value']
    def inventory(self,op,member):return self.call(member,op,'inventory')
    def execute(self,effect,authorities):
        state=self.control.store.read();op=state['operations'][effect['token']['operation_id']]
        return self.call(effect['payload']['member'],op,'execute',effect=effect,authorities=authorities)
    def verify_running(self,op,member,container,guard):return self.call(member,op,'verify_running',container=container,guard=guard)
    def snapshot(self,op,head):return self.call(op['spec']['recipe']['members'][0],op,'snapshot',container=head)
    def mutate(self,effect):
        op=self.control.view(effect['token']['operation_id'])
        return self.call(op['spec']['recipe']['members'][0],op,'mutate',effect=effect)
    def configuration(self,op):
        transport=Transport(self.runtime);head=op['spec']['recipe']['members'][0];binding=transport.identity(head)
        bootstrap=self.control.native_bootstrap(op['id'],binding)
        if bootstrap['binding']!=binding:raise Refused('native control host birth changed')
        files=[]
        for target,body in ((bootstrap['config_path'],bootstrap['config']),
                            (str(Path(bootstrap['state_dir'])/'owner.json'),{'owner':token_for(op).document(),'binding':binding})):
            content=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
            files.append({'source':{'content':base64.b64encode(content).decode(),'binding':binding},
                          'target':target,'sha256':hashlib.sha256(content).hexdigest(),'usage':'new_path'})
        prepared=self.control.prepare_request(binding['machine_id'],files,op['id']+':control-config',op['source_identity'])
        for index,item in enumerate(files):
            if any(e['id']==prepared['id']+':artifact:'+str(index) and e['state']=='verified' for e in prepared['effects']):continue
            effect=self.control.prepare_issue(prepared['id'],index)
            body=base64.b64decode(item['source']['content']);common={'effect_id':effect['id'],'target':item['target'],
                'sha256':item['sha256'],'usage':item['usage'],'binding':binding,'source_binding':binding}
            found=transport.call(head,'stage',**common)
            if found.get('state')!='verified':
                for offset in range(found['offset'],len(body),CHUNK):
                    transport.call(head,'append',**common,offset=offset,chunk=base64.b64encode(body[offset:offset+CHUNK]).decode())
                found=transport.call(head,'publish',**common)
            self.control.prepare_outcome(prepared['id'],effect['id'],found)
        return {key:bootstrap[key] for key in ('config_path','state_dir')}
