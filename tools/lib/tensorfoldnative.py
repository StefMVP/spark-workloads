"""Aligned TensorFold TP2 leaf. Direct API, fixed pair, and native child models."""
import copy
import json

from clustercontrol import EffectState, Refused, token_for
from nativeinstalled import Driver as InstalledDriver
import opencodeprofile

MODEL='sparks-glm-flash/GLM-5.3-Flash'
AGENTS=('general','explore')

class Driver(InstalledDriver):
    def step(self,identifier):
        from profilecli import source_identity
        op=self.control.view(identifier)
        from nativeinstalled import EXPIRED_GUARD_SOURCE, expired_guard_refusal
        if op['source_identity']==EXPIRED_GUARD_SOURCE and len(op['effects'])==2:
            op=self.control.record_expired_native_guard_refusal(identifier)
            refusal=expired_guard_refusal(op)
            self.evidence(op)
            effect=op['effects'][1]
            proof=self.transport.call(effect['payload']['member'],'expired-guard-recovery',op=op,effect=effect,refusal=refusal)
            op=self.control.repair_expired_native_guard(identifier,source_identity(self.runtime.root),proof)
        if len(op['effects'])==1:
            effect=op['effects'][0]
            if effect['state']==EffectState.UNCERTAIN and effect['payload'].get('action')=='guard-start':
                candidates=[o for o in self.control.store.read()['operations'].values()
                    if o.get('outcome')=='released' and (o.get('absence_retirement') or o.get('intent')=='retire')
                    and set(op['resources'])<=set(o['resources'])]
                for old in candidates:
                    try:
                        proof=self.transport.call(effect['payload']['member'],'unaccepted',op=op,effect=effect,
                            retired={'token':token_for(old).document(),'manifest':old['spec']['installed_native']})
                    except Refused:
                        continue
                    op=self.control.repair_unaccepted_native(identifier,old['id'],source_identity(self.runtime.root),proof)
                    break
        if op.get('intent')=='retire' and not any(e['state']==EffectState.VERIFIED and e['payload'].get('action')=='offline-settings' for e in op['effects']):
            ev=self.evidence(op)
            op=self.control.update_evidence(identifier,ev,[])
            return self.issue(op,ev,op['spec']['installed_native']['head'],'offline-settings',op['spec']['name'])
        return super().step(identifier)

    def offline(self):
        path=self.runtime.config
        data=json.loads(path.read_text());before=copy.deepcopy(data)
        data=release_config(data)
        if data!=before:
            opencodeprofile.idle()
            from clusterprofile import atomic
            atomic(path.resolve(),data)
        opencodeprofile.refresh(path,data)

    def ready(self,op,ev,manifest):
        health=self.runtime.api(manifest['head'],manifest['port'],'/health')
        if health is not None:
            from clusterprofile import metrics
            observed=metrics(self.runtime.api(manifest['head'],manifest['port'],'/metrics') or '')
            if (observed.get('tensorfold:prompt_cache_can_keep_state',0)<1
                    or observed.get('tensorfold:prompt_cache_budget_bytes',0)<=0
                    or health.get('max_batch_size')!=8):
                raise Refused('GLM prompt cache or parallel capacity differs from saved profile')
        return super().ready(op,ev,manifest)

    def settings(self,manifest):
        path=self.runtime.config
        data=json.loads(path.read_text());before=copy.deepcopy(data)
        data=client_config(data,manifest)
        if data!=before:
            opencodeprofile.idle()
            from clusterprofile import atomic
            atomic(path.resolve(),data)
        opencodeprofile.refresh(path,data)
        models=opencodeprofile.command('api','GET','/api/model')['data']
        selected=next((m for m in models if m['providerID']=='sparks-glm-flash' and m['id']=='GLM-5.3-Flash'),{})
        default=next((v for v in selected.get('variants',[]) if v['id']=='default'),None)
        if default is None or any(default.get(k) for k in ('settings','headers','body')):
            raise Refused('OpenCode default GLM variant must preserve the existing request body')
        actual=opencodeprofile.command('api','GET','/api/agent')['data']
        for name in AGENTS:
            agent=next((a for a in actual if a['id']==name),{})
            if not agent or agent.get('model') is not None:
                raise Refused('OpenCode native child model differs: '+name)

def client_config(original,manifest):
    data=copy.deepcopy(original)
    provider=data.setdefault('providers',{}).setdefault('sparks-glm-flash',{})
    settings=dict(provider.get('settings',{}),baseURL=manifest['endpoint'])
    provider.update(name='Sparks GLM Flash (Spark 1 + Spark 4)',package='@opencode/ai/providers/openai-compatible',
                    settings=settings,models={manifest['model']:{
                        'name':'GLM-5.3-Flash','limit':{'context':1048576,'output':32768},
                        'capabilities':{'tools':True,'input':['text','image'],'output':['text']},
                        'variants':[{'id':'default'},*[{'id':level,'settings':{'reasoningEffort':level}} for level in ('low','medium','high')]],
                        'cost':{'input':0,'output':0}}})
    data['model']=MODEL
    clear_owned_agents(data)
    return data

def clear_owned_agents(data):
    for name in AGENTS:
        agent=data.get('agents',{}).get(name,{})
        model=agent.get('model')
        if model==MODEL or model=={'providerID':'sparks-glm-flash','model':'GLM-5.3-Flash'}:
            agent.pop('model')
    return data


def release_config(original):
    data=copy.deepcopy(original)
    clear_owned_agents(data)
    if data.get('model')==MODEL:
        data['model']='sparks-offline/offline'
        data.setdefault('providers',{}).setdefault('sparks-offline',{
            'name':'Sparks offline','package':'@opencode/ai/providers/openai-compatible',
            'settings':{'baseURL':'http://127.0.0.1:9/v1'},'models':{'offline':{'name':'Sparks offline'}}})
    return data
