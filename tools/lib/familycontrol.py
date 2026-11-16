"""Exact native-family member effects and their durable per-host authority."""
from __future__ import annotations
import copy
import fcntl
import json
import hashlib
from pathlib import Path

from clustercontrol import ADAPTER,Birth,EffectKind,Refused,canonical,digest,durable

FAMILY='spark-ds41f'
ACTIONS={'create':EffectKind.HOST_START,'start':EffectKind.HOST_START,'guard-start':EffectKind.HOST_START,
         'stop':EffectKind.HOST_STOP,'guard-stop':EffectKind.HOST_STOP}

def live_guards(effects):
    live=[]
    for row in effects:
        payload=row.get('payload',{})
        if payload.get('family')!=FAMILY:continue
        if payload['action']=='guard-start':live.append(row)
        elif payload['action']=='guard-stop' and (row.get('outcome') or {}).get('verified') is True:
            live=[old for old in live if old['token']!=row['token'] or (old.get('outcome') or {}).get('physical',{}).get('guard_birth')!=payload['guard_birth']]
    return live

def retired_members(operation):
    effects=operation['effects']
    if live_guards(effects) or any(e['state']=='may_have_applied' for e in effects):return False
    for row in effects:
        if row.get('payload',{}).get('family')==FAMILY and row['payload']['action']=='start' and row.get('outcome'):
            if not any(e.get('payload',{}).get('action')=='stop' and (e.get('outcome') or {}).get('verified') is True
                and e['token']==row['token'] and e['payload']['plan']['binding']==row['outcome']['physical']['container'] for e in effects):return False
    return True

def member_payload(recipe,action,member,footprint,guard_birth=None,**parameters):
    from profile_recipes.native_leaf import leaf
    plan=leaf(recipe).materialize_member_plan(recipe,action,member,footprint,**parameters)
    root=Path(__file__).parent
    policy={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob('*.py'))}
    result={'adapter':ADAPTER,'family':FAMILY,'recipe':recipe,'action':action,'member':member,'policy':policy,
            'resources':list(footprint),'parameters':parameters,'plan':plan}
    if action=='guard-stop':Birth.parse(guard_birth);result['guard_birth']=guard_birth
    elif guard_birth is not None:raise Refused('guard birth is only valid for guard-stop')
    return result

def validate_effect(effect,resource,recipe=None):
    payload=effect.get('payload',{});token=effect.get('token',{})
    if payload.get('family')!=FAMILY or payload.get('action') not in ACTIONS or effect.get('kind')!=ACTIONS[payload['action']]:
        raise Refused('unsupported native family effect kind')
    footprint=token.get('resources')
    if (not isinstance(footprint,list) or len(footprint)!=4 or footprint!=sorted(set(footprint))
        or any(not canonical(r) for r in footprint) or resource not in footprint
        or type(token.get('generation')) is not int or token['generation']<1
        or not isinstance(token.get('operation_id'),str) or not token['operation_id']
        or not effect.get('id','').startswith(token['operation_id']+':')):
        raise Refused('native family requires the exact full-footprint generation')
    if recipe is not None and {k:v for k,v in recipe.items() if k!='package'}!={k:v for k,v in payload['recipe'].items() if k!='package'}:
        raise Refused('host recipe projection differs from issued source identity')
    try:expected=member_payload(recipe or payload['recipe'],payload['action'],payload['member'],footprint,guard_birth=payload.get('guard_birth'),**payload['parameters'])
    except (ValueError,KeyError,TypeError) as error:raise Refused('native member source/plan identity differs') from error
    expected['recipe']=payload['recipe']
    if expected!=payload:raise Refused('native member kind/command/digest/source/footprint differs from pure plan')
    if payload['action'] in ('stop','guard-stop') and effect.get('birth')!=(payload.get('guard_birth') if payload['action']=='guard-stop' else payload['plan']['binding']['process_birth']):
        raise Refused('native stop lacks exact running process birth')
    return expected['plan']

def prerequisite(effect, effects):
    """A member can only advance from its exact verified physical outcome."""
    plan=effect['payload']['plan'];action=plan['action']
    if action=='create':return
    if action=='guard-stop':
        if not any(row.get('token')==effect['token'] and row.get('payload',{}).get('action')=='guard-start'
                   and row['payload']['plan']['binding']==plan['binding']
                   and (row.get('outcome') or {}).get('physical',{}).get('guard_birth')==effect['payload']['guard_birth'] for row in effects):
            raise Refused('guard stop lacks its exact verified startup/running guard birth')
        return
    wanted='create' if action=='start' or action=='guard-start' and plan['guard']['birth'] is None else 'start'
    matches=[row for row in effects if row.get('payload',{}).get('family')==FAMILY
             and row['payload']['member']==plan['member'] and row['payload']['action']==wanted
             and row.get('token')==effect['token'] and row.get('outcome',{}).get('verified') is True
             and row['outcome']['physical'].get('container')==plan['binding']]
    if not matches:raise Refused('native member lacks exact verified '+wanted+' prerequisite')
    if action=='start':
        guards=[row for row in effects if row.get('payload',{}).get('action')=='guard-start'
                and row['payload']['plan'].get('guard',{}).get('phase')=='startup'
                and row['payload']['plan']['binding']==plan['binding'] and row.get('token')==effect['token']
                and row.get('outcome',{}).get('verified') is True]
        if not guards:raise Refused('native start lacks its exact startup guard')

def validate_outcome(effect,evidence):
    plan=effect['payload']['plan']
    if (evidence.get('member')!=plan['member'] or evidence.get('action')!=plan['action']
        or evidence.get('plan_digest')!=digest(plan) or evidence.get('physical',{}).get('state')!='verified'):
        raise Refused('native outcome differs from exact member plan')
    physical=evidence['physical'];action=plan['action']
    if action in ('create','start'):
        container=physical.get('container',{})
        if action=='create':
            if container.get('image_id')!=plan['binding']['image_id'] or container.get('started') is not None or container.get('process_birth') is not None:
                raise Refused('created object outcome has unexpected image/runtime')
        else:
            if any(container.get(k)!=plan['binding'][k] for k in ('id','image_id','created')):raise Refused('start changed immutable container object')
            Birth.parse(container.get('process_birth'))
            if not container.get('started'):raise Refused('start physical birth unknown')
    elif action=='guard-start':
        Birth.parse(physical.get('guard_birth'))
        if physical.get('container')!=plan['binding'] or physical.get('policy')!=plan['guard']:raise Refused('guard binding/policy differs')

class FamilyJournal:
    """HostAccept, physical apply, HostOutcome and wire reply are separate cuts."""
    def __init__(self,path,resource,executor,observe_authority):
        if not canonical(resource):raise Refused('host physical GPU identity unknown')
        self.path,self.resource,self.executor,self.observe_authority=Path(path),resource,executor,observe_authority
    def read(self):
        if not self.path.exists():return {'schema':1,'resource':self.resource,'owner':None,'effects':{}}
        value=json.loads(self.path.read_text())
        if value.get('schema')!=1 or value.get('resource')!=self.resource or not isinstance(value.get('effects'),dict):
            raise Refused('invalid native family journal')
        return value
    def execute(self,effect,authorities):
        plan=validate_effect(effect,self.resource,self.executor.recipe_context(effect['payload']) if hasattr(self.executor,'recipe_context') else None)
        from profileplan import sealed_authorities
        fresh={peer:self.observe_authority(peer,prior) for peer,prior in authorities.items()}
        sealed_authorities({'id':effect['token']['operation_id'],'generation':effect['token']['generation'],
                            'resources':effect['token']['resources'],'observations':{'authorities':fresh}})
        target=fresh.get(plan['member'],{}).get('binding',{})
        if target.get('resource_id')!=self.resource or self.executor.resource()!=self.resource:
            raise Refused('member placement does not match this host physical resource')
        if hasattr(self.executor,'boot') and self.executor.boot()!=target['birth']['boot_id']:
            raise Refused('host boot differs from exact Studio resource binding')
        # These reads are independent of the controller's saved observations.
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.read();prior=state['effects'].get(effect['id']);token=effect['token']
            if state['owner'] and state['owner']!=token and state['owner']['generation']>=token['generation']:
                raise Refused('native host owner/generation is stale')
            if prior:
                if prior['digest']!=digest(effect):raise Refused('native effect ID reused for another exact plan')
                if prior['state']=='Verified':
                    outcome=copy.deepcopy(prior['outcome'])
                    validate_outcome(effect,outcome)
                    for birth in (outcome['physical'].get('container',{}).get('process_birth'),outcome['physical'].get('guard_birth')):
                        if birth is not None and birth['boot_id']!=target['birth']['boot_id']:raise Refused('cached physical outcome belongs to another host boot')
                    return outcome
                # An uncertain create/start is observed by immutable labels/object
                # identity. Absence does not authorize another physical send.
                observed=self.executor.observe(plan,effect,self.path)
                if observed.get('state')!='verified':raise Refused('native family effect remains MayHaveApplied; no second send')
            else:
                owner=state['owner']
                if owner and owner!=token and (owner['generation']>=token['generation'] or any(r['state']!='Verified' for r in state['effects'].values())):
                    raise Refused('native host owner/generation is stale or unsettled')
                if owner and owner!=token and not retired_members({'effects':[r['effect']|{'outcome':r['outcome'],'state':'verified' if r['state']=='Verified' else 'may_have_applied'} for r in state['effects'].values()]}):
                    raise Refused('memory guard generation handoff is not verified; native adoption blocked')
                prerequisite(effect,[r['effect']|{'outcome':r['outcome']} for r in state['effects'].values() if r.get('outcome')])
                self.executor.preflight(plan,effect['payload'],effect,self.path)
                observed=self.executor.observe(plan,effect,self.path)
                if observed.get('state') not in ('verified','unsent'):raise Refused('native effect precondition unknown')
                if plan['action']=='create' and observed['state']=='verified':raise Refused('created object has no prior HostAccept journal')
                state['owner']=copy.deepcopy(token)
                state['effects'][effect['id']]={'digest':digest(effect),'state':'MayHaveApplied','effect':copy.deepcopy(effect),'outcome':None}
                durable(self.path,state)
                if observed['state']=='unsent':self.executor.apply(plan,effect,self.path)
                observed=self.executor.observe(plan,effect,self.path)
                if observed.get('state')!='verified':raise Refused('native family physical outcome unknown')
            outcome={'effect_id':effect['id'],'token':copy.deepcopy(token),'verified':True,'member':plan['member'],
                     'action':plan['action'],'plan_digest':digest(plan),'physical':observed}
            validate_outcome(effect,outcome)
            for birth in (observed.get('container',{}).get('process_birth'),observed.get('guard_birth')):
                if birth is not None and birth['boot_id']!=target['birth']['boot_id']:raise Refused('physical outcome belongs to another host boot')
            state['effects'][effect['id']].update(state='Verified',outcome=outcome)
            durable(self.path,state)
            return outcome
