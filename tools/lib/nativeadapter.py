"""Finite TP4 lifecycle over exact member journals and the in-server barrier."""
import copy
from clustercontrol import EffectKind,EffectState,Refused,digest,token_for
from familycontrol import member_payload,ACTIONS
from profile_recipes.spark_ds41f_control import parse_request,parse_snapshot
from profile_recipes.native_leaf import leaf

class NativeAdapter:
    def __init__(self,controller,transport):self.control,self.transport=controller,transport
    def inspect(self,op,bindings,authorities):
        recipe=op['spec']['recipe'];facts={};inventory={};blockers=[]
        for member in recipe['members']:
            try:
                found=self.transport.inventory(op,member)
                if found['resource']!=bindings[member]['resource_id']:raise Refused('native inventory and Studio physical binding differ')
                inventory[member]=found;facts[member]=found['facts']
                if 'gpu:'+facts[member]['gpu_uuid']!=found['resource']:raise Refused('native fact identity differs from inventory')
            except (OSError,ValueError,KeyError,Refused) as error:blockers.append({'resource':bindings[member]['resource_id'],'reason':str(error)})
        # Startup capacity was measured before our own load. Reuse that exact
        # owned preflight; current running safety comes from the bound guard.
        inputs=copy.deepcopy(facts);saved=op.get('native_preflight')
        if saved:
            if saved['bindings']!=bindings:raise Refused('native startup binding changed')
            started={e['payload']['member'] for e in op['effects'] if e.get('payload',{}).get('action')=='start' and e['state']==EffectState.VERIFIED}
            for member in started:
                if member not in inventory:continue
                prior=saved['inventory'][member]
                if inventory[member]['image_id']!=prior['image_id']:raise Refused('native startup image identity changed')
                inputs[member]['available_memory_gib']=prior['facts']['available_memory_gib']
        check=leaf(recipe).preflight(recipe,inputs)
        blockers += [{'resource':'*','reason':reason} for reason in check['blockers']]
        if not blockers and saved is None:self.control.capture_native_preflight(op['id'],inventory,bindings)
        containers={};guards={}
        for effect in op['effects']:
            if effect['state']!=EffectState.VERIFIED or effect.get('payload',{}).get('family')!='spark-ds41f':continue
            plan=effect['payload']['plan'];physical=effect['outcome']['physical']
            if plan['action']=='start':containers[plan['member']]=physical['container']
            if plan['action']=='guard-start' and plan['guard']['phase']=='steady':guards[plan['member']]=physical
        snapshot=None
        if len(containers)==4 and len(guards)==4:
            try:
                for member in recipe['members']:
                    self.transport.verify_running(op,member,containers[member],guards[member])
                snapshot=self.transport.snapshot(op,containers[recipe['members'][0]])
                parse_snapshot(snapshot,instance_id=op['id'],expected_birth=snapshot['birth'],footprint=op['resources'])
                if snapshot['state'] not in ('Draining','Sealed','Open'):raise Refused('native service admission state is unknown')
                if snapshot['owner'] not in (None,token_for(op).document()):raise Refused('native barrier belongs to another operation')
            except (OSError,ValueError,KeyError,Refused) as error:blockers.append({'resource':'*','reason':str(error)})
        complete=len(containers)==4 and len(guards)==4 and snapshot is not None and snapshot['state'] in ('Sealed','Open')
        state='verified' if complete and not blockers else 'unknown'
        return {'blockers':blockers,'changes':[] if complete else ['native member lifecycle'],
                'inventory':inventory,'native':snapshot,'containers':containers,'guards':guards,
                'artifacts':{'state':'verified' if not blockers else 'unknown','inventory':inventory},
                'capabilities':{'state':'verified' if not blockers else 'unknown','family':'spark-ds41f'},
                'services':{'state':state,'containers':containers,'guards':guards,'native':snapshot},
                'settings':{'state':'verified','site_digest':recipe['site_digest']}}
    def execute(self,effect,authorities):
        if effect['kind']==EffectKind.NATIVE_FENCE:
            raw=self.transport.mutate(effect)
            request=effect['payload']['request']
            snapshot=parse_snapshot(raw,instance_id=request['instance_id'],expected_birth=request['birth'],
                                    footprint=request['resources'],minimum_revision=request['expected_revision']+1)
            owner={k:request[k] for k in ('operation_id','generation','resources')}
            if request['action']=='release':
                if snapshot['state']!='Open' or snapshot['owner'] is not None:raise Refused('native release outcome unknown')
            elif request['action']=='acquire':
                if snapshot['owner']!=owner or snapshot['state']!='Draining':raise Refused('native ingress acquisition outcome unknown')
            elif snapshot['owner']!=owner or snapshot['state']!='Sealed':raise Refused('native pause/device synchronization outcome unknown')
            if request['action']!='acquire' and not any(e['state']=='Verified' and e['owner']==owner and e['birth']==request['birth']
                       and e['kind']==('resume' if request['action']=='release' else 'pause')
                       for key,e in snapshot['effects'].items() if key==request['effect_id']):raise Refused('native exact effect outcome unknown')
            return {'effect_id':effect['id'],'token':effect['token'],'verified':True,'native':snapshot}
        return self.transport.execute(effect,authorities)
    def issue(self,op,payload,kind,birth=None):
        effect=self.control.issue(op['id'],kind,payload,[e['id'] for e in op['effects'] if e['state']==EffectState.VERIFIED],birth=birth)
        try:return self.control.outcome(op['id'],effect['id'],self.execute(effect,op['observations']['authorities']))
        except (Refused,OSError,ValueError) as error:return self.control.block(op['id'],[{'resource':'*','reason':str(error),'effect_id':effect['id']}])
    def member(self,op,action,member,**parameters):
        payload=member_payload(op['spec']['recipe'],action,member,op['resources'],**parameters)
        return self.issue(op,payload,ACTIONS[action],payload.get('guard_birth') if action=='guard-stop' else payload['plan']['binding'].get('process_birth'))
    def barrier(self,op,action,snapshot):
        suffix=':retire:' if op.get('retirement') else ':'
        request={'version':1,'action':action,'expected_revision':snapshot['revision'],'instance_id':snapshot['instance_id'],
                 'birth':snapshot['birth'],**token_for(op).document(),'effect_id':op['id']+':native'+suffix+action}
        request['key']=digest(request);parse_request(request)
        observations=copy.deepcopy(op['observations']);observations.setdefault('requirements',{})['native']=snapshot
        self.control.update_evidence(op['id'],observations,[])
        op=self.control.view(op['id'])
        return self.issue(op,{'adapter':'spark-ds41f-ingress-v1','request':request},EffectKind.NATIVE_FENCE)
    def step(self,op):
        checks=op['observations']['requirements'];recipe=op['spec']['recipe']
        if self.control.view(op['id']).get('allocation_current'):
            snapshot=checks['native']
            if snapshot['state']!='Open':return self.barrier(op,'release',snapshot)
            return None
        # Frozen worker-first order. Every create/start/guard has a separate
        # durable effect, immutable object result and exact prerequisite.
        order=leaf(recipe).START_ORDER if recipe['engine']=='tensorfold-mia-tp4' else recipe['members'][1:]+recipe['members'][:1]
        for member in order:
            rows=[e for e in op['effects'] if e['state']==EffectState.VERIFIED and e.get('payload',{}).get('family')=='spark-ds41f' and e['payload']['member']==member]
            def outcome(action,phase=None):
                return next((e['outcome']['physical'] for e in rows if e['payload']['action']==action
                             and (phase is None or e['payload']['plan'].get('guard',{}).get('phase')==phase)),None)
            created=outcome('create')
            if not created:
                inventory=checks['inventory'][member]
                parameters={'creation_key':digest({'operation':op['id'],'generation':op['generation'],'member':member}),
                            'image_id':inventory['image_id']}
                if member==recipe['members'][0]:
                    parameters.update(control=self.transport.configuration(op),image_environment=inventory['image_environment'])
                return self.member(op,'create',member,**parameters)
            if not outcome('guard-start','startup'):return self.member(op,'guard-start',member,container=created['container'],phase='startup')
            running=outcome('start')
            if not running:return self.member(op,'start',member,container=created['container'])
            if not outcome('guard-start','steady'):return self.member(op,'guard-start',member,container=running['container'],phase='steady')
        snapshot=checks['native']
        if snapshot['state']=='Draining':return self.barrier(op,'drain',snapshot)
        if snapshot['state']!='Sealed':raise Refused('native target must remain Sealed before target commit')
        return None
