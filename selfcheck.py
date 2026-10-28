"""Validate task checkers with known-broken fixtures and known-good reference code.

Operator only. References must never be exposed to contestants.
"""
import contextlib
import io
import json
import uuid
from suite import CASES, RUNS, prepare, check
import light10_cases  # noqa: F401  L6-L11 (the sandbox uses them too); references below since 2026-10-28

REFS = {
'L0': {'calculator.py': 'def add(a,b):\n    return a+b\n'},
'L1': {'answer.json':json.dumps(dict(removed_fields=['id','meta','model','output'],preserves_original=True,preserves_content=True))},
'L2': {'fetcher.py': 'def fetch(client,url):\n    return client.get(url, timeout=30)\n'},
'L3': {'protocol.py': '''
def next_action(result):
    status=result['status']
    if status=='running': return dict(tool='get_process_status',process_id=result['process_id'])
    if status in ('completed','failed'): return dict(tool='final_report',success=status=='completed' and result['exit_code']==0)
    raise ValueError(status)
'''},
'L4': {'settings.py': '''
def effective_limit(value,default=10):
    if value is None: return default
    if value<0: raise ValueError(value)
    return value
'''},
'L5': {'cache.py':'def key(user_id,job_id,profile_version):\n    return (user_id,job_id,profile_version)\n'},
'L6': {'retry.py': '''
def should_retry(status,error):
    if status is None: return error in ('timeout','connection')
    return status==429 or 500<=status<=599
'''},
'L7': {'api.py': '''
from queueing import payload
REGIONS=('worldwide','europe','americas')
def submit(owner,enqueue,region='worldwide'):
    if not isinstance(region,str) or region not in REGIONS: raise ValueError(region)
    return enqueue(payload(owner,region))
''', 'queueing.py': "def payload(owner,region='worldwide'):\n    return {'owner':owner,'region':region}\n",
     'worker.py': "def execute(task,collect):\n    return collect(task['owner'],task.get('region','worldwide'))\n"},
'L8': {'recovery.py': '''
def next_recovery(task):
    state=task['state']
    if state=='running': return 'wait' if task.get('lease_live') else 'retry'
    if state=='queued': return 'retry'
    if state=='failed': return 'retry' if task['attempts']<task['max_attempts'] else 'escalate'
    raise ValueError(state)
'''},
'L9': {'config_reader.py': '''
def read_settings(text):
    out={}
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith('#'): continue
        if '=' not in line: raise ValueError(line)
        key,value=(part.strip() for part in line.split('=',1))
        if key not in ('endpoint','enabled') or key in out: raise ValueError(key)
        out[key]=value
    if set(out)!={'endpoint','enabled'} or out['enabled'] not in ('true','false'): raise ValueError(out)
    out['enabled']=out['enabled']=='true'
    return out
'''},
'L10': {'monitor.py': '''
def is_stuck(events,now,grace):
    latest={}
    for e in events:
        if e['job_id'] not in latest or e['at']>=latest[e['job_id']]['at']: latest[e['job_id']]=e
    return sorted(j for j,e in latest.items() if e['state']=='running' and now-e['at']>grace)
'''},
'L11': {'urls.py': '''
def canonical_url(url):
    if not isinstance(url,str): raise ValueError(url)
    scheme,sep,rest=url.partition('://')
    if not sep or scheme.lower() not in ('http','https'): raise ValueError(url)
    rest=rest.split('#',1)[0]
    cut=min([i for i in (rest.find('/'),rest.find('?')) if i>=0] or [len(rest)])
    host,tail=rest[:cut],rest[cut:]
    if not host: raise ValueError(url)
    path,mark,query=tail.partition('?')
    if path!='/' and path.endswith('/'): path=path[:-1]
    return scheme.lower()+'://'+host.lower()+path+mark+query
'''},
'L12': {'pagination.py': '''
def page(items,number,size):
    if number<1 or size<1: raise ValueError((number,size))
    return list(items[(number-1)*size:number*size])
'''},
'L13': {'skills.py': '''
def has_skill(skills,wanted):
    wanted=wanted.strip().lower()
    return bool(wanted) and any(s.strip().lower()==wanted for s in skills)
'''},
'L14': {'report.py': '''
import math
def percent(part,total):
    if part<0 or total<0: raise ValueError((part,total))
    if total==0: return '0%'
    return f"{math.floor(part*100/total+0.5)}%"
'''},
'L15': {'config.py': '''
def with_defaults(options,defaults):
    merged=dict(defaults)
    for k,v in options.items():
        if v is not None or k not in defaults: merged[k]=v
    return merged
'''},
'M1': {'jobs.py': '''
def unique_jobs(rows):
    seen=set(); output=[]
    for row in rows:
        ident=row.get('source_id')
        key=(row['source'],ident)
        if ident is None or ident=='': output.append(row)
        elif key not in seen: seen.add(key); output.append(row)
    return output
'''},
'M2': {'api.py': '''
from queueing import payload
def submit(owner,enqueue,batch_size=20):
    if type(batch_size) is not int or not 1<=batch_size<=100: raise ValueError(batch_size)
    return enqueue(payload(owner,batch_size))
''', 'queueing.py':'def payload(owner,batch_size=20):\n    return dict(owner=owner,batch_size=batch_size)\n',
'worker.py':'def execute(task,process):\n    return process(task["owner"],task.get("batch_size",20))\n'},
'M3': {'cache.py':'def make_key(text,profile,criteria):\n    return (text,profile,criteria)\n'},
'M4': {'amount.py': '''
import math
def parse_amount(text):
    value=float(text.strip().replace(',','.'))
    if not math.isfinite(value) or value<0: raise ValueError(text)
    return value
'''},
'M5': {'view.py': '''
class View:
    def __init__(self):
        self.rows=[]; self.notes={}; self.loading=False; self.error=None; self.seq=0
    def begin(self,query):
        self.seq+=1; self.loading=True; self.error=None
        return self.seq
    def finish(self,request_id,rows):
        if request_id!=self.seq or not self.loading: return False
        self.rows=list(rows); self.loading=False
        return True
'''},
'H1': {'api.py': '''
from queueing import payload
def submit(owner,enqueue,mode='all'):
    if mode not in ('all','remote','hybrid','office'): raise ValueError(mode)
    return enqueue(payload(owner,mode))
''','queueing.py':"def payload(owner,mode='all'):\n    return dict(owner=owner,mode=mode)\n",
'worker.py': '''
def execute(task,jobs):
    mode=task.get('mode','all')
    return [j for j in jobs if mode=='all' or j.get('mode') is None or j['mode']==mode]
'''},
'H2': {'queueing.py': '''
from threading import Lock
class Queue:
    def __init__(self): self.jobs={}; self.lock=Lock(); self.seq=0
    def add(self,job_id):
        with self.lock: self.jobs.setdefault(job_id,dict(state='queued'))
    def claim(self,worker,now,ttl):
        if ttl<=0: raise ValueError(ttl)
        with self.lock:
            for job_id,j in self.jobs.items():
                if j['state']=='queued' or (j['state']=='running' and j['expiry']<=now):
                    self.seq+=1
                    j.update(state='running',worker=worker,token=self.seq,expiry=now+ttl)
                    return dict(job_id=job_id,worker=worker,token=self.seq)
            return None
    def valid(self,lease,now):
        j=self.jobs.get(lease.get('job_id'),{})
        return j.get('state')=='running' and j.get('worker')==lease.get('worker') and j.get('token')==lease.get('token') and j['expiry']>now
    def renew(self,lease,now,ttl):
        if ttl<=0: raise ValueError(ttl)
        with self.lock:
            if not self.valid(lease,now): return False
            self.jobs[lease['job_id']]['expiry']=now+ttl
            return True
    def complete(self,lease,now):
        with self.lock:
            if not self.valid(lease,now): return False
            self.jobs[lease['job_id']]['state']='completed'
            return True
'''},
'H3': {'service.py': '''
def evaluate(store,task,profile,compute):
    if task['owner']!=profile['owner']: raise PermissionError('owner')
    if task['version']!=profile['version']: return None
    key=(task['owner'],profile['version'],task['criteria_version'],task['job_text'])
    if key not in store: store[key]=compute()
    return store[key]
'''},
'H4': {'controller.py': '''
class Controller:
    def __init__(self):
        self.generation=0; self.rows=[]; self.decisions={}; self.busy=False; self.state='idle'
    def start(self):
        self.generation+=1; self.busy=True; self.state='running'
        return self.generation
    def poll_status(self,generation,status):
        if generation!=self.generation or self.state!='running': return False
        if status=='completed': self.state='pending'; return True
        if status=='failed': self.state='failed'; self.busy=False
        return False
    def finish(self,generation,rows):
        if generation!=self.generation or self.state!='pending': return False
        self.rows=list(rows); self.busy=False; self.state='done'; return True
    def set_decision(self,job_id,value): self.decisions[job_id]=value
'''},
'H5': {'engine.py': '''
from threading import Lock
from copy import deepcopy
class Engine:
    def __init__(self): self.tasks={}; self.lock=Lock(); self.seq=0; self.fence=0
    def submit(self,owner,key):
        with self.lock:
            for ident,t in self.tasks.items():
                if t['owner']==owner and t['key']==key and t['state'] in ('queued','running'): return ident
            self.seq+=1; self.tasks[self.seq]=dict(owner=owner,key=key,state='queued'); return self.seq
    def claim(self,worker,now,ttl):
        if ttl<=0: raise ValueError(ttl)
        with self.lock:
            for ident,t in self.tasks.items():
                if t['state']=='queued' or (t['state']=='running' and t['expiry']<=now):
                    self.fence+=1; t.update(state='running',expiry=now+ttl,token=self.fence,worker=worker)
                    return dict(task_id=ident,worker=worker,token=self.fence,owner=t['owner'])
            return None
    def cancel(self,owner,task_id):
        with self.lock:
            t=self.tasks.get(task_id)
            if t is None: return False
            if t['owner']!=owner: raise PermissionError('owner')
            if t['state'] not in ('queued','running'): return False
            t['state']='cancelled'; return True
    def publish(self,lease,now,result):
        with self.lock:
            t=self.tasks.get(lease.get('task_id'),{})
            if not (t.get('state')=='running' and t['expiry']>now and t['worker']==lease.get('worker') and t['token']==lease.get('token') and t['owner']==lease.get('owner')): return False
            t['result']=deepcopy(result); t['state']='completed'; return True
    def results(self,owner):
        with self.lock: return deepcopy([t['result'] for t in self.tasks.values() if t['owner']==owner and t['state']=='completed'])
'''},
'H0': {'report.py': '''
def summarize(jobs):
    counts=dict(new=0,applied=0,rejected=0,other=0)
    for job in jobs:
        status=job.get('status')
        status=status.strip().lower() if isinstance(status,str) else None
        counts[status if status in ('new','applied','rejected') else 'other']+=1
    return counts
'''},
'H6': {'models.py': '''
MODES={'yes':'remote','no':'office',None:'unknown'}
def upgrade(record):
    if record.get('version')==2: return dict(record)
    remote=record.get('remote')
    if remote not in MODES: raise ValueError(remote)
    return dict(version=2,id=record['id'],title=record['name'],work_mode=MODES[remote])
''', 'storage.py': '''
from models import upgrade
def load_all(rows):
    return [upgrade(r) for r in rows]
def save(record):
    return upgrade(record)
'''},
'H7': {'limiter.py': '''
class Limiter:
    def __init__(self,rate,burst):
        if rate<=0 or burst<1: raise ValueError((rate,burst))
        self.rate=rate; self.burst=burst; self.state={}
    def allow(self,key,now):
        tokens,last=self.state.get(key,(self.burst,now))
        if now<last: raise ValueError(now)
        tokens=min(self.burst,tokens+(now-last)*self.rate)
        if tokens>=1:
            self.state[key]=(tokens-1,now); return True
        self.state[key]=(tokens,now); return False
'''},
'H8': {'money.py': "def parse_salary(text):\n    return int(text.replace(' ', ''))\n",
       'rules.py': "def keep(jobs, minimum):\n    return [j for j in jobs if j['salary'] >= minimum]\n"},
'H9': {'pages.py': '''
def list_page(items,cursor,size):
    if size<1: raise ValueError(size)
    ordered=sorted(items,key=lambda x:(-x['score'],x['id']))
    if cursor is not None:
        try:
            score,ident=cursor.split(':',1); score=int(score)
        except (ValueError,AttributeError): raise ValueError(cursor)
        ordered=[x for x in ordered if (-x['score'],x['id'])>(-score,ident)]
    page=ordered[:size]
    nxt=f"{page[-1]['score']}:{page[-1]['id']}" if len(ordered)>size else None
    return page,nxt
'''},
}


def main():
    run='selfcheck-'+uuid.uuid4().hex[:8]
    results=[]
    for key in CASES:
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(run,key)
            root=RUNS/run/key
            broken=check(root,selftest=True)
            for name,source in REFS[key].items(): (root/'workspace'/name).write_text(source.lstrip(),encoding='utf-8')
            good=check(root,selftest=True)
        results.append(dict(case=key,rejects_broken=not broken,accepts_reference=good))
    report=dict(kind='checker_validation_not_model_results',cases=results,
                passed=all(x['rejects_broken'] and x['accepts_reference'] for x in results))
    (RUNS/run/'selfcheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(RUNS/run/'selfcheck.json')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
