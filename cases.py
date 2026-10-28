"""Frozen synthetic tasks. Verifiers stay outside the model's working directory."""
from textwrap import dedent

CASES = {}


def case(key, title, requirement, files, checks, bridge=False):
    CASES[key] = dict(id=key, level=key[0], title=title, requirement=requirement,
                     files={p: dedent(s).lstrip() for p, s in files.items()},
                     checks=dedent(checks).lstrip(), bridge=bridge)


case('L0', 'Baseline: fix addition',
     'In calculator.py, fix add(a, b) so it returns the sum. Keep its name and two arguments. '
     'Read calculator.py, edit it, run the supplied tests with run_tests, then report the result. '
     'candidate_tests.py is supplied and must not be edited.',
     {'calculator.py': 'def add(a, b):\n    return a - b\n',
      'candidate_tests.py': 'from calculator import add\nassert add(2, 3) == 5\nassert add(-2, 2) == 0\n'}, '''
from calculator import add
assert add(2, 3) == 5
assert add(-2, 2) == 0
assert add(0, 0) == 0
''')


case('L1', 'Read message sanitization',
     'Read messages.py. Create answer.json: keys removed_fields (sorted array), '
     'preserves_original (boolean), preserves_content (boolean). Report actual behavior; do not edit messages.py.',
     {'messages.py': '''
def sanitize(messages):
    result = []
    for message in messages:
        clean = dict(message)
        for key in ('id', 'output', 'model', 'meta'):
            clean.pop(key, None)
        result.append(clean)
    return result
'''}, '''
import json
a = json.loads((ROOT/'answer.json').read_text())
assert a == dict(removed_fields=['id','meta','model','output'], preserves_original=True, preserves_content=True)
''')

case('L2', 'Timeout without changing API',
     'Make fetch(client, url) use timeout=30. Preserve arguments, return value and exception propagation. No retries.',
     {'fetcher.py': 'def fetch(client, url):\n    return client.get(url)\n'}, '''
import inspect
from fetcher import fetch
assert list(inspect.signature(fetch).parameters) == ['client','url']
class Client:
    calls = 0
    def get(self, url, **kw):
        self.calls += 1
        assert url == 'https://example.invalid' and kw == {'timeout':30}
        return object_marker
object_marker = object()
c = Client()
assert fetch(c,'https://example.invalid') is object_marker and c.calls == 1
class Broken(Client):
    def get(self,*a,**kw):
        self.calls += 1
        raise TimeoutError('expected')
b = Broken()
try: fetch(b, 'https://example.invalid')
except TimeoutError: pass
else: raise AssertionError('exception swallowed')
assert b.calls == 1
class Echo:
    def get(self,url,**kw): return url,kw
assert fetch(Echo(),'another-url') == ('another-url',{'timeout':30})
''')

case('L3', 'Asynchronous command state',
     'Implement next_action(result) -> dict. running => {tool:get_process_status, process_id:<same id>}; '
     'completed with exit_code 0 => {tool:final_report, success:True}; failed or completed nonzero => '
     '{tool:final_report, success:False}. Do not repeat run_command. Unknown status => ValueError.',
     {'protocol.py': 'def next_action(result):\n    return {"tool":"run_command"}\n'}, '''
from protocol import next_action
for pid in ['p17','other-process']:
    assert next_action(dict(status='running',process_id=pid)) == dict(tool='get_process_status',process_id=pid)
for status, code, success in [('completed',0,True),('completed',2,False),('failed',1,False)]:
    assert next_action(dict(status=status,exit_code=code)) == dict(tool='final_report',success=success)
try: next_action(dict(status='mystery'))
except ValueError: pass
else: raise AssertionError('unknown status accepted')
''')

case('L4', 'Zero is a valid setting',
     'Fix settings.effective_limit: None means default; zero means disabled and must stay zero; '
     'positive int overrides default; negative int raises ValueError. Preserve function signature.',
     {'settings.py': 'def effective_limit(value, default=10):\n    return value or default\n',
      'run.log': 'WARN optional icons missing\nINFO requested_limit=0 effective_limit=10\n'}, '''
from settings import effective_limit
assert effective_limit(None) == 10
assert effective_limit(None,7) == 7
assert effective_limit(0,7) == 0
assert effective_limit(3,7) == 3
try: effective_limit(-1)
except ValueError: pass
else: raise AssertionError('negative limit accepted')
''')

# L12-L15 added 2026-10-28 as reserve Light tasks (first written as L6-L9, which collided with the verified
# L6-L11 in light10_cases.py). Same shape: one small module, one trap, assert checks.
case('L12', 'Pagination is 1-based',
     'Fix pagination.page(items, number, size): pages are numbered from 1; the last page may be shorter; '
     'a page past the end returns an empty list; number < 1 or size < 1 raises ValueError. '
     'Return a new list and do not modify items. Keep the signature.',
     {'pagination.py': 'def page(items, number, size):\n    return items[number * size:(number + 1) * size]\n'}, '''
from pagination import page
items=list(range(1,8))
assert page(items,1,3) == [1,2,3]
assert page(items,2,3) == [4,5,6]
assert page(items,3,3) == [7]
assert page(items,4,3) == []
assert page([],1,5) == []
for bad in ((0,3),(-1,3),(1,0)):
    try: page(items,*bad)
    except ValueError: pass
    else: raise AssertionError(f'accepted {bad}')
first=page(items,1,3); first.append(99)
assert items == list(range(1,8))
''')

case('L13', 'Skill match is whole-word',
     'Fix skills.has_skill(skills, wanted): True when one listed skill equals wanted ignoring case and '
     'surrounding spaces. A skill that only contains the word does not count ("Java" is not "JavaScript"). '
     'Empty or blank wanted returns False. Keep the signature.',
     {'skills.py': 'def has_skill(skills, wanted):\n    return any(wanted.lower() in s.lower() for s in skills)\n'}, '''
from skills import has_skill
skills=['Python', ' JavaScript ', 'SQL', 'Power BI']
assert has_skill(skills,'python')
assert has_skill(skills,'  sql ')
assert has_skill(skills,'javascript')
assert has_skill(skills,'POWER BI')
assert not has_skill(skills,'Java')
assert not has_skill(skills,'Power')
assert not has_skill(skills,'')
assert not has_skill(skills,'   ')
assert not has_skill([],'Python')
''')

case('L14', 'Percent rounds half up',
     'Fix report.percent(part, total): whole-number percent string like "13%", rounded half up '
     '(12.5 -> 13, 62.5 -> 63), not round-half-to-even. total == 0 returns "0%". '
     'Negative part or total raises ValueError. Keep the signature.',
     {'report.py': 'def percent(part, total):\n    return f"{round(part / total * 100)}%"\n'}, '''
from report import percent
assert percent(1,8) == '13%'
assert percent(5,8) == '63%'
assert percent(3,8) == '38%'
assert percent(1,3) == '33%'
assert percent(2,3) == '67%'
assert percent(8,8) == '100%'
assert percent(0,5) == '0%'
assert percent(0,0) == '0%'
for bad in ((-1,5),(1,-5)):
    try: percent(*bad)
    except ValueError: pass
    else: raise AssertionError(f'accepted {bad}')
''')

case('L15', 'Defaults without side effects',
     'Fix config.with_defaults(options, defaults): return a NEW dict with every default key; a value from '
     'options wins unless it is None (None means "use the default"); keys only in options are kept. '
     'Neither argument may be modified. Keep the signature.',
     {'config.py': 'def with_defaults(options, defaults):\n    defaults.update(options)\n    return defaults\n'}, '''
from config import with_defaults
defaults={'limit':10,'remote':False,'country':'NL'}
options={'limit':0,'remote':None,'tag':'x'}
merged=with_defaults(options,defaults)
assert merged == {'limit':0,'remote':False,'country':'NL','tag':'x'}
assert defaults == {'limit':10,'remote':False,'country':'NL'}
assert options == {'limit':0,'remote':None,'tag':'x'}
merged['country']='DE'
assert defaults['country'] == 'NL'
assert with_defaults({},defaults) == defaults
assert with_defaults({},defaults) is not defaults
''')

case('L5', 'Cache respects profile changes',
     'Fix stale scores across profiles: get_score(store, user_id, job_id, profile_version, compute) '
     'must cache by ALL three identifiers. Cache hits including zero must not call compute. '
     'Keep all public signatures. Inspect all three files.',
     {'cache.py': 'def key(user_id, job_id, profile_version):\n    return job_id\n',
      'service.py': '''
from cache import key
def get_score(store, user_id, job_id, profile_version, compute):
    k = key(user_id, job_id, profile_version)
    if k not in store:
        store[k] = compute()
    return store[k]
''', 'demo.py': 'from service import get_score\n# Same job can have different scores per owner/version.\n'}, '''
from service import get_score
s={}; calls=[]
def compute(value):
    def run(): calls.append(value); return value
    return run
assert get_score(s,'a','j',1,compute(0)) == 0
assert get_score(s,'a','j',1,compute(99)) == 0
assert get_score(s,'b','j',1,compute(80)) == 80
assert get_score(s,'a','j',2,compute(30)) == 30
assert calls == [0,80,30]
assert get_score(s,'a|b','c',1,compute(11)) == 11
assert get_score(s,'a','b|c',1,compute(22)) == 22
''', bridge=True)

case('M1', 'Stable deduplication',
     'Fix unique_jobs(rows): stable first occurrence wins by (source, source_id); do not merge across sources '
     'or by title. Missing/None/empty source_id means keep the row separately. Do not mutate input.',
     {'jobs.py': '''
def unique_jobs(rows):
    return list({r['title']:r for r in rows}.values())
'''}, '''
from copy import deepcopy
from jobs import unique_jobs
rows=[dict(source='x',source_id='1',title='A'),dict(source='x',source_id='1',title='B'),
      dict(source='y',source_id='1',title='A'),dict(source='x',title='A'),
      dict(source='x',source_id=None,title='A'),dict(source='x',source_id='',title='A')]
saved=deepcopy(rows)
assert unique_jobs(rows) == [rows[i] for i in [0,2,3,4,5]]
assert rows == saved and unique_jobs([]) == []
zero=[dict(source='x',source_id=0,title='A'),dict(source='x',source_id=0,title='B')]
assert unique_jobs(zero)==zero[:1]
''')

case('M2', 'Propagate a setting',
     'Add optional batch_size=20 to submit(owner, enqueue, batch_size=20). Validate integer 1..100, '
     'reject bool and invalid values with ValueError BEFORE enqueue. Queue payload includes batch_size; '
     'worker.execute uses it; legacy queued payloads without batch_size use 20. '
     'Existing submit(owner, enqueue) remains valid. Preserve owner in payload.',
     {'api.py': '''
from queueing import payload
def submit(owner, enqueue):
    return enqueue(payload(owner))
''', 'queueing.py': 'def payload(owner):\n    return {"owner":owner}\n',
      'worker.py': 'def execute(task, process):\n    return process(task["owner"], 20)\n'}, '''
from api import submit
from worker import execute
seen=[]
def enqueue(x): seen.append(x); return x
for size in [1,37,100]:
    task=submit('alice',enqueue,batch_size=size)
    assert task['owner']=='alice' and task['batch_size']==size
    assert execute(task,lambda owner,n:(owner,n)) == ('alice',size)
assert submit('bob',enqueue)['batch_size']==20
assert execute({'owner':'legacy'},lambda owner,n:n)==20
before=len(seen)
for size in [0,-1,101,True,1.5,'3',None]:
    try: submit('alice',enqueue,batch_size=size)
    except ValueError: pass
    else: raise AssertionError('invalid batch size accepted')
assert len(seen)==before
''')

case('M3', 'Two cache invalidation inputs',
     'Fix score(store, job_text, profile_version, criteria_version, compute): all three inputs affect caching; '
     'changing any input recomputes, unchanged inputs reuse even None. Avoid ambiguous concatenation keys.',
     {'service.py': '''
from cache import make_key
def score(store, job_text, profile_version, criteria_version, compute):
    k=make_key(job_text,profile_version,criteria_version)
    if k not in store: store[k]=compute()
    return store[k]
''', 'cache.py': 'def make_key(text, profile, criteria):\n    return text\n',
      'run.log': 'INFO profile=2 criteria=1 cache_hit=true returned_profile=1\nWARN optional metrics unavailable\n'}, '''
from service import score
s={}; calls=[]
cases=[(('a',1,1),0),(('a',2,1),False),(('a',2,2),None),(('b',2,2),''),(('ab','c','d'),'text'),(('a','bc','d'),0.0)]
for args,val in cases:
    def comp(v=val): calls.append(1); return v
    for _ in range(2):
        got=score(s,*args,comp); assert got==val and type(got) is type(val), (args, got)
assert len(calls)==6
assert score(s,'a',1,1,lambda: calls.append(1) or 'new')==0 and len(calls)==6, 're-read of the first key recomputed'
''')

case('M4', 'Recover from an unavailable test tool',
     'Fix parse_amount(text): accept decimal dot/comma, spaces around input, zero; reject empty, '
     'negative and non-finite values with ValueError. No dependencies. README contains an obsolete test command: '
     'run the obsolete command once, then find the available standard-library smoke check and run it. '
     'Report what each command actually returned. Do not install the missing package.',
     {'amount.py': 'def parse_amount(text):\n    return float(text)\n',
      'README.md': 'Old command: python -m definitely_missing_test_package\nAvailable fallback: python smoke.py\n',
      'smoke.py': '''
from amount import parse_amount
assert parse_amount(' 12,50 ')==12.5
assert parse_amount('0')==0
for value in ['','-2','nan','inf','-inf']:
    try: parse_amount(value)
    except ValueError: continue
    raise AssertionError(value)
print('SMOKE_OK')
'''}, '''
from amount import parse_amount
for raw,want in [(' 12,50 ',12.5),('0',0),('2.75',2.75),(' 100 ',100)]:
    assert parse_amount(raw)==want
for value in ['','  ','-2','nan','NaN','inf','-inf','hello']:
    try: parse_amount(value)
    except ValueError: pass
    else: raise AssertionError('invalid amount accepted')
''')

case('M5', 'Ignore stale responses',
     'Implement View.begin(query)->monotonically increasing request id; View.finish(id, rows)->bool. '
     'Only the latest begun request can update rows, once. Older or duplicate replies return False. '
     'begin clears loading errors but keeps existing rows until valid finish. Expose loading boolean; '
     'copy accepted rows. Preserve notes (job_id -> note) through all requests.',
     {'view.py': '''
class View:
    def __init__(self):
        self.rows=[]; self.notes={}; self.loading=False; self.error=None; self.seq=0
    def begin(self, query):
        self.seq+=1
        return self.seq
    def finish(self, request_id, rows):
        self.rows=rows
        return True
'''}, '''
from view import View
v=View(); v.rows=['existing']; v.notes['j']='keep'; v.error='old'
a=v.begin('old'); b=v.begin('new')
assert b>a and v.loading and v.rows==['existing'] and v.error is None
assert v.finish(a,['stale']) is False and v.loading
data=['fresh']; assert v.finish(b,data) is True
data.append('mutated'); assert v.rows==['fresh'] and not v.loading
assert v.finish(b,['duplicate']) is False
c=v.begin('third'); assert c>b
assert v.finish(b,['late']) is False
assert v.finish(c,[]) is True and v.rows==[] and v.notes=={'j':'keep'}
''', bridge=True)

case('H1', 'Setting from API to execution',
     'Implement mode propagation across api.py -> queueing.py -> worker.py. '
     'submit(owner, enqueue, mode="all") accepts all/remote/hybrid/office only; invalid mode raises ValueError '
     'before enqueue. Worker defaults missing legacy mode to all. Unknown job mode means missing or None, '
     'and such jobs stay included for every filter; input job modes otherwise belong to remote/hybrid/office. '
     'excludes known mismatches, preserves input order and input objects. Preserve owner field and old call signature.',
     {'api.py': 'from queueing import payload\ndef submit(owner, enqueue):\n    return enqueue(payload(owner))\n',
      'queueing.py': 'def payload(owner):\n    return {"owner":owner}\n',
      'worker.py': 'def execute(task, jobs):\n    return list(jobs)\n'}, '''
from copy import deepcopy
from api import submit
from worker import execute
jobs=[{'id':1,'mode':'remote'},{'id':2,'mode':'office'},{'id':3}, {'id':4,'mode':'hybrid'},{'id':5,'mode':None}]
old=deepcopy(jobs); calls=[]
def enq(x): calls.append(x); return x
for mode,ids in [('remote',[1,3,5]),('office',[2,3,5]),('hybrid',[3,4,5]),('all',[1,2,3,4,5])]:
    t=submit('owner',enq,mode=mode)
    assert t['owner']=='owner' and t['mode']==mode
    assert [j['id'] for j in execute(t,jobs)]==ids
assert len(execute(submit('owner',enq),jobs))==5
assert len(execute({'owner':'legacy'},jobs))==5
n=len(calls)
try: submit('owner',enq,mode='invalid')
except ValueError: pass
else: raise AssertionError('invalid mode')
assert len(calls)==n and jobs==old
''')

QUEUE_SPEC = '''
Implement Queue as a thread-safe in-memory queue with a lock; injected numeric now, no sleeps.
add(job_id) adds a queued record once. claim(worker, now, ttl) atomically returns a lease dict
{job_id,worker,token} or None. Eligible: queued or running whose expiry <= now. IDs chosen in insertion order.
Every claim gets a NEW unique token, including reclaim by the same worker. ttl must be >0.
renew(lease, now, ttl) and complete(lease, now) return boolean; only exact current token/worker
with expiry > now may act. renew extends expiry to now+ttl. completed jobs never reclaim.
Caller mutation of returned lease must not alter internal state. Preserve class and method signatures.
'''
QUEUE_FILES = {'queueing.py': '''
class Queue:
    def __init__(self): self.jobs={}
    def add(self, job_id): self.jobs[job_id]={'state':'queued'}
    def claim(self, worker, now, ttl): return None
    def renew(self, lease, now, ttl): return False
    def complete(self, lease, now): return False
'''}
QUEUE_CHECKS = '''
from queueing import Queue
from concurrent.futures import ThreadPoolExecutor
import threading
q=Queue(); q.add('a'); gate=threading.Barrier(16)
def race(i): gate.wait(); return q.claim(str(i),0,10)
with ThreadPoolExecutor(max_workers=16) as pool:
    leases=list(pool.map(race,range(16)))
valid=[x for x in leases if x is not None]; assert len(valid)==1
old=valid[0]; assert q.renew(old,5,10) is True
assert q.claim('new',10,5) is None
new=q.claim(old['worker'],15,5); assert new and new['token']!=old['token']
assert q.complete(old,16) is False and q.renew(old,16,5) is False
forged=dict(new,worker='intruder'); assert q.complete(forged,16) is False
assert q.complete(new,16) is True
q.add('a'); assert q.claim('again',100,5) is None
q.add('b'); copy=q.claim('x',100,5); altered=dict(copy); altered['token']='bad'
assert q.complete(altered,101) is False and q.complete(copy,101) is True
q.add('c'); returned=q.claim('x',110,5); saved=dict(returned); returned['token']='external-mutation'
assert q.complete(saved,111) is True
try: q.claim('x',100,0)
except ValueError: pass
else: raise AssertionError('zero ttl accepted')
'''
case('H2', 'Lease fencing and concurrent claims', QUEUE_SPEC, QUEUE_FILES, QUEUE_CHECKS)

case('H3', 'Owner-safe versioned cache',
     'Implement service.evaluate(store, task, profile, compute). task and profile contain owner and version; '
     'task additionally job_text and criteria_version. Owner mismatch raises PermissionError BEFORE reading/writing '
     'store or calling compute. task.version mismatch profile.version returns None without side effects. '
     'Cache key includes owner, profile.version, criteria_version, job_text; unchanged input reuses even 0/None. '
     'Exceptions from compute propagate and are not cached. Preserve signature.',
     {'service.py': '''
def evaluate(store, task, profile, compute):
    key=task['job_text']
    if key not in store: store[key]=compute()
    return store[key]
'''}, '''
from service import evaluate
s={}; calls=[]
p={'owner':'a','version':1}; t={'owner':'a','version':1,'criteria_version':1,'job_text':'text'}
def comp(): calls.append(1); return 0
assert evaluate(s,t,p,comp)==0 and evaluate(s,t,p,comp)==0 and len(calls)==1
before=dict(s)
try: evaluate(s,dict(t,owner='b'),p,comp)
except PermissionError: pass
else: raise AssertionError('cross-owner accepted')
assert s==before and len(calls)==1
class Untouchable(dict):
    def __contains__(self,key): raise AssertionError('store accessed before authorization/version check')
    def __getitem__(self,key): raise AssertionError('store accessed before authorization/version check')
    def get(self,*args): raise AssertionError('store accessed before authorization/version check')
    def __setitem__(self,*args): raise AssertionError('store written before authorization/version check')
try: evaluate(Untouchable(),dict(t,owner='b'),p,comp)
except PermissionError: pass
else: raise AssertionError('cross-owner accepted')
assert evaluate(Untouchable(),dict(t,version=0),p,comp) is None
assert evaluate(s,dict(t,version=0),p,comp) is None and s==before and len(calls)==1
for task,prof in [(dict(t,criteria_version=2),p),(dict(t,job_text='changed'),p),
                  (dict(t,owner='b'),dict(p,owner='b')),(dict(t,version=2),dict(p,version=2))]:
    evaluate(s,task,prof,comp)
assert len(calls)==5
none_calls=[]
def none_comp(): none_calls.append(1); return None
none_task=dict(t,job_text='none-value')
assert evaluate(s,none_task,p,none_comp) is None
assert evaluate(s,none_task,p,none_comp) is None and len(none_calls)==1
def boom(): raise RuntimeError('expected')
fresh=dict(t,job_text='exception')
for i in range(2):
    try: evaluate(s,fresh,p,boom)
    except RuntimeError: pass
    else: raise AssertionError('exception swallowed')
''')

case('H4', 'Refresh, versions and user decisions',
     'Repair Controller. start() issues an increasing generation, sets busy=True, preserves rows and decisions. '
     'poll_status(generation,status) returns whether caller should fetch full results: only the latest running '
     'generation transitioning to completed returns True ONCE. queued/running keep busy, failed ends busy '
     'without fetching. Stale/duplicate statuses return False. finish(generation,rows) accepts only latest '
     'completed pending results once, copies rows, ends busy; stale/unrequested finish returns False. '
     'set_decision(job_id,value) persists independent of all refreshes. No network calls.',
     {'controller.py': '''
class Controller:
    def __init__(self):
        self.generation=0; self.rows=[]; self.decisions={}; self.busy=False
    def start(self):
        self.generation+=1; self.rows=[]; self.decisions={}
        return self.generation
    def poll_status(self,generation,status): return True
    def finish(self,generation,rows): self.rows=rows; return True
    def set_decision(self,job_id,value): self.decisions[job_id]=value
'''}, '''
from controller import Controller
c=Controller(); c.rows=['saved']; c.set_decision('j','star')
a=c.start(); b=c.start(); assert b>a and c.busy and c.rows==['saved']
assert c.poll_status(a,'completed') is False
for status in ['queued','running','running']: assert c.poll_status(b,status) is False
assert c.finish(b,['early']) is False
assert c.poll_status(b,'completed') is True
assert c.poll_status(b,'completed') is False
assert c.finish(a,['old']) is False
data=['new']; assert c.finish(b,data) is True; data.append('external')
assert c.rows==['new'] and not c.busy
assert c.finish(b,['duplicate']) is False
g=c.start(); assert c.poll_status(g,'failed') is False and not c.busy
assert c.poll_status(g,'completed') is False
assert c.decisions=={'j':'star'} and c.rows==['new']
''')

case('H5', 'Cancellation, restart and stale publication',
     'Implement Engine in engine.py. All operations thread-safe; no disk/network dependencies. '
     'submit(owner,key) returns task id: duplicate active (queued/running) owner+key returns same id; '
     'after cancel/completion allows a new id. claim(worker,now,ttl) returns lease dict '
     '{task_id,worker,token,owner}; queued or expired running eligible, insertion order. '
     'Every claim uses new unique fencing token. cancel(owner,task_id) raises PermissionError for another owner, '
     'returns False for missing/terminal, otherwise cancels. publish(lease,now,result) succeeds only for '
     'current unexpired lease and active task, exactly once; returns bool. '
     'results(owner) returns completed results for that owner in submission order as a deep copy '
     '(including nested result dictionaries). '
     'Expired/cancelled/stale workers cannot publish; reclaim after crash works; ttl>0 required. '
     'A lease claimed at time t with ttl expires at t+ttl: it is valid while now < t+ttl.',
     {'engine.py': '''
class Engine:
    def __init__(self): pass
    def submit(self,owner,key): return 1
    def claim(self,worker,now,ttl): return None
    def cancel(self,owner,task_id): return False
    def publish(self,lease,now,result): return False
    def results(self,owner): return []
'''}, '''
from engine import Engine
from concurrent.futures import ThreadPoolExecutor
import threading
def race(fn, n=16):
    gate=threading.Barrier(n)
    def go(_): gate.wait(); return fn()
    with ThreadPoolExecutor(max_workers=n) as pool: return list(pool.map(go, range(n)))
def dedup():
    e=Engine(); ids=race(lambda: e.submit('a','same')); assert len(set(ids))==1, 'concurrent duplicates'
    a=ids[0]; assert e.submit('b','same')!=a, 'other owner shares id'
    assert e.cancel('a',a) is True; again=e.submit('a','same'); assert again!=a, 'no new id after cancel'
    lease=e.claim('w',0,10); lease=lease if lease['task_id']==again else e.claim('w',0,10)
    assert e.publish(lease,1,'done') is True; assert e.submit('a','same') not in (a,again), 'no new id after completion'
def permissions():
    e=Engine(); t=e.submit('a','k')
    try: e.cancel('b',t)
    except PermissionError: pass
    else: raise AssertionError('cross-owner cancel')
    assert e.cancel('a','missing') is False
    l=e.claim('w',0,10); assert l['owner']=='a' and l['task_id']==t
    assert e.publish(l,1,'r') is True and e.cancel('a',t) is False, 'terminal task cancelled'
def lease_expiry():
    e=Engine(); t=e.submit('a','k'); l=e.claim('w',1,3); assert l['task_id']==t
    assert e.claim('w2',3,3) is None, 'claimed while lease valid'
    r=e.claim('w2',4,10); assert r and r['task_id']==t and r['token']!=l['token'], 'no reclaim after expiry'
    e2=Engine(); e2.submit('z','x'); l2=e2.claim('w',20,1); assert e2.publish(l2,21,'late') is False, 'expired publish'
    try: e.claim('w',30,0)
    except ValueError: pass
    else: raise AssertionError('ttl validation')
def stale_token():
    e=Engine(); e.submit('a','k'); old=e.claim('w',1,3); new=e.claim('w2',4,10)
    assert e.publish(old,5,'stale') is False, 'stale token published'
    assert e.publish(dict(new,worker='wrong'),5,'forged') is False, 'forged worker published'
    assert e.publish(dict(new,token='bad'),5,'forged') is False, 'forged token published'
    assert e.publish(new,5,'ok') is True
def cancellation():
    e=Engine(); t=e.submit('a','k'); l=e.claim('w',0,10)
    assert e.cancel('a',t) is True and e.cancel('a',t) is False
    assert e.publish(l,1,'cancelled-result') is False and e.results('a')==[]
def publish_once():
    e=Engine(); e.submit('a','k'); l=e.claim('w',0,10)
    assert sum(race(lambda: e.publish(l,1,{'job':'a'})))==1, 'concurrent publish not exactly once'
    assert e.publish(l,2,'twice') is False
    copy=e.results('a'); copy[0]['job']='mutated'; assert e.results('a')==[{'job':'a'}], 'results not a deep copy'
    o=Engine(); first=o.submit('a','first'); second=o.submit('a','second')
    one=o.claim('w1',0,10); two=o.claim('w2',0,10)
    assert one['task_id']==first and two['task_id']==second, 'claim order'
    assert o.publish(two,1,'second') is True and o.publish(one,1,'first') is True
    assert o.results('a')==['first','second'], 'results not in submission order'
failed=[]
for name, fn in [('dedup',dedup),('permissions',permissions),('lease_expiry',lease_expiry),
                 ('stale_token',stale_token),('cancellation',cancellation),('publish_once',publish_once)]:
    try: fn(); print('PASS', name)
    except Exception as ex: failed.append(name); print('FAIL', name, type(ex).__name__, ex)
assert not failed, 'failed groups: ' + ', '.join(failed)
''')


# H0, H6-H9 added 2026-10-28 (user: ten High tasks; H0 is a Medium-difficulty warm-up, like L0 for Light).
case('H0', 'Warm-up: status summary',
     'Fix report.summarize(jobs): return a dict with keys new, applied, rejected, other (always all four, '
     'in that order) counting jobs by job["status"]. A missing, None or unknown status counts as other; '
     'status matching ignores case and surrounding spaces. Do not modify the input. Keep the signature.',
     {'report.py': '''
def summarize(jobs):
    counts = {}
    for job in jobs:
        counts[job['status']] = counts.get(job['status'], 0) + 1
    return counts
''', 'demo.py': 'from report import summarize\n# summarize([{"status": "new"}, {"title": "no status"}])\n'}, '''
from copy import deepcopy
from report import summarize
jobs=[dict(status='new'),dict(status=' Applied '),dict(status='REJECTED'),dict(status='archived'),
      dict(title='x'),dict(status=None),dict(status='new')]
saved=deepcopy(jobs)
result=summarize(jobs)
assert result == dict(new=2,applied=1,rejected=1,other=3)
assert list(result) == ['new','applied','rejected','other']
assert jobs == saved
assert summarize([]) == dict(new=0,applied=0,rejected=0,other=0)
''')

case('H6', 'Storage migration v1 -> v2',
     'Records are stored as v1 {"id","name","remote": "yes"/"no"} or v2 {"version": 2,"id","title",'
     '"work_mode": "remote"/"office"/"unknown"}. Fix models.upgrade(record): v1 -> v2 (name becomes title; '
     'remote yes -> remote, no -> office, missing -> unknown, any other value raises ValueError); a v2 record '
     'is returned as an equal copy (upgrade is idempotent). Never modify the input. storage.load_all(rows) must '
     'return v2 records for mixed v1/v2 input and storage.save(record) must always return a v2 record. '
     'Inspect models.py, storage.py and README.md.',
     {'models.py': '''
def upgrade(record):
    record['title'] = record.pop('name')
    record['work_mode'] = 'remote' if record.pop('remote') == 'yes' else 'office'
    record['version'] = 2
    return record
''', 'storage.py': '''
from models import upgrade

def load_all(rows):
    return [upgrade(r) for r in rows if 'version' not in r]

def save(record):
    return record
''', 'README.md': 'Old installs still hold v1 rows; new code writes v2 only. Migrations may run twice.\n'}, '''
from copy import deepcopy
from models import upgrade
from storage import load_all, save
v1=[dict(id=1,name='A',remote='yes'),dict(id=2,name='B',remote='no'),dict(id=3,name='C')]
v2=dict(version=2,id=4,title='D',work_mode='office')
rows=v1+[v2]; saved=deepcopy(rows)
out=load_all(rows)
assert out == [dict(version=2,id=1,title='A',work_mode='remote'),dict(version=2,id=2,title='B',work_mode='office'),
               dict(version=2,id=3,title='C',work_mode='unknown'),v2]
assert rows == saved
for r in out:
    assert upgrade(r) == r and upgrade(r) is not r
assert upgrade(upgrade(v1[0])) == upgrade(v1[0])
assert save(dict(id=5,name='E',remote='no')) == dict(version=2,id=5,title='E',work_mode='office')
assert save(v2) == v2
try: upgrade(dict(id=6,name='F',remote='maybe'))
except ValueError: pass
else: raise AssertionError('unknown remote value accepted')
''')

case('H7', 'Rate limiter per key',
     'Implement limiter.Limiter(rate, burst): token bucket per key. allow(key, now) refills rate tokens per '
     'second of injected time (now, float seconds) up to burst, then consumes one token and returns True, or '
     'returns False without consuming when less than one token is available. A new key starts full. Keys are '
     'independent. now going backwards for a key raises ValueError. rate <= 0 or burst < 1 raises ValueError. '
     'No sleeps, no real clock.',
     {'limiter.py': '''
class Limiter:
    def __init__(self, rate, burst):
        self.left = burst

    def allow(self, key, now):
        if self.left > 0:
            self.left -= 1
            return True
        return False
'''}, '''
from limiter import Limiter
lim=Limiter(1,2)
assert lim.allow('a',0.0) and lim.allow('a',0.0)
assert not lim.allow('a',0.0)
assert not lim.allow('a',0.5)
assert lim.allow('a',1.0)
assert not lim.allow('a',1.0)
assert lim.allow('b',1.0) and lim.allow('b',1.0) and not lim.allow('b',1.0)
assert lim.allow('a',100.0) and lim.allow('a',100.0) and not lim.allow('a',100.0)
try: lim.allow('a',50.0)
except ValueError: pass
else: raise AssertionError('time went backwards')
fast=Limiter(4,1)
assert fast.allow('k',0.0) and not fast.allow('k',0.0) and fast.allow('k',0.25)
for bad in ((0,1),(-1,1),(1,0)):
    try: Limiter(*bad)
    except ValueError: pass
    else: raise AssertionError(f'accepted {bad}')
''')

case('H8', 'Bug from a log',
     'pipeline.run(lines, minimum) fails in production; see run.log. Find the real cause and fix it. '
     'Modules: parse.py, clean.py, money.py, rules.py, pipeline.py. Only change the module(s) that are '
     'actually wrong; healthy modules must stay byte-for-byte unchanged. Expected behaviour: salaries may '
     'contain spaces as thousands separators ("1 200 000"); a job whose salary equals minimum is kept.',
     {'parse.py': '''
def parse_line(line):
    job_id, title, salary = line.split(';')
    return dict(id=job_id.strip(), title=title, salary=salary.strip())
''', 'clean.py': '''
def clean_title(title):
    # TODO: title casing is deprecated, kept for old exports
    return ' '.join(title.split()).title()
''', 'money.py': '''
def parse_salary(text):
    return int(text.replace(' ', '', 1))
''', 'rules.py': '''
def keep(jobs, minimum):
    return [j for j in jobs if j['salary'] > minimum]
''', 'pipeline.py': '''
from parse import parse_line
from clean import clean_title
from money import parse_salary
from rules import keep

def run(lines, minimum):
    jobs = []
    for line in lines:
        job = parse_line(line)
        job['title'] = clean_title(job['title'])
        job['salary'] = parse_salary(job['salary'])
        jobs.append(job)
    return keep(jobs, minimum)
''', 'run.log': '''WARN clean.py: title casing is deprecated (old exports)
INFO loaded 3 lines
ERROR Traceback (most recent call last):
  File "pipeline.py", line 11, in run
    job['salary'] = parse_salary(job['salary'])
  File "money.py", line 2, in parse_salary
    return int(text.replace(' ', '', 1))
ValueError: invalid literal for int() with base 10: '1200 000'
INFO after hotfix attempt: job 7 with salary 50 000 and minimum 50000 missing from results
'''}, '''
from pipeline import run
lines=['7; data analyst ;50 000','8;Senior  engineer;1 200 000','9;intern;20 000']
assert [j['id'] for j in run(lines,50000)] == ['7','8']
assert run(lines,50000)[1] == dict(id='8',title='Senior Engineer',salary=1200000)
assert [j['id'] for j in run(lines,1200001)] == []
HEALTHY = {
    'parse.py': "def parse_line(line):\\n    job_id, title, salary = line.split(';')\\n"
                "    return dict(id=job_id.strip(), title=title, salary=salary.strip())\\n",
    'clean.py': "def clean_title(title):\\n    # TODO: title casing is deprecated, kept for old exports\\n"
                "    return ' '.join(title.split()).title()\\n",
}
for name, text in HEALTHY.items():
    assert (ROOT/name).read_text(encoding='utf-8').replace('\\r\\n','\\n') == text, name + ' changed'
import hashlib
pipe=(ROOT/'pipeline.py').read_text(encoding='utf-8').replace('\\r\\n','\\n').encode()
assert hashlib.sha256(pipe).hexdigest() == '0147b9ea73d3209f78234413e12642eae65e4b28cea8122f0e96d9370a6ef108', 'pipeline.py changed'
''')

case('H9', 'Cursor pagination with ties',
     'Implement pages.list_page(items, cursor, size) -> (page, next_cursor). Order: score descending, then id '
     'ascending (ids are unique strings). cursor None means the first page; next_cursor is None after the last '
     'page. Walking all pages must return every item exactly once even when many items share a score. The '
     'cursor is an opaque string that must not contain titles or other item data besides score and id. '
     'size < 1 or a malformed cursor raises ValueError. Do not modify items.',
     {'pages.py': '''
def list_page(items, cursor, size):
    ordered = sorted(items, key=lambda x: -x['score'])
    start = 0 if cursor is None else int(cursor)
    page = ordered[start:start + size]
    nxt = str(start + size) if start + size < len(ordered) else None
    return page, nxt
'''}, '''
from copy import deepcopy
from pages import list_page
items=[dict(id='j%02d' % i,score=s,title='Secret title %d' % i) for i,s in
       enumerate([5,3,5,5,1,3,5,2,3,5,1])]
import random; random.Random(7).shuffle(items)
saved=deepcopy(items)
expected=[x['id'] for x in sorted(items,key=lambda x:(-x['score'],x['id']))]
for size in (1,2,3,4,11,20):
    seen=[]; cursor=None; steps=0
    while True:
        page,cursor=list_page(items,cursor,size); steps+=1
        assert len(page) <= size
        seen += [x['id'] for x in page]
        if cursor is None: break
        assert isinstance(cursor,str) and 'Secret' not in cursor and 'title' not in cursor
        assert steps < 50
    assert seen == expected, (size, seen)
assert items == saved
assert list_page([],None,3) == ([],None)
for bad in ((None,0),('garbage',2)):
    try: list_page(items,*bad)
    except ValueError: pass
    else: raise AssertionError('accepted %r' % (bad,))
''')
