"""Additional verified Light tasks for the Qwen 4B vs Granite 4.2 8B round."""
from cases import case

case('L6', 'Classify a retry without amplifying an outage',
     'Implement should_retry(status, error). Retry status 429 and all 5xx; never retry other 4xx. '
     'When status is None, retry only error values timeout or connection. Unknown errors must be false. '
     'Preserve the function signature. Read run.log as evidence before editing.',
     {'retry.py': 'def should_retry(status, error):\n    return True\n',
      'run.log': 'WARN status=401 retry=false\nWARN error=timeout retry=true\n'}, '''
from retry import should_retry
for status in [500, 501, 599, 429]: assert should_retry(status,None) is True
for status in [0,200,400,401,403,404,418]: assert should_retry(status,None) is False
assert should_retry(None,'timeout') is True and should_retry(None,'connection') is True
for error in [None,'dns','permission','']: assert should_retry(None,error) is False
''')

case('L7', 'Trace a setting through three components',
     'Add optional region="worldwide" to api.submit, queueing.payload and worker.execute. '
     'Allowed values are worldwide, europe, americas; invalid values raise ValueError before enqueue. '
     'Legacy queue records without region mean worldwide. Preserve all old call signatures and owner.',
     {'api.py': 'from queueing import payload\ndef submit(owner, enqueue):\n    return enqueue(payload(owner))\n',
      'queueing.py': 'def payload(owner):\n    return {"owner":owner}\n',
      'worker.py': 'def execute(task, collect):\n    return collect(task["owner"], "worldwide")\n'}, '''
from api import submit
from worker import execute
seen=[]
def enqueue(x): seen.append(x); return x
for region in ['worldwide','europe','americas']:
 t=submit('u',enqueue,region=region)
 assert t=={'owner':'u','region':region}
 assert execute(t,lambda owner,value:(owner,value))==('u',region)
assert submit('legacy',enqueue)=={'owner':'legacy','region':'worldwide'}
assert execute({'owner':'legacy'},lambda owner,value:value)=='worldwide'
n=len(seen)
for bad in ['',None,'asia',1]:
 try: submit('u',enqueue,region=bad)
 except ValueError: pass
 else: raise AssertionError('invalid region')
assert len(seen)==n
''')

case('L8', 'Choose the smallest safe recovery action',
     'Implement next_recovery(task). Return one of wait, retry, escalate. Running with a live lease means wait. '
     'Queued means retry. Failed means retry only when attempts < max_attempts; otherwise escalate. '
     'Unknown state must raise ValueError. Do not mutate the input.',
     {'recovery.py': 'def next_recovery(task):\n    return "retry"\n',
      'incident.txt': 'Never claim a second task while the original lease is live.\n'}, '''
from copy import deepcopy
from recovery import next_recovery
data=[({'state':'running','lease_live':True,'attempts':1,'max_attempts':3},'wait'),
      ({'state':'queued','lease_live':False,'attempts':0,'max_attempts':3},'retry'),
      ({'state':'failed','lease_live':False,'attempts':1,'max_attempts':3},'retry'),
      ({'state':'failed','lease_live':False,'attempts':3,'max_attempts':3},'escalate')]
for task,want in data:
 old=deepcopy(task); assert next_recovery(task)==want and task==old
try: next_recovery({'state':'mystery','lease_live':False,'attempts':0,'max_attempts':1})
except ValueError: pass
else: raise AssertionError('unknown state accepted')
''')

case('L9', 'Read an unfamiliar configuration format',
     'Implement read_settings(text) without external packages. Input is lines key=value; blank lines and lines '
     'starting # are ignored; surrounding whitespace is ignored. Required keys endpoint and enabled. enabled is '
     'true/false and returns a boolean. Duplicate keys, malformed lines, unknown keys or missing keys raise ValueError.',
     {'config_reader.py': 'def read_settings(text):\n    return {}\n',
      'settings.ini': '# service settings\nendpoint = http://127.0.0.1:11434\nenabled = true\n'}, '''
from config_reader import read_settings
assert read_settings('endpoint=http://x\\nenabled=false\\n')=={'endpoint':'http://x','enabled':False}
assert read_settings(' # a\\n endpoint = url \\n\\nenabled = true\\n')=={'endpoint':'url','enabled':True}
for text in ['', 'endpoint=x\\n', 'enabled=true\\n', 'endpoint=x\\nenabled=yes\\n',
             'endpoint=x\\nenabled=true\\nextra=1\\n', 'endpoint=x\\nenabled=true\\nendpoint=y\\n', 'bad\\n']:
 try: read_settings(text)
 except ValueError: pass
 else: raise AssertionError(text)
''')

case('L10', 'Find a stuck task from incomplete evidence',
     'Implement is_stuck(events, now, grace). Each event has job_id, state and at. Return sorted job_ids whose '
     'latest event is running and older than grace seconds. queued and completed are never stuck. Equal to grace '
     'is not stuck. Do not mutate events. Read events.log before editing.',
     {'monitor.py': 'def is_stuck(events, now, grace):\n    return []\n',
      'events.log': 'job=a running at=10; job=a completed at=20; job=b running at=5\n'}, '''
from copy import deepcopy
from monitor import is_stuck
events=[{'job_id':'a','state':'running','at':10},{'job_id':'a','state':'completed','at':20},
        {'job_id':'b','state':'queued','at':1},{'job_id':'c','state':'running','at':10},
        {'job_id':'d','state':'running','at':15},{'job_id':'b','state':'running','at':2}]
old=deepcopy(events)
assert is_stuck(events,20,5)==['b','c'] and events==old
assert is_stuck([],20,5)==[] and is_stuck(events,15,5)==['b']
''')

case('L11', 'Normalize job URLs before deduplication',
     'Implement canonical_url(url). Accept only http or https URLs. Lowercase scheme and host; remove fragment; '
     'remove a single trailing slash from non-root paths; preserve query exactly. Reject non-strings, missing host, '
     'and other schemes with ValueError. The restricted adapter has no URL library: implement it with string operations.',
     {'urls.py': 'def canonical_url(url):\n    return url\n',
      'README.md': 'Duplicates differ mostly by host case, fragments and a trailing slash.\n'}, '''
from urls import canonical_url
assert canonical_url('HTTPS://EXAMPLE.COM/jobs/42/#top')=='https://example.com/jobs/42'
assert canonical_url('http://Example.com/')=='http://example.com/'
assert canonical_url('https://EXAMPLE.com/a/?q=A#x')=='https://example.com/a?q=A'
assert canonical_url('https://example.com/a//')=='https://example.com/a/'
for value in [None,3,'/relative','ftp://example.com/a','https:///missing']:
 try: canonical_url(value)
 except ValueError: pass
 else: raise AssertionError(value)
''')
