"""Serial L/M/H tool episodes through Ollama. No arbitrary shell/filesystem tool.

Historical entry point retained; run_medium.py/run_high.py select other levels.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request, urllib.error
from suite import BASE, RUNS, CASES, prepare, digest

MODELS={'granite':'granite3.3:8b','gemma':'gemma4:e4b-t',
        'ministral':'ministral-3:8b-t','qwen4b':'qwen3.5:4b-t',
        'qwen9b':'qwen3.5:9b-t','k2q5':'k2-horizon:7b-q5-t',
        'k2q8':'k2-horizon:7b-q8-t','gptoss':'gpt-oss:20b'}
LEVELS={
    'light':dict(cases=['L0','L2','L3','L4','L5'],calls=12,seconds=300,num_ctx=8192,num_predict=2048,
                 models=['granite','gemma','ministral','qwen4b']),
    'medium':dict(cases=['M1','M2','M3','M4','M5'],calls=30,seconds=900,num_ctx=16384,num_predict=4096,
                 models=['granite','gemma','ministral','qwen4b','qwen9b','k2q5']),
    'high':dict(cases=['H1','H2','H3','H4','H5'],calls=60,seconds=720,num_ctx=16384,num_predict=8192,
               models=['k2q5','k2q8','qwen9b','gptoss'])}
SYSTEM='''You are a coding assistant in a controlled Light benchmark.
Each response must contain exactly one JSON object (no markdown fences).
Available actions:
{"action":"list_files"}
{"action":"read_file","path":"filename"}
{"action":"write_file","path":"filename","content":"complete file contents"}
{"action":"run_tests"} executes candidate_tests.py. Write this file yourself using assert statements
and imports from task modules. No shell, filesystem IO, networking, eval/exec or runtime introspection.
Only task modules, math, and json.loads/json.dumps are available to your Python code.
{"action":"final","report":"short factual report","tests_passed":true}
Use false for tests_passed if you have not observed successful run_tests. Your report must describe
checks actually performed. Read task source files yourself. Do not modify TASK.md or protected files.
No access outside this workspace. A tool error is returned as data; choose your own next action.
You have at most 12 tool calls and 5 minutes. Write a final action when finished.
'''


def system_prompt(level):
    spec=LEVELS[level]
    text=SYSTEM.replace('controlled Light benchmark',f'controlled {level.title()} benchmark')
    text=text.replace('12 tool calls and 5 minutes',f"{spec['calls']} tool calls and {spec['seconds']//60} minutes")
    if level!='light':
        text=text.replace('Only task modules, math, and json.loads/json.dumps are available to your Python code.',
            'Available Python imports: task modules, math, json.loads/json.dumps, copy.copy/deepcopy, '
            'threading.Lock/RLock/Barrier/Event/Thread, concurrent.futures.ThreadPoolExecutor, '
            'hashlib.sha256, uuid.uuid4. Private instance fields such as self._lock are allowed. '
            'Asyncio, unittest and arbitrary runtime introspection are unavailable; use assert-based tests.')
        text+='''\nAdditional action: {"action":"run_command","command":"python smoke.py"}.
Only three exact commands are accepted: python smoke.py (when a fixture provides it),
python candidate_tests.py, python -m definitely_missing_test_package.
The missing-module command actually executes and returns its error. No arbitrary shell commands.
Do not modify smoke.py, README.md, demo.py or run.log; these are fixed evidence/fixtures.
'''
    return text


# Tool protocol (2026-10-28). 'json': the model writes one JSON action as text (format=json), the historical
# contest protocol. 'native': the actions are offered as Ollama tools, as OpenCode does; models trained on
# native tool calling (gpt-oss, K2, glm, nemotron) answered the JSON protocol with tool calls and empty text.
# In both modes a native tool call is accepted and translated into the same action.
TOOL_MODE='json'
NATIVE_INTRO=('Use the provided tools: exactly one tool call per response, no prose. '
              'The tools mirror these actions (arguments as shown):')


def tool_specs(level):
    def tool(name,description,**props):
        return dict(type='function',function=dict(name=name,description=description,parameters=dict(
            type='object',properties={k:dict(type=t,description=d) for k,(t,d) in props.items()},
            required=list(props))))
    specs=[tool('list_files','List the files in the workspace.'),
           tool('read_file','Read one workspace file.',path=('string','plain file name')),
           tool('write_file','Create or replace one workspace file.',path=('string','plain file name'),
                content=('string','complete file contents')),
           tool('run_tests','Execute candidate_tests.py and return its output.'),
           tool('final','Finish the task.',report=('string','short factual report of checks actually performed'),
                tests_passed=('boolean','true only after an observed successful run_tests'))]
    if level!='light':
        specs.insert(4,tool('run_command','Run one of the documented fixed Python commands.',
                            command=('string','exact command text')))
    return specs


def action_from_tool_call(call):
    """A native tool call -> our action dict ("repo_browser.list_files", {"action": ...} wrappers included)."""
    fn=call.get('function') or {}
    args=fn.get('arguments') or {}
    if isinstance(args,str): args=json.loads(args) if args.strip() else {}
    if not isinstance(args,dict): raise ValueError('Tool arguments must be an object')
    if isinstance(args.get('action'),str): return dict(args)
    return dict(args,action=str(fn.get('name','')).rsplit('.',1)[-1])


def reasoning_profile(label,show,level,requested):
    caps=show.get('capabilities',[])
    desired=('off' if level=='light' else 'on') if requested=='auto' else requested
    if label=='granite':
        if desired not in ('off','on'): raise ValueError('Granite supports control on/off, not reasoning levels')
        if desired=='on' and 'thinking' not in show.get('template',''):
            raise ValueError('Granite control thinking not found in template')
        return dict(think_request=None,control_thinking=desired=='on',
                    reasoning_status='granite_control_on_requested' if desired=='on' else 'control_off')
    if label=='gptoss':
        # Installed template advertises ThinkLevel; GPT-OSS uses low/medium/high, not bool off.
        if desired=='off': raise ValueError('GPT-OSS off unsupported; choose low/medium/high')
        value=('high' if level=='high' else 'medium') if desired=='on' else desired
        if value not in ('low','medium','high') or 'ThinkLevel' not in show.get('template',''):
            raise ValueError('GPT-OSS native reasoning level not confirmed')
        return dict(think_request=value,control_thinking=False,reasoning_status=f'native_{value}_requested')
    if 'thinking' in caps:
        if desired not in ('off','on'): raise ValueError(f'{label}: discrete reasoning levels unverified; choose on/off')
        return dict(think_request=desired=='on',control_thinking=False,
                    reasoning_status=f'native_{desired}_requested')
    if requested not in ('auto','off'):
        raise ValueError(f'{label}: native reasoning control not advertised')
    return dict(think_request=None,control_thinking=False,reasoning_status='model_default_unverified')


def api(route, payload=None, timeout=15):
    req=urllib.request.Request('http://127.0.0.1:11434/api/'+route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as res: return json.load(res)
    except urllib.error.HTTPError as e:
        # keep Ollama's own reason (tool-call parse error, OOM...) in the row; before 2026-10-28 only "500" was kept
        body=e.read().decode('utf-8','replace')[:300]
        raise urllib.error.HTTPError(e.url,e.code,f'{e.msg}: {body}',e.headers,None) from None


def append(path, obj):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')


def worker(work,key,kind,timeout=15):
    res=subprocess.run([sys.executable,'-I','-B',str(BASE/'light_worker.py'),str(work),key,kind],
        cwd=BASE,capture_output=True,text=True,timeout=timeout)
    return dict(success=res.returncode==0,exit_code=res.returncode,output=res.stdout[-4000:],error=res.stderr[-3000:])


def parser_failure(exc):
    """HTTP 500 from Ollama's tool-call parser: the model produced an unparsable call (seen 2026-10-28)."""
    text=str(exc or '')
    return '500' in text and ('error parsing tool call' in text or 'arg_key and arg_value' in text)


def m4_recovery(commands):
    """M4 (Astra review 2026-10-28): the obsolete command really ran and failed, then the fallback smoke check
    really ran and passed after it; a generic 'tests ran' flag does not prove this."""
    old=[i for i,c in enumerate(commands) if c['command']=='python -m definitely_missing_test_package' and not c['success']]
    return bool(old) and any(c['command']=='python smoke.py' and c['success'] for c in commands[old[0]+1:])


def path_for(work, name):
    if not isinstance(name,str) or not name or Path(name).name!=name or ':' in name or name.startswith('.'):
        raise ValueError('Only plain relative filenames allowed')
    candidate=work/name
    if candidate.is_symlink() or candidate.is_junction(): raise ValueError('Links forbidden')
    candidate.resolve().relative_to(work.resolve())
    return candidate


def act(work,key,action):
    task=CASES[key]
    op=action.get('action')
    if op=='list_files': return dict(files=sorted(p.name for p in work.iterdir() if p.is_file()))
    if op=='read_file':
        path=path_for(work,action['path'])
        if path.stat().st_size>30000: raise ValueError('Read limit')
        return dict(content=path.read_text(encoding='utf-8'))
    if op=='write_file':
        path=path_for(work,action['path'])
        permitted=set(task['files'])|{'candidate_tests.py'}|({'answer.json'} if key=='L1' else set())
        protected={'TASK.md','smoke.py','README.md','demo.py','run.log'}|({'messages.py'} if key=='L1' else set())|({'candidate_tests.py'} if key=='L0' else set())
        if path.name not in permitted or path.name in protected: raise ValueError('Write not allowed')
        content=action['content']
        if path.suffix=='.json' and isinstance(content,(dict,list)):
            content=json.dumps(content,ensure_ascii=False)
        if not isinstance(content,str) or len(content)>30000: raise ValueError('Write limit')
        if path.suffix=='.py':
            from light_worker import validate
            validate(content)
        path.write_text(content,encoding='utf-8')
        return dict(written=path.name)
    if op=='run_tests': return worker(work,key,'candidate')
    if op=='run_command' and task['level'] in ('M','H'):
        command=action.get('command')
        if command=='python candidate_tests.py': return worker(work,key,'candidate')
        if command=='python smoke.py' and key=='M4': return worker(work,key,'smoke')
        if command=='python -m definitely_missing_test_package':
            res=subprocess.run([sys.executable,'-I','-m','definitely_missing_test_package'],
                cwd=BASE,capture_output=True,text=True,timeout=10)
            return dict(success=res.returncode==0,exit_code=res.returncode,output=res.stdout[-4000:],error=res.stderr[-3000:])
        raise ValueError('Command not allowed; use documented fixed Python commands')
    raise ValueError('Unknown tool/action')


def preflight(selected, output, level='light', overrides=None, reasoning='auto'):
    tags={m['name']:m for m in api('tags')['models']}
    passport={}
    for label in selected:
        tag=MODELS[label]
        if tag not in tags: raise ValueError(f'Missing model {tag}')
        show=api('show',dict(model=tag))
        template=show.get('template','')
        capabilities=show.get('capabilities',[])
        passport[label]=dict(tag=tag,digest=tags[tag]['digest'],details=show.get('details'),
            capabilities=capabilities,template_hash=digest(template),parameters=show.get('parameters'),
            **reasoning_profile(label,show,level,reasoning))
        (output/f'{label}-show.json').write_text(json.dumps(show,indent=2),encoding='utf-8')
    spec=LEVELS[level]
    options=dict(num_ctx=spec['num_ctx'],num_predict=spec['num_predict'],temperature=0.3,top_p=0.95,top_k=20,
                 min_p=0.0,repeat_penalty=1.0,presence_penalty=0.0,seed=42)
    options.update(overrides or {})
    result=dict(interface='ollama-custom-json-tools-v2',ollama=api('version'),models=passport,
        level=level,reasoning_argument=reasoning,
        runner_hash=digest(Path(__file__).read_text(encoding='utf-8')),
        worker_hash=digest((BASE/'light_worker.py').read_text(encoding='utf-8')),
        cases_hash=digest((BASE/'cases.py').read_text(encoding='utf-8')),
        config_id=f"{level}-{options['num_ctx']}ctx-{options['num_predict']}out-{reasoning}",
        limits=dict(calls=spec['calls'],seconds=spec['seconds']),options=options,
        sampling='common controlled profile; not official per-model optimum',
        runtime='restricted Python subset; not an OS security sandbox')
    (output/'configuration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def episode(output,label,key,configuration):
    run=output.name+'-'+label
    root=RUNS/run/key
    if root.exists(): raise ValueError(f'Partial task exists: {root}; inspect, do not overwrite')
    with contextlib.redirect_stdout(io.StringIO()): prepare(run,key)
    work=root/'workspace'; transcript=root/'transcript.jsonl'
    info=configuration['models'][label]
    level=configuration.get('level','light'); spec=LEVELS[level]
    instructions=system_prompt(level)
    native=TOOL_MODE=='native'
    if native:
        instructions=instructions.replace('Each response must contain exactly one JSON object (no markdown fences).\n'
                                          'Available actions:',NATIVE_INTRO)
        instructions+='Finish by calling final.\n'
    if key=='L0':
        instructions=instructions.replace('Write this file yourself using assert statements',
            'For this task candidate_tests.py is already supplied; do not edit it. It uses assert statements')
    task_text=(work/'TASK.md').read_text()
    if info.get('control_thinking'):
        # Granite injects its thinking instructions only if no custom system message is present.
        messages=[dict(role='control',content='thinking'),
                  dict(role='user',content=instructions+'\n\n'+task_text)]
    else:
        messages=[dict(role='system',content=instructions),dict(role='user',content=task_text)]
    calls=0; seen_test_success=False; final=None; violations=[]; infra_error=None; time_limit=False; commands_run=[]
    format_errors=0; native_calls=0; parse_failures=0; raise_or=None
    sums={k:0 for k in ['load_duration','prompt_eval_duration','eval_duration','prompt_eval_count','eval_count']}
    began=time.perf_counter(); started=datetime.now(timezone.utc).isoformat(); truncated=False
    # A final report may require one extra generation after 12 calls, but no 13th tool is executed.
    for turn in range(spec['calls']+2):
        left=spec['seconds']-(time.perf_counter()-began)
        if left<=0: time_limit=True; break   # owner 2026-10-28: the time limit is a model outcome, not infra
        request=dict(model=info['tag'],messages=messages,stream=False,
                     options=configuration['options'],keep_alive='5m')
        if native: request['tools']=tool_specs(level)
        else: request['format']='json'
        if info['think_request'] is not None: request['think']=info['think_request']
        append(transcript,dict(event='request',turn=turn,payload=request))
        response=None
        for attempt in range(3):
            try:
                response=api('chat',request,timeout=max(1,spec['seconds']-(time.perf_counter()-began)))
                break
            except Exception as exc:
                if not parser_failure(exc) or time.perf_counter()-began>=spec['seconds']: raise_or=exc; break
                # Ollama could not parse the model's tool call (gpt-oss: raw='', GLM: arg_key/arg_value mismatch):
                # the model's output, not infra (owner 2026-10-28). Retry with another seed, at most twice.
                parse_failures+=1
                append(transcript,dict(event='parser_failure',turn=turn,attempt=attempt,error=str(exc)[:400]))
                request=dict(request,options=dict(request['options'],seed=(request['options'].get('seed') or 0)+1000+attempt))
                raise_or=exc
        if response is None and parser_failure(raise_or):
            format_errors+=1   # three unparsable tool calls in a row: tell the model and go on, it costs the turn
            messages.append(dict(role='user',content='Your last tool call could not be parsed (invalid or empty '
                                 'arguments). Call exactly one tool with valid JSON arguments.'))
            continue
        if response is None:
            exc=raise_or
            if 'timed out' in str(exc) and time.perf_counter()-began>=spec['seconds']-2: time_limit=True
            else: infra_error=f'{type(exc).__name__}: {exc}'
            break
        append(transcript,dict(event='response',turn=turn,payload=response))
        for field in sums: sums[field]+=response.get(field,0) or 0
        truncated |= response.get('done_reason')=='length'
        message=response.get('message',{})
        content=message.get('content','')
        tool_calls=message.get('tool_calls') or []
        if native and tool_calls:
            messages.append(dict(role='assistant',content=content,tool_calls=tool_calls[:1]))
        else:
            messages.append(dict(role='assistant',content=content))
        # gpt-oss must see its own reasoning between tool calls until the final answer (OpenAI raw-CoT guide;
        # its Ollama template renders Thinking after the last user message). Before 2026-10-28 it was dropped.
        # Other models keep the old history so earlier comparisons stay valid.
        if str(info['tag']).startswith('gpt-oss') and message.get('thinking'):
            messages[-1]['thinking']=message['thinking']
        try:
            action_text=content.strip()
            if tool_calls and (native or not action_text):
                # A native tool call is the move (json mode: only when the text is empty, as gpt-oss answers).
                native_calls+=1
                action_text=json.dumps(action_from_tool_call(tool_calls[0]))
            if info.get('control_thinking'):
                if action_text.startswith('<think>') and '</think>' in action_text:
                    action_text=action_text.split('</think>',1)[1].strip()
                if action_text.startswith('<response>') and action_text.endswith('</response>'):
                    action_text=action_text[len('<response>'):-len('</response>')].strip()
            action=json.loads(action_text)
            if not isinstance(action,dict): raise ValueError('Expected JSON object')
            if 'action' not in action and isinstance(action.get('report'),str):
                action['action']='final'   # gpt-oss in native mode: {"report": ...} as text = the final move
            if action.get('action')=='final':
                if not isinstance(action.get('report'),str) or not action['report'].strip():
                    raise ValueError('Final needs a nonempty report')
                if 'tests_passed' not in action:
                    action['tests_passed']=seen_test_success
                    action['tests_passed_source']='observed_run_tests'
                elif type(action['tests_passed']) is not bool:
                    raise ValueError('tests_passed must be boolean when supplied')
                final=action; break
            if calls>=spec['calls']: raise ValueError('Tool budget exhausted')
            calls+=1
            result=act(work,key,action)
            if action.get('action') in ('run_tests','run_command') and result.get('success'): seen_test_success=True
            if action.get('action')=='write_file': seen_test_success=False
            if action.get('action')=='run_command': commands_run.append(dict(command=action.get('command'),success=bool(result.get('success'))))
        except Exception as exc:
            calls+=0 if content.strip().startswith('{') else 1
            result=dict(error=f'{type(exc).__name__}: {exc}')
            if isinstance(exc,json.JSONDecodeError) or str(exc)=='Expected JSON object': format_errors+=1
            if 'Write not allowed' in str(exc) or 'Links forbidden' in str(exc): violations.append(str(exc))
        append(transcript,dict(event='tool_result',turn=turn,result=result))
        if native and tool_calls:
            messages.append(dict(role='tool',content=json.dumps(result),
                                 tool_name=str((tool_calls[0].get('function') or {}).get('name',''))))
        else:
            messages.append(dict(role='user',content='Tool result: '+json.dumps(result)))
        if calls>=spec['calls']:
            messages.append(dict(role='user',content='Tool budget exhausted; give final action now.'))
    wall=time.perf_counter()-began
    try: acceptance=worker(work,key,'acceptance')
    except Exception as exc: acceptance=dict(success=False,error=str(exc)); infra_error=infra_error or 'acceptance_runtime_error'
    # Candidate code success and workflow completion are provisional until report is reviewed.
    row=dict(model=info['tag'],digest=info['digest'],case_id=key,
        config_id=configuration.get('config_id','light-common8k2k'),seed=configuration['options'].get('seed',42),
        started_at=started,ended_at=datetime.now(timezone.utc).isoformat(),wall_seconds=round(wall,3),
        tool_calls=calls,usage=sums,truncation=truncated,infra_error=infra_error,time_limit=time_limit,final=final,
        tool_mode=TOOL_MODE,native_tool_calls=native_calls,format_errors=format_errors,parser_failures=parse_failures,
        code_pass=acceptance['success'],acceptance=acceptance,violations=violations,
        observed_candidate_test_pass=seen_test_success,commands_run=commands_run,
        m4_recovery_observed=(None if key!='M4' else m4_recovery(commands_run)),
        automated_workflow_pass=bool(acceptance['success'] and final and not infra_error and not violations
            and seen_test_success and final.get('tests_passed') and key!='M4' or bool(acceptance['success'] and final
            and not infra_error and not violations and seen_test_success and final.get('tests_passed') and m4_recovery(commands_run))),
        final_pass=None,review_required=True,transcript=str(transcript))
    (root/'episode.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
    append(output/'episodes.jsonl',row)
    print(f"{label} {key}: code_pass={row['code_pass']} time={wall:.1f}s review_required",flush=True)
    return row


def main(default_level='light'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level',choices=LEVELS,default=default_level)
    p.add_argument('--models',nargs='+',choices=MODELS)
    p.add_argument('--run',required=True)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--reasoning',choices=['auto','off','on','low','medium','high'],default='auto')
    p.add_argument('--num-ctx',type=int)
    p.add_argument('--num-predict',type=int)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    args.models=args.models or LEVELS[args.level]['models']
    overrides={'seed':args.seed}
    for field,value in [('num_ctx',args.num_ctx),('num_predict',args.num_predict)]:
        if value is not None:
            if value<=0: p.error(f'{field} must be positive')
            overrides[field]=value
    if not args.run or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run):
        raise SystemExit('Invalid run name')
    output=RUNS/args.run
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'supervisor.lock'
    import os
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise SystemExit('Run lock exists; inspect its PID before resuming')
    try:
        os.write(fd,str(os.getpid()).encode())
        path=output/'configuration.json'
        configuration=json.loads(path.read_text()) if path.exists() else preflight(args.models,output,args.level,overrides,args.reasoning)
        if configuration.get('level')!=args.level or configuration.get('reasoning_argument')!=args.reasoning:
            raise ValueError('Category or reasoning differs from saved configuration; use a new run')
        for field,value in overrides.items():
            if configuration['options'].get(field)!=value: raise ValueError('Options changed; use a new run')
        for name,file in [('runner_hash',Path(__file__)),('worker_hash',BASE/'light_worker.py'),('cases_hash',BASE/'cases.py')]:
            if configuration.get(name)!=digest(file.read_text(encoding='utf-8')):
                raise ValueError('Runner, worker or tasks changed; use a new run')
        tags={m['name']:m for m in api('tags')['models']}
        for label in args.models:
            info=configuration['models'][label]
            if tags.get(info['tag'],{}).get('digest')!=info['digest']:
                raise ValueError(f'Model changed since preflight: {label}; use a new run')
        if args.preflight_only:
            print(f'PREFLIGHT_OK {output} (no model generation)'); return
        old=[]
        if (output/'episodes.jsonl').exists(): old=[json.loads(x) for x in (output/'episodes.jsonl').read_text().splitlines()]
        done={(r['model'],r['case_id']) for r in old}
        (output/'status.json').write_text(json.dumps(dict(state='running',completed=len(done))))
        # Rotate models each case to reduce fixed order effects. One inference at a time.
        for i,key in enumerate(LEVELS[args.level]['cases']):
            models=args.models[i%len(args.models):]+args.models[:i%len(args.models)]
            for label in models:
                if (MODELS[label],key) in done: continue
                row=episode(output,label,key,configuration)
                done.add((MODELS[label],key))
                (output/'status.json').write_text(json.dumps(dict(state='running',completed=len(done),last_case=key,last_model=label)))
                if row['infra_error']:
                    (output/'status.json').write_text(json.dumps(dict(state='stopped',completed=len(done),
                        last_case=key,last_model=label,error=row['infra_error'])))
                    raise RuntimeError('Infrastructure failure recorded; diagnose before continuing')
        (output/'status.json').write_text(json.dumps(dict(state='completed',completed=len(done),review_required=True)))
        print(f'{args.level.upper()}_EXECUTION_COMPLETE; review transcripts before choosing champion',flush=True)
    finally:
        os.close(fd); lock.unlink(missing_ok=True)


if __name__=='__main__': main()
