"""Prepare isolated candidates and verify round 2; never invokes an LLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from cases import CASES

BASE = Path(__file__).resolve().parent
RUNS = BASE / 'runs'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prepare(run, key):
    if not run or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run):
        raise ValueError('run must contain letters, numbers, hyphen or underscore')
    task = CASES[key]
    root = RUNS / run / key
    if root.exists():
        raise FileExistsError(f'Already exists, never overwrite a candidate: {root}')
    work = root / 'workspace'
    work.mkdir(parents=True)
    for name, source in task['files'].items():
        (work / name).write_text(source, encoding='utf-8')
    prompt = (f"Task {key}: {task['title']}\n\n{task['requirement']}\n\n"
              "Work only inside the supplied workspace. Use Python standard library only. "
              "Inspect the files, implement the requirements, run available checks, and give a short final report "
              "describing changes and checks actually performed. Do not modify TASK.md, smoke.py or the evaluation "
              "infrastructure. Do not install packages or access the internet. Do not claim a check passed without running it.\n")
    (work / 'TASK.md').write_text(prompt, encoding='utf-8')
    protected = ['TASK.md'] + (['messages.py'] if key == 'L1' else []) + (['smoke.py'] if key == 'M4' else []) + (['candidate_tests.py'] if key == 'L0' else [])
    manifest = dict(case=key, suite_version='round2-reviewed-v3', created_at=datetime.now(timezone.utc).isoformat(),
                    requirement_hash=digest(task['requirement']),
                    fixture_hash=digest(json.dumps(task['files'], sort_keys=True)),
                    checks_hash=digest(task['checks']),
                    protected={name:digest((work/name).read_text(encoding='utf-8')) for name in protected})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(work)


def check(root, selftest=False):
    root = root.resolve()
    root.relative_to(RUNS.resolve())
    meta = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    task = CASES[meta['case']]
    work = root/'workspace'
    if meta['checks_hash'] != digest(task['checks']):
        raise ValueError('Verifier changed since preparation; use a new run')
    if meta.get('requirement_hash') != digest(task['requirement']):
        raise ValueError('Old or changed task instructions; prepare a fresh run')
    violations=[]
    for name, expected in meta['protected'].items():
        if not (work/name).is_file() or digest((work/name).read_text(encoding='utf-8')) != expected:
            violations.append(f'Protected file changed: {name}')
    code = "import sys\nfrom pathlib import Path\nROOT=Path(sys.argv[1])\nsys.path.insert(0,str(ROOT))\n" + task['checks']
    started=time.perf_counter()
    try:
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(work)], cwd=work,
                              capture_output=True,text=True,timeout=20,
                              env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        row=dict(case=meta['case'], code_pass=result.returncode==0 and not violations,
                 exit_code=result.returncode, output=result.stdout[-6000:],error=result.stderr[-6000:])
    except subprocess.TimeoutExpired:
        row=dict(case=meta['case'],code_pass=False,exit_code=None,error='Verifier timeout (20 seconds)')
    row.update(violations=violations,checked_at=datetime.now(timezone.utc).isoformat(),
               verification_seconds=round(time.perf_counter()-started,4),selftest=selftest)
    with (root/'checks.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False))
    return row['code_pass']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    prepare_p=sub.add_parser('prepare'); prepare_p.add_argument('--run',required=True)
    prepare_p.add_argument('--case',required=True,choices=CASES)
    check_p=sub.add_parser('check'); check_p.add_argument('path',type=Path)
    args=p.parse_args()
    if args.command=='list':
        for task in CASES.values(): print(task['id'], task['title'], '(bridge)' if task['bridge'] else '')
    elif args.command=='prepare': prepare(args.run,args.case)
    else: sys.exit(0 if check(args.path) else 1)


if __name__=='__main__': main()
