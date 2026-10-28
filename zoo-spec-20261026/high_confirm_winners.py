"""Confirm the six owner-selected High winners on seeds22/33, H0/H3/H8.

Freeze actual screen options from results.jsonl; change seed only. Standard zoo
runner, native tools and original syshint adapter. No subsequent contest launch.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import zoo_night as z
import light_matrix  # installs the SAME syshint handling as high_matrix

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'runs/zoo-high-matrix-20261028'
WINNERS = {
    'k2-horizon-7b-q5': 'q5-template-syshint',
    'gpt-oss-20b': 'low',
    'glm-4.7-flash': 'agent-th-syshint',
    'qwen3.5-9b': 'nt-general',
    'nemotron-3.5-lightning-30b': 'vendor-th',
    'qwen3-coder-30b': 'vendor',
}
TASKS = ['H0', 'H3', 'H8']
SEEDS = [22, 33]


def plan():
    rows = [json.loads(s) for s in (SOURCE/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows) != 60:
        raise ValueError('Screen is not complete at60 rows')
    runs = []
    # Watching mode: every model finishes this task on both seeds before next task.
    for task in TASKS:
        for key, cfg in WINNERS.items():
            matches = [r for r in rows if r.get('model_key')==key and r.get('config_id')==cfg and r.get('task_id')==task and r.get('seed')==11]
            if len(matches) != 1 or matches[0].get('outcome') != 'episode':
                raise ValueError(f'Missing or duplicate screen cell: {key}/{cfg}/{task}')
            original = matches[0]
            for seed in SEEDS:
                r = copy.deepcopy({k:v for k,v in original.items() if k not in ('row','outcome','telemetry')})
                r['seed'] = seed
                r['options']['seed'] = seed
                r['purpose'] = 'High winner confirmation; frozen screen profile, new seed only'
                r['est_seconds'] = min(720, max(60, original['row'].get('wall_seconds',300)*1.3))
                runs.append(r)
    source_plan=json.loads(sorted(SOURCE.glob('plan-*.json'))[-1].read_text(encoding='utf-8'))
    return dict(originals={k:next(r['tag'] for r in runs if r['model_key']==k) for k in WINNERS}, missing=[],baked=[],runs=runs,
                source=str(SOURCE),source_sha256=hashlib.sha256((SOURCE/'results.jsonl').read_bytes()).hexdigest(),
                source_plan=str(sorted(SOURCE.glob('plan-*.json'))[-1]),ordering='task-major, model, seed')


def build_plan(inv, only=None, profile='all', levels=('high',)):
    p=plan()
    screen_inventory=json.loads(sorted(SOURCE.glob('inventory-*.json'))[-1].read_text(encoding='utf-8'))
    for tag in p['originals'].values():
        if tag not in inv:
            raise ValueError('Missing installed model: '+tag)
        if inv[tag]['digest'] != screen_inventory[tag]['digest']:
            raise ValueError('Digest differs from screen: '+tag)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--name',default='zoo-high-winners-confirm-20261028')
    args=parser.parse_args()
    p=plan()
    if args.plan:
        print(json.dumps(p,ensure_ascii=False,indent=2))
        return
    root=HERE.parent/'runs'/args.name
    if root.exists():
        raise ValueError('Run folder already exists; inspect before any resume')
    lock=HERE.parent/'high-winners-confirm.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        z.build_plan=build_plan
        z.run(argparse.Namespace(name=args.name,hours=12.0,new_deadline=False,models=None,profile='all',levels=['high']))
    finally:
        os.close(fd)
        lock.unlink()


if __name__=='__main__':
    main()
