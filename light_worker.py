"""Execute synthetic tasks with mediated imports/builtins. Not an OS sandbox.

Historical filename retained for Light compatibility; supports L/M/H tasks.
"""
import ast
import builtins
import json
from pathlib import Path
import sys
import types

BANNED = {'open','eval','exec','compile','getattr','setattr','delattr','globals','locals','vars',
          'breakpoint','input','help','exit','quit','memoryview'}


def validate(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise ValueError('Runtime introspection attributes are unavailable in benchmark adapter')
        if isinstance(node, ast.Name) and (node.id in BANNED or node.id.startswith('__') and node.id != '__name__'):
            raise ValueError(f'Unavailable runtime operation: {node.id}')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError('Relative imports unavailable in Light adapter')
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
            raise ValueError('Python asyncio unavailable in Light adapter')
    return tree


def execute(work, task, code):
    allowed = {Path(name).stem:name for name in task['files'] if name.endswith('.py')}
    allowed['candidate_tests'] = 'candidate_tests.py'
    extended = task['level'] in ('M','H')
    modules = {}
    original_import = builtins.__import__
    public = {name:getattr(builtins,name) for name in (
        'abs','all','any','bool','bytes','callable','chr','dict','enumerate','filter','float','format',
        'frozenset','hash','int','isinstance','issubclass','iter','len','list','map','max','min','next',
        'object','ord','pow','print','property','range','repr','reversed','round','set','slice','sorted',
        'staticmethod','classmethod','str','sum','super','tuple','type','zip','Exception','ValueError',
        'TypeError','KeyError','RuntimeError','TimeoutError','AssertionError','PermissionError','IndexError',
        'StopIteration','NotImplementedError','ZeroDivisionError','AttributeError')}
    public['__build_class__'] = builtins.__build_class__

    def importer(name, globals=None, locals=None, fromlist=(), level=0, *, trusted=False):
        # Trust is a closure parameter, never a variable candidate code can overwrite.
        if level: raise ValueError('Relative imports unavailable')
        if name in allowed:
            if name not in modules:
                source=(work/allowed[name]).read_text(encoding='utf-8')
                validate(source)
                module=types.ModuleType(name)
                module.__dict__.update(__builtins__={**public,'__import__':importer})
                modules[name]=module
                exec(compile(source,allowed[name],'exec'),module.__dict__)
            return modules[name]
        if not trusted:
            if name=='math':
                import math
                return math
            if name=='json': return types.SimpleNamespace(loads=json.loads,dumps=json.dumps)
            if extended and name=='copy':
                from copy import copy, deepcopy
                return types.SimpleNamespace(copy=copy,deepcopy=deepcopy)
            if extended and name=='threading':
                from threading import Lock,RLock,Barrier,Event,Thread
                return types.SimpleNamespace(Lock=Lock,RLock=RLock,Barrier=Barrier,Event=Event,Thread=Thread)
            if extended and name in ('concurrent','concurrent.futures'):
                from concurrent.futures import ThreadPoolExecutor
                futures=types.SimpleNamespace(ThreadPoolExecutor=ThreadPoolExecutor)
                return futures if fromlist else types.SimpleNamespace(futures=futures)
            if extended and name=='hashlib':
                from hashlib import sha256
                return types.SimpleNamespace(sha256=sha256)
            if extended and name=='uuid':
                from uuid import uuid4
                return types.SimpleNamespace(uuid4=uuid4)
            raise ValueError(f'Import unavailable in benchmark adapter: {name}')
        return original_import(name,globals,locals,fromlist,level)

    def operator_import(name,globals=None,locals=None,fromlist=(),level=0):
        return importer(name,globals,locals,fromlist,level,trusted=True)

    trusted = dict(ROOT=work,__builtins__={**vars(builtins),'__import__':operator_import})
    # The checker is operator-authored; contestant modules get restricted globals above.
    exec(compile(code,'operator_checks','exec'),trusted)


if __name__=='__main__':
    # sys.path contains only the operator directory; candidate imports are mediated.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from cases import CASES
    # Optional extension module only registers extra frozen fixtures; it does
    # not expose host capabilities to candidate code.
    try:
        import light10_cases  # noqa: F401
    except ImportError:
        pass
    root=Path(sys.argv[1]).resolve()
    key=sys.argv[2]
    task=CASES[key]
    if key not in CASES: raise SystemExit('Unknown benchmark task')
    if sys.argv[3]=='acceptance':
        checks=task['checks']
    elif sys.argv[3]=='smoke':
        if key!='M4': raise SystemExit('No fixture smoke command for this task')
        checks='import smoke'
    else:
        checks=(root/'candidate_tests.py').read_text(encoding='utf-8')
        validate(checks)
        # Execute candidate tests as a contestant module, not with operator globals.
        checks='import candidate_tests'
    execute(root,task,checks)
    print('CHECKS_PASSED')
