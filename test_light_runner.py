"""Operator tests. No LLM calls or model solutions exposed to contestants."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
from suite import prepare, RUNS, CASES
from selfcheck import REFS
from run_light import act, worker, episode, path_for


class AdapterTests(unittest.TestCase):
    def setUp(self): self.run='adapter-validation-'+uuid.uuid4().hex[:8]

    def fresh(self,key):
        with contextlib.redirect_stdout(io.StringIO()): prepare(self.run,key)
        return RUNS/self.run/key/'workspace'

    def test_five_light_references_and_broken(self):
        for key in ['L1','L2','L3','L4','L5']:
            work=self.fresh(key)
            self.assertFalse(worker(work,key,'acceptance')['success'],key)
            for name,code in REFS[key].items():
                act(work,key,dict(action='write_file',path=name,content=code.lstrip()))
            self.assertTrue(worker(work,key,'acceptance')['success'],key)

    def test_paths_and_protected_writes(self):
        work=self.fresh('L1')
        for name in ['../cases.py','C:/Windows/win.ini','..','file:stream','.hidden']:
            with self.assertRaises(ValueError): path_for(work,name)
        with self.assertRaises(ValueError): act(work,'L1',dict(action='write_file',path='messages.py',content=''))
        with self.assertRaises(ValueError): act(work,'L1',dict(action='write_file',path='TASK.md',content=''))

    def test_baseline_supplied_tests(self):
        work=self.fresh('L0')
        self.assertFalse(worker(work,'L0','acceptance')['success'])
        with self.assertRaises(ValueError):
            act(work,'L0',dict(action='write_file',path='candidate_tests.py',content=''))
        act(work,'L0',dict(action='write_file',path='calculator.py',content='def add(a, b):\n    return a + b\n'))
        self.assertTrue(worker(work,'L0','candidate')['success'])
        self.assertTrue(worker(work,'L0','acceptance')['success'])

    def test_runtime_refuses_host_access(self):
        work=self.fresh('L2')
        for code in ['open("secret")','eval("1")','x.__class__', '__import__("os")']:
            with self.assertRaises(ValueError): act(work,'L2',dict(action='write_file',path='candidate_tests.py',content=code))
        act(work,'L2',dict(action='write_file',path='candidate_tests.py',content='import os\nos.listdir()'))
        self.assertFalse(worker(work,'L2','candidate')['success'])

    def test_complete_tool_episode_without_model(self):
        output=RUNS/(self.run+'-episode'); output.mkdir()
        config=dict(models={'granite':dict(tag='fake-fixture',digest='fake',think_request=None)},options={})
        actions=[dict(action='read_file',path='fetcher.py'),
            dict(action='write_file',path='fetcher.py',content=REFS['L2']['fetcher.py']),
            dict(action='write_file',path='candidate_tests.py',content='from fetcher import fetch\nassert callable(fetch)'),
            dict(action='run_tests'),dict(action='final',report='Edited fetcher and ran my checks.',tests_passed=True)]
        import json
        responses=iter(dict(message=dict(content=json.dumps(a)),done_reason='stop') for a in actions)
        with patch('run_light.api',side_effect=lambda *a,**kw:next(responses)):
            with contextlib.redirect_stdout(io.StringIO()): result=episode(output,'granite','L2',config)
        self.assertTrue(result['code_pass'])
        self.assertTrue(result['automated_workflow_pass'])
        self.assertEqual(result['tool_calls'],4)
        self.assertIsNone(result['final_pass'])

    def test_final_uses_observed_test_result_when_flag_omitted(self):
        output=RUNS/(self.run+'-optional-final'); output.mkdir()
        config=dict(models={'granite':dict(tag='fake-fixture',digest='fake',think_request=None)},options={})
        import json
        actions=[dict(action='read_file',path='calculator.py'),
            dict(action='write_file',path='calculator.py',content='def add(a, b):\n    return a + b\n'),
            dict(action='run_tests'),dict(action='final',report='Fixed addition and ran supplied checks.')]
        responses=iter(dict(message=dict(content=json.dumps(a)),done_reason='stop') for a in actions)
        with patch('run_light.api',side_effect=lambda *a,**kw:next(responses)):
            with contextlib.redirect_stdout(io.StringIO()): result=episode(output,'granite','L0',config)
        self.assertTrue(result['automated_workflow_pass'])
        self.assertEqual(result['final']['tests_passed_source'],'observed_run_tests')


class NativeToolTests(unittest.TestCase):
    """2026-10-28: gpt-oss answered the JSON protocol with native tool calls and empty text."""
    actions=[dict(action='read_file',path='calculator.py'),
             dict(action='write_file',path='calculator.py',content='def add(a, b):\n    return a + b\n'),
             dict(action='run_tests'),dict(action='final',report='Fixed addition and ran supplied checks.',tests_passed=True)]

    def run_episode(self,replies,mode):
        import json
        import run_light
        output=RUNS/('native-tools-'+uuid.uuid4().hex[:8]); output.mkdir()
        config=dict(models={'granite':dict(tag='fake-fixture',digest='fake',think_request=None)},options={})
        requests=[]
        def fake(route,payload=None,timeout=15):
            requests.append(json.loads(json.dumps(payload)))
            return next(replies)
        with patch.object(run_light,'TOOL_MODE',mode), patch('run_light.api',side_effect=fake):
            with contextlib.redirect_stdout(io.StringIO()): result=episode(output,'granite','L0',config)
        return result,requests

    @staticmethod
    def call(name,args):
        return dict(message=dict(content='',tool_calls=[dict(function=dict(name=name,arguments=args))]),done_reason='stop')

    def test_json_mode_accepts_gpt_oss_style_tool_calls(self):
        replies=iter([self.call('repo_browser.read_file',dict(path='calculator.py')),
                      self.call('developer',dict(action='write_file',path='calculator.py',
                                                 content='def add(a, b):\n    return a + b\n')),
                      self.call('run_tests',{}),
                      self.call('final',dict(report='Fixed addition and ran supplied checks.',tests_passed=True))])
        result,requests=self.run_episode(replies,'json')
        self.assertTrue(result['automated_workflow_pass'])
        self.assertEqual((result['native_tool_calls'],result['format_errors']),(4,0))
        self.assertEqual(requests[0]['format'],'json')
        self.assertNotIn('tools',requests[0])

    def test_native_mode_offers_tools_and_returns_tool_messages(self):
        replies=iter([self.call(a['action'],{k:v for k,v in a.items() if k!='action'}) for a in self.actions])
        result,requests=self.run_episode(replies,'native')
        self.assertTrue(result['automated_workflow_pass'])
        self.assertNotIn('format',requests[0])
        self.assertEqual({t['function']['name'] for t in requests[0]['tools']},
                         {'list_files','read_file','write_file','run_tests','final'})
        self.assertIn('Use the provided tools',requests[0]['messages'][0]['content'])
        last=requests[-1]['messages']
        self.assertEqual(last[-1]['role'],'tool')
        self.assertEqual(last[-2]['tool_calls'][0]['function']['name'],'run_tests')
        self.assertEqual(result['tool_mode'],'native')

    def test_native_mode_still_accepts_a_json_text_move(self):
        import json
        replies=iter(dict(message=dict(content=json.dumps(a)),done_reason='stop') for a in self.actions)
        result,_=self.run_episode(replies,'native')
        self.assertTrue(result['automated_workflow_pass'])
        self.assertEqual(result['native_tool_calls'],0)

    def test_report_without_action_is_the_final_move(self):
        import json
        moves=[json.dumps(a) for a in self.actions[:3]]+[json.dumps(dict(report='Fixed addition and ran checks.',tests_passed=True))]
        replies=iter(dict(message=dict(content=m),done_reason='stop') for m in moves)
        result,_=self.run_episode(replies,'native')
        self.assertTrue(result['automated_workflow_pass'])

    def test_unparseable_text_counts_as_format_error(self):
        import json
        replies=iter([dict(message=dict(content='Let me look at the files.'),done_reason='stop')]+
                     [dict(message=dict(content=json.dumps(a)),done_reason='stop') for a in self.actions])
        result,_=self.run_episode(replies,'json')
        self.assertEqual(result['format_errors'],1)
        self.assertTrue(result['automated_workflow_pass'])


if __name__=='__main__': unittest.main()
