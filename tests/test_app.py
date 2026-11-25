import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import app
import benchmark
import provider
from collections import Counter


class AppTest(unittest.TestCase):
    """Runs the real HTTP handler on a random local port."""

    def setUp(self):
        quiet = type('QuietHandler', (app.Handler,), {'log_message': lambda *a: None})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), quiet)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return json.load(r)

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.load(e)

    def test_status_has_no_predictions(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            status = self.get('/api/status')
        self.assertNotIn('rows', status)
        self.assertFalse(status['live_available'])

    def test_published_evaluation_loads(self):
        with tempfile.TemporaryDirectory() as d:
            payload = {'kind': 'real', 'run': {'model': 'm'}, 'dataset': {'dataset': 'x'},
                       'rows': [{'confidence': .9, 'correct': True}]}
            (Path(d) / 'run-a.json').write_text(json.dumps(payload))
            with mock.patch.object(app, 'PUBLISHED', Path(d)):
                self.assertEqual(self.get('/api/status')['evaluations'], [{'name': 'run-a', 'model': 'm', 'dataset': 'x', 'n': 1}])
                self.assertEqual(self.get('/api/evaluations/run-a')['rows'], payload['rows'])
                for bad in ('missing', '..%2Fapp', '.hidden%2F..'):
                    with self.assertRaises(urllib.error.HTTPError) as e:
                        self.get('/api/evaluations/' + bad)
                    self.assertEqual(e.exception.code, 404)
                    e.exception.close()

    def test_no_evaluations_is_empty_not_synthetic(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(app, 'PUBLISHED', Path(d)):
            self.assertEqual(self.get('/api/status')['evaluations'], [])

    def test_metrics_and_validation(self):
        code, m = self.post('/api/metrics', {'rows': [{'confidence': 1, 'correct': True}, {'confidence': 1, 'correct': False}],
                                             'threshold': .5, 'wrong_cost': 10, 'defer_cost': 1})
        self.assertEqual((code, m['auroc']), (200, .5))
        code, m = self.post('/api/metrics', {'rows': [{'confidence': 1, 'correct': True}], 'threshold': .5, 'wrong_cost': 10, 'defer_cost': 1})
        self.assertEqual((code, m['auroc']), (200, None))
        code, err = self.post('/api/metrics', {'rows': [{'confidence': 2, 'correct': True}], 'threshold': .5, 'wrong_cost': 10, 'defer_cost': 1})
        self.assertEqual(code, 400)
        code, _ = self.post('/api/metrics', {'rows': []}, {'Origin': 'http://evil.example'})
        self.assertEqual(code, 403)

    def test_live_requires_provider(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            code, err = self.post('/api/live', {'model': 'm', 'count': 14})
            self.assertEqual(code, 502)
            self.assertIn('No API provider configured', err['error'])
            self.assertFalse(self.get('/api/status')['live_available'])

    def test_plan_and_run_mark_cached_and_new(self):
        env = {'TRUSTGATE_API_KEY': 'k', 'TRUSTGATE_API_BASE': 'https://llm.example.org/v1'}
        calls = []

        def fake_request(cfg, method, path, body=None):
            calls.append(body)
            return {'choices': [{'message': {'content': 'Answer: A\nConfidence: 70'}, 'finish_reason': 'stop'}]}

        with tempfile.TemporaryDirectory() as d, mock.patch.dict('os.environ', env, clear=True), \
                mock.patch.object(provider, 'RESULTS_DIR', Path(d)), mock.patch.object(provider, '_request', side_effect=fake_request), \
                mock.patch('builtins.print'):
            code, plan = self.post('/api/live/plan', {'model': 'm', 'count': 14})
            self.assertEqual((code, plan), (200, {'total': 14, 'cached': 0, 'new': 14}))
            code, first = self.post('/api/live', {'model': 'm', 'count': 14})
            self.assertEqual((code, first['counts']['cached'], first['counts']['new'], len(calls)), (200, 0, 14, 14))
            self.assertTrue(all(r['origin'] == 'new' for r in first['rows']))
            code, plan = self.post('/api/live/plan', {'model': 'm', 'count': 70})
            self.assertEqual(plan, {'total': 70, 'cached': 14, 'new': 56})
            code, second = self.post('/api/live', {'model': 'm', 'count': 70})
            self.assertEqual((second['counts']['cached'], second['counts']['new'], len(calls)), (14, 56, 70))
            origin = {r['id']: r['origin'] for r in second['rows']}
            self.assertEqual(sorted(origin.values()).count('cached'), 14)
            self.assertTrue(all(origin[r['id']] == 'cached' for r in first['rows']))
            self.assertIsNotNone(app.evaluate(second['rows'])['holdout'])  # 70 keeps both halves
            self.assertTrue(all('raw_output' not in r for r in second['rows']))
            for bad in ({'model': 'm', 'count': 30}, {'model': 'bad model', 'count': 14}):
                self.assertEqual(self.post('/api/live/plan', bad)[0], 400)


class SubsetTest(unittest.TestCase):
    def test_even_deterministic_nested(self):
        sizes = {}
        for per_cat, total in ((1, 14), (5, 70), (10, 140)):
            meta, items = benchmark.subset(per_cat)
            self.assertEqual(len(items), total)
            self.assertEqual(set(Counter(q['category'] for q in items).values()), {per_cat})
            self.assertEqual(items, benchmark.subset(per_cat)[1])
            sizes[total] = {q['id'] for q in items}
        self.assertTrue(sizes[14] <= sizes[70] <= sizes[140])
        self.assertEqual(Counter(q['split'] for q in benchmark.subset(5)[1]), {'calibration': 42, 'test': 28})
        self.assertIn('subset', benchmark.subset(1)[0]['sampling'])
        self.assertNotIn('subset', benchmark.subset(10)[0]['sampling'])
        with self.assertRaises(ValueError):
            benchmark.subset(3)


class PublishTest(unittest.TestCase):
    def test_publish_writes_only_sanitized_json(self):
        import evaluate_model
        env = {'TRUSTGATE_API_KEY': 'k', 'TRUSTGATE_API_BASE': 'https://llm.example.org/v1'}
        reply = {'choices': [{'message': {'content': 'long private reasoning\nAnswer: A\nConfidence: 70'}, 'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict('os.environ', env, clear=True), \
                mock.patch.object(provider, 'RESULTS_DIR', Path(d) / 'results'), \
                mock.patch.object(evaluate_model, 'RESULTS_DIR', Path(d) / 'results'), \
                mock.patch.object(evaluate_model, 'PUBLISHED_DIR', Path(d) / 'evaluations'), \
                mock.patch.object(provider, '_request', return_value=reply), mock.patch('builtins.print'):
            evaluate_model.main(['--questions', '14', '--skip-model-check', '--publish'])
            published = list((Path(d) / 'evaluations').iterdir())
            self.assertEqual([f.suffix for f in published], ['.json'])
            text = published[0].read_text()
            for leak in ('private reasoning', 'raw_output', 'provider_id', 'llm.example.org'):
                self.assertNotIn(leak, text)
            self.assertIn('private reasoning', next((Path(d) / 'results').glob('*.jsonl')).read_text())


if __name__ == '__main__':
    unittest.main()
