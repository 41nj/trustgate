import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import provider
from fixtures import questions

DATASET = {'dataset': 'test', 'revision': 'rev-1', 'sample_sha256': 'abc'}
from engine import validate
from provider import ProviderError, dashboard_payload, evaluate_run, load_config, parse_prediction

SECRET = 'sk-test-SECRET-123'


class ParseTest(unittest.TestCase):
    def test_formats(self):
        cases = {
            'Answer: B\nConfidence: 85': ('B', .85),
            '**Answer:** C\n**Confidence:** 70%': ('C', .7),
            'answer: (D)\nconfidence = 40': ('D', .4),
            '{"answer": "A", "confidence": 90}': ('A', .9),
            '```json\n{"Answer": "b", "Confidence": "65%"}\n```': ('B', .65),
            'Option A looks plausible, but option C is right.\nAnswer: C\nConfidence: 92': ('C', .92),
            '<think>Answer: A\nConfidence: 10</think>Answer: B\nConfidence: 100': ('B', 1.0),
            'Answer: A\nConfidence: 0': ('A', 0.0),
            'Answer: A\nConfidence: 1': ('A', .01),  # 0–100 scale: 1 means 1 %
            'Answer: I think it is B.\nAnswer: J\nConfidence: 55': ('J', .55),
            'The answer is (E). Confidence: 80': ('E', .8),
        }
        for text, (letter, p) in cases.items():
            with self.subTest(text=text):
                got = parse_prediction(text, 'ABCDEFGHIJ')
                self.assertEqual(got[0], letter)
                self.assertAlmostEqual(got[1], p)

    def test_rejects(self):
        for text in ['', '   ', None, 'I am not sure.', 'Answer: B', 'Confidence: 80',
                     'Answer: E\nConfidence: 50', 'Answer: B\nConfidence: 150',
                     'Answer: B\nConfidence: 0.85',  # ambiguous scale, not guessed
                     '{"answer": "E", "confidence": 50}', '{"answer": "A", "confidence": true}',
                     'The answer is a guess. Confidence: 50']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_prediction(text, 'ABCD')

    def test_variable_options(self):
        self.assertEqual(parse_prediction('Answer: J\nConfidence: 50', 'ABCDEFGHIJ')[0], 'J')
        with self.assertRaises(ValueError):
            parse_prediction('Answer: H\nConfidence: 50', 'ABCDEFG')


class ConfigTest(unittest.TestCase):
    def test_defaults_and_override(self):
        cfg = load_config({})
        self.assertEqual(cfg.base_url, 'https://api.openai.com/v1')
        self.assertIsNone(cfg.key)
        cfg = load_config({'TRUSTGATE_API_BASE': 'https://llm.example.org/api/', 'TRUSTGATE_MODEL': 'google/gemma-x',
                           'TRUSTGATE_API_KEY': SECRET, 'TRUSTGATE_MAX_TOKENS': '2048', 'TRUSTGATE_WORKERS': '2'})
        self.assertEqual((cfg.base_url, cfg.model, cfg.max_tokens, cfg.workers), ('https://llm.example.org/api', 'google/gemma-x', 2048, 2))
        self.assertEqual(load_config({'OPENAI_API_KEY': SECRET}).key, SECRET)

    def test_key_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'keys.env'
            f.write_text(f'KEY_0=other\nKEY_1={SECRET}\n')
            self.assertEqual(load_config({'TRUSTGATE_API_KEY_FILE': str(f), 'TRUSTGATE_API_KEY_NAME': 'KEY_1'}).key, SECRET)
            with self.assertRaises(ValueError) as e:
                load_config({'TRUSTGATE_API_KEY_FILE': str(f), 'TRUSTGATE_API_KEY_NAME': 'KEY_9'})
            self.assertNotIn(SECRET, str(e.exception))

    def test_invalid(self):
        for env in [{'TRUSTGATE_API_BASE': 'http://llm.example.org'}, {'TRUSTGATE_API_BASE': 'ftp://x'},
                    {'TRUSTGATE_WORKERS': '9'}, {'TRUSTGATE_TEMPERATURE': '-1'}, {'TRUSTGATE_MODEL': 'bad model'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                load_config(env)
        self.assertEqual(load_config({'TRUSTGATE_API_BASE': 'http://localhost:8080/v1'}).base_url, 'http://localhost:8080/v1')

    def test_repr_hides_key(self):
        cfg = load_config({'TRUSTGATE_API_KEY': SECRET})
        self.assertNotIn(SECRET, repr(cfg))
        identity = provider.run_identity(cfg, 'm', {'revision': 'r', 'sample_sha256': 's'})
        self.assertNotIn(SECRET, json.dumps(identity))
        self.assertNotIn('api.openai.com', json.dumps(identity))


def fake_api(outputs):
    """Replace HTTP transport; outputs maps question text -> content, or an exception."""
    def request(cfg, method, path, body=None):
        if path == '/models':
            return {'data': [{'id': 'm1'}, {'id': 'google/gemma-x'}]}
        self_q = body['messages'][1]['content'].split('\n')[0]
        out = outputs.get(self_q, 'Answer: A\nConfidence: 50')
        if isinstance(out, Exception):
            raise out
        content, finish = out if isinstance(out, tuple) else (out, 'stop')
        return {'choices': [{'message': {'content': content}, 'finish_reason': finish}]}
    return mock.patch.object(provider, '_request', side_effect=request)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'run.jsonl'
        self.cfg = load_config({'TRUSTGATE_API_BASE': 'https://internal.example.edu/api', 'TRUSTGATE_API_KEY': SECRET})
        self.qs = questions()[:4]
        self.identity = provider.run_identity(self.cfg, 'google/gemma-x', DATASET)

    def tearDown(self):
        self.dir.cleanup()

    def run_eval(self, outputs, qs=None):
        with fake_api(outputs) as m:
            result = evaluate_run(self.cfg, qs or self.qs, 'google/gemma-x', self.path, self.identity, log=lambda _: None, retries=2, backoff=0)
        return result, m.call_count

    def test_models(self):
        with fake_api({}):
            self.assertEqual(provider.list_models(self.cfg), ['google/gemma-x', 'm1'])

    def test_records_invalid_and_truncated(self):
        q = self.qs
        (meta, recs, failed), _ = self.run_eval({q[0]['question']: 'Answer: B\nConfidence: 90',
                                                 q[1]['question']: 'No idea.',
                                                 q[2]['question']: ('Answer: B\nConf', 'length')})
        self.assertEqual(failed, [])
        self.assertEqual([r['status'] for r in recs], ['ok', 'invalid', 'invalid', 'ok'])
        self.assertFalse(recs[0]['correct'])  # fixture question 1 has gold answer A
        self.assertIn('truncated', recs[2]['error'])
        payload = dashboard_payload(meta, recs)
        self.assertEqual(len(payload['rows']), 2)
        self.assertIn('2 unparseable excluded', payload['source'])
        validate(payload['rows'])

    def test_resume_and_failures(self):
        target = self.qs[1]['question']
        (_, recs, failed), calls = self.run_eval({target: ProviderError('API returned HTTP 503')})
        self.assertEqual(failed, [self.qs[1]['id']])
        self.assertEqual(len(recs), 3)
        self.assertEqual(calls, 3 + 2)  # failing question retried twice
        (_, recs, failed), calls = self.run_eval({})
        self.assertEqual((failed, len(recs), calls), ([], 4, 1))  # only the failed one is re-requested

    def test_no_retry_on_auth_error(self):
        (_, _, failed), calls = self.run_eval({q['question']: ProviderError('API returned HTTP 401') for q in self.qs}, self.qs[:1])
        self.assertEqual((len(failed), calls), (1, 1))

    def test_cache_identity(self):
        self.run_eval({})
        base = {'TRUSTGATE_API_KEY': SECRET, 'TRUSTGATE_API_BASE': 'https://internal.example.edu/api'}
        variants = {
            'provider_id': (load_config({**base, 'TRUSTGATE_API_BASE': 'https://other.example.com/v1'}), DATASET),
            'temperature': (load_config({**base, 'TRUSTGATE_TEMPERATURE': '0.7'}), DATASET),
            'max_tokens': (load_config({**base, 'TRUSTGATE_MAX_TOKENS': '1024'}), DATASET),
            'dataset_revision': (self.cfg, {**DATASET, 'revision': 'newer'}),
            'sample_sha256': (self.cfg, {**DATASET, 'sample_sha256': 'changed'}),
        }
        for field, (cfg, dataset) in variants.items():
            identity = provider.run_identity(cfg, 'google/gemma-x', dataset)
            with self.subTest(field=field), fake_api({}) as m, self.assertRaises(ValueError) as e:
                evaluate_run(cfg, self.qs, 'google/gemma-x', self.path, identity, log=lambda _: None)
            self.assertIn(field, str(e.exception))
            self.assertEqual(m.call_count, 0)  # nothing reused, nothing requested
        # Same model ID at another provider gets its own cache file.
        other = load_config({**base, 'TRUSTGATE_API_BASE': 'https://other.example.com/v1'})
        self.assertNotEqual(provider.cache_path(self.cfg, 'google/gemma-x'), provider.cache_path(other, 'google/gemma-x'))
        self.assertNotIn('example', provider.cache_path(self.cfg, 'google/gemma-x').name)

    def test_payload_is_sanitized(self):
        q = self.qs
        (meta, recs, _), _ = self.run_eval({q[0]['question']: 'secret reasoning\nAnswer: B\nConfidence: 90',
                                             q[1]['question']: 'private chain of thought, no answer'})
        payload = json.dumps(dashboard_payload(meta, recs, dataset=DATASET))
        for leak in ('secret reasoning', 'private chain', 'raw_output', 'provider_id', 'finish_reason', 'internal.example.edu', SECRET):
            self.assertNotIn(leak, payload)
        self.assertEqual(json.loads(payload)['excluded'][0]['status'], 'invalid')

    def test_prompt_has_no_answer_key(self):
        seen = []
        def request(cfg, method, path, body=None):
            seen.append(json.dumps(body))
            return {'choices': [{'message': {'content': 'Answer: A\nConfidence: 50'}, 'finish_reason': 'stop'}]}
        from benchmark import mmlu_pro
        q = dict(mmlu_pro()[1][0], cot_content='SECRET EXPLANATION')
        with mock.patch.object(provider, '_request', side_effect=request):
            rec = provider.predict(self.cfg, {k: v for k, v in q.items() if k != 'cot_content'}, 'm', 1, 0)
        body = json.loads(seen[0])
        letters = 'ABCDEFGHIJ'[:len(q['options'])]
        expected_user = q['question'] + '\n' + '\n'.join(f'{l}. {o}' for l, o in zip(letters, q['options']))
        self.assertEqual(body['messages'], [{'role': 'system', 'content': provider.system_prompt(letters)},
                                            {'role': 'user', 'content': expected_user}])
        self.assertNotIn('SECRET', seen[0])
        self.assertEqual((rec['category'], rec['split'], rec['question_id']), (q['category'], q['split'], q['question_id']))

    def test_no_secrets_in_results(self):
        self.run_eval({})
        text = self.path.read_text()
        self.assertNotIn(SECRET, text)
        self.assertNotIn('internal.example.edu', text)
        first = json.loads(text.splitlines()[0])
        self.assertEqual((first['model'], first['temperature'], first['max_tokens']), ('google/gemma-x', 0.0, 4096))


class TransportTest(unittest.TestCase):
    def test_http_error_message_is_sanitized(self):
        import io, urllib.error
        cfg = load_config({'TRUSTGATE_API_BASE': 'https://internal.example.edu/api', 'TRUSTGATE_API_KEY': SECRET})
        err = urllib.error.HTTPError('https://internal.example.edu/api/models', 401, 'Unauthorized',
                                     {'Authorization': 'Bearer ' + SECRET}, io.BytesIO(b'{"error":{"message":"bad key"}}'))
        with mock.patch('urllib.request.urlopen', side_effect=err), self.assertRaises(ProviderError) as e:
            provider.list_models(cfg)
        self.assertEqual(str(e.exception), 'API returned HTTP 401: bad key')
        leaky = (b'{"detail":"Model x not found at https://internal.example.edu/api/v1 for key ' + SECRET.encode() +
                 b' (user jane@uni.example.edu, host gpu7.internal.example.edu, token abcdefghijklmnopqrstuvwxyz123456)"}')
        err = urllib.error.HTTPError('u', 404, 'Not Found', {}, io.BytesIO(leaky))
        with mock.patch('urllib.request.urlopen', side_effect=err), self.assertRaises(ProviderError) as e:
            provider.list_models(cfg)
        msg = str(e.exception)
        for leak in (SECRET, 'internal', 'example', 'jane', 'gpu7', 'abcdefghijklmnop', 'https'):
            self.assertNotIn(leak, msg)
        self.assertTrue(msg.startswith('API returned HTTP 404: Model x not found'))
        self.assertLessEqual(len(msg), 150)
        with mock.patch('urllib.request.urlopen', side_effect=urllib.error.URLError('internal.example.edu')), \
                self.assertRaises(ProviderError) as e:
            provider.list_models(cfg)
        self.assertNotIn('internal', str(e.exception))


if __name__ == '__main__':
    unittest.main()
