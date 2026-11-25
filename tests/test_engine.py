import json
import unittest
from engine import evaluate, auc, validate
from fixtures import synthetic_rows

class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.rows=[{'confidence':.9,'correct':True},{'confidence':.8,'correct':False},{'confidence':.2,'correct':False},{'confidence':.1,'correct':True}]
    def test_metrics(self):
        m=evaluate(self.rows,.8,10,1)
        self.assertAlmostEqual(m['accuracy'],.5)
        self.assertAlmostEqual(m['auroc'],.5)
        self.assertAlmostEqual(m['brier'],.375)
        self.assertEqual(m['selected']['accepted'],2)
        self.assertEqual(m['selected']['errors'],1)
        self.assertEqual(m['selected']['cost'],12)
        self.assertEqual(m['baseline_cost'],20)
        self.assertEqual(m['best']['threshold'],.9)
    def test_ties(self):
        rows=[{'confidence':.5,'correct':True},{'confidence':.5,'correct':False}]
        self.assertEqual(auc(rows),.5)
        self.assertEqual(len(evaluate(rows)['curve']),2)
    def test_none_accepted(self):
        m=evaluate(self.rows,1)
        self.assertIsNone(m['selected']['risk'])
        self.assertEqual(m['selected']['coverage'],0)
    def test_boundaries(self):
        m=evaluate([{'confidence':1,'correct':True},{'confidence':0,'correct':False}],1)
        self.assertEqual(m['ece'],0)
        self.assertEqual(m['brier'],0)
        self.assertEqual(m['auroc'],1)
        self.assertEqual(m['bins'][-1]['count'],1)
    def test_one_class(self):
        self.assertIsNone(auc([{'confidence':.9,'correct':True}]))
    def test_invalid(self):
        for p in [-.1,1.1,float('nan'),True,'0.5']:
            with self.assertRaises(ValueError): validate([{'confidence':p,'correct':True}])
    def test_fixture(self):
        self.assertEqual(synthetic_rows(),synthetic_rows())
        validate(synthetic_rows())
        self.assertTrue(all(r['correct']==(r['prediction']==r['answer']) for r in synthetic_rows()))

class EdgeCaseTest(unittest.TestCase):
    def test_constant_confidence_with_errors(self):
        # Constant confidence cannot rank: AUROC is defined and exactly 0.5; only one policy step.
        rows=[{'confidence':1.0,'correct':c} for c in [True]*7+[False]*3]
        m=evaluate(rows,.8,10,1)
        self.assertEqual(m['auroc'],.5)
        self.assertEqual(len(m['curve']),2)
        self.assertAlmostEqual(m['ece'],.3)
        self.assertAlmostEqual(m['brier'],.3)
        self.assertEqual(m['bins'][9]['count'],10)
        self.assertEqual(m['best']['threshold'],None)  # 30 cost for errors vs 10 for deferring all
        self.assertEqual(m['best']['cost'],10)

    def test_single_class_auroc_undefined(self):
        for c in (True,False):
            m=evaluate([{'confidence':p,'correct':c} for p in (.2,.6,.9)])
            self.assertIsNone(m['auroc'])
            self.assertTrue(json.dumps(m,allow_nan=False))  # no NaN leaks to the client

    def test_ece_and_brier_by_hand(self):
        rows=[{'confidence':.95,'correct':True},{'confidence':.95,'correct':False},{'confidence':.35,'correct':True},{'confidence':.05,'correct':False}]
        m=evaluate(rows)
        # bins: [.9,1): conf .95, acc .5, w .5 ; [.3,.4): conf .35, acc 1, w .25 ; [0,.1): conf .05, acc 0, w .25
        self.assertAlmostEqual(m['ece'],.5*.45+.25*.65+.25*.05)
        self.assertAlmostEqual(m['brier'],(.05**2+.95**2+.65**2+.05**2)/4)

    def test_risk_coverage_monotone_coverage(self):
        m=evaluate(synthetic_rows())
        cov=[p['coverage'] for p in m['curve']]
        self.assertEqual(cov,sorted(cov))
        self.assertEqual((cov[0],cov[-1]),(0,1))
        self.assertEqual(m['curve'][-1]['cost'],m['baseline_cost'])
        self.assertEqual(m['best']['cost'],min(p['cost'] for p in m['curve']))

    def test_cost_threshold_brute_force(self):
        import random
        rng=random.Random(1)
        rows=[{'confidence':round(rng.random(),1),'correct':rng.random()<.6} for _ in range(60)]
        for wc,dc in [(10,1),(1,1),(5,0),(1,10)]:
            best=evaluate(rows,0,wc,dc)['best']['cost']
            brute=min(sum((not r['correct'])*wc if r['confidence']>=t else dc for r in rows) for t in [i/10 for i in range(11)]+[2])
            self.assertAlmostEqual(best,brute)

    def test_holdout(self):
        cal=[{'confidence':.9,'correct':True,'split':'calibration'},{'confidence':.4,'correct':False,'split':'calibration'}]
        test=[{'confidence':.95,'correct':False,'split':'test'},{'confidence':.5,'correct':True,'split':'test'}]
        h=evaluate(cal+test,.5,10,1)['holdout']
        self.assertEqual(h['threshold'],.9)  # chosen on calibration only
        self.assertEqual((h['test']['accepted'],h['test']['errors'],h['test']['cost']),(1,1,11))
        self.assertEqual((h['test_answer_all_cost'],h['test_defer_all_cost']),(10,2))
        self.assertIsNone(evaluate(cal)['holdout'])
        self.assertIsNone(evaluate(self_rows())['holdout'])

    def test_holdout_defer_all(self):
        rows=[{'confidence':1,'correct':False,'split':'calibration'},{'confidence':1,'correct':True,'split':'test'}]
        h=evaluate(rows,.5,10,1)['holdout']
        self.assertIsNone(h['threshold'])
        self.assertEqual(h['test']['accepted'],0)


def self_rows():
    return [{'confidence':.5,'correct':True},{'confidence':.6,'correct':False}]


if __name__=='__main__':unittest.main()
