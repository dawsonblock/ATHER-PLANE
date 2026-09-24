from awa.evaluation.matrix import mean_ci95

def test_mean_ci95_continuous_metric():
    m,ci=mean_ci95([1.0,2.0,3.0]); assert m==2.0 and ci>0
