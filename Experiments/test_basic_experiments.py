import unittest
import numpy as np
from scipy.stats import ncx2
from run_basic_experiments import exact_confusion, intervals, steering, F


class BasicExperimentTests(unittest.TestCase):
    def test_laplace_limits_and_symmetry(self):
        sites=np.array([[-5,-5,25],[5,-15,25],[15,-25,25]],float)
        low=exact_confusion(sites,1e-9)
        high=exact_confusion(sites,100)
        np.testing.assert_allclose(low.sum(1),1)
        np.testing.assert_allclose(low[:,[0,2]],.5,atol=1e-7)
        np.testing.assert_allclose(high,np.eye(3),atol=1e-10)
        c=exact_confusion(sites,.1)
        np.testing.assert_allclose(c,c[::-1,::-1])

    def test_single_antenna_and_rayleigh_controls(self):
        sites=np.array([[20,40,100],[-40,20,80]],float)
        a=steering(sites,1)
        np.testing.assert_allclose(np.abs(a@a.conj().T)**2,1)
        # kappa=0 removes noncentrality regardless of geometric correlation.
        q=ncx2.cdf(2/.7,2,0)
        self.assertAlmostEqual(q,1-np.exp(-1/.7))

    def test_scenario_average_and_bound(self):
        q=np.array([[.1,.8],[.4,.4]])
        lo,hi=intervals(q,10000,4)
        exact=F(q).mean(1)
        self.assertTrue(np.all((lo<exact)&(exact<hi)))
        self.assertGreater(abs(F(q[0]).mean()-F(q[0].mean())),.01)


if __name__=='__main__':unittest.main()
