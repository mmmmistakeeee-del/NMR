"""Physics-facing regression tests, independent synthetic truth and malformed input."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fit_nmr_batch import derivative_lorentz, fit_segment, split_sweeps, main

class FitTests(unittest.TestCase):
    def synthetic(self,sign=1):
        f=np.linspace(15990000.,16030000.,260)
        # Generate the physical formula directly, not using the fitted implementation.
        delta=f-16010000.
        y=-2*1800.**2*(sign*3200.)*delta/(1800.**2+delta**2)**2+.08+2e-6*(f-16010000.)
        y+=np.random.default_rng(45).normal(0,.025,len(f))
        return np.column_stack((f,y))

    def test_physical_recovery_and_covariance(self):
        for sign in [1,-1]:
            row,cov=fit_segment(self.synthetic(sign))
            self.assertLess(abs(row['f0_Hz']-16010000.),4*row['f0_se_Hz'])
            self.assertLess(abs(row['hwhm_Hz']-1800.),4*row['hwhm_se_Hz'])
            self.assertLess(abs(row['L0']-sign*3200.),4*row['L0_se'])
            self.assertTrue(np.all(np.linalg.eigvalsh(cov)>-1e-7))
            self.assertAlmostEqual(row['fwhm_se_Hz'],2*row['hwhm_se_Hz'])
            self.assertAlmostEqual(row['L0_se']**2,cov[2,2])
            # Frequency-translation invariance also checks stable centering.
            shifted=self.synthetic(sign); shifted[:,0]+=1e8
            other,_=fit_segment(shifted)
            self.assertAlmostEqual(other['f0_Hz']-1e8,row['f0_Hz'],places=3)
            self.assertAlmostEqual(other['hwhm_se_Hz'],row['hwhm_se_Hz'],places=3)

    def test_constant_background_and_descending(self):
        a=self.synthetic(); a[:,1]-=2e-6*(a[:,0]-16010000.)
        row,cov=fit_segment(a[::-1],linear=False)
        self.assertLess(abs(row['f0_Hz']-16010000.),4*row['f0_se_Hz'])
        self.assertEqual(row['baseline_slope_per_Hz'],0)
        self.assertTrue(np.all(cov[4,:]==0))

    def test_sweeps_preserve_all_samples(self):
        f=np.array([1,2,2,3,2,1,0,1,2.])
        a=np.column_stack((f,np.arange(len(f))))
        legs=split_sweeps(a)
        self.assertEqual([x[0] for x in legs],['up','down','up'])
        np.testing.assert_array_equal(np.concatenate([x[1] for x in legs]),a)

    def test_failed_file_and_new_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'out'
            np.savetxt(root/'20261029_KIT_x1.0_z2.0_t.dat',self.synthetic())
            (root/'20261029_KIT_x2.0_z2.0.dat').write_text('bad data')
            with contextlib.redirect_stdout(io.StringIO()):
                code=main([str(root),'--output',str(out),'--no-plots'])
            self.assertEqual(code,1)
            with (out/'fit_results.csv').open(encoding='utf-8-sig') as h: rows=list(csv.DictReader(h))
            self.assertEqual(len(rows),2)
            self.assertEqual(sum(r['fit_status']=='failed' for r in rows),1)
            self.assertEqual(rows[0]['x'],'1.0')
            # Date takes priority over a larger suffix. Compare same-day suffixes numerically.
            np.savetxt(root/'20261030_KIT_x1.0_z2.0_20.dat',self.synthetic(-1))
            np.savetxt(root/'20261031_KIT_x1.0_z2.0_9.dat',self.synthetic(-1))
            np.savetxt(root/'20261031_KIT_x1.0_z2.0_10.dat',self.synthetic())
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(root),'--output',str(out),'--no-plots'])
            with (out/'fit_results.csv').open(encoding='utf-8-sig') as h:
                updated=list(csv.DictReader(h))
            self.assertEqual(len(updated),2)
            self.assertEqual([r['filename'] for r in updated if r['x']=='1.0'],
                             ['20261031_KIT_x1.0_z2.0_10.dat'])
            summary=json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual({r['filename'] for r in summary['superseded_files']},
                             {'20261029_KIT_x1.0_z2.0_t.dat', '20261030_KIT_x1.0_z2.0_20.dat',
                              '20261031_KIT_x1.0_z2.0_9.dat'})


if __name__=='__main__': unittest.main()
