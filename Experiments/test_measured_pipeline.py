"""Physical checks and pipeline integration on explicitly synthetic fixtures."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from measured_pipeline import align, freeze, evaluate


class PipelineTests(unittest.TestCase):
    def test_covariance_phase_invariance(self):
        rng = np.random.default_rng(4)
        h = rng.normal(size=(30, 8)) + 1j*rng.normal(size=(30, 8))
        phased = h * np.exp(1j*rng.uniform(-np.pi, np.pi, (30, 1)))
        np.testing.assert_allclose(h.T @ h.conj(), phased.T @ phased.conj(), atol=1e-12)

    def test_decode_alignment_and_iq_order(self):
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            path = root/'20230316_154153_n64_l1_s4_sr2_f2.610G_g30.0.bin'
            raw = np.arange(4*100*64*2, dtype=np.int16).reshape(4, 100, 64, 2)
            raw.astype('>i2').tofile(path)
            track = root/'track.csv'
            pd.DataFrame({'datetime(utc)':['2023-03-16T15:41:53Z']*2,
                          'time(millisecond)':[0, 3], 'latitude':[50.,50.],
                          'longitude':[4.,4.], 'height_above_takeoff(feet)':[100.,100.]}).to_csv(track,index=False)
            # Alignment deliberately requires >=20 points in production, so test a longer fixture.
            path = root/'20230316_154153_n64_l1_s40_sr2_f2.610G_g30.0.bin'
            raw = np.tile(raw, (10,1,1,1))
            raw.astype('>i2').tofile(path)
            frame = pd.read_csv(track); frame['time(millisecond)']=[0,39]; frame.to_csv(track,index=False)
            out=root/'aligned.npz'
            align(argparse.Namespace(channel=path,trajectory=[track],output=out,stride=1,subcarrier=50,
                                     bs_lat=50.,bs_lon=4.,clock_offset=0,max_gps_gap=2))
            with np.load(out) as archive:
                data={k:archive[k] for k in archive.files}
            expected=(raw[:,25,:,1].astype(float)+1j*raw[:,25,:,0].astype(float))*2**-8
            np.testing.assert_allclose(data['h'],expected)
            np.testing.assert_allclose(data['xyz'][:,:2],0)

    def test_frozen_map_and_curve(self):
        rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder)
            centers=np.array([[i*30+5,5,105] for i in range(12)])
            xyz=np.repeat(centers,12,axis=0)+rng.uniform(-1,1,(144,3))
            a=np.exp(1j*np.arange(8)[None,:]*np.repeat(np.arange(12)*.3,12)[:,None])
            h=a+0.1*(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))
            np.savez(root/'paired.npz',xyz=xyz,h=h)
            freeze(argparse.Namespace(input=root/'paired.npz',output=root/'map.npz',cell_m=10,
                                      guard_m=2,train_fraction=.7,min_samples=5,seed=4))
            with np.load(root/'map.npz') as archive:
                d={k:archive[k] for k in archive.files}
            self.assertEqual(len(np.intersect1d(d['test_id'],d['train_id'])),0)
            np.testing.assert_allclose(np.linalg.norm(d['beams'],axis=1),1)
            evaluate(argparse.Namespace(input=root/'map.npz',output=root/'results',samples=200,budgets=3,
                                        eta_min=.1,eta_max=10,r0=10,alpha=.1,scenario_cell_m=10,
                                        reference_snr_db=10,spectral_efficiency=1,seed=5))
            self.assertTrue((root/'results/privacy_curve.pdf').exists())
            self.assertTrue(json.loads((root/'results/run.json').read_text())['limitations'])


if __name__=='__main__':
    unittest.main()
