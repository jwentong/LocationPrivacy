"""Measured location/CSI -> frozen local covariance beam map -> private outage.

No WRF training or instantaneous test CSI is supplied to the beam controller.
Run with --help. All paths are relative to the current working directory.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def align(args):
    """Decode the published 64-antenna binary convention, one subcarrier."""
    match = re.fullmatch(r'(\d{8})_(\d{6})_n(\d+)_l(\d+)_s(\d+)_sr(\d+)_f[^_]+_g[^.]+\.0\.bin', args.channel.name)
    # Gain formatting varies; only the acquisition fields are needed here.
    if match is None:
        match = re.match(r'(\d{8})_(\d{6})_n(\d+)_l(\d+)_s(\d+)_sr(\d+)_', args.channel.name)
    if match is None:
        raise ValueError('Unrecognized measurement filename')
    date, clock, m, layer, n, sr = match.groups()
    m, n, sr = int(m), int(n), int(sr)
    if m != 64 or layer != '1':
        raise ValueError('This audited decoder supports n64_l1 only')
    if not 0 <= args.subcarrier < 100:
        raise ValueError('subcarrier must be in [0,99]')
    expected = n * m * 100 * 4
    if args.channel.stat().st_size != expected:
        raise ValueError(f'Incomplete/unexpected binary: expected {expected} bytes')
    frames = [pd.read_csv(p) for p in args.trajectory]
    df = pd.concat(frames, ignore_index=True)
    required = ['datetime(utc)', 'time(millisecond)', 'latitude', 'longitude']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Missing trajectory fields: {required}')
    # Reconstruct each flight clock separately, as in the provider script.
    for frame in frames:
        start = pd.to_datetime(frame['datetime(utc)'].iloc[0], utc=True)
        frame['_utc'] = start + pd.to_timedelta(frame['time(millisecond)'] - frame['time(millisecond)'].iloc[0], unit='ms')
    df = pd.concat(frames).sort_values('_utc').drop_duplicates('_utc')
    height = 'height_above_takeoff(m)'
    if height not in df:
        df[height] = df['height_above_takeoff(feet)'] / 3.28084
    # Height remains relative to takeoff; BS altitude is not silently assumed.
    lat, lon = np.deg2rad(df.latitude.to_numpy()), np.deg2rad(df.longitude.to_numpy())
    lat0, lon0 = np.deg2rad([args.bs_lat, args.bs_lon])
    positions = np.column_stack([6378137 * np.cos(lat0) * (lon-lon0), 6378137 * (lat-lat0), df[height]])
    gps_t = df['_utc'].to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9
    ids = np.arange(0, n, args.stride)
    start = pd.to_datetime(date + clock, format='%Y%m%d%H%M%S', utc=True).timestamp()
    times = start + ids * 0.0005 * sr + args.clock_offset
    right = np.searchsorted(gps_t, times, side='right')
    left = np.clip(right-1, 0, len(gps_t)-1)
    right = np.clip(right, 0, len(gps_t)-1)
    valid = (times >= gps_t[0]) & (times <= gps_t[-1]) & ((gps_t[right]-gps_t[left]) <= args.max_gps_gap)
    xyz = np.column_stack([np.interp(times, gps_t, positions[:, k]) for k in range(3)])
    raw = np.memmap(args.channel, dtype='>i2', mode='r', shape=(n, 100, m, 2))
    # For n64 the provider stores even subcarriers before odd subcarriers.
    storage = list(range(0, 100, 2)) + list(range(1, 100, 2))
    k = storage.index(args.subcarrier)
    iq = np.array(raw[ids, k], dtype=np.float32, copy=True)
    del raw
    h = (iq[..., 1] + 1j * iq[..., 0]) * 2**-8
    valid &= np.isfinite(xyz).all(axis=1) & np.isfinite(h).all(axis=1) & (np.linalg.norm(h, axis=1) > 0)
    if valid.sum() < 20:
        raise ValueError('Too few aligned samples; inspect clocks and trajectory coverage')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, xyz=xyz[valid], h=h[valid], time=times[valid], frame_id=ids[valid])
    meta = dict(channel=str(args.channel), channel_sha256=digest(args.channel), trajectory=[str(p) for p in args.trajectory],
                trajectory_sha256=[digest(p) for p in args.trajectory], subcarrier=args.subcarrier,
                stride=args.stride, clock_offset_seconds=args.clock_offset, accepted=int(valid.sum()),
                rejected=int((~valid).sum()), height_reference='takeoff-relative; not absolute BS-relative',
                xy_reference=[args.bs_lat, args.bs_lon], note='Array order is provider order, not inferred physical UPA order.')
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


def freeze(args):
    with np.load(args.input) as archive:
        data = {k: archive[k] for k in archive.files}
    xyz, h = data['xyz'], data['h']
    if len(xyz) != len(h) or h.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Expected paired xyz[N,3], h[N,M]')
    cells = np.floor(xyz / args.cell_m).astype(int)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    groups = np.unique(group)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    if len(groups) < 4:
        raise ValueError('Need at least four spatial cells')
    cut = max(1, min(len(groups)-1, int(len(groups)*args.train_fraction)))
    train = np.isin(group, groups[:cut])
    # Guard band removes training samples close to a test position.
    test = ~train
    distance = cKDTree(xyz[test]).query(xyz[train])[0]
    train_ids = np.flatnonzero(train)[distance >= args.guard_m]
    test_ids = np.flatnonzero(test)
    beams, centers, counts = [], [], []
    for g in groups[:cut]:
        local = train_ids[group[train_ids] == g]
        if len(local) < args.min_samples:
            continue
        z = h[local]
        covariance = z.T @ z.conj() / len(z)
        _, v = np.linalg.eigh(covariance)
        beams.append(v[:, -1])
        centers.append(xyz[local].mean(axis=0))
        counts.append(len(local))
    if len(beams) < 2:
        raise ValueError('Insufficient map cells after spatial split/guard band')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, map_xyz=np.array(centers), beams=np.array(beams), counts=counts,
                        test_xyz=xyz[test_ids], test_h=h[test_ids], test_id=test_ids, train_id=train_ids)
    meta = dict(input_sha256=digest(args.input), seed=args.seed, cell_m=args.cell_m, guard_m=args.guard_m,
                min_samples=args.min_samples, train_fraction=args.train_fraction, map_cells=len(beams),
                test_samples=len(test_ids), method='local principal covariance eigenvector; nearest-cell lookup',
                map_fit_uses_test_csi=False)
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


def evaluate(args):
    with np.load(args.input) as archive:
        data = {k: archive[k] for k in archive.files}
    x, h = data['test_xyz'], data['test_h']
    map_x, beams = data['map_xyz'], data['beams']
    # Public location grid uses a single documented datum, independent of budget.
    candidate = np.unique(np.floor(x/args.scenario_cell_m).astype(int), axis=0)
    candidate = (candidate + 0.5) * args.scenario_cell_m
    # This is a binned-location empirical experiment, not exact on-grid CSI.
    tree = cKDTree(candidate)
    group = tree.query(x)[1]
    beam_for_candidate = cKDTree(map_x).query(candidate)[1]
    prior = np.full(len(candidate), 1/len(candidate))
    etas = np.geomspace(args.eta_min, args.eta_max, args.budgets)
    rng = np.random.default_rng(args.seed)
    rows, references = [], []
    ref_power = float(np.median(np.sum(h * h.conj(), axis=1).real))
    scale = 10**(args.reference_snr_db/10) / ref_power
    threshold = 2**args.spectral_efficiency - 1
    for j in range(len(candidate)):
        ids = np.flatnonzero(group == j)
        draw = rng.choice(ids, args.samples, replace=True)
        true_h = h[draw]
        radius = rng.gamma(3, 1, args.samples)
        unit = rng.normal(size=(args.samples, 3))
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        anchor = candidate[j]
        # Noise is added to public cell centers; within-cell CSI is resampled.
        oracle_w = beams[beam_for_candidate[j]]
        oracle_gain = np.abs(true_h @ oracle_w.conj())**2
        q_ref = float(np.mean(scale * oracle_gain < threshold))
        references.append(dict(scenario=j, q=q_ref))
        for eta in etas:
            epsilon = eta / args.r0
            z = anchor + radius[:, None] / epsilon * unit
            inferred = np.empty(args.samples, dtype=int)
            for start in range(0, args.samples, 1024):
                zz = z[start:start+1024]
                score = np.log(prior)[None, :] - epsilon*np.linalg.norm(zz[:, None, :]-candidate[None, :, :], axis=2)
                inferred[start:start+len(zz)] = score.argmax(axis=1)
            w = beams[beam_for_candidate[inferred]]
            gain = np.abs(np.sum(w.conj()*true_h, axis=1))**2
            q = float(np.mean(scale * gain < threshold))
            a = args.alpha
            aoii = a*q / ((1-(1-a)*q)*(1-(1-2*a)*q))
            rows.append(dict(eta=float(eta), epsilon=float(epsilon), scenario=j, q=q, aoii_slots=aoii,
                             normalized_gain=float(np.mean(gain/np.sum(np.abs(true_h)**2, axis=1))),
                             location_error_m=float(np.mean(np.linalg.norm(candidate[inferred]-anchor, axis=1)))))
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'per_scenario.csv', index=False)
    summary = frame.groupby('eta', as_index=False)[['epsilon', 'q', 'aoii_slots', 'normalized_gain', 'location_error_m']].mean()
    summary.to_csv(args.output/'privacy_curve.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, column, label in zip(axs, ['q', 'aoii_slots'], ['Outage probability', 'IID-resampled AoII / T']):
        ax.semilogx(summary.eta, summary[column], 'o-')
        ax.set(xlabel=r'Single-report budget $\eta$', ylabel=label)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output/'privacy_curve.pdf')
    fig.savefig(args.output/'privacy_curve.png', dpi=160)
    plt.close(fig)
    meta = dict(input_sha256=digest(args.input), **{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'func'},
                candidate_count=len(candidate), prior='uniform', scenario_weights='uniform',
                reference_power=ref_power, snr_scale=scale, no_report_noise_reference=references,
                limitations=['Empirical cell-center scenarios; within-cell position mismatch remains.',
                             'Samples bootstrap measured CSI independently; not actual chronological AoII.',
                             'Absolute SNR is configured, not a calibrated measured link budget.',
                             'No finite-sample population confidence guarantee; not LAPIS confirmation results.'])
    (args.output/'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Wrote {len(summary)} operating points to {args.output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('align')
    a.add_argument('--channel', type=Path, required=True)
    a.add_argument('--trajectory', type=Path, nargs='+', required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--stride', type=int, default=100)
    a.add_argument('--subcarrier', type=int, default=50)
    a.add_argument('--bs-lat', type=float, default=50.86208865861035)
    a.add_argument('--bs-lon', type=float, default=4.685513023420431)
    a.add_argument('--clock-offset', type=float, default=0)
    a.add_argument('--max-gps-gap', type=float, default=2)
    a.set_defaults(func=align)
    a = sub.add_parser('freeze')
    a.add_argument('--input', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--cell-m', type=float, default=10)
    a.add_argument('--guard-m', type=float, default=2)
    a.add_argument('--train-fraction', type=float, default=.7)
    a.add_argument('--min-samples', type=int, default=5)
    a.add_argument('--seed', type=int, default=20261027)
    a.set_defaults(func=freeze)
    a = sub.add_parser('evaluate')
    a.add_argument('--input', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--samples', type=int, default=2000)
    a.add_argument('--budgets', type=int, default=21)
    a.add_argument('--eta-min', type=float, default=.1)
    a.add_argument('--eta-max', type=float, default=10)
    a.add_argument('--r0', type=float, default=10)
    a.add_argument('--alpha', type=float, default=.1)
    a.add_argument('--scenario-cell-m', type=float, default=10)
    a.add_argument('--reference-snr-db', type=float, default=10)
    a.add_argument('--spectral-efficiency', type=float, default=1)
    a.add_argument('--seed', type=int, default=20261028)
    a.set_defaults(func=evaluate)
    args = p.parse_args()
    if hasattr(args, 'alpha') and not 0 < args.alpha <= .5:
        p.error('alpha must be in (0, .5]')
    args.func(args)


if __name__ == '__main__':
    main()
