"""Read-only model audit: raw features, frozen embeddings and emission evidence.

Runs no training. Outputs new files only; all summaries are exploratory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rho(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def summarize(a):
    a = np.asarray(a, dtype=float)
    return dict(min=float(a.min()), max=float(a.max()),
                first15_median=float(np.median(a[:15])),
                last15_median=float(np.median(a[-15:])),
                quarter_medians=[float(np.median(x)) for x in np.array_split(a, 4)],
                time_rho=rho(np.arange(len(a)), a),
                lag1_rho=rho(a[:-1], a[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    latent_path = args.repo / 'runs/latent_analysis_z8_20260801/latent_trajectories.csv'
    old_dir = args.repo / 'results/2026-09-16_hmm_interpreter_exploratory'
    new_dir = args.repo / 'results/2026-09-16_hmm_interpreter_log_filter_check'
    latent = read_csv(latent_path)
    old = read_csv(old_dir / 'per_step_states.csv')
    new = read_csv(new_dir / 'per_step_states.csv')
    key = lambda row: (row['bearing_id'], int(row['step_id']))
    assert len(set(map(key, latent))) == len(latent)
    assert list(map(key, old)) == list(map(key, new)) == list(map(key, latent))
    comparison = {}
    for col in ['hmm_expected_stage', 'hmm_confidence'] + [f'hmm_probability_{k}' for k in range(3)]:
        differences = np.abs(np.array([float(r[col]) for r in old]) - np.array([float(r[col]) for r in new]))
        comparison[col] = {'max_absolute_difference': float(differences.max()),
                           'nonzero_difference_count': int(np.count_nonzero(differences))}
    comparison['stage_mismatch_count'] = sum(a['hmm_stage'] != b['hmm_stage'] for a, b in zip(old, new))
    params_path = new_dir / 'hmm_parameters.json'
    p = json.loads(params_path.read_text())
    mean = np.array(p['feature_mean'])
    scale = np.array(p['feature_std'])
    mu = np.array(p['emission_means_standardized'])
    variance = np.array(p['emission_variances_standardized'])
    z_all = np.array([[float(r[f'z_{j}']) for j in range(8)] for r in latent])
    train = np.array([r['split'] == 'train' for r in latent])
    _, singular, vt = np.linalg.svd(z_all[train] - z_all[train].mean(axis=0), full_matrices=False)
    pc1 = vt[0]
    rows_out, summaries, files = [], {}, []
    for bearing in sorted({r['bearing_id'] for r in latent}):
        indices = [i for i, r in enumerate(latent) if r['bearing_id'] == bearing]
        z = z_all[indices]
        x = (z - mean) / scale
        emission = -.5 * (np.log(2*np.pi*variance)[None] + (x[:, None]-mu[None])**2 / variance[None]).sum(axis=2)
        winners = emission.argmax(axis=1)
        baseline = z[:15].mean(axis=0)
        distance = np.linalg.norm(z-baseline, axis=1)
        movement = np.r_[0., np.linalg.norm(np.diff(z, axis=0), axis=1)]
        bearing_rows = []
        expected_files = list(range(1, len(indices)+1))
        folder = args.root / '35Hz12kN' / bearing
        assert sorted(int(f.stem) for f in folder.glob('*.csv')) == expected_files
        print('AUDIT', bearing, len(indices), flush=True)
        for local, i in enumerate(indices):
            row = latent[i]
            assert int(row['step_id']) == local
            source = folder / f'{local+1}.csv'
            with source.open(encoding='utf-8-sig') as stream:
                assert stream.readline().strip() == 'Horizontal_vibration_signals,Vertical_vibration_signals'
                raw = np.loadtxt(stream, delimiter=',')
            assert raw.shape == (32768, 2) and np.isfinite(raw).all()
            files.append({'bearing': bearing, 'measurement': local+1, 'sha256': sha(source)})
            centered = raw-raw.mean(axis=0)
            rms = np.sqrt((raw**2).mean(axis=0))
            var = (centered**2).mean(axis=0)
            kurt = (centered**4).mean(axis=0)/(var**2)
            out = dict(bearing_id=bearing, step_id=local, measurement_number=local+1,
                       rms_h=float(rms[0]), rms_v=float(rms[1]),
                       kurtosis_h=float(kurt[0]), kurtosis_v=float(kurt[1]),
                       peak_h=float(np.abs(raw[:,0]).max()), peak_v=float(np.abs(raw[:,1]).max()),
                       latent_pc1=float(z[local]@pc1), local_distance=float(distance[local]),
                       movement=float(movement[local]), reconstruction_mse=float(row['reconstruction_mse']),
                       level=new[i]['distance_level'], hmm_stage=int(new[i]['hmm_stage']),
                       hmm_expected=float(new[i]['hmm_expected_stage']), emission_winner=int(winners[local]))
            out.update({f'log_emission_{k}': float(emission[local,k]) for k in range(3)})
            bearing_rows.append(out)
        rows_out.extend(bearing_rows)
        stats = {col: summarize([r[col] for r in bearing_rows]) for col in
                 ['rms_h','rms_v','kurtosis_h','kurtosis_v','latent_pc1','local_distance','movement','reconstruction_mse']}
        stats['raw_latent_rho'] = {c: rho(np.array([r[c] for r in bearing_rows]), z@pc1) for c in ['rms_h','rms_v']}
        stats['emission_winner_counts'] = np.bincount(winners, minlength=3).tolist()
        stats['hmm_stage_counts'] = np.bincount([r['hmm_stage'] for r in bearing_rows], minlength=3).tolist()
        stats['emission_hmm_disagreements_after15'] = int(sum(r['emission_winner'] != r['hmm_stage'] for r in bearing_rows[15:]))
        stats['state1_log_margin_after15'] = summarize(emission[15:,1] - np.maximum(emission[15:,0], emission[15:,2]))
        top = np.argsort(movement)[-3:][::-1]
        stats['largest_latent_moves'] = [dict(measurement_number=int(t+1), movement=float(movement[t]),
             raw_at_event={c: float(bearing_rows[t][c]) for c in ['rms_h','rms_v']},
             before_median_distance=float(np.median(distance[max(0,t-10):t])) if t else None,
             after_median_distance=float(np.median(distance[t:min(len(z),t+10)]))) for t in top]
        summaries[bearing] = stats
    args.output.mkdir(parents=True)
    with (args.output/'aligned_evidence.csv').open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows_out[0])); writer.writeheader(); writer.writerows(rows_out)
    report = dict(status='completed', scope='exploratory observational audit, no fitting',
                  numerical_comparison=comparison, training_pc1_variance_ratio=float(singular[0]**2/(singular**2).sum()),
                  bearings=summaries, files_checked=len(files), samples_per_file=32768,
                  provenance={str(f):sha(f) for f in [latent_path,params_path,old_dir/'per_step_states.csv',new_dir/'per_step_states.csv']},
                  limitations=['No repeated measurement at fixed physical state; randomness is not identified.',
                    'RMS and kurtosis describe signals, not measured physical damage.',
                    'Archived latents are aligned by bearing and index; encoder extraction is not rerun.',
                    'Temporal dependence invalidates naive independent-sample significance claims.',
                    'Emission-only assignment is a diagnostic ablation, not a retrained alternative model.'])
    (args.output/'audit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (args.output/'raw_file_hashes.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    print(json.dumps({'comparison':comparison, 'Bearing1_4':summaries['Bearing1_4']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
