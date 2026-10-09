"""Frozen development replay: resettable events versus the archived v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(bearing, steps, values, provenance):
    from state_interpreter.episodic_weak_event import EpisodeConfig, OnlineEpisodicWeakEvent
    detector = OnlineEpisodicWeakEvent(EpisodeConfig(provenance=provenance), bearing_id=bearing)
    rows = []
    for step, value in zip(steps, values):
        out = detector.update(bearing_id=bearing, step_id=int(step), value=float(value))
        row = asdict(out)
        row['phase'] = out.phase.value
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root / 'src'))
    from state_interpreter.features.bearing_features import FEATURE_NAMES
    from state_interpreter.weak_event_development import evaluate_bearing
    old_path = root / 'configs/xjtu_condition3_weak_event_development_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    dependencies = ['src/state_interpreter/episodic_weak_event.py',
                    'src/state_interpreter/weak_event_detector.py',
                    'src/state_interpreter/weak_event_development.py',
                    'src/state_interpreter/features/bearing_features.py',
                    'configs/xjtu_condition3_weak_event_development_v1.json',
                    'results/2026-09-19_condition3_weak_event_development_v1/candidate_event_trajectories.csv']
    expected = {'experiment_id': 'condition3_episodic_v2_development_20261009',
                'development': ['Bearing3_1', 'Bearing3_2', 'Bearing3_3'],
                'protected_not_read': ['Bearing3_4', 'Bearing3_5'],
                'parameters': {'calibration_steps': 15, 'sigma_multiplier': 2.0,
                               'enter_steps': 5, 'exit_steps': 5, 'trend_window': 10,
                               'min_reference_std': 1e-8},
                'input': old['input'], 'dependencies': {p: digest(root / p) for p in dependencies},
                'runner_sha256': digest(__file__),
                'output': 'results/2026-10-09_condition3_episodic_v2_development',
                'record': 'records/condition3_episodic_v2_20261009/evaluation_record.json',
                'criteria': ['six valid references', 'finite continuous outputs',
                             'exact causal prefix equality at steps 15,25,half,N-1',
                             'entry and exit only after five consecutive observations',
                             'first v2 entry equals v1 confirmation', 'archived v1 exact replay'],
                'interpretation': 'algorithmic sustained vibration-change episodes; no physical onset accuracy',
                'validation_release': 'not automatic; separate prospective protocol required'}
    if args.freeze:
        write_json(args.freeze, expected)
        print('Frozen config created; no NPZ parsed.')
        return
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config != expected:
        raise ValueError('Frozen configuration/dependency mismatch')
    cache = root / config['input']['cache_directory']
    if digest(cache / 'manifest.json') != config['input']['manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    for entry in config['input']['cache_entries']:
        if digest(cache / entry['path']) != entry['sha256']:
            raise ValueError('Cache hash mismatch')
    output, record = root / config['output'], root / config['record']
    if output.exists() or record.exists():
        raise FileExistsError('Existing output/record; refusing overwrite')
    if not args.execute:
        print('Preflight passed; no NPZ parsed.')
        return
    record.parent.mkdir(parents=True, exist_ok=True)
    write_json(record, {'status': 'started', 'config_sha256': digest(args.config)})
    output.mkdir(parents=True, exist_ok=False)
    with (root / dependencies[-1]).open(encoding='utf-8', newline='') as f:
        archive = list(csv.DictReader(f))
    trajectories, summaries, episodes = [], [], []
    archived_cursor = 0
    for entry in config['input']['cache_entries']:
        bearing = entry['bearing_id']
        with np.load(cache / entry['path'], allow_pickle=False) as data:
            features, steps = data['features'], data['step_ids']
        if len(steps) != entry['sample_count']:
            raise ValueError('Count mismatch')
        v1_rows, _ = evaluate_bearing(bearing, features, steps)
        for row, archived in zip(v1_rows, archive[archived_cursor:archived_cursor + len(v1_rows)]):
            for key, value in row.items():
                actual = archived[key]
                if value is None:
                    assert actual == ''
                elif isinstance(value, bool):
                    assert actual == str(value)
                elif isinstance(value, (int, float)):
                    assert float(actual) == float(value)
                else:
                    assert actual == str(value)
        archived_cursor += len(v1_rows)
        for kind, names in [('rms', ['h_rms', 'v_rms']), ('kurtosis', ['h_kurtosis', 'v_kurtosis'])]:
            values = (features[:, FEATURE_NAMES.index(names[0])].astype(float) +
                      features[:, FEATURE_NAMES.index(names[1])].astype(float)) / 2
            rows = replay(bearing, steps, values, kind + '_episodic_v2')
            for count in sorted({15, 25, len(rows)//2, len(rows)-1}):
                assert replay(bearing, steps[:count], values[:count], kind + '_episodic_v2') == rows[:count]
            entries = [r for r in rows if r['entered_now']]
            exits = [r for r in rows if r['exited_now']]
            old_rows = [r for r in v1_rows if r['provenance'] == kind + '_threshold_v1']
            assert (entries[0]['step_id'] if entries else None) == old_rows[-1]['event_confirm_step']
            assert rows[-1]['threshold'] is not None
            for i, row in enumerate(rows):
                assert all(np.isfinite(row[k]) for k in ('deviation_z', 'trend') if row[k] is not None)
                if row['entered_now']:
                    assert all(values[j] > row['threshold'] for j in range(i-4, i+1))
                if row['exited_now']:
                    assert all(values[j] <= row['threshold'] for j in range(i-4, i+1))
            for start in entries:
                end = next((r for r in exits if r['step_id'] > start['step_id']), None)
                episodes.append({'bearing_id': bearing, 'feature': kind,
                                 'episode_id': start['episode_count'], 'candidate_start_step': start['candidate_start_step'],
                                 'confirm_step': start['step_id'], 'exit_confirm_step': None if end is None else end['step_id'],
                                 'active_measurements': (int(steps[-1])+1 if end is None else end['step_id'])-start['step_id'],
                                 'right_censored_at_record_end': end is None})
            summaries.append({'bearing_id': bearing, 'feature': kind, 'measurement_count': len(rows),
                              'episode_count': len(entries), 'exit_count': len(exits),
                              'first_confirm_step': entries[0]['step_id'] if entries else None,
                              'last_confirm_step': entries[-1]['step_id'] if entries else None,
                              'active_at_record_end': rows[-1]['active_event'],
                              'v1_active_measurements': sum(r['event_flag'] for r in old_rows),
                              'v2_active_measurements': sum(r['active_event'] is True for r in rows),
                              'reference_mean': rows[-1]['reference_mean'], 'reference_std': rows[-1]['reference_std'],
                              'threshold': rows[-1]['threshold']})
            trajectories.extend(rows)
    assert archived_cursor == len(archive)
    write_csv(output / 'trajectories.csv', trajectories)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'episodes.csv', episodes)
    report = {'status': 'completed', 'summaries': summaries, 'trajectory_rows': len(trajectories),
              'criteria_passed': config['criteria'], 'protected_not_read': config['protected_not_read'],
              'interpretation': config['interpretation'], 'config_sha256': digest(args.config)}
    write_json(output / 'report.json', report)
    # Keep started record immutable; completion is a separate artifact.
    write_json(record.parent / 'completion_record.json', {**report, 'output_sha256': {
        p.name: digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
