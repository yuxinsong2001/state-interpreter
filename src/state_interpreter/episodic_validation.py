"""Fixed operational validation of causal feature-event streams."""
from dataclasses import asdict
import numpy as np

from state_interpreter.episodic_weak_event import EpisodeConfig
from state_interpreter.relative_event_state import RelativeEventStateInterpreter

ALLOWED_BEARINGS=('Bearing3_1','Bearing3_2','Bearing3_3','Bearing3_4')

def evaluate_event_stream(bearing_id, step_ids, feature_pairs):
    """Input pairs: [measurement, 2], RMS then kurtosis; no physical labels."""
    if bearing_id not in ALLOWED_BEARINGS:
        raise PermissionError('unknown/protected bearing')
    steps=np.asarray(step_ids)
    pairs=np.asarray(feature_pairs)
    if steps.ndim!=1 or steps.dtype.kind not in 'iu' or len(steps)<15:
        raise ValueError('at least 15 integer step IDs required')
    if np.any(steps<0) or np.any(np.diff(steps)<=0):
        raise ValueError('step IDs must increase strictly')
    if pairs.shape!=(len(steps),2) or not np.isfinite(pairs).all():
        raise ValueError('finite RMS/kurtosis pairs required')
    if np.any(np.diff(steps[:15])!=1):
        raise ValueError('calibration must be consecutive')
    rows,ends,summaries=[],[],[]
    for column,feature in enumerate(('rms','kurtosis')):
        config=EpisodeConfig(provenance=feature+'_episodic_v2')
        def replay(count):
            d=RelativeEventStateInterpreter(config,bearing_id=bearing_id)
            outputs=[d.update(bearing_id=bearing_id,step_id=int(steps[i]),value=float(pairs[i,column])) for i in range(count)]
            return d,outputs
        detector,outputs=replay(len(steps))
        for count in sorted({15,min(25,len(steps)),max(15,len(steps)//2),len(steps)-1}):
            if count<15:continue
            assert replay(count)[1]==outputs[:count], 'prefix mismatch'
        for i,state in enumerate(outputs):
            raw=state.observation
            if raw.entered_now or raw.exited_now:
                assert i>=4 and np.all(np.diff(steps[i-4:i+1])==1)
                window=pairs[i-4:i+1,column]
                assert raw.threshold is not None
                assert np.all(window>raw.threshold) if raw.entered_now else np.all(window<=raw.threshold)
            for value in (state.feature_deviation_z,state.feature_trend_z_per_measurement):
                assert value is None or np.isfinite(value)
            flat=asdict(raw);flat['phase']=raw.phase.value
            flat.update(schema_version=state.schema_version,
                        feature_deviation_z=state.feature_deviation_z,
                        feature_trend_z_per_measurement=state.feature_trend_z_per_measurement,
                        trend_observation_count=state.trend_observation_count,
                        transition=state.transition,
                        termination_now=asdict(state.termination_now) if state.termination_now else None,
                        last_termination=asdict(state.last_termination) if state.last_termination else None)
            rows.append(flat)
            if state.termination_now:ends.append({'bearing_id':bearing_id,'feature':feature,**asdict(state.termination_now)})
        terminal=detector.finish_record()
        if terminal:ends.append({'bearing_id':bearing_id,'feature':feature,**asdict(terminal)})
        ref=outputs[14].observation
        valid=ref.threshold is not None
        summaries.append({'bearing_id':bearing_id,'feature':feature,'measurements':len(steps),
                          'reference_valid':valid,'reference_mean':ref.reference_mean,
                          'reference_std':ref.reference_std,'threshold':ref.threshold,
                          'episodes':outputs[-1].observation.episode_count,
                          'active_observed_measurements':sum(s.observation.active_event is True for s in outputs),
                          'unknown_gap_steps':sum(s.observation.phase.value=='UNKNOWN_GAP' for s in outputs),
                          'first_confirmation_step':next((s.observation.step_id for s in outputs if s.observation.entered_now),None),
                          'rules_and_prefix_checks_passed':True})
    checks={'input_valid':True,'references_valid':all(s['reference_valid'] for s in summaries),
            'finite_present_outputs':True,'causal_prefix_equality':True,'entry_exit_rules_valid':True,
            'termination_metadata_complete':len(ends)==sum(s['episodes'] for s in summaries)}
    return rows,ends,summaries,checks
