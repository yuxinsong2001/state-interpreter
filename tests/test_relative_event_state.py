import json
import pytest
from state_interpreter.episodic_weak_event import EpisodeConfig, OnlineEpisodicWeakEvent
from state_interpreter.relative_event_state import RelativeEventStateInterpreter

def make():
    return RelativeEventStateInterpreter(EpisodeConfig(provenance='rms_episodic_v2'),bearing_id='A')

def run(detector, values, start=0):
    return [detector.update(bearing_id='A',step_id=i,value=v) for i,v in enumerate(values,start)]

def calibration():
    return [0.,2.]*7+[1.]

def test_exit_preserves_completed_duration_and_reentry():
    d=make();run(d,calibration())
    entered=run(d,[4.]*5,15)[-1]
    assert entered.transition=='entered'
    exited=run(d,[1.]*5,20)[-1]
    end=exited.termination_now
    assert end.reason=='consecutive_non_exceedances' and not end.right_censored
    assert end.confirm_step==19 and end.last_observed_active_step==23
    assert end.termination_observed_step==24 and end.observed_active_measurements==5
    assert exited.observation.active_event is False
    again=run(d,[4.]*5,25)[-1]
    assert again.observation.active_episode_id==2 and again.last_termination==end
    assert again.termination_now is None

def test_gap_is_censored_unknown_and_trend_restarts():
    d=make();run(d,calibration()+[4.]*5)
    gap=d.update(bearing_id='A',step_id=22,value=4.)
    assert gap.observation.active_event is None
    assert gap.termination_now.reason=='measurement_gap' and gap.termination_now.right_censored
    assert gap.termination_now.observed_active_measurements==1
    assert gap.feature_trend_z_per_measurement is None and gap.trend_observation_count==1
    nxt=d.update(bearing_id='A',step_id=23,value=4.)
    assert nxt.feature_trend_z_per_measurement==0.

def test_end_of_record_does_not_invent_exit():
    d=make();states=run(d,calibration()+[4.]*7)
    end=d.finish_record()
    assert end.reason=='record_end' and end.right_censored
    assert end.observed_active_measurements==3
    assert states[-1].observation.active_event is True and not states[-1].observation.exited_now
    with pytest.raises(ValueError):d.update(bearing_id='A',step_id=22,value=1.)
    with pytest.raises(ValueError):d.finish_record()
    d.reset(bearing_id='A')
    assert run(d,calibration())[0].last_termination is None

def test_atomic_rejection_and_prefix_equivalence():
    values=calibration()+[4.]*5+[1.]*5+[4.]*6
    d=make();run(d,calibration())
    with pytest.raises(ValueError):d.update(bearing_id='A',step_id=15,value=float('nan'))
    after=d.update(bearing_id='A',step_id=15,value=4.)
    assert after==run(make(),values[:16])[-1]
    full=run(make(),values)
    assert run(make(),values[:26])==full[:26]
    base=OnlineEpisodicWeakEvent(EpisodeConfig(provenance='rms_episodic_v2'),bearing_id='A')
    for i,(v,state) in enumerate(zip(values,full)):
        assert state.observation==base.update(bearing_id='A',step_id=i,value=v)
    json.dumps(full[-1].to_dict(),allow_nan=False)

def test_invalid_reference_does_not_claim_valid_state():
    d=make();out=run(d,[1.]*16)[-1]
    assert out.feature_deviation_z is None and out.feature_trend_z_per_measurement is None
    assert out.observation.active_event is None and d.finish_record() is None

def test_positive_deviation_without_confirmed_event():
    d=make();run(d,calibration())
    state=d.update(bearing_id='A',step_id=15,value=4.)
    assert state.feature_deviation_z>2
    assert state.observation.active_event is False and state.transition=='none'
    assert state.feature_trend_z_per_measurement is None

def test_idle_gap_and_cross_bearing_rejection():
    d=make();run(d,calibration())
    d.update(bearing_id='A',step_id=15,value=1.)
    gap=d.update(bearing_id='A',step_id=18,value=1.)
    assert gap.observation.active_event is None and gap.termination_now is None
    with pytest.raises(ValueError):d.update(bearing_id='B',step_id=19,value=1.)
    d.reset(bearing_id='B')
    state=d.update(bearing_id='B',step_id=0,value=1.)
    assert state.last_termination is None and state.observation.episode_count==0
