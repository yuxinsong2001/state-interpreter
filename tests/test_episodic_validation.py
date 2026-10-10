import numpy as np
import pytest
from state_interpreter.episodic_validation import evaluate_event_stream

def data():
    seq=np.array([0.,2.]*7+[1.]+[4.]*5+[1.]*5+[4.]*5)
    return np.arange(len(seq)),np.column_stack([seq,seq])

def test_complete_and_censored_episodes():
    steps,pairs=data()
    rows,ends,summaries,checks=evaluate_event_stream('Bearing3_4',steps,pairs)
    assert len(rows)==60 and len(ends)==4 and all(checks.values())
    assert all(s['episodes']==2 for s in summaries)
    assert [e['reason'] for e in ends]==['consecutive_non_exceedances','record_end']*2
    assert ends[0]['observed_active_measurements']==5

def test_invalid_reference_recorded_as_failure_not_health():
    _,_,summary,checks=evaluate_event_stream('Bearing3_4',np.arange(20),np.ones((20,2)))
    assert checks['references_valid'] is False
    assert all(s['episodes']==0 for s in summary)

def test_gap_interrupts_active_event_without_recovery():
    steps,pairs=data();steps[20:]+=2
    rows,ends,_,checks=evaluate_event_stream('Bearing3_4',steps,pairs)
    assert all(checks.values())
    assert ends[0]['reason']=='measurement_gap' and ends[0]['right_censored']
    assert rows[20]['active_event'] is None

def test_protected_bearing_rejected():
    steps,pairs=data()
    with pytest.raises(PermissionError):evaluate_event_stream('Bearing3_5',steps,pairs)

@pytest.mark.parametrize('kind',['nonfinite','reversed','calibration_gap'])
def test_input_failure(kind):
    steps,pairs=data()
    if kind=='nonfinite':pairs[25,0]=np.nan
    if kind=='reversed':steps[20]=19
    if kind=='calibration_gap':steps[5:]+=1
    with pytest.raises(ValueError):evaluate_event_stream('Bearing3_4',steps,pairs)
