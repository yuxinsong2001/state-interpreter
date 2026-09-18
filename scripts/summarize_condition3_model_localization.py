"""Aggregate Condition 3 Arm A/B evidence and exploratory two-sided Level."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-09-18_condition3_model_localization_summary'
if OUT.exists(): raise FileExistsError(OUT)
arm_a=pd.read_csv(ROOT/'results/2026-09-18_condition3_arm_a_lobo/lobo_summary.csv')
paths=[ROOT/f'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_{s}/lobo_summary.csv' for s in (20260918,20260921,20260924)]
frames=[]
for base,path in zip((20260918,20260921,20260924),paths):
 d=pd.read_csv(path); d['base_seed']=base; frames.append(d)
all_b=pd.concat(frames,ignore_index=True)
aggregate=all_b.groupby('test_bearing').agg(level_rho_mean=('level_spearman','mean'),level_rho_std=('level_spearman','std'),level_rho_min=('level_spearman','min'),level_rho_max=('level_spearman','max'),positive_seed_fraction=('level_spearman',lambda x:float(np.mean(x>0))),delta_vs_baseline_mean=('delta_vs_baseline','mean'),reconstruction_mse_mean=('heldout_reconstruction_mse','mean')).reset_index()
states=pd.read_csv(ROOT/'results/2026-09-18_condition3_arm_a_lobo/lobo_state_trajectories.csv'); states=states[(states.step_id>=25)&states.level.notna()]
explore=[]
for bearing,g in states.groupby('bearing_id',sort=False):
 explore.append({'bearing_id':bearing,'signed_rho':float(spearmanr(g.normalized_lifetime,g.level).statistic),'absolute_axis_displacement_rho':float(spearmanr(g.normalized_lifetime,g.level.abs()).statistic),'delta':float(spearmanr(g.normalized_lifetime,g.level.abs()).statistic-spearmanr(g.normalized_lifetime,g.level).statistic)})
explore=pd.DataFrame(explore)
OUT.mkdir(parents=True); all_b.to_csv(OUT/'arm_b_all_seeds.csv',index=False); aggregate.to_csv(OUT/'arm_b_multiseed_summary.csv',index=False); explore.to_csv(OUT/'exploratory_two_sided_level.csv',index=False)
decision={'arm_a_interpreter_only_supported':False,'arm_b_reconstruction_encoder_adaptation_supported':False,'evidence':{'B3_3_negative_all_arm_b_seeds':bool((all_b[all_b.test_bearing=='Bearing3_3'].level_spearman<0).all()),'B3_1_sign_stable_across_arm_b_seeds':bool((all_b[all_b.test_bearing=='Bearing3_1'].level_spearman>0).all() or (all_b[all_b.test_bearing=='Bearing3_1'].level_spearman<0).all()),'exploratory_two_sided_positive_all_bearings':bool((explore.absolute_axis_displacement_rho>0).all())},'next_candidate':'Preregister a two-sided axial-displacement Level; do not call the post-hoc development check validation. Compare against signed Level on Condition 1 development evidence before deciding whether to unlock Bearing3_4.','protected_bearings_not_read':['Bearing3_4','Bearing3_5']}
report={'status':'completed','decision':decision,'warning':'The two-sided Level was inspected after observing signed-Level failures and is hypothesis-generating only.','fallacy_scan_11_of_11':{'simpsons_paradox':'All decisions remain per bearing; no pooled time series.','ecological_fallacy':'Bearing, not time point, is the evaluation unit.','berkson_bias':'Laboratory run-to-failure selection limits scope.','collider_bias':'No conditioned causal estimate.','base_rate_neglect':'No fault-event classification in this summary.','regression_to_mean':'No intervention comparison.','survivorship_bias':'Field censoring is absent.','look_elsewhere_effect':'Two-sided Level is explicitly labeled post-hoc exploratory.','researcher_degrees_of_freedom':'Arm A/B protocols were preregistered; the new candidate is not treated as confirmation.','correlation_causation':'Temporal association is not physical damage causation.','reverse_causality':'No causal direction is claimed.'}}
(OUT/'experiment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(aggregate.to_string(index=False)); print(explore.to_string(index=False)); print(json.dumps(decision,indent=2))
