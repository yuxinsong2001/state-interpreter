"""Validate Condition 3 preregistration without opening the dataset root."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from state_interpreter.condition3_access import Condition3AccessPolicy
def main():
 a=argparse.ArgumentParser(); a.add_argument('--config',required=True); x=a.parse_args()
 policy=Condition3AccessPolicy.load(x.config,ROOT); verified=policy.verify_artifacts()
 report={'status':'preflight_passed_no_bearing3_data_read','permitted_development_bearings':policy.config['split']['development'],'protected_validation':policy.config['split']['validation'],'protected_blind':policy.config['split']['blind'],'verified_frozen_artifacts':verified,'next_command_requires_explicit_development_token':policy.config['tokens']['development']}
 print(json.dumps(report,indent=2))
if __name__=='__main__': main()
