"""Re-extract Bearing1_4 with the archived checkpoint; inspect endpoint sensitivity."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo.resolve()/'src'))
    from state_interpreter.data import ChannelStandardizer, materialize_stft_data
    from state_interpreter.adapters import XJTUSYDatasetAdapter
    from state_interpreter.encoders import SmallConvAutoEncoder
    from state_interpreter.preprocessing import LogSTFTPreprocessor
    output=args.audit_dir/'alignment_verification.json'
    if output.exists(): raise FileExistsError(output)
    checkpoint=torch.load(args.repo/'runs/ae_baseline_z8_20260801/best_checkpoint.pt',map_location='cpu',weights_only=True)
    model=SmallConvAutoEncoder(**checkpoint['model'])
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    stft=checkpoint['stft']
    preprocessor=LogSTFTPreprocessor(n_fft=stft['n_fft'],win_length=stft['win_length'],hop_length=stft['hop_length'],output_size=tuple(stft['output_size']))
    adapter=XJTUSYDatasetAdapter(args.root,conditions=['35Hz12kN'],bearings=['Bearing1_4'])
    torch.set_num_threads(2)
    data=materialize_stft_data(adapter,preprocessor)
    inputs=ChannelStandardizer(**checkpoint['standardizer']).transform(data.tensors)
    with torch.inference_mode():
        z=torch.cat([model(inputs[i:i+32]).z for i in range(0,len(inputs),32)]).numpy()
    with (args.repo/'runs/latent_analysis_z8_20260801/latent_trajectories.csv').open(encoding='utf-8') as f:
        archived=[r for r in csv.DictReader(f) if r['bearing_id']=='Bearing1_4']
    previous=np.array([[float(r[f'z_{j}']) for j in range(8)] for r in archived])
    assert [int(r['step_id']) for r in archived]==list(data.step_ids)
    with (args.audit_dir/'aligned_evidence.csv').open(encoding='utf-8') as f:
        rows=[r for r in csv.DictReader(f) if r['bearing_id']=='Bearing1_4']
    sensitivity={}
    for name in ['rms_h','rms_v','latent_pc1','local_distance']:
        values=np.array([float(r[name]) for r in rows])
        trimmed=values[:-1]
        sensitivity[name]={'without_final_rho':float(spearmanr(np.arange(len(trimmed)),trimmed).statistic),
                           'final_over_previous':float(values[-1]/values[-2]),
                           'median_measurements_16_45':float(np.median(values[15:45])),
                           'median_measurements_92_121':float(np.median(values[91:121]))}
    result={'rows':len(z),'max_latent_absolute_difference':float(np.max(np.abs(z-previous))),
            'allclose_rtol_1e-5_atol_1e-5':bool(np.allclose(z,previous,rtol=1e-5,atol=1e-5)),
            'endpoint_sensitivity_descriptive_only':sensitivity,
            'torch_version':torch.__version__,
            'note':'Original metrics unchanged; endpoint exclusion is a declared diagnostic, not model selection.'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
