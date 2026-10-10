"""Frozen first-generation z8 audit; explicitly excludes holdout bearings."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr,kurtosis
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(root/'src'))
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.data import ChannelStandardizer
from state_interpreter.preprocessing import LogSTFTPreprocessor
from state_interpreter.adapters import XJTUSYDatasetAdapter
allowed={'Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4'}
cp=root/'runs/ae_baseline_z8_20260801/best_checkpoint.pt'
archive=root/'runs/latent_analysis_z8_20260801/latent_trajectories.csv'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cp,archive)}
with archive.open(encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if r['bearing_id'] in allowed]
assert len(rows)==564
checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
assert checkpoint['split']['train']==['Bearing1_1','Bearing1_2','Bearing1_3']
assert checkpoint['split']['validation']==['Bearing1_4']
model=SmallConvAutoEncoder(**checkpoint['model']);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
st=checkpoint['preprocessing'] if 'preprocessing' in checkpoint else checkpoint['stft']
prep=LogSTFTPreprocessor(n_fft=int(st['n_fft']),win_length=int(st['win_length']),hop_length=int(st['hop_length']),output_size=tuple(st['output_size']))
norm=ChannelStandardizer(mean=checkpoint['standardizer']['mean'],std=checkpoint['standardizer']['std'])
z=np.array([[float(r[f'z_{d}']) for d in range(8)] for r in rows]);train=np.array([r['split']=='train' for r in rows])
pca=PCA().fit(z[train]);pc=pca.transform(z);eig=pca.explained_variance_;prob=eig/eig.sum()
rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0]))))
out.mkdir(exist_ok=False)
summary=[];joined=[];matches=[]
torch.set_num_threads(2)
for bearing in sorted(allowed):
    indices=[i for i,r in enumerate(rows) if r['bearing_id']==bearing];zz=z[indices];tt=np.arange(len(indices))
    own=np.linalg.norm(zz-zz[:15].mean(axis=0),axis=1)
    movement=np.r_[np.nan,np.linalg.norm(np.diff(zz,axis=0),axis=1)]
    rms=[];kur=[]
    choices={0,len(indices)//2,len(indices)-1};sample_images=[]
    for local,idx in enumerate(indices):
        r=rows[idx];path=Path(r['source_path'])
        assert path.parent.name==bearing and bearing in allowed
        raw=XJTUSYDatasetAdapter(path.parent)._read_vibration(path)
        a=raw.numpy().astype(float)
        rms.append(float(np.sqrt(np.mean(a*a,axis=1)).mean()));kur.append(float(kurtosis(a,axis=1,fisher=True,bias=False).mean()))
        if local in choices:
            x=norm.transform(prep(raw).unsqueeze(0))
            with torch.no_grad():o=model(x)
            error=float((o.reconstruction-x).square().mean())
            matches.append({'bearing':bearing,'step_id':int(r['step_id']),'max_z_difference':float(np.max(np.abs(o.z.numpy()[0]-z[idx]))),'mse_difference':abs(error-float(r['reconstruction_mse']))})
            sample_images.append((local,x[0,0].numpy(),o.reconstruction[0,0].numpy()))
        joined.append({'bearing':bearing,'split':r['split'],'step_id':r['step_id'],'rms_pair_mean':rms[-1],'excess_kurtosis_pair_mean':kur[-1],'pc1_train_fitted':float(pc[idx,0]),'own_reference_distance':float(own[local]),'global_reference_distance':float(r['health_center_distance']),'movement':None if local==0 else float(movement[local]),'reconstruction_mse':float(r['reconstruction_mse'])})
    def rho(a,b):return float(spearmanr(np.asarray(a,dtype=float),np.asarray(b,dtype=float)).statistic)
    mse=np.array([float(rows[i]['reconstruction_mse']) for i in indices])
    summary.append({'bearing':bearing,'split':rows[indices[0]]['split'],'n':len(indices),'pc1_rms_rho':rho(pc[indices,0],rms),'pc1_kurtosis_rho':rho(pc[indices,0],kur),'pc1_time_rho':rho(pc[indices,0],tt),'global_distance_time_rho':rho([rows[i]['health_center_distance'] for i in indices],tt),'own_distance_time_rho':rho(own,tt),'mse_time_rho':rho(mse,tt),'mse_median':float(np.median(mse)),'mse_last':float(mse[-1]),'max_movement':float(np.nanmax(movement)),'max_movement_step':int(tt[np.nanargmax(movement)]),'dimension_stds':zz.std(axis=0).tolist()})
    fig,axes=plt.subplots(8,1,sharex=True,figsize=(10,12),constrained_layout=True)
    for d,ax in enumerate(axes):ax.plot(tt,zz[:,d],lw=1);ax.set_ylabel('z'+str(d));ax.grid(alpha=.2)
    axes[-1].set_xlabel('Measurement step ID');fig.suptitle(bearing+' | frozen z8 dimensions');fig.savefig(out/(bearing+'_dimensions.png'),dpi=130);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(10,9),constrained_layout=True)
    for j,(step,x,recon) in enumerate(sample_images):
        lo=min(x.min(),recon.min());hi=max(x.max(),recon.max())
        for k,img in enumerate((x,recon,x-recon)):
            m=axes[j,k].imshow(img,origin='lower',aspect='auto',vmin=lo if k<2 else None,vmax=hi if k<2 else None,cmap='viridis');axes[j,k].set_title(f'step {step}: '+('input','reconstruction','residual')[k]);axes[j,k].set_xlabel('Pooled time bin');axes[j,k].set_ylabel('Pooled frequency bin');fig.colorbar(m,ax=axes[j,k],shrink=.7)
    fig.suptitle(bearing+' | horizontal channel | standardized log-STFT units');fig.savefig(out/(bearing+'_reconstruction.png'),dpi=130);plt.close(fig)
    print('Audited '+bearing,flush=True)
for name,data in [('sample_checks.csv',matches),('feature_latent_mapping.csv',joined)]:
    with (out/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (cp,archive)}
report={'status':'completed','scope':'original z8 baseline only, first condition, train1_1-3 and already-used validation1_4','input_sha256':hashes,'inputs_unchanged':True,'training_pca_variance_ratio':pca.explained_variance_ratio_.tolist(),'training_effective_rank':rank,'training_dimension_std':z[train].std(axis=0).tolist(),'summaries':summary,'sample_checks':matches,'raw_measurements_read':564,'excluded':['Bearing1_5','condition2','condition3'],'limitations':['no health labels','single checkpoint, no new multiseed claims','post-hoc descriptive correlations, no p-values','PCA fit on train only','reconstruction images horizontal channel only','sample inference checks not full re-extraction']}
with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(report,indent=2))
