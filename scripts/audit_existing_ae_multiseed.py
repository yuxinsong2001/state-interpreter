"""Post-hoc frozen AE audit: development bearings only; no training."""
import sys, json, hashlib, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr, kurtosis
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, data_root, out = map(Path, sys.argv[1:4])
sys.path.insert(0, str(root/'src'))
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.data import ChannelStandardizer
from state_interpreter.preprocessing import LogSTFTPreprocessor
from state_interpreter.adapters import XJTUSYDatasetAdapter

bearings = ['Bearing3_1','Bearing3_2','Bearing3_3']
bases = [20260918,20260921,20260924]
paths = [root/f'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_{s}/{b}_checkpoint.pt' for s in bases for b in bearings]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(root)):sha(p) for p in paths}
assert not out.exists(), 'Immutable output already exists'
out.mkdir(parents=True)
torch.set_num_threads(2)
started=time.time()
raw={}; features={}
prep=LogSTFTPreprocessor()
for b in bearings:
    directory=data_root/'40Hz10kN'/b
    adapter=XJTUSYDatasetAdapter(directory)
    files=adapter._ordered_csv_files(directory)
    xs=[]; fs=[]
    for i,p in enumerate(files):
        a=adapter._read_vibration(p)
        xs.append(prep(a))
        v=a.numpy().astype(np.float64)
        fs.append([float(np.sqrt(np.mean(v*v,axis=1)).mean()),float(kurtosis(v,axis=1,fisher=True,bias=False).mean())])
        if (i+1)%500==0:print(f'{b}: {i+1}/{len(files)} materialized',flush=True)
    raw[b]=torch.stack(xs);features[b]=np.array(fs)
    print(f'{b}: completed {len(files)}',flush=True)

rows=[]; latent_rows=[]
for base in bases:
    for test in bearings:
        p=root/f'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_{base}/{test}_checkpoint.pt'
        # Trusted project-generated checkpoint includes a numpy axis.
        cp=torch.load(p,map_location='cpu',weights_only=False)
        train=cp['train'];assert cp['test']==test and set(train)==set(bearings)-{test}
        model=SmallConvAutoEncoder(2,8);model.load_state_dict(cp['model_state_dict']);model.eval()
        norm=ChannelStandardizer(**cp['standardizer'])
        zs={};mses={}
        with torch.inference_mode():
            for b in bearings:
                chunks=[];errors=[]
                for start in range(0,len(raw[b]),64):
                    x=norm.transform(raw[b][start:start+64]);o=model(x)
                    chunks.append(o.z.numpy());errors.append((o.reconstruction-x).square().mean(dim=(1,2,3)).numpy())
                zs[b]=np.concatenate(chunks);mses[b]=np.concatenate(errors)
        train_z=np.concatenate([zs[b] for b in train])
        pca=PCA().fit(train_z);prob=pca.explained_variance_ratio_;positive=prob[prob>0]
        rank=float(np.exp(-np.sum(positive*np.log(positive))))
        z=zs[test];pc=pca.transform(z)[:,0];steps=np.arange(len(z));own=np.linalg.norm(z-z[:15].mean(axis=0),axis=1)
        global_center=np.concatenate([zs[b][:15] for b in train]).mean(axis=0)
        global_distance=np.linalg.norm(z-global_center,axis=1)
        def rho(a,b):return float(spearmanr(a,b).statistic)
        rows.append(dict(base_seed=base,actual_seed=int(cp['seed']),test_bearing=test,train_bearings='+'.join(train),n=len(z),pc1_variance_ratio=float(prob[0]),effective_rank=rank,pc1_rms_abs_rho=abs(rho(pc,features[test][:,0])),pc1_kurtosis_abs_rho=abs(rho(pc,features[test][:,1])),own_distance_time_rho=rho(own,steps),global_distance_time_rho=rho(global_distance,steps),mse_time_rho=rho(mses[test],steps),mean_reconstruction_mse=float(mses[test].mean()),min_training_dimension_std=float(train_z.std(axis=0).min())))
        for i in range(len(z)):
            latent_rows.append(dict(base_seed=base,test_bearing=test,step_id=i,rms_pair_mean=features[test][i,0],excess_kurtosis_pair_mean=features[test][i,1],pc1=pc[i],own_distance=own[i],global_distance=global_distance[i],reconstruction_mse=mses[test][i],**{f'z_{d}':z[i,d] for d in range(8)}))
        print(f'Audited base {base}, {test}: rank={rank:.3f}, PC1={prob[0]:.4f}',flush=True)
for name,data in [('metrics.csv',rows),('heldout_latents.csv',latent_rows)]:
    with (out/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
assert hashes=={str(p.relative_to(root)):sha(p) for p in paths}
summary={}
for b in bearings:
    rr=[r for r in rows if r['test_bearing']==b]
    summary[b]={key:{'min':min(r[key] for r in rr),'max':max(r[key] for r in rr),'mean':float(np.mean([r[key] for r in rr]))} for key in ['pc1_variance_ratio','effective_rank','pc1_rms_abs_rho','own_distance_time_rho','global_distance_time_rho']}
report=dict(status='completed',verification_status='ANALYZED',scope='frozen nine reconstruction-AE checkpoints on already-used condition3 development bearings',raw_measurements_read=sum(len(v) for v in raw.values()),protected_bearings_not_read=['Bearing3_4','Bearing3_5'],no_training=True,input_sha256=hashes,inputs_unchanged=True,elapsed_seconds=time.time()-started,summary=summary,health_aware_latent_audit='not executable: original run did not save weights',ts2vec_latent_audit='not included: different input/scaling/context requires separate matched audit',limitations=['post-hoc descriptive audit, no p-values or confidence intervals','three seeds per fold are not independent bearings','within-bearing lifetime association is not a physical health label','higher effective rank is not automatically better','PCA fitted on each fold training bearings only','global-distance mapping is diagnostic, not the original signed-axis interpreter'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,key,label in zip(axes,['pc1_variance_ratio','pc1_rms_abs_rho','own_distance_time_rho'],['Train PC1 explained variance (fraction)','Held-out |Spearman PC1 vs RMS|','Held-out Spearman distance vs step']):
    for j,b in enumerate(bearings):
        vals=[r[key] for r in rows if r['test_bearing']==b]
        ax.scatter([j-.12,j,j+.12],vals,s=40)
    ax.set_xticks(range(3),bearings);ax.set_ylabel(label);ax.grid(alpha=.2)
fig.suptitle('Frozen AE audit | three seeds per LOBO fold | source: metrics.csv')
fig.savefig(out/'multiseed_audit.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2),flush=True)
