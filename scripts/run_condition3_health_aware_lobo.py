from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer,materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.preprocessing import LogSTFTPreprocessor
def smooth(x,w): return np.array([np.mean(x[max(0,i-w+1):i+1]) for i in range(len(x))])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--confirm',required=True);a=p.parse_args()
 if a.confirm!='EXECUTE_HEALTH_AWARE_LOBO_ONCE':raise PermissionError('token')
 c=json.loads(Path(a.config).read_text());out=ROOT/c['output_directory'];
 if out.exists():raise FileExistsError(out)
 if c['bearings']!=['Bearing3_1','Bearing3_2','Bearing3_3'] or c['protected']!=['Bearing3_5']:raise ValueError('boundary')
 cfg=c['model'];prep=LogSTFTPreprocessor();raw={};meta={};started=time.time()
 for b in c['bearings']:
  d=materialize_stft_data(XJTUSYDatasetAdapter(Path(a.root),conditions=[c['condition']],bearings=[b]),prep);raw[b]=d.tensors;s=np.asarray(d.step_ids,int);meta[b]=(s,s/max(s))
 rows=[];curves=[]
 for fi,f in enumerate(c['folds']):
  seed=cfg['seed']+fi;random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);train=f['train'];test=f['test'];std=ChannelStandardizer.fit(torch.cat([raw[b] for b in train]));xs=[];ys=[];bs=[]
  for bi,b in enumerate(train):xs.append(std.transform(raw[b]));ys.append(torch.tensor(meta[b][1],dtype=torch.float32));bs.append(torch.full((len(raw[b]),),bi))
  X=torch.cat(xs);Y=torch.cat(ys);B=torch.cat(bs);loader=DataLoader(TensorDataset(X,Y,B),batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed));ae=SmallConvAutoEncoder(2,cfg['latent_dim']);head=nn.Linear(cfg['latent_dim'],1);opt=torch.optim.Adam(list(ae.parameters())+list(head.parameters()),lr=cfg['learning_rate']);hist=[]
  for ep in range(1,cfg['epochs']+1):
   total=rank_total=n=0
   for x,y,b in loader:
    opt.zero_grad(set_to_none=True);o=ae(x);h=head(o.z).squeeze(1);recon=(o.reconstruction-x).square().mean();dy=y[:,None]-y[None,:];same=b[:,None].eq(b[None,:]);mask=same&(dy.abs()>=cfg['minimum_lifetime_gap']);dh=h[:,None]-h[None,:];rank=torch.nn.functional.softplus(-torch.sign(dy[mask])*dh[mask]).mean() if mask.any() else h.sum()*0;loss=recon+cfg['ranking_weight']*rank;loss.backward();opt.step();total+=float(recon)*len(x);rank_total+=float(rank)*len(x);n+=len(x)
   hist.append({'fold':test,'epoch':ep,'reconstruction_loss':total/n,'ranking_loss':rank_total/n})
  values=std.transform(raw[test]);hs=[]
  with torch.inference_mode():
   for i in range(0,len(values),32):hs.append(head(ae.encode(values[i:i+32])).squeeze(1).cpu())
  h=torch.cat(hs).numpy();h=smooth(h-h[:c['calibration_steps']].mean(),c['temporal_window']);steps,life=meta[test];mask=steps>=c['score_start_step'];rho=float(spearmanr(life[mask],h[mask]).statistic);rows.append({'test_bearing':test,'train_bearings':'+'.join(train),'seed':seed,'rho':rho,'first_last_delta':float(h[mask][-1]-h[mask][0]),'backward_fraction':float(np.mean(np.diff(h[mask])<0))})
  curves.extend({'bearing_id':test,'step_id':int(s),'normalized_lifetime':float(l),'health_score':float(v)} for s,l,v in zip(steps,life,h))
 out.mkdir(parents=True);summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(curves).to_csv(out/'trajectories.csv',index=False);pd.DataFrame(hist).to_csv(out/'training_curves.csv',index=False);decision={'all_positive':bool((summary.rho>0).all()),'continue_health_aware_family':bool((summary.rho>0).all()),'protected_bearing_not_read':'Bearing3_5'};(out/'report.json').write_text(json.dumps({'status':'completed','elapsed_seconds':time.time()-started,'decision':decision,'fallacy_scan_11_of_11':{'simpsons':'per-bearing','ecological':'bearing unit','berkson':'lab run-to-failure','collider':'no causal conditioning','base_rate':'no event claim','regression':'no intervention','survivorship':'no field censoring','look_elsewhere':'locked folds/metrics','forking_paths':'locked hyperparameters','causation':'time is proxy','reverse_causality':'no causal claim'}},indent=2));print(summary.to_string(index=False));print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
