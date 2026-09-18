"""Condition 3 Arm B: deterministic LOBO AutoEncoder adaptation for Level."""
from __future__ import annotations

import argparse, copy, hashlib, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.preprocessing import LogSTFTPreprocessor
from state_interpreter.signed_axis import fit_signed_axis, signed_level


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def evaluate(model: nn.Module, values: torch.Tensor, batch_size: int) -> float:
    model.eval(); total = 0.0
    with torch.inference_mode():
        for begin in range(0, len(values), batch_size):
            batch = values[begin:begin + batch_size]
            total += float((model(batch).reconstruction - batch).square().sum())
    return total / values.numel()


def encode(model: nn.Module, values: torch.Tensor, batch_size: int) -> np.ndarray:
    chunks=[]; model.eval()
    with torch.inference_mode():
        for begin in range(0, len(values), batch_size):
            chunks.append(model.encode(values[begin:begin + batch_size]).cpu())
    return torch.cat(chunks).numpy()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True); parser.add_argument("--config",required=True); parser.add_argument("--confirm",required=True); args=parser.parse_args()
    if args.confirm != "EXECUTE_CONDITION3_ARM_B_ENCODER_LOBO_ONCE": raise PermissionError("Exact Arm B token required")
    config_path=Path(args.config).resolve(); cfg=json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["status"]!="preregistered_development_only" or cfg["bearings"] != ["Bearing3_1","Bearing3_2","Bearing3_3"] or cfg["protected_bearings"] != ["Bearing3_4","Bearing3_5"]: raise ValueError("Protected split changed")
    expected={x["test"]:tuple(x["train"]) for x in cfg["folds"]}
    if expected != {"Bearing3_1":("Bearing3_2","Bearing3_3"),"Bearing3_2":("Bearing3_1","Bearing3_3"),"Bearing3_3":("Bearing3_1","Bearing3_2")}: raise ValueError("LOBO folds changed")
    if not (ROOT/cfg["arm_a_result"]).is_file(): raise FileNotFoundError("Arm A must complete before Arm B")
    out=ROOT/cfg["output_directory"]
    if out.exists(): raise FileExistsError(f"Immutable output exists: {out}")
    model_cfg=cfg["model"]; seed=int(model_cfg["seed"]); random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    preprocessor=LogSTFTPreprocessor(); raw={}; meta={}; started=time.time()
    for bearing in cfg["bearings"]:
        adapter=XJTUSYDatasetAdapter(Path(args.root),conditions=[cfg["condition"]],bearings=[bearing]); data=materialize_stft_data(adapter,preprocessor)
        raw[bearing]=data.tensors; steps=np.asarray(data.step_ids,int); meta[bearing]=(steps,steps/max(steps))
    baseline=pd.read_csv(ROOT/cfg["source_frozen_result"]).set_index("bearing_id"); arm_a=pd.read_csv(ROOT/cfg["arm_a_result"]).set_index("test_bearing")
    rows=[]; state_rows=[]; histories={}; checkpoints={}
    for fold_index,fold in enumerate(cfg["folds"]):
        test=fold["test"]; train=fold["train"]; fold_seed=seed+fold_index
        random.seed(fold_seed); torch.manual_seed(fold_seed)
        train_raw=torch.cat([raw[b] for b in train]); standardizer=ChannelStandardizer.fit(train_raw); train_values=standardizer.transform(train_raw); test_values=standardizer.transform(raw[test])
        generator=torch.Generator().manual_seed(fold_seed); loader=DataLoader(TensorDataset(train_values),batch_size=int(model_cfg["batch_size"]),shuffle=True,generator=generator)
        model=SmallConvAutoEncoder(2,int(model_cfg["latent_dim"])); optimizer=torch.optim.Adam(model.parameters(),lr=float(model_cfg["learning_rate"])); loss_fn=nn.MSELoss(); history=[]
        for epoch in range(1,int(model_cfg["epochs"])+1):
            model.train(); total=0.0; count=0
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True); output=model(batch); loss=loss_fn(output.reconstruction,batch); loss.backward(); optimizer.step(); total+=float(loss.detach())*len(batch); count+=len(batch)
            history.append({"epoch":epoch,"train_mse":total/count,"heldout_reconstruction_mse":evaluate(model,test_values,int(model_cfg["batch_size"]))})
        histories[test]=history
        train_z=[]
        for b in train: train_z.append(encode(model,standardizer.transform(raw[b]),int(model_cfg["batch_size"])))
        test_z=encode(model,test_values,int(model_cfg["batch_size"])); axis=fit_signed_axis(train_z,calibration_steps=int(cfg["calibration_steps"])); level=signed_level(test_z,axis,calibration_steps=int(cfg["calibration_steps"]),temporal_window=int(cfg["temporal_window"]))
        steps,life=meta[test]; mask=(steps>=int(cfg["score_start_step"]))&np.isfinite(level); scored=level[mask]; rho=float(spearmanr(life[mask],scored).statistic); backward=float(np.mean(np.diff(scored)<-1e-9))
        rows.append({"test_bearing":test,"train_bearings":"+".join(train),"seed":fold_seed,"score_samples":int(mask.sum()),"level_spearman":rho,"level_first_last_delta":float(scored[-1]-scored[0]),"backward_step_fraction":backward,"heldout_reconstruction_mse":history[-1]["heldout_reconstruction_mse"],"baseline_level_spearman":float(baseline.loc[test].level_spearman),"delta_vs_baseline":rho-float(baseline.loc[test].level_spearman),"arm_a_level_spearman":float(arm_a.loc[test].level_spearman),"delta_vs_arm_a":rho-float(arm_a.loc[test].level_spearman)})
        for s,l,v in zip(steps,life,level): state_rows.append({"bearing_id":test,"step_id":int(s),"normalized_lifetime":float(l),"level":v})
        checkpoints[test]={"model_state_dict":copy.deepcopy(model.state_dict()),"standardizer":{"mean":standardizer.mean,"std":standardizer.std},"axis":axis,"train":train,"test":test,"seed":fold_seed}
    out.mkdir(parents=True)
    summary=pd.DataFrame(rows); summary.to_csv(out/"lobo_summary.csv",index=False); pd.DataFrame(state_rows).to_csv(out/"lobo_levels.csv",index=False)
    for test,history in histories.items(): pd.DataFrame(history).to_csv(out/f"{test}_training_curve.csv",index=False)
    for test,checkpoint in checkpoints.items(): torch.save(checkpoint,out/f"{test}_checkpoint.pt")
    decision={"all_level_rho_positive":bool((summary.level_spearman>0).all()),"majority_improved_vs_baseline":bool((summary.delta_vs_baseline>0).sum()>=2),"encoder_adaptation_supported":bool((summary.level_spearman>0).all() and (summary.delta_vs_baseline>0).sum()>=2)}
    report={"experiment_id":cfg["experiment_id"],"status":"completed","config_sha256":sha256(config_path),"protected_bearings_not_read":cfg["protected_bearings"],"elapsed_seconds":time.time()-started,"movement_not_scored_reason":cfg["scope_note"],"decision":decision,"fallacy_scan_11_of_11":{"simpsons_paradox":"Per-bearing LOBO results; no pooled correlation.","ecological_fallacy":"Bearing is the independent evaluation unit.","berkson_bias":"Laboratory run-to-failure selection limits deployment generalization.","collider_bias":"No causal conditioning claim.","base_rate_neglect":"No event classification is performed in Arm B.","regression_to_mean":"No intervention claim.","survivorship_bias":"Complete trajectories omit field censoring.","look_elsewhere_effect":"Folds, epochs, seed and metrics were preregistered.","researcher_degrees_of_freedom":"Fixed five epochs; held-out bearing is not used for checkpoint selection.","correlation_causation":"Lifetime correlation is not physical damage causation.","reverse_causality":"Temporal order is preserved without causal inference."}}
    (out/"experiment_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(10,5)); states=pd.DataFrame(state_rows)
    for b,g in states.groupby("bearing_id",sort=False): ax.plot(g.normalized_lifetime,g.level,label=b)
    ax.set(xlabel="Normalized lifetime",ylabel="Arm B LOBO Level"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out/"lobo_levels.png",dpi=180); plt.close(fig)
    print(summary.to_string(index=False)); print(json.dumps(decision,indent=2))


if __name__=="__main__": main()
