"""Train capacity-matched AttRNet or residual CNN baselines on saved splits."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from attrnet_ofdm.baselines import CapacityAttRNet, ResidualCNN
from attrnet_ofdm.data import SavedOFDMDataset
from attrnet_ofdm.model import parameter_count

def mse(model,loader,device):
    model.eval(); total=count=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); v=(model(x)-y).square().sum(1).mean(); total+=v.item()*len(x); count+=len(x)
    return total/count

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["attnet","rescnn"],required=True); p.add_argument("--attention",choices=["fc","conv"],default="fc"); p.add_argument("--features",type=int,default=24); p.add_argument("--blocks",type=int,default=3); p.add_argument("--pilots",type=int,choices=[12,16,24,48],default=24); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--seed",type=int,default=2025); p.add_argument("--output",type=Path,required=True); p.add_argument("--workers",type=int,default=0); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); a.output.mkdir(parents=True,exist_ok=True)
    train=SavedOFDMDataset(a.data_dir,"train"); val=SavedOFDMDataset(a.data_dir,"validation"); test=SavedOFDMDataset(a.data_dir,"test")
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=="cuda"); vl=DataLoader(val,batch_size=a.batch_size,num_workers=a.workers); tl=DataLoader(test,batch_size=a.batch_size,num_workers=a.workers)
    model=(CapacityAttRNet(a.attention,a.features,a.blocks) if a.model=="attnet" else ResidualCNN(a.features,a.blocks)).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); best=float("inf"); history=[]
    print(f"device={device}; model={a.model}; params={parameter_count(model):,}")
    for epoch in range(1,a.epochs+1):
        model.train(); total=count=0; start=time.perf_counter()
        for x,y in tr:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=(model(x)-y).square().sum(1).mean(); loss.backward(); opt.step(); total+=loss.item()*len(x); count+=len(x)
        train_mse=total/count; val_mse=mse(model,vl,device); history.append({"epoch":epoch,"train_mse":train_mse,"val_mse":val_mse,"seconds":time.perf_counter()-start}); print(f"epoch {epoch:03d}: train_mse={train_mse:.6g} val_mse={val_mse:.6g}")
        if val_mse<best:
            best=val_mse; torch.save({"state_dict":model.state_dict(),"model":a.model,"attention":a.attention,"features":a.features,"blocks":a.blocks,"pilots":a.pilots,"seed":a.seed,"val_mse":best},a.output/"best.pt")
    (a.output/"train_log.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); (a.output/"config.json").write_text(json.dumps({**vars(a),"parameters":parameter_count(model)},default=str,indent=2),encoding="utf-8")
    ck=torch.load(a.output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["state_dict"]); model.eval(); groups={}
    with torch.no_grad():
        for i in range(len(test)):
            x,y=test[i]; pred=model(x.unsqueeze(0).to(device)).cpu()[0]; snr=float(test.snr_db[i]); row=groups.setdefault(snr,[0.,0.,0]); err=(pred-y).square().sum(0); row[0]+=err.sum().item(); row[1]+=y.square().sum().item(); row[2]+=err.numel()
    metrics={s:{"mse":e/n,"nmse":e/max(q,1e-30),"nmse_db":10*np.log10(max(e/max(q,1e-30),1e-30)),"samples":800} for s,(e,q,n) in groups.items()}
    (a.output/"metrics.json").write_text(json.dumps({"per_snr":metrics,"best_val_mse":best},indent=2),encoding="utf-8")
if __name__=="__main__": main()
