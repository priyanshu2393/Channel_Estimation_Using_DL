"""Train PACE-Net with the repository's saved split and validation checkpoint rule."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from attrnet_ofdm.data import SavedOFDMDataset
from attrnet_ofdm.pacenet import PACEConfig, PACENet, loss_fn, parameter_count, to_pilot_tokens
from pilot_design.pilot_utils import FullGridCandidateDataset, pilot_coordinates, validate_pilot_indices


class LabeledSavedDataset(torch.utils.data.Dataset):
    def __init__(self, source): self.source = source
    def __len__(self): return len(self.source)
    def __getitem__(self, i): return (*self.source[i], torch.tensor(float(self.source.snr_db[i])))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def coords_from_manifest(manifest):
    pc = manifest["pilot_configuration"]
    if "frequency_indices" in pc:
        f, t = pc["frequency_indices"], pc["time_indices"]
    else:
        f = (np.asarray(pc["frequency_indices_matlab_one_based"]) - 1).tolist()
        t = (np.asarray(pc["time_indices_matlab_one_based"]) - 1).tolist()
    return np.asarray([(a,b) for a in f for b in t], dtype=np.int64)


@torch.no_grad()
def evaluate(model, loader, coords, device, max_batches=None):
    model.eval(); aggregate = {}
    for batch_i, item in enumerate(loader):
        if max_batches is not None and batch_i >= max_batches: break
        x, y = item[0].to(device), item[1].to(device)
        snr = item[2].cpu().numpy() if len(item) > 2 else np.zeros(len(x))
        tokens, pc, pad = to_pilot_tokens(x, coords)
        pred = model(tokens.to(device), pc.to(device), pad)
        err = (pred-y).square().sum(dim=1)
        power = y.square().sum(dim=1)
        for j, s in enumerate(snr):
            row = aggregate.setdefault(float(s), [0.0,0.0,0])
            row[0] += err[j].sum().item(); row[1] += power[j].sum().item(); row[2] += err[j].numel()
    return {s: {"mse": e/n, "nmse": e/max(p,1e-30), "nmse_db": 10*np.log10(max(e/max(p,1e-30),1e-30)), "resource_elements": n} for s,(e,p,n) in aggregate.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pilots",type=int,choices=[12,16,24,48],default=24)
    p.add_argument("--data-dir",type=Path,default=None)
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-2)
    p.add_argument("--dropout",type=float,default=0.1)
    p.add_argument("--warmup-epochs",type=float,default=5.0)
    p.add_argument("--output",type=Path,default=None); p.add_argument("--seed",type=int,default=2025)
    p.add_argument("--preset",choices=["pace_s","pace_m"],default="pace_s")
    p.add_argument("--pilot-indices",type=Path,default=None,help="Optional ordered flat 72x14 indices for an irregular mask")
    p.add_argument("--max-train-batches",type=int,default=None,help="Optional batch cap for smoke runs")
    p.add_argument("--max-eval-batches",type=int,default=None,help="Optional validation/test batch cap for smoke runs")
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--L",type=int,default=None); p.add_argument("--K",type=int,default=None); p.add_argument("--delay-step",type=float,default=None)
    p.add_argument("--no-basis",action="store_true"); p.add_argument("--no-linear-branch",action="store_true"); p.add_argument("--fixed-basis",action="store_true")
    p.add_argument("--signed-linear",action="store_true"); p.add_argument("--no-refine",action="store_true")
    args=p.parse_args()
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        p.error("Require learning-rate > 0, weight-decay >= 0, and dropout in [0,1).")
    if args.epochs < 1 or args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        p.error("Require epochs >= 1 and 0 <= warmup-epochs < epochs.")
    seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir=args.data_dir or Path("data")/f"epa7_{args.pilots}pilots"
    cfg=PACEConfig.preset(args.preset)
    if args.L is not None: cfg.L=args.L
    if args.K is not None: cfg.K=args.K
    if args.delay_step is not None: cfg.delay_step=args.delay_step
    cfg.dropout=args.dropout
    cfg.use_basis=not args.no_basis; cfg.use_linear_branch=not args.no_linear_branch
    cfg.learnable_basis=not args.fixed_basis; cfg.linear_attn_signed=args.signed_linear; cfg.use_refine=not args.no_refine
    if args.pilot_indices:
        indices=validate_pilot_indices(np.load(args.pilot_indices,allow_pickle=False),args.pilots)
        coords=pilot_coordinates(indices)
        train=FullGridCandidateDataset(data_dir,"train",seed=args.seed); val=FullGridCandidateDataset(data_dir,"validation",seed=args.seed); test=FullGridCandidateDataset(data_dir,"test",seed=args.seed)
    else:
        train=SavedOFDMDataset(data_dir,"train"); val=LabeledSavedDataset(SavedOFDMDataset(data_dir,"validation")); test=LabeledSavedDataset(SavedOFDMDataset(data_dir,"test"))
        if train.pilots != args.pilots: p.error(f"--pilots={args.pilots} differs from manifest ({train.pilots})")
        coords=coords_from_manifest(train.manifest)
    cfg.Nf,cfg.Nt=72,14
    output=args.output or Path(f"runs/pacenet_{args.pilots}_{args.preset}_d{cfg.d}_{'joint' if args.pilot_indices else 'uniform'}_seed{args.seed}_adamw")
    output.mkdir(parents=True,exist_ok=True)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=="cuda")
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=="cuda")
    test_loader=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=="cuda")
    model=PACENet(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    steps_per_epoch=min(len(train_loader),args.max_train_batches) if args.max_train_batches is not None else len(train_loader)
    if steps_per_epoch < 1: p.error("The training loader must provide at least one batch.")
    total_steps=args.epochs*steps_per_epoch
    warmup_steps=min(int(round(args.warmup_epochs*steps_per_epoch)),total_steps-1)
    global_step=0
    training_started=time.perf_counter()
    print(f"device={device}; parameters={parameter_count(model):,}; preset={args.preset}; dropout={cfg.dropout:g}; optimizer=AdamW; lr={args.learning_rate:g}; weight_decay={args.weight_decay:g}; warmup={args.warmup_epochs:g} epochs then cosine; coords={len(coords)}; data={data_dir}")
    best=float("inf"); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; count=0; start=time.perf_counter(); epoch_lrs=[]
        for batch_i,item in enumerate(train_loader):
            if args.max_train_batches is not None and batch_i >= args.max_train_batches: break
            if global_step < warmup_steps:
                lr_scale=(global_step+1)/max(warmup_steps,1)
            else:
                progress=(global_step-warmup_steps)/max(total_steps-warmup_steps,1)
                lr_scale=0.5*(1.0+np.cos(np.pi*min(progress,1.0)))
            lr=args.learning_rate*float(lr_scale)
            for group in optimizer.param_groups: group["lr"]=lr
            epoch_lrs.append(lr)
            x,y=item[0].to(device),item[1].to(device)
            h,pc,pad=to_pilot_tokens(x,coords)
            optimizer.zero_grad(set_to_none=True); pred=model(h,pc,pad); loss=loss_fn(pred,y); loss.backward(); optimizer.step()
            total += loss.item()*len(x); count += len(x)
            global_step += 1
        train_mse=total/count; val_metrics=evaluate(model,val_loader,coords,device,args.max_eval_batches)
        # validation MSE is pooled over samples and SNR strata (all fixed-size here).
        val_mse=float(np.mean([v["mse"] for v in val_metrics.values()]))
        duration=time.perf_counter()-start
        history.append({"epoch":epoch,"train_mse":train_mse,"val_mse":val_mse,"lr_start":epoch_lrs[0],"lr_end":epoch_lrs[-1],"seconds":duration})
        print(f"epoch {epoch:03d}/{args.epochs}: train_mse={train_mse:.6g} val_mse={val_mse:.6g} lr={epoch_lrs[-1]:.3g} seconds={duration:.2f}")
        if val_mse < best:
            best=val_mse
            torch.save({"state_dict":model.state_dict(),"config":cfg.__dict__,"pilots":args.pilots,"seed":args.seed,"val_mse":best,"pilot_coords":coords.tolist()},output/"best.pt")
    (output/"train_log.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    config={**vars(args),"config":cfg.__dict__,"optimizer":"AdamW","scheduler":"linear_warmup_then_cosine","warmup_steps":warmup_steps,"total_optimizer_steps":total_steps,"pilot_coords":coords.tolist(),"parameter_count":parameter_count(model),"data_dir":str(data_dir),"training_seconds_total":time.perf_counter()-training_started}
    (output/"config.json").write_text(json.dumps(config,default=str,indent=2),encoding="utf-8")
    ckpt=torch.load(output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ckpt["state_dict"])
    metrics=evaluate(model,test_loader,coords,device,args.max_eval_batches)
    (output/"metrics.json").write_text(json.dumps({"test_per_snr":metrics,"best_val_mse":best},indent=2),encoding="utf-8")
    print(f"best val MSE={best:.7g}; test metrics saved to {output/'metrics.json'}")

if __name__=="__main__": main()
