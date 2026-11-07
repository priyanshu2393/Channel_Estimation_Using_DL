"""Evaluate a PACE-Net checkpoint and report per-SNR accuracy and runtime."""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from attrnet_ofdm.data import SavedOFDMDataset
from attrnet_ofdm.pacenet import PACEConfig, PACENet, parameter_count, to_pilot_tokens
from train_pacenet import LabeledSavedDataset, evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--split",choices=["validation","test"],default="test"); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--latency-runs",type=int,default=100); p.add_argument("--output",type=Path,default=None); a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False); cfg=PACEConfig(**ck["config"]); model=PACENet(cfg).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    data=LabeledSavedDataset(SavedOFDMDataset(a.data_dir,a.split)); loader=DataLoader(data,batch_size=a.batch_size,shuffle=False)
    coords=np.asarray(ck["pilot_coords"],dtype=np.int64)
    metrics=evaluate(model,loader,coords,device)
    sample=data[0]; h,pc,_=to_pilot_tokens(sample[0].unsqueeze(0).to(device),coords)
    with torch.no_grad():
        for _ in range(10): model(h,pc)
        activities=[ProfilerActivity.CPU]
        if device.type=="cuda": activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities,with_flops=True) as prof: model(h,pc)
        flops=sum(int(getattr(event,"flops",0) or 0) for event in prof.key_averages())
        if device.type=="cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(max(100,a.latency_runs)):
            start=time.perf_counter(); model(h,pc)
            if device.type=="cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
    mean_nmse=float(np.mean([v["nmse"] for v in metrics.values()])) if metrics else None
    report={"split":a.split,"device":str(device),"parameters":parameter_count(model),"batch1_latency_ms_median":statistics.median(times),"latency_runs":len(times),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,"per_snr":metrics,"mean_nmse_across_snr":mean_nmse,"mean_nmse_db_across_snr":10*np.log10(mean_nmse) if mean_nmse else None,"torch_profiler_forward_flops":flops,"torch_profiler_forward_macs_approx":flops//2}
    out=a.output or a.checkpoint.parent/ f"{a.split}_metrics.json"; out.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
