from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from attrnet_ofdm.data import SavedOFDMDataset
from attrnet_ofdm.model import AttRNet, parameter_count as attnet_parameter_count
from attrnet_ofdm.pacenet import DelayDopplerBasis, PACEConfig, PACENet, loss_fn, parameter_count, to_pilot_tokens


def coords(n):
    f=np.rint(np.linspace(0,71,n//4)).astype(int); t=np.rint(np.linspace(0,13,4)).astype(int)
    return torch.tensor([(i,j) for i in f for j in t])


@pytest.mark.parametrize("n",[12,16,24,48])
def test_shapes_finite(n):
    model=PACENet(PACEConfig(d=16,heads=4))
    out=model(torch.randn(2,n,2),coords(n))
    assert out.shape==(2,2,72,14) and torch.isfinite(out).all()


def test_gradients_reach_components():
    model=PACENet(PACEConfig(d=16,heads=4)); x=torch.randn(2,24,2); c=coords(24); y=torch.randn(2,2,72,14)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn(model(x,c),y).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    loss_fn(model(x,c),y).backward()
    groups=[model.tokenizer,model.encoder,model.coefficient_head,model.basis,model.grid_embed,model.linear_interp,model.gate,model.refine]
    for group in groups:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in group.parameters())
    assert model.latent_queries.grad is not None and model.latent_queries.grad.abs().sum()>0


def test_permutation_invariance():
    torch.manual_seed(2); model=PACENet(PACEConfig(d=16,heads=4)).eval(); x=torch.randn(1,24,2); c=coords(24); perm=torch.randperm(24)
    with torch.no_grad(): a=model(x,c); b=model(x[:,perm],c[perm])
    assert torch.max(torch.abs(a-b))<1e-4


def test_padding_invariance():
    torch.manual_seed(3); model=PACENet(PACEConfig(d=16,heads=4)).eval(); x=torch.randn(1,24,2); c=coords(24)
    xp=torch.cat((x,torch.randn(1,5,2)*100),1); cp=torch.cat((c,torch.randint(0,72,(5,2))),0); pad=torch.tensor([[False]*24+[True]*5])
    with torch.no_grad(): a=model(x,c); b=model(xp,cp,pad)
    assert torch.max(torch.abs(a-b))<1e-5


def test_zero_initialized_refinement_is_identity():
    cfg=PACEConfig(d=16,heads=4); refined=PACENet(cfg).eval(); bare_cfg=deepcopy(cfg); bare_cfg.use_refine=False; bare=PACENet(bare_cfg).eval(); bare.load_state_dict({k:v for k,v in refined.state_dict().items() if k in bare.state_dict()})
    x=torch.randn(2,24,2); c=coords(24)
    with torch.no_grad(): assert torch.equal(refined(x,c),bare(x,c))


def test_basis_initialization_matches_formula():
    b=DelayDopplerBasis(); f=torch.arange(72)[:,None]; tau=torch.arange(6)[None,:]*0.5
    t=torch.arange(14)[:,None]-6.5; nu=(torch.arange(3)[None,:]-1)*0.1
    assert torch.allclose(b.bf_re,torch.cos(2*torch.pi*f*tau/72),atol=1e-6)
    assert torch.allclose(b.bf_im,-torch.sin(2*torch.pi*f*tau/72),atol=1e-6)
    assert torch.allclose(b.bt_re,torch.cos(2*torch.pi*t*nu/14),atol=1e-6)
    assert torch.allclose(b.bt_im,torch.sin(2*torch.pi*t*nu/14),atol=1e-6)


def test_one_batch_overfit():
    torch.manual_seed(4); model=PACENet(PACEConfig(d=16,heads=4)); x=torch.randn(4,24,2); c=coords(24); y=torch.zeros(4,2,72,14)
    opt=torch.optim.Adam(model.parameters(),lr=3e-3); start=loss_fn(model(x,c),y).item()
    for _ in range(200): opt.zero_grad(); loss=loss_fn(model(x,c),y); loss.backward(); opt.step()
    assert loss_fn(model(x,c),y).item() < start/10


def test_adapter_matches_manifest_and_zero_grid():
    root=Path(__file__).resolve().parents[1]/"matlab_epa7_24pilots"
    if not root.exists(): pytest.skip("MATLAB dataset not present")
    ds=SavedOFDMDataset(root,"train"); pc=ds.manifest["pilot_configuration"]
    cr=torch.tensor([(f-1,t-1) for f in pc["frequency_indices_matlab_one_based"] for t in pc["time_indices_matlab_one_based"]])
    x=torch.from_numpy(np.array(ds.x[:2],copy=True)); h,c,_=to_pilot_tokens(x,cr)
    full=torch.zeros(2,2,72,14); full[:, :, cr[:,0],cr[:,1]]=h.permute(0,2,1)
    gathered,c2,_=to_pilot_tokens(full,cr)
    assert torch.equal(h,gathered) and torch.equal(c,c2)


def test_loss_parity():
    pred=torch.randn(3,2,72,14); target=torch.randn_like(pred)
    expected=(pred-target).square().sum(dim=1).mean()
    assert torch.equal(loss_fn(pred,target),expected)


def test_mixed_precision_basis_math_float32_and_finite():
    if not torch.cuda.is_available(): pytest.skip("CUDA autocast unavailable")
    model=PACENet(PACEConfig(d=16,heads=4)).cuda(); x=torch.randn(2,24,2,device="cuda"); c=coords(24).cuda()
    with torch.autocast("cuda",dtype=torch.float16): out=model(x,c)
    assert torch.isfinite(out).all()
    assert model.basis.bf_re.dtype==torch.float32


def test_presets_and_attnet_parameter_counts():
    pace_s=PACEConfig.preset("pace_s")
    pace_m=PACEConfig.preset("pace_m")
    assert pace_s.d==16 and pace_s.d%pace_s.heads==0
    assert pace_m.d==64 and pace_m.d%pace_m.heads==0
    assert parameter_count(PACENet(pace_s))==11457
    assert parameter_count(PACENet(pace_m))==128529
    assert attnet_parameter_count(AttRNet())==10254
