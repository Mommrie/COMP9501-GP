"""Convex four-coefficient response matching; no labels or validation data."""
import torch


def simplex(x):
    values=x.sort(descending=True).values
    sums=values.cumsum(0)-1
    idx=torch.arange(1,len(x)+1,device=x.device,dtype=x.dtype)
    valid=values-sums/idx>0
    rho=int(valid.nonzero()[-1])
    return (x-sums[rho]/(rho+1)).clamp_min(0)


def solve(g,b,beta=0.):
    g=(g.double()+g.double().T)/2;b=b.double()
    scale=g.trace().clamp_min(1e-12)
    g,b=g/scale,b/scale
    u=torch.full_like(b,1/len(b));ridge=beta/len(b)
    step=1/(2*torch.linalg.eigvalsh(g).max().clamp_min(1e-12)+2*ridge)
    a=u.clone()
    for iteration in range(5000):
        nxt=simplex(a-step*(2*g@a-2*b+2*ridge*(a-u)))
        if (nxt-a).abs().max()<1e-11:
            a=nxt;break
        a=nxt
    objective=lambda v:float(v@g@v-2*b@v+ridge*(v-u).square().sum())
    assert objective(a)<=objective(u)+1e-9
    assert torch.isfinite(a).all() and abs(float(a.sum())-1)<1e-8
    return a,{'iterations':iteration+1,'uniform_objective':objective(u),'objective':objective(a)}
