"""Bounded-moment feature orientation repair; no target labels are accepted."""
from itertools import combinations_with_replacement
import numpy as np


def probes(d, degree=3):
    return [p for k in range(1, degree+1) for p in combinations_with_replacement(range(d), k)]


def incidence(ps, d):
    a = np.zeros((len(ps), d), dtype=np.uint8)
    for r,p in enumerate(ps):
        for j in p: a[r,j] ^= 1
    return a


def moments(z, ps):
    u = np.tanh(np.asarray(z, dtype=float))
    # Chunking bounds peak memory for large source sets.
    means, variances = [], []
    for start in range(0,len(ps),256):
        block = np.stack([np.prod(u[:,p],axis=1) for p in ps[start:start+256]],axis=1)
        means.extend(block.mean(axis=0)); variances.extend(block.var(axis=0,ddof=1))
    return np.array(means), np.array(variances)


def gf2_solve(a,b):
    """Return RREF particular solution, nullspace basis, rank, consistency."""
    a=np.asarray(a,dtype=np.uint8).copy(); b=np.asarray(b,dtype=np.uint8).copy()
    n,d=a.shape; r=0; piv=[]
    for j in range(d):
        found=np.flatnonzero(a[r:,j])
        if not len(found): continue
        q=r+found[0]; a[[r,q]]=a[[q,r]]; b[[r,q]]=b[[q,r]]
        rows=np.flatnonzero(a[:,j]); rows=rows[rows!=r]
        a[rows]^=a[r]; b[rows]^=b[r]
        piv.append(j); r+=1
        if r==n: break
    ok=not np.any(b[r:]); x=np.zeros(d,dtype=np.uint8)
    for i,j in enumerate(piv): x[j]=b[i]
    null=[]
    for j in range(d):
        if j in piv: continue
        v=np.zeros(d,dtype=np.uint8); v[j]=1
        for i,q in enumerate(piv): v[q]=a[i,j]
        null.append(v)
    return x,np.array(null,dtype=np.uint8).reshape(-1,d),r,ok


def confidence_system(ms,mt,ps,d,ns,nt,delta=.05):
    """Simultaneous Hoeffding sign constraints for fixed bounded probes.

    Conditional on a fixed preprocessing map, independent source/reference and
    target samples, and a pure reflection shift, true orientation is feasible
    with probability at least 1-delta. Absence of constraints is informative.
    """
    m=len(ps); rs=np.sqrt(2*np.log(4*m/delta)/ns); rt=np.sqrt(2*np.log(4*m/delta)/nt)
    keep=(np.abs(ms)>rs)&(np.abs(mt)>rt)
    a=incidence(ps,d)[keep]; b=(ms[keep]*mt[keep]<0).astype(np.uint8)
    x,null,rank,consistent=gf2_solve(a,b)
    return dict(solution=x,nullspace=null,rank=rank,consistent=consistent,rows=int(keep.sum()),radius_source=rs,radius_target=rt)


def repair(ms,mt,vs,vt,ps,d,ns,nt,seed=0,starts=12):
    """Heuristic weighted moment matching; returns signs, objective, metadata.

    Gaussian elimination supplies a confidence-ranked initialization; redundant
    equations are retained in a multistart coordinate descent objective. This
    estimate is distinct from the conservative confidence system above.
    """
    a=incidence(ps,d); valid=a.any(axis=1)
    a=a[valid]; ms=ms[valid]; mt=mt[valid]; vs=vs[valid]; vt=vt[valid]
    # Variance floor avoids domination by nearly deterministic source probes.
    var=vs/ns+vt/nt+1/ns**2+1/nt**2
    c=ms*mt/var
    # Sum terms with the same parity: repeated powers share an orientation bit.
    groups={}
    for row,weight in zip(a,c):
        key=tuple(np.flatnonzero(row)); groups[key]=groups.get(key,0.)+weight
    keys=list(groups); c=np.array([groups[k] for k in keys]); a=np.zeros((len(keys),d),dtype=np.uint8)
    for i,k in enumerate(keys): a[i,list(k)]=1
    b=(c<0).astype(np.uint8)
    basis=[]; rhs=[]; rank=0
    for idx in np.argsort(-np.abs(c)):
        trial=np.array(basis+[a[idx].tolist()],dtype=np.uint8)
        _,_,newrank,_=gf2_solve(trial,np.zeros(len(trial),dtype=np.uint8))
        if newrank>rank:
            basis.append(a[idx].tolist()); rhs.append(int(b[idx])); rank=newrank
        if rank==d: break
    x,_,_,_=gf2_solve(np.array(basis,dtype=np.uint8).reshape(-1,d),np.array(rhs,dtype=np.uint8))
    rng=np.random.default_rng(seed)
    initial=[1-2*x.astype(float),np.ones(d),-np.ones(d)]
    initial += [rng.choice([-1.,1.],d) for _ in range(max(0,starts-3))]
    members=[np.flatnonzero(a[:,j]) for j in range(d)]
    best=None; bestscore=-np.inf
    for sign in initial:
        sign=sign.copy(); parity=np.prod(np.where(a,sign[None,:],1.),axis=1)
        for _ in range(100):
            gain=np.array([-2*np.dot(c[idx],parity[idx]) for idx in members])
            j=int(np.argmax(gain))
            if gain[j]<=1e-10: break
            sign[j]*=-1; parity[members[j]]*=-1
        score=float(c@parity)
        if score>bestscore: bestscore=score; best=sign.copy()
    return best,bestscore,dict(parity_rows=len(keys),basis_rank=rank)


def select_degree(z,n,repetitions=8,seed=0):
    """Choose moment order using only clean source training inputs.

    Each resampling partitions source inputs into independent reference and
    pseudo-target subsets. Known random reflections provide sign supervision;
    neither class labels nor any real target input is used for selection.
    """
    rng=np.random.default_rng(seed+314159); d=z.shape[1]; losses=np.zeros(3)
    for rep in range(repetitions):
        idx=rng.permutation(len(z)); nt=min(n,len(z)//3); ref=z[idx[nt:]]; target=z[idx[:nt]]
        sign=rng.choice([-1.,1.],d)
        for degree in [1,2,3]:
            ps=probes(d,degree); ms,vs=moments(ref,ps); mt,vt=moments(target*sign,ps)
            est,_,_=repair(ms,mt,vs,vt,ps,d,len(ref),nt,seed=seed+rep)
            losses[degree-1]+=np.mean(est!=sign)
    losses/=repetitions
    return int(np.argmin(losses)+1),losses
