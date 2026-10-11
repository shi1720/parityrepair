import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data import datasets
from run_experiments import split
from parityrepair import probes,moments,incidence,repair

def main():
 rows=[]
 for name,(x,y) in datasets('data').items():
  if x.shape[1]>11: continue
  d=x.shape[1]; signs=1-2*((np.arange(2**d)[:,None]>>np.arange(d))&1)
  for seed in range(200,220):
   tr,ref,cal,te=split(x,y,seed);sc=StandardScaler().fit(x[tr]);zs=sc.transform(x[ref]);zt=sc.transform(x[cal])[:32]*np.random.default_rng(seed+423).choice([-1.,1.],d);ps=probes(d,3);ms,vs=moments(zs,ps); mt,vt=moments(zt,ps); a=incidence(ps,d)
   est,_,_=repair(ms,mt,vs,vt,ps,d,len(ref),len(zt),seed=seed)
   c=ms*mt/(vs/len(ref)+vt/len(zt)+1/len(ref)**2+1/len(zt)**2); valid=a.any(1);c=c[valid];a=a[valid]
   parity=np.prod(np.where(a[None,:,:],signs[:,None,:],1),axis=2);scores=parity@c
   obtained=np.prod(np.where(a,est,1),axis=1)@c; gap=float(scores.max()-obtained)
   rows.append(dict(dataset=name,seed=seed,gap=gap,exact_objective=gap<1e-8))
 pd.DataFrame(rows).to_csv('results/solver_audit.csv',index=False)
 print(pd.DataFrame(rows).groupby('dataset')[['gap','exact_objective']].mean())
if __name__=='__main__':main()
