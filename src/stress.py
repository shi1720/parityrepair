"""Mechanism-mismatch stress tests: additive offsets/noise after reflection."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from data import datasets
from run_experiments import split
from parityrepair import probes,moments,repair,select_degree

def main():
 rows=[]; start=time.time()
 with threadpool_limits(1):
  for name,(x,y) in datasets('data').items():
   for seed in range(200,210):
    tr,ref,cal,te=split(x,y,seed); sc=StandardScaler().fit(x[tr]); zs=sc.transform(x[ref]); zc=sc.transform(x[cal])[:32]; zt=sc.transform(x[te]); d=x.shape[1]
    model=MLPClassifier(hidden_layer_sizes=(64,32),alpha=.01,batch_size=64,max_iter=300,early_stopping=True,validation_fraction=.2,n_iter_no_change=25,random_state=seed).fit(sc.transform(x[tr]),y[tr])
    degree,_=select_degree(sc.transform(x[tr]),32,seed=seed); ps=probes(d,degree); ms,vs=moments(zs,ps)
    for fault in ['offset','noise']:
     for level in [0.,.25,.5,1.]:
      rng=np.random.default_rng(seed+1234); sign=rng.choice([-1.,1.],d); direction=rng.choice([-1.,1.],d)
      target=zc*sign; test=zt*sign
      if fault=='offset': target=target+level*direction; test=test+level*direction
      else: target=target+level*rng.normal(size=target.shape); test=test+level*rng.normal(size=test.shape)
      mt,vt=moments(target,ps); est,_,_=repair(ms,mt,vs,vt,ps,d,len(ref),len(target),seed=seed)
      for method,z in {'Unrepaired':test,'ParityRepair':test*est,'KnownSigns':test*sign}.items():
       rows.append(dict(dataset=name,seed=seed,fault=fault,level=level,method=method,accuracy=np.mean(model.predict(z)==y[te]),sign_accuracy=np.mean(est==sign),selected_degree=degree))
   pd.DataFrame(rows).to_csv('results/stress.csv',index=False); print('stress',name,round(time.time()-start,1),flush=True)
 Path('results/stress_execution.json').write_text(json.dumps({'seconds':time.time()-start,'seeds':list(range(200,210))},indent=2))
if __name__=='__main__': main()
