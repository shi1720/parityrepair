import time,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from parityrepair import probes,moments,repair,confidence_system

# All four degree-three Walsh coefficients are rho; all degree <=2 vanish.
STATES=np.array([[1 if (i>>j)&1 else -1 for j in range(4)] for i in range(16)],dtype=float)
TRIPLES=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
def sample(n,rho,rng):
 p=(1+rho*sum(np.prod(STATES[:,q],axis=1) for q in TRIPLES))/16
 assert np.all(p>=0) and abs(sum(p)-1)<1e-12
 return 3*STATES[rng.choice(16,n,p=p)]

def main():
 out=Path('results'); rows=[]; start=time.time()
 with threadpool_limits(1):
  for seed in range(50):
   rng=np.random.default_rng(seed+90210)
   source=sample(4096,.24,rng); fit=sample(2048,.24,rng); test=sample(4096,.24,rng); calibration=sample(4096,.24,rng)
   net=MLPClassifier(hidden_layer_sizes=(16,16),max_iter=200,random_state=seed).fit(fit,(fit[:,0]>0).astype(int))
   for degree in [1,2,3]:
    ps=probes(4,degree); ms,vs=moments(source,ps)
    for n in [16,32,64,128,256,512,1024,4096]:
     mt0,vt=moments(calibration[:n],ps)
     for mask in range(16):
      sign=STATES[mask]; mt=mt0*np.array([np.prod(sign[list(q)]) for q in ps]); est,_,_=repair(ms,mt,vs,vt,ps,4,4096,n,seed=seed)
      system=confidence_system(ms,mt,ps,4,4096,n)
      rows.append(dict(seed=seed,degree=degree,n=n,mask=mask,exact=float(np.all(est==sign)),sign_accuracy=np.mean(est==sign),accuracy=np.mean(net.predict(test*sign*est)==(test[:,0]>0)),rank=system['rank'],consistent=system['consistent']))
   if seed%10==0: print('synthetic',seed,round(time.time()-start,1),flush=True)
 pd.DataFrame(rows).to_csv(out/'synthetic.csv',index=False)
 (out/'synthetic_execution.json').write_text(json.dumps({'seconds':time.time()-start,'rho':.24,'seeds':50,'source':4096,'fit':2048,'test':4096,'masks':16},indent=2))
if __name__=='__main__': main()
