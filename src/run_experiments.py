"""Run disjoint source/reference/calibration/test reflection experiments."""
import argparse, json, time, platform, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from data import datasets
from parityrepair import probes,moments,repair,confidence_system,select_degree
warnings.filterwarnings('ignore',category=ConvergenceWarning)


def split(x,y,seed):
 ids=np.arange(len(y)); train,rest=train_test_split(ids,test_size=.5,stratify=y,random_state=seed)
 reference,rest=train_test_split(rest,test_size=.6,stratify=y[rest],random_state=seed+100)
 calib,test=train_test_split(rest,test_size=2/3,stratify=y[rest],random_state=seed+200)
 # 50% fit (including validation), 20% independent reference, 10% adaptation, 20% test
 return train,reference,calib,test


def coral(z,source,target):
 cs=np.cov(source,rowvar=False)+.1*np.eye(source.shape[1]); ct=np.cov(target,rowvar=False)+.1*np.eye(target.shape[1])
 et,ut=np.linalg.eigh(ct); es,us=np.linalg.eigh(cs)
 return (z-target.mean(0))@((ut*et**-.5)@ut.T)@((us*es**.5)@us.T)+source.mean(0)


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=10); ap.add_argument('--start-seed',type=int,default=100); ap.add_argument('--out',default='results'); ap.add_argument('--cache',default='data'); args=ap.parse_args()
 out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time(); rows=[]; timing=[]
 with threadpool_limits(limits=1):
  for name,(x,y) in datasets(args.cache).items():
   for seed in range(args.start_seed,args.start_seed+args.seeds):
    tr,ref,cal,te=split(x,y,seed); scaler=StandardScaler().fit(x[tr]); zs=scaler.transform(x[ref]); zc=scaler.transform(x[cal]); zt=scaler.transform(x[te]); d=x.shape[1]
    models={'MLP':MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',alpha=.01,batch_size=64,learning_rate_init=.001,max_iter=300,early_stopping=True,validation_fraction=.2,n_iter_no_change=25,random_state=seed),'LR':LogisticRegression(C=1,max_iter=2000)}
    for model in models.values(): model.fit(scaler.transform(x[tr]),y[tr])
    for n in [16,32,64,128]:
     if n>len(cal): continue
     clean=zc[:n]; stats={}
     selected,cv_losses=select_degree(scaler.transform(x[tr]),n,seed=seed)
     for degree in [1,2,3]:
      ps=probes(d,degree); ms,vs=moments(zs,ps); mt,vt=moments(clean,ps); stats[degree]=(ps,ms,vs,mt,vt)
     ps,ms,vs,mt,vt=stats[3]; keep=np.array([len(q)<=2 or len(set(q))==1 for q in ps]); stats[4]=([q for q,k in zip(ps,keep) if k],ms[keep],vs[keep],mt[keep],vt[keep])
     for mask_id,p in enumerate([0.,.25,.5,.5,.75,1.]):
      rng=np.random.default_rng(seed*1000+mask_id+77); sign=np.ones(d); sign[rng.choice(d,round(p*d),replace=False)]=-1
      target=clean*sign; corrupted=zt*sign; versions={'Unrepaired':corrupted,'Oracle':zt,'CORAL':coral(corrupted,zs,target)}
      q=np.linspace(.01,.99,99); sq=np.quantile(zs,q,axis=0); tq=np.quantile(target,q,axis=0); tnq=np.quantile(-target,q,axis=0)
      marginal=np.where(np.mean(abs(sq-tq),axis=0)<=np.mean(abs(sq-tnq),axis=0),1.,-1.)
      versions['Marginal']=corrupted*marginal; signs={}; ranks={}; runtimes={}
      for degree in [1,2,3,4]:
       ps,ms,vs,mt0,vt=stats[degree]; parity=np.array([np.prod(sign[list(q)]) for q in ps]); mt=mt0*parity
       t=time.perf_counter(); est,_,_=repair(ms,mt,vs,vt,ps,d,len(ref),n,seed=seed); elapsed=time.perf_counter()-t
       method='P2+U3' if degree==4 else 'P'+str(degree); versions[method]=corrupted*est; signs[method]=est; runtimes[method]=elapsed
       system=confidence_system(ms,mt,ps,d,len(ref),n); ranks[method]=system['rank']
      versions['ParityRepair']=versions['P'+str(selected)]; signs['ParityRepair']=signs['P'+str(selected)]; ranks['ParityRepair']=ranks['P'+str(selected)]; runtimes['ParityRepair']=runtimes['P'+str(selected)]
      for modelname,model in models.items():
       for method,z in versions.items():
        prob=model.predict_proba(z); pred=model.classes_[prob.argmax(1)]
        rows.append(dict(dataset=name,seed=seed,n=n,mask=mask_id,fraction=p,model=modelname,method=method,accuracy=accuracy_score(y[te],pred),balanced_accuracy=balanced_accuracy_score(y[te],pred),nll=log_loss(y[te],prob,labels=model.classes_),sign_accuracy=float(np.mean(signs[method]==sign)) if method in signs else np.nan,exact=float(np.all(signs[method]==sign)) if method in signs else np.nan,confidence_rank=ranks.get(method,np.nan),repair_seconds=runtimes.get(method,np.nan),selected_degree=selected,cv_loss_1=cv_losses[0],cv_loss_2=cv_losses[1],cv_loss_3=cv_losses[2],d=d,n_ref=len(ref),n_test=len(te)))
     pd.DataFrame(rows).to_csv(out/'real.csv',index=False)
    print(name,seed,'elapsed',round(time.time()-start,1),flush=True)
 (out/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,python=sys.version,platform=platform.platform(),seeds=args.seeds,start_seed=args.start_seed,protocol='50/20/10/20; clean source internal validation; fixed hyperparameters',numpy=np.__version__),indent=2))
if __name__=='__main__': main()
