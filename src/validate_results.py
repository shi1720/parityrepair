"""Validate saved evaluation provenance and the key numerical claims."""
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 x=pd.read_csv('results/final/real.csv')
 assert len(x)==34560 and not x.method.isna().any()
 assert set(x.seed)==set(range(200,220))
 assert x[['dataset','seed']].drop_duplicates().shape[0]==100
 assert set(x.model)=={'MLP','LR'}
 assert (x.accuracy.between(0,1)&x.balanced_accuracy.between(0,1)&(x.nll>=0)).all()
 oracle=x[x.method=='Oracle']
 assert oracle.groupby(['dataset','seed','model']).accuracy.nunique().max()==1
 s=pd.read_csv('results/synthetic.csv');assert len(s)==19200
 assert s[(s.degree==3)&(s.n>=512)].exact.eq(1).all()
 assert s[(s.degree==3)&(s.n>=1024)]['rank'].eq(4).all()
 assert s[(s.degree<=2)]['rank'].eq(0).all()
 audit=pd.read_csv('results/solver_audit.csv'); assert len(audit)==60 and (abs(audit.gap)<1e-8).all()
 assert len(pd.read_csv('results/stress.csv'))==1200
 for p in ['synthetic','sample_size','stress']:
  assert (Path('figures')/(p+'.png')).stat().st_size>10000
 print('Saved results validated: 100 dataset/split evaluations, 34,560 primary rows, 19,200 synthetic rows, 1,200 stress rows, 60 exact-objective audits.')
if __name__=='__main__':main()
