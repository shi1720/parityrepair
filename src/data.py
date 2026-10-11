from pathlib import Path
import hashlib, io, json
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer

URLS={
 'red':'https://archive.ics.uci.edu/static/public/186/wine+quality.zip',
 'white':'https://archive.ics.uci.edu/static/public/186/wine+quality.zip',
 'banknote':'https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip',
 'ionosphere':'https://archive.ics.uci.edu/static/public/52/ionosphere.zip'}

def datasets(cache):
 import zipfile
 cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
 out={}; meta={}
 for name,url in URLS.items():
  file=cache/(name+'.zip')
  if not file.exists():
   r=requests.get(url,timeout=60); r.raise_for_status(); file.write_bytes(r.content)
  z=zipfile.ZipFile(file)
  if name in ['red','white']:
   f=next(f for f in z.namelist() if f.endswith('winequality-'+name+'.csv'))
   data=pd.read_csv(z.open(f),sep=';'); x=data.iloc[:,:-1].to_numpy(); y=(data.iloc[:,-1].to_numpy()>=6).astype(int)
  else:
   f=next(f for f in z.namelist() if f.endswith('.txt' if name=='banknote' else '.data'))
   data=pd.read_csv(z.open(f),header=None); x=data.iloc[:,:-1].to_numpy(dtype=float)
   y=data.iloc[:,-1].to_numpy(); y=(y=='g').astype(int) if name=='ionosphere' else y.astype(int)
   if name=='ionosphere': x=np.delete(x,1,axis=1) # published constant zero column
  out[name]=(x,y); meta[name]={'url':url,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'n':len(y),'d':x.shape[1]}
 cancer=load_breast_cancer(); out['cancer']=(cancer.data,cancer.target)
 meta['cancer']={'source':'sklearn.datasets.load_breast_cancer','n':len(cancer.target),'d':30,'array_sha256':hashlib.sha256(cancer.data.tobytes()+cancer.target.tobytes()).hexdigest()}
 for name,(x,y) in list(out.items()):
  _,idx=np.unique(np.column_stack([x,y]),axis=0,return_index=True); idx=np.sort(idx)
  out[name]=(x[idx],y[idx]); meta[name]['n_after_dedup']=len(idx)
 (cache/'manifest.json').write_text(json.dumps(meta,indent=2))
 return out
