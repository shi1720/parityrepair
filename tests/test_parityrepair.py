import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from itertools import product
import numpy as np
from parityrepair import probes,incidence,moments,gf2_solve,confidence_system,repair


def test_moment_equivariance_repeated_powers():
 rng=np.random.default_rng(0); x=rng.normal(size=(53,5)); ps=probes(5,3)
 for signs in product([-1,1],repeat=5):
  signs=np.array(signs); m,v=moments(x,ps); mt,vt=moments(x*signs,ps)
  parity=np.array([np.prod(signs[list(p)]) for p in ps])
  np.testing.assert_allclose(mt,m*parity,atol=1e-14); np.testing.assert_allclose(vt,v,atol=1e-14)


def test_solver_matches_exhaustive_cosets():
 rng=np.random.default_rng(42)
 for d in range(1,8):
  for _ in range(20):
   a=rng.integers(0,2,size=(rng.integers(0,12),d),dtype=np.uint8); truth=rng.integers(0,2,d,dtype=np.uint8); b=a@truth%2
   x,null,r,ok=gf2_solve(a,b); assert ok
   brute={tuple(t) for t in product([0,1],repeat=d) if np.array_equal(a@np.array(t)%2,b)}
   expanded={tuple((x+np.array(c,dtype=np.uint8)@null)%2) for c in product([0,1],repeat=len(null))}
   assert brute==expanded and len(brute)==2**(d-r)


def test_inconsistent():
 _,_,_,ok=gf2_solve(np.array([[1,0],[1,0]]),np.array([0,1])); assert not ok


def test_pair_gauge_and_triple_fullrank():
 a=incidence([(i,j) for i in range(4) for j in range(i+1,4)],4)
 assert gf2_solve(a,np.zeros(6,dtype=np.uint8))[2]==3
 a=incidence([(0,1,2),(0,1,3),(0,2,3),(1,2,3)],4)
 assert gf2_solve(a,np.zeros(4,dtype=np.uint8))[2]==4


def test_population_blindness_and_identification():
 states=np.array(list(product([-1.,1.],repeat=4))); triples=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
 p=(1+.24*sum(np.prod(states[:,q],axis=1) for q in triples))/16
 ps=probes(4,3); a=incidence(ps,4)
 ms=np.array([p@np.prod(np.tanh(3*states[:,q]),axis=1) for q in ps]); v=np.ones(len(ps))
 low=len(probes(4,2)); informative=a[:low].any(axis=1)
 assert np.max(np.abs(ms[:low][informative]))<1e-14
 for signs in product([-1,1],repeat=4):
  signs=np.array(signs); mt=ms*np.prod(np.where(a,signs,1),axis=1)
  est,_,_=repair(ms,mt,v,v,ps,4,4096,4096)
  assert np.array_equal(est,signs)
  system=confidence_system(ms,mt,ps,4,4096,4096)
  assert system['rank']==4 and system['consistent']


def test_confidence_null_case():
 ps=probes(4,3); z=np.zeros(len(ps)); s=confidence_system(z,z,ps,4,1000,1000)
 assert s['rank']==0 and len(s['nullspace'])==4
