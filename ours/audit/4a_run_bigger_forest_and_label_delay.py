import sys, os, time, numpy as np
sys.path.insert(0,"ours")
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import run_ext as r
from river import ensemble
from river.drift import ADWIN
OUT="results/ext/extra"; os.makedirs(OUT,exist_ok=True)

def run(kind, ds, seed, par):
    out=f"{OUT}/{kind}__{ds}__{par}__s{seed}.npz"
    if os.path.exists(out): return out
    tr,ytr,te,yte,_=r.load(ds)
    if kind=="trees":
        m=ensemble.AdaptiveRandomForestClassifier(n_models=int(par),drift_detector=ADWIN(),seed=seed)
        p1,pred,lat=r.prequential(m,tr,ytr,te,yte)
    else:  # delayed labels: predict now, learn label D flows later
        D=int(par); m=ensemble.AdaptiveRandomForestClassifier(n_models=3,drift_detector=ADWIN(),seed=seed)
        for x,y in zip(tr,ytr): m.learn_one(x,int(y))
        q=deque(); n=len(te); p1=np.zeros(n); pred=np.zeros(n,int); lat=np.zeros(n)
        for i,(x,y) in enumerate(zip(te,yte)):
            pr=m.predict_proba_one(x); p1[i]=pr.get(1,0.0); pred[i]=max(pr,key=pr.get) if pr else 0
            q.append((x,int(y)))
            if len(q)>D:
                xx,yy=q.popleft(); m.learn_one(xx,yy)
    np.savez(out,p1=p1,pred=pred,lat=lat,y=yte); return out

if __name__=="__main__":
    jobs=[("trees",ds,s,10) for ds in r.DATA for s in (1,2,3)]+[("delay",ds,s,D) for ds in r.DATA for s in (1,2,3) for D in (50,200,1000)]
    t0=time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs=[ex.submit(run,*j) for j in jobs]
        for i,f in enumerate(futs): print(i+1,len(jobs),f.result(),round(time.time()-t0),flush=True)
