import sys, numpy as np
sys.path.insert(0,"ours")
from analyse import win_err, combine, load_npz, rolling_mean
from run_ext import BASE, load, DATA
for ds in DATA:
    tr,ytr,te,yte,dr=load(ds); n=len(yte); out=[]
    for s in (1,2,3):
        z=[load_npz("base",ds,b,s) for b in BASE]; E=np.stack([(q["pred"]!=yte) for q in z]).astype(float); P=np.stack([q["p1"] for q in z])
        p=combine(P,win_err(E,None,False)); err=((p>=.5).astype(int)!=yte).astype(float)
        sig={"model disagreement (std of 4 probs)":P.std(0),"uncertainty min(p,1-p)":np.minimum(p,1-p),"mean prob of 'attack'":p}
        for W in (200,):
            re=rolling_mean(err,W); r={}
            for k,v in sig.items():
                rv=rolling_mean(v,W); r[k]=np.corrcoef(rv[W:],re[W:])[0,1]
                # best lag: signal leads error by L flows
            out.append(r)
    print("=====",ds,"correlation of rolling-200 signal with rolling-200 ensemble error (mean of 3 seeds)")
    for k in out[0]: print("  %-36s r = %.2f"%(k,np.mean([o[k] for o in out])))
