import sys, numpy as np, pandas as pd
sys.path.insert(0,"ours")
from analyse import win_err, combine, load_npz, rolling_mean
from run_ext import BASE, load, DATA
for ds in DATA:
    tr,ytr,te,yte,dr=load(ds); n=len(yte)
    print("=====",ds,"n_test",n)
    # data structure
    X=pd.DataFrame(te); dup=X.duplicated().mean(); print("duplicate rows in test stream: %.1f%%"%(dup*100))
    runs=np.diff(np.where(np.diff(np.concatenate([[-1],yte,[-1]]))!=0)[0]); print("label runs: count",len(runs),"median len",int(np.median(runs)),"max",runs.max())
    # class prior and error per 1000 for seed1 ARF-ADWIN
    z=[load_npz("base",ds,b,1) for b in BASE]; E=np.stack([(q["pred"]!=yte) for q in z]).astype(float); P=np.stack([q["p1"] for q in z])
    seg=np.arange(n)//1000; pri=pd.Series(yte).groupby(seg).mean(); err=pd.Series(E[0]).groupby(seg).mean()
    print("attack-share per 1000:", (pri.values*100).round(0).astype(int).tolist()[:26])
    print("ARF-ADWIN err%/1000  :", (err.values*100).round(1).tolist()[:26])
    print("corr(prior change, error) per 1000 blocks: %.2f"%np.corrcoef(np.abs(np.diff(pri.values)),err.values[1:])[0,1])
    # diversity
    k=len(BASE); dis=np.mean([ (z[i]["pred"]!=z[j]["pred"]).mean() for i in range(k) for j in range(i+1,k)])
    allw=(E.sum(0)==4).mean(); anyc=(E.sum(0)<4).mean(); acc=[1-e.mean() for e in E]
    print("pairwise disagreement %.2f%% | all four wrong %.2f%% | oracle(any correct) acc %.2f%% | single accs %s"%(dis*100,allw*100,anyc*100,np.round(np.array(acc)*100,2)))
    # error correlation: given ARF-ADWIN wrong, prob others wrong
    w=E[0]==1; print("P(other wrong | ARF-ADWIN wrong):",[round(E[j][w].mean(),2) for j in range(1,4)])
    # ensemble gain concentration
    ens=(combine(P,win_err(E,None,False))>=0.5).astype(int)!=yte
    gain=E[0]-ens.astype(float)  # +1 ensemble fixes single's error, -1 ensemble breaks
    post=np.zeros(n,bool)
    for g in dr: post[g:g+600]=True
    burst=rolling_mean(E[0],200)>0.02
    for name,mask in [("within 600 after marked drift",post),("error-burst (roll200>2%)",burst),("everything else",~(post|burst))]:
        print("  %-30s share of stream %.1f%% | net errors fixed by ensemble: %d (single made %d errors there)"%(name,mask.mean()*100,gain[mask].sum(),E[0][mask].sum()))
