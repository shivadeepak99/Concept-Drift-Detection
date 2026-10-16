import sys, numpy as np, glob
sys.path.insert(0,"ours")
from analyse import full_metrics, load_npz
from run_ext import load, DATA, BASE
for ds in DATA:
    tr,ytr,te,yte,dr=load(ds); last=int(ytr[-1]); print("=====",ds)
    def agg(files,label):
        M=[]
        for f in files:
            z=np.load(f); M.append(full_metrics(z["y"],z["pred"],z["p1"],z["lat"],dr,last))
        k=["acc","macro_f1","mcc","kappa_t","worst_acc_w500","drop","recovery","lat_ms","thr"]
        print("%-26s"%label," ".join("%s=%.3f±%.3f"%(c,np.mean([m[c] for m in M]),np.std([m[c] for m in M],ddof=1)) if c in("mcc","kappa_t") else "%s=%.2f±%.2f"%(c,np.mean([m[c] for m in M])*(100 if c in("acc","macro_f1","worst_acc_w500","drop") else 1),np.std([m[c] for m in M],ddof=1)*(100 if c in("acc","macro_f1","worst_acc_w500","drop") else 1)) for c in k))
    agg([f"results/ext/base__{ds}__ARF-ADWIN__s{s}.npz" for s in (1,2,3)],"ARF-ADWIN 3 trees")
    agg([f"results/ext/extra/trees__{ds}__10__s{s}.npz" for s in (1,2,3)],"ARF-ADWIN 10 trees")
    for D in (50,200,1000): agg([f"results/ext/extra/delay__{ds}__{D}__s{s}.npz" for s in (1,2,3)],f"ARF-ADWIN label delay {D}")
