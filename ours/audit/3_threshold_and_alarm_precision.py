import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"ours")
from analyse import win_err, combine, load_npz, rolling_mean
from run_ext import BASE, load, DATA
from sklearn.metrics import matthews_corrcoef, f1_score
from river.drift import ADWIN
for ds in DATA:
    tr,ytr,te,yte,dr=load(ds); n=len(yte); print("=====",ds)
    rows=[]
    for s in (1,2,3):
        z=[load_npz("base",ds,b,s) for b in BASE]; E=np.stack([(q["pred"]!=yte) for q in z]).astype(float); P=np.stack([q["p1"] for q in z])
        p=combine(P,win_err(E,None,False)); rows.append([ (matthews_corrcoef(yte,(p>=t).astype(int)), f1_score(yte,(p>=t).astype(int),average="macro")) for t in (0.2,0.3,0.4,0.5,0.6,0.7,0.8)])
    r=np.array(rows).mean(0); print("threshold  :",[0.2,0.3,0.4,0.5,0.6,0.7,0.8]); print("MCC        :",r[:,0].round(3).tolist()); print("macro-F1   :",(r[:,1]*100).round(2).tolist())
    # precision of feature-drift alarms
    Xtr=pd.DataFrame(tr); Xte=pd.DataFrame(te); mu,sd=Xtr.mean(),Xtr.std().replace(0,1); Z=((Xte-mu)/sd).clip(-5,5).values
    al=[]
    for j in range(Z.shape[1]):
        a=ADWIN(delta=0.002)
        for t,v in enumerate(Z[:,j]):
            a.update(float(v))
            if a.change_detected: al.append((t,j))
    if not al: print("no feature alarms"); continue
    al=np.array(al); cnt=np.bincount(al[:,0]//50,minlength=n//50+1); blocks=np.where(cnt>=3)[0]*50
    # cluster blocks closer than 500
    cl=[]; 
    for b in blocks:
        if not cl or b-cl[-1][-1]>500: cl.append([b])
        else: cl[-1].append(b)
    starts=[c[0] for c in cl]
    z=[load_npz("base",ds,b,1) for b in BASE]; e=(z[0]["pred"]!=yte).astype(float); roll=rolling_mean(e,200)
    hit=[bool((roll[s:s+1000]>0.02).any()) for s in starts]
    base_rate=np.mean([(roll[t:t+1000]>0.02).any() for t in range(0,n-1000,250)])
    print("alarm clusters %d; followed by an error burst (>2%% over 200) within 1000 flows: %d (%.0f%%) | chance level for a random point: %.0f%%"%(len(starts),sum(hit),100*np.mean(hit),100*base_rate))
