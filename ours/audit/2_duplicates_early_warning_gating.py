import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"ours")
from analyse import win_err, combine, load_npz, rolling_mean
from run_ext import BASE, load, DATA
from river.drift import ADWIN, EDDM
from sklearn.metrics import f1_score
for ds in DATA:
    tr,ytr,te,yte,dr=load(ds); n=len(yte); Xte=pd.DataFrame(te); Xtr=pd.DataFrame(tr)
    print("=====",ds)
    z=[load_npz("base",ds,b,1) for b in BASE]; E=np.stack([(q["pred"]!=yte) for q in z]).astype(float); P=np.stack([q["p1"] for q in z])
    ens_pred=(combine(P,win_err(E,None,False))>=0.5).astype(int); ens_err=(ens_pred!=yte).astype(float)
    # --- duplicates
    allX=pd.concat([Xtr,Xte],ignore_index=True); seen=allX.duplicated(keep="first").values[len(Xtr):]
    allY=np.concatenate([ytr,yte]); grp=allX.groupby(list(allX.columns)).ngroup().values
    lab_nunique=pd.Series(allY).groupby(grp).nunique(); conflict=(lab_nunique>1).mean()
    print("rows already seen earlier (train or test): %.1f%% | groups with conflicting labels: %.2f%%"%(seen.mean()*100,conflict*100))
    for nm,e in [("ARF-ADWIN",E[0]),("PWPAE-style ensemble",ens_err)]:
        print("  %-20s error on repeated rows %.2f%% | on first-seen rows %.2f%%"%(nm,e[seen].mean()*100,e[~seen].mean()*100))
    # --- input-drift early warning: ADWIN per feature, standardized by train stats
    mu,sd=Xtr.mean(),Xtr.std().replace(0,1); Z=((Xte-mu)/sd).clip(-5,5).values
    alarms=[]
    for j in range(Z.shape[1]):
        a=ADWIN(delta=0.002); 
        for t,v in enumerate(Z[:,j]):
            a.update(float(v))
            if a.change_detected: alarms.append((t,j))
    al=np.array(alarms) if alarms else np.zeros((0,2),int)
    cnt=np.bincount(al[:,0]//50,minlength=n//50+1) if len(al) else np.zeros(n//50+1,int)
    multi=np.where(cnt>=3)[0]*50   # blocks with >=3 feature alarms
    roll=rolling_mean(E[0],200); hot=roll>0.02
    onsets=[t for t in range(500,n) if hot[t] and not hot[t-500:t].any()]
    print("feature alarms total %d; multi-feature alarm blocks %d; error-burst onsets %d: %s"%(len(al),len(multi),len(onsets),onsets[:12]))
    leads=[]
    for o in onsets:
        near=multi[(multi>=o-1000)&(multi<=o+1000)]
        leads.append(int(near.min()-o) if len(near) else None)
    print("  nearest multi-feature alarm relative to onset (neg = earlier):",leads)
    # error-based alarm for comparison
    d=EDDM(); ea=[t for t,v in enumerate(E[0]) if d.update(float(v))[0]]
    print("  EDDM alarms near onsets:",[ (int(min([a-o for a in ea if o-1000<=a<=o+1000],key=abs)) if any(o-1000<=a<=o+1000 for a in ea) else None) for o in onsets])
    # --- gating estimate (optimistic: other learners assumed already trained)
    for src,alarms_t in [("EDDM-on-error alarms",ea)]:
        for hold in (300,600,1500):
            act=np.zeros(n,bool)
            for t in alarms_t: act[t:t+hold]=True
            pred=np.where(act,ens_pred,(P[0]>=0.5).astype(int))
            print("  gate(%s,hold=%d): ensemble active %.0f%% of flows | acc %.2f%% | macroF1 %.2f | [single %.2f%% / %.2f | full ens %.2f%% / %.2f]"%(src,hold,act.mean()*100,(pred==yte).mean()*100,f1_score(yte,pred,average="macro")*100,(1-E[0].mean())*100,f1_score(yte,(P[0]>=.5).astype(int),average="macro")*100,(ens_pred==yte).mean()*100,f1_score(yte,ens_pred,average="macro")*100))
