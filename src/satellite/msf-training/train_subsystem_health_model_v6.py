#!/usr/bin/env python3
"""Finalize V6 subsystem health models.

High-dimensional subsystems (power/thermal/attitude/data) use validation-only
selection among MSF-MOFS runs. Communication/propulsion intentionally use all
local features because the final search study did not justify applying MSF to
communication and propulsion is only 24-D.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
META={"scenarioId","split","tick","state","label","faultScope","faultKey","faultPair","faultMode","magnitude","startTick","readiness"}
MSF_SUBSYSTEMS=['power','thermal','attitude','data']
ALL_FEATURE_SUBSYSTEMS=['communication','propulsion']
ALL_SUBSYSTEMS=MSF_SUBSYSTEMS+ALL_FEATURE_SUBSYSTEMS


def gmean(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    tnr=tn/max(1,tn+fp); tpr=tp/max(1,tp+fn)
    return float(math.sqrt(max(0,tnr*tpr))), float(tnr), float(tpr)


def feature_meta(name:str)->dict:
    m=re.match(r'^(param_\d+)__([^_]+)__w(\d+)$',name)
    if m: return {'name':name,'featureType':'univariate','parameterKey':m.group(1),'descriptor':m.group(2),'scale':int(m.group(3))}
    m=re.match(r'^rel_(param_\d+)_(param_\d+)__([^_]+)__w(\d+)$',name)
    if m: return {'name':name,'featureType':'relation','parameterKeys':[m.group(1),m.group(2)],'descriptor':m.group(3),'scale':int(m.group(4))}
    return {'name':name,'featureType':'unknown'}


def choose_msf_candidate(root:Path, sub:str, feats:list[str], Xtr,ytr,Xv,yv):
    exp_path=root/'experiments'/'v6_final'/f'{sub}_92_101'/'experiment_results.json'
    exp=json.loads(exp_path.read_text())
    candidates=[]
    for run in exp['runs']:
        m=run['methods']['MSF']; names=m['chosenFeatures']; idx=np.array([feats.index(n) for n in names],dtype=int)
        knn=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)); knn.fit(Xtr[:,idx],ytr)
        vg,_,_=gmean(yv,knn.predict(Xv[:,idx]))
        candidates.append({'seed':run['seed'],'names':names,'idx':idx,'k':len(idx),'valGmean':vg})
    best=max(c['valGmean'] for c in candidates)
    eligible=[c for c in candidates if c['valGmean']>=best-0.01]
    chosen=sorted(eligible,key=lambda c:(c['k'],-c['valGmean'],c['seed']))[0]
    return chosen,best,exp['dataset']['guidedCardinalities']


def fit_evidence(Xtr,ytr,Xv,yv,Xt,yt,idx,normal_quantile=0.99):
    # KNN final report: refit train+validation only after subset selection.
    Xfit=np.vstack([Xtr[:,idx],Xv[:,idx]]); yfit=np.r_[ytr,yv]
    knn=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)); knn.fit(Xfit,yfit)
    kg,kspec,krec=gmean(yt,knn.predict(Xt[:,idx]))

    # Online evidence: train on TRAIN only; validation is reserved for threshold
    # calibration. This avoids choosing an operating threshold on fitted data.
    scaler=StandardScaler().fit(Xtr[:,idx]); Xtrz=scaler.transform(Xtr[:,idx]); Xvz=scaler.transform(Xv[:,idx]); Xtz=scaler.transform(Xt[:,idx])
    logit=LogisticRegression(max_iter=3000,class_weight='balanced',solver='liblinear',random_state=2608)
    logit.fit(Xtrz,ytr)
    pval=logit.predict_proba(Xvz)[:,1]
    ptest=logit.predict_proba(Xtz)[:,1]

    # Health monitoring needs a much lower false-alarm rate than a one-shot
    # classifier. Choose the larger of the validation G-mean optimum and the
    # 99th percentile of validation-normal probability.
    best_thr=0.5; best_g=-1
    for thr in np.linspace(0.05,0.95,181):
        gg,_,_=gmean(yv,(pval>=thr).astype(int))
        if gg>best_g: best_g=gg; best_thr=float(thr)
    normal_val=pval[yv==0]
    q99=float(np.quantile(normal_val,normal_quantile)) if len(normal_val) else 0.5
    deploy_thr=float(min(0.995,max(best_thr,q99)))
    vg,vspec,vrec=gmean(yv,(pval>=deploy_thr).astype(int))
    lg,lspec,lrec=gmean(yt,(ptest>=deploy_thr).astype(int))
    return {
        'knn':(kg,kspec,krec), 'logistic':(lg,lspec,lrec),
        'evidence':{
            'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':logit.coef_[0].tolist(),'intercept':float(logit.intercept_[0]),
            'threshold':deploy_thr,'validationGmeanOptimumThreshold':best_thr,'validationNormalQuantile':normal_quantile,'validationNormalQuantileProbability':q99,
            'validationGmeanAtThreshold':vg,'validationSpecificityAtThreshold':vspec,'validationRecallAtThreshold':vrec,
            'riskLogitSlope':2.5
        }
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(HERE)); ap.add_argument('--out',default=str(HERE/'../health-v2-service/core/models/subsystem-health-model-v6.json'))
    args=ap.parse_args(); root=Path(args.root); out_path=Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True)
    result={
        'schemaVersion':'msf-subsystem-health-v6-2.0',
        'createdAt':pd.Timestamp.utcnow().isoformat(),
        'method':{
            'name':'hybrid subsystem-local health evidence',
            'msfSubsystems':MSF_SUBSYSTEMS,
            'directSubsystems':ALL_FEATURE_SUBSYSTEMS,
            'msfSelectionRule':'best validation G-mean; within 0.01 choose smallest feature subset; test untouched until finalization',
            'directRule':'use full local feature set when MSF is not statistically justified or dimensionality is too small'
        },
        'subsystems':{}
    }
    summaries=[]
    for sub in ALL_SUBSYSTEMS:
        df=pd.read_csv(root/f'telemetry-msf-v6-{sub}.csv',low_memory=False)
        feats=[c for c in df.columns if c not in META]
        tr=df[df.split=='train'].copy(); va=df[df.split=='val'].copy(); te=df[df.split=='test'].copy()
        med=tr[feats].median().fillna(0.0)
        for part in (tr,va,te): part.loc[:,feats]=part[feats].fillna(med)
        Xtr=tr[feats].to_numpy(float); ytr=tr.label.to_numpy(int); Xv=va[feats].to_numpy(float); yv=va.label.to_numpy(int); Xt=te[feats].to_numpy(float); yt=te.label.to_numpy(int)
        if sub in MSF_SUBSYSTEMS:
            chosen,best,guided=choose_msf_candidate(root,sub,feats,Xtr,ytr,Xv,yv)
            names=chosen['names']; idx=chosen['idx']; selected_seed=chosen['seed']; val_g=chosen['valGmean']; method='MSF-MOFS'; best_val=best
        else:
            names=list(feats); idx=np.arange(len(feats),dtype=int); selected_seed=None; guided=[]; method='all-local-features'
            knn=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)); knn.fit(Xtr,ytr); val_g,_,_=gmean(yv,knn.predict(Xv)); best_val=val_g
        normal_q=0.995 if sub in {'thermal','communication'} else 0.99
        fitted=fit_evidence(Xtr,ytr,Xv,yv,Xt,yt,idx,normal_quantile=normal_q); kg,kspec,krec=fitted['knn']; lg,lspec,lrec=fitted['logistic']
        result['subsystems'][sub]={
            'selectionMethod':method,
            'candidateFeatureCount':len(feats),'guidedCardinalities':guided,'selectedSeed':selected_seed,
            'selectedFeatureCount':len(idx),'selectionValidationGmean':float(val_g),'bestValidationGmean':float(best_val),
            'selectedFeatures':[feature_meta(n) for n in names],
            'imputationMedians':{n:float(med[n]) for n in names},
            'logisticEvidence':fitted['evidence'],
            'test':{'knnGmean':kg,'knnSpecificity':kspec,'knnRecall':krec,'logisticGmean':lg,'logisticSpecificity':lspec,'logisticRecall':lrec},
        }
        summaries.append({'subsystem':sub,'method':method,'D':len(feats),'seed':selected_seed,'K':len(idx),'valGmean':val_g,'testKnnGmean':kg,'testLogisticGmean':lg,'logisticSpecificity':lspec,'logisticRecall':lrec})
    out_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    csv_path=out_path.with_suffix('.csv'); pd.DataFrame(summaries).to_csv(csv_path,index=False)
    print(pd.DataFrame(summaries).to_string(index=False)); print('model',out_path)

if __name__=='__main__': main()
