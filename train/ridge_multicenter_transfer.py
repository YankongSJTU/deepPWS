"""Shanghai nested10CV Ridge and direct/50-record prior-centered transfer."""
import argparse
import json
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from local_neural_phase_cv import BASE, clinical as sh_clinical
from local_score_rank import partition, transform, blocks, add_composite, joint, regression_metrics
from local_spatial_ridge import mm
from local_three_class import cutoffs, labels, metrics as classification_metrics

TARGETS=['color','area','overall_equal_weight']
ALPHAS=np.logspace(-1,5,25)
SEEDS=list(range(20261013,20261023))


def load_multicenter():
    records=pd.read_csv(BASE/'outputs/records.csv')
    records=records[records.status.eq('eligible')].reset_index(drop=True)
    sh=records[records.center.eq('Shanghai')].reset_index(drop=True)
    sc=sh_clinical(sh);clinical=pd.DataFrame(index=records.index,columns=sc.columns)
    clinical.loc[records.center.eq('Shanghai')]=sc.to_numpy()
    audits=[]
    for center in ['Chongqing','Zhejiang']:
        mask=records.center.eq(center);q=records[mask]
        table=pd.read_excel(BASE.parent/'data'/f'{center}.score.xls')
        table.columns=[str(c).strip() for c in table.columns]
        idcolumn=table.columns[1]
        selected=table.iloc[q.excel_row.astype(int).to_numpy()-2]
        normalize=lambda x:x.astype(str).str.strip().str.replace(r'\.0$','',regex=True)
        assert np.array_equal(normalize(selected[idcolumn]).to_numpy(),normalize(q.patient).to_numpy()),center+' Excel patient mismatch'
        scores=selected.iloc[:,2:4].apply(pd.to_numeric,errors='raise').to_numpy(float)
        assert np.allclose(scores,q[['color','area']].to_numpy(float)),center+' Excel score mismatch'
        assert set(sc.columns)<=set(table.columns)
        clinical.loc[mask]=selected[sc.columns].fillna('missing').astype(str).to_numpy()
        audits.append(dict(center=center,records=len(q),patients=q.split_group.nunique(),exact_excel_row_ids_verified=True,actual_scores_verified=True))
    assert not clinical.isna().any().any()
    phases={p:[] for p in ['before','after']};hashgroups=defaultdict(set);centerhash=defaultdict(set)
    for q in records.itertuples():
        pairs=json.loads(q.pairs)
        for phase in phases:
            embeddings=[]
            for pair in pairs:
                sha=pair[phase+'_hash'];centerhash[q.center].add(sha);hashgroups[(q.center,sha)].add(q.split_group)
                with np.load(BASE/'outputs/raw_features'/f'{sha}.npz') as z:embeddings.append(z['embedding'])
            phases[phase].append(np.mean(embeddings,0))
    assert all(len(g)==1 for g in hashgroups.values()),'Images shared across patients'
    assert all(not centerhash['Shanghai']&centerhash[c] for c in ['Chongqing','Zhejiang'])
    features={p:[np.asarray(values,np.float32),np.zeros((len(records),1),np.float32),np.zeros((len(records),1),np.float32)] for p,values in phases.items()}
    assert all(np.isfinite(features[p][0]).all() for p in phases)
    return records,features,clinical.astype(str),audits


def matrix(data):return np.concatenate(blocks(data),1).astype(np.float64)


def raw_prediction(x,state):return mm(x,state['coefficient'])+state['intercept']


def predict_saved_model(model_path, features, clinical):
    """Inference on frozen phase features/baseline fields, without scores.

    features={'before':[N x768], 'after':[N x768]}. Returns color,area,overall.
    The stored source preprocessing is reused, never refitted on new patients.
    """
    with open(model_path,'rb') as f:saved=pickle.load(f)
    x=matrix(transform(features,clinical,saved['preprocessing']))
    return add_composite(raw_prediction(x,saved['linear']).clip(0,10))


def linear_grid(x,y,z,alphas,prior=None):
    """min ||Y-XB-b||^2+alpha||B-B_source||^2; free intercept.

    prior=None gives ordinary Ridge. Frozen source preprocessing for adaptation.
    Residuals use raw source scores, not previously clipped predictions.
    """
    residual=y if prior is None else y-raw_prediction(x,prior)
    mean,ym=x.mean(0),residual.mean(0);xx=x-mean
    vals,vecs=eigh(mm(xx,xx.T),check_finite=False)
    projection=vecs.T@(residual-ym)
    predictions,states=[],[]
    for alpha in alphas:
        dual=vecs@(projection/(np.maximum(vals,0)[:,None]+alpha))
        delta=mm(xx.T,dual);coef=delta if prior is None else prior['coefficient']+delta
        intercept=ym-mm(mean[None],delta)[0]
        if prior is not None:intercept=intercept+prior['intercept']
        state=dict(coefficient=coef,intercept=intercept,alpha=float(alpha),prior_centered=prior is not None)
        predictions.append(raw_prediction(z,state).clip(0,10));states.append(state)
    return predictions,states


def select_predictions(y,predictions,alphas):
    results=[dict(alpha=float(a),**joint(y,p)) for a,p in zip(alphas,predictions)]
    guard=1.05*min(r['mae'] for r in results)
    winner=max((i for i,r in enumerate(results) if r['mae']<=guard),key=lambda i:(results[i]['spearman'],-results[i]['mae']))
    return float(alphas[winner]),dict(selected=results[winner],mae_guard=guard,candidates=results)


def source_selection(features,clinical,scores,records,train):
    predictions=[np.empty((len(train),2),float) for _ in ALPHAS]
    for tr,va in GroupKFold(3).split(train,groups=records.split_group.iloc[train]):
        ti,vi=train[tr],train[va]
        assert not set(records.split_group.iloc[ti])&set(records.split_group.iloc[vi])
        a,b,_=partition(features,clinical,ti,vi)
        ps,_=linear_grid(matrix(a),scores[ti],matrix(b),ALPHAS)
        for target,p in zip(predictions,ps):target[va]=p
    return select_predictions(scores[train],predictions,ALPHAS)


def transfer_selection(x,y,groups,prior):
    predictions=[np.empty_like(y,dtype=float) for _ in ALPHAS]
    for tr,va in GroupKFold(3).split(x,groups=groups):
        assert not set(groups[tr])&set(groups[va])
        ps,_=linear_grid(x[tr],y[tr],x[va],ALPHAS,prior)
        for target,p in zip(predictions,ps):target[va]=p
    return select_predictions(y,predictions,ALPHAS)


def fifty_record_groups(ids,groups,seed,count=50):
    """Randomized subset-sum gives exactly50 records, retaining whole patients."""
    rng=np.random.default_rng(seed);unique=np.unique(groups[ids]);dp={0:[]}
    for group in rng.permutation(unique):
        weight=int(np.sum(groups[ids]==group))
        for total in sorted(list(dp),reverse=True):
            if total+weight<=count and total+weight not in dp:dp[total+weight]=dp[total]+[group]
        if count in dp:break
    if count not in dp:raise ValueError('Cannot select exact count with intact patient groups')
    train=ids[np.isin(groups[ids],dp[count])];test=ids[~np.isin(groups[ids],dp[count])]
    assert len(train)==count and len(test)>0 and not set(groups[train])&set(groups[test])
    return train,test


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='ridge_shanghai_10cv_external50_10seed');parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    threadpool_limits(4)
    records,features,clinical,audit=load_multicenter()
    print('Cohort audit',audit,flush=True)
    if args.audit_only:return
    out=BASE/'outputs'/args.run;out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Preserve existing experiment; select new run')
    protocol=dict(source='Shanghai',model='Ridge continuous color/area scores, overall derived mean',features='frozen global CLIP before/after/difference plus baseline clinical fields',source_outer_cv='existing10 patient-group folds; already explored',source_selection='3fold grouped inner CV, maximize joint true Spearman with MAE <=1.05 best candidate MAE',alpha_grid=ALPHAS.tolist(),source_final='full Shanghai refit after internal configuration selection, not a best outer-test fold',transfer='prior-centered Ridge correction around Shanghai coefficients with free intercept',target_selection='3fold grouped CV on only50 training records',external_train_records=50,seeds=SEEDS,target_split='random group-permutation subset-sum to exactly50 records, whole patients kept intact',preprocessing='Shanghai training preprocessing frozen during target adaptation',classification='fixed3/6 and source-training-only tertiles; external boundaries never fitted',external_direct='full cohort plus matched test subset for every seed',seed_summary='mean/std across overlapping random splits, not independent cohorts',targets=TARGETS)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'input_audit.json').write_text(json.dumps(audit,indent=2))
    scores=records[['color','area']].to_numpy(np.float32);truth=add_composite(scores);groups=records.split_group.astype(str).to_numpy()
    sh=np.flatnonzero(records.center.eq('Shanghai'));rows,classrows,metricrows,splits,choices=[],[],[],[],[]

    def add(pred,ids,source_train,protocol,center,seed=-1,fold=-1):
        p=add_composite(pred.clip(0,10))
        for j,target in enumerate(TARGETS):
            metricrows.append(dict(protocol=protocol,center=center,seed=seed,fold=fold,target=target,n_test=len(ids),test_patients=len(np.unique(groups[ids])),**regression_metrics(truth[ids,j],p[:,j])))
            rows.extend(dict(protocol=protocol,center=center,seed=seed,fold=fold,target=target,record_key=records.record_key[i],truth=float(t),prediction=float(v)) for i,t,v in zip(ids,truth[ids,j],p[:,j]))
        for scheme in ['fixed','train_tertiles']:
            cuts=cutoffs(truth,source_train,scheme);tc,pc=labels(truth[ids],cuts),labels(p,cuts)
            for j,target in enumerate(TARGETS):
                classrows.extend(dict(protocol=protocol,center=center,seed=seed,fold=fold,scheme=scheme,target=target,record_key=records.record_key[i],truth=int(t),prediction=int(v),cut_low=float(cuts[j,0]),cut_high=float(cuts[j,1])) for i,t,v in zip(ids,tc[:,j],pc[:,j]))

    def save(path,linear,state,ids,test,probe,choice):
        with open(path,'wb') as f:pickle.dump(dict(linear=linear,preprocessing=state,training_keys=records.record_key.iloc[ids].tolist(),test_keys=records.record_key.iloc[test].tolist(),input_probe=probe[:2],score_probe=raw_prediction(probe[:2],linear).clip(0,10),selection=choice),f,protocol=5)

    canonical=pd.read_csv(BASE/'outputs/neural_real_phase_10cv/splits.csv')
    for fold,frame in canonical.groupby('fold'):
        train=np.flatnonzero(records.record_key.isin(frame.loc[frame.role.eq('train'),'record_key']));test=np.flatnonzero(records.record_key.isin(frame.loc[frame.role.eq('test'),'record_key']))
        assert set(train)|set(test)==set(sh) and not set(groups[train])&set(groups[test])
        alpha,choice=source_selection(features,clinical,scores,records,train)
        a,b,state=partition(features,clinical,train,test);ps,states=linear_grid(matrix(a),scores[train],matrix(b),[alpha])
        save(out/f'shanghai_fold_{fold:02d}.pkl',states[0],state,train,test,matrix(b),choice)
        add(ps[0],test,train,'shanghai_10cv','Shanghai',fold=int(fold));choices.append(dict(stage='shanghai_10cv',fold=int(fold),**choice))
        print('Shanghai fold',fold+1,'complete alpha',alpha,flush=True)
    alpha,choice=source_selection(features,clinical,scores,records,sh)
    a,b,state=partition(features,clinical,sh,np.arange(len(records)));x=matrix(b)
    ps,states=linear_grid(matrix(a),scores[sh],x,[alpha]);source=states[0]
    save(out/'shanghai_source.pkl',source,state,sh,[],x,choice);choices.append(dict(stage='source_full',**choice))
    unknown=[]
    for center in ['Chongqing','Zhejiang']:
        ids=np.flatnonzero(records.center.eq(center))
        for column,categories in zip(state['clinical_columns'],state['clinical_categories']):
            unknown.append(dict(center=center,clinical_field=column,unknown_category_records=int((~clinical.iloc[ids][column].isin(categories)).sum()),records=len(ids)))
        add(raw_prediction(x[ids],source),ids,sh,'direct_full',center)
        for seed in SEEDS:
            train,test=fifty_record_groups(ids,groups,seed)
            for role,indices in [('train',train),('test',test)]:splits.extend(dict(center=center,seed=seed,role=role,record_key=records.record_key[i],split_group=groups[i]) for i in indices)
            alpha,selection=transfer_selection(x[train],scores[train],groups[train],source)
            pp,ss=linear_grid(x[train],scores[train],x[test],[alpha],source)
            save(out/f'{center}_seed{seed}_transfer50.pkl',ss[0],state,train,test,x[test],selection)
            add(raw_prediction(x[test],source),test,sh,'direct_matched',center,seed=seed)
            add(pp[0],test,sh,'transfer50',center,seed=seed)
            choices.append(dict(stage='transfer50',center=center,seed=seed,train_records=len(train),train_patients=len(np.unique(groups[train])),test_records=len(test),**selection))
            print(center,'seed',seed,'50 train',len(test),'test complete',flush=True)
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);pd.DataFrame(classrows).to_csv(out/'class_predictions.csv',index=False)
    pd.DataFrame(metricrows).to_csv(out/'split_metrics.csv',index=False);pd.DataFrame(splits).to_csv(out/'external_splits.csv',index=False)
    pd.DataFrame(unknown).to_csv(out/'external_unknown_categories.csv',index=False);(out/'choices.json').write_text(json.dumps(choices,indent=2))
    summary=[]
    for target,r in pd.DataFrame(rows).query("protocol=='shanghai_10cv'").groupby('target'):
        summary.append(dict(protocol='shanghai_10cv_pooled',center='Shanghai',target=target,n=len(r),**regression_metrics(r.truth.to_numpy(),r.prediction.to_numpy())))
    pd.DataFrame(summary).to_csv(out/'shanghai_summary.csv',index=False)
    cs=[]
    for keys,r in pd.DataFrame(classrows).groupby(['protocol','center','seed','scheme','target']):
        cs.append(dict(zip(['protocol','center','seed','scheme','target'],keys),n=len(r),**classification_metrics(r.truth.to_numpy(),r.prediction.to_numpy())))
    pd.DataFrame(cs).to_csv(out/'classification_metrics.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Shanghai and external transfer experiments complete',flush=True)


if __name__=='__main__':main()
