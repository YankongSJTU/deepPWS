"""Paired continuous-score regression with nested selection of ranking losses.

Soft ranks use sigmoid pairwise comparisons, not the permutahedron algorithm.
Training uses minibatch soft Spearman; evaluation uses scipy hard average ranks.
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from local_neural_phase_cv import BASE, load, scaler_state
from local_nested_stability import prepare
from local_spatial_ridge import mm, ridge_grid
from local_three_class import cutoffs, labels, metrics as class_metrics

METHODS = ['regression', 'pairwise', 'soft_spearman']
TARGETS = ['color', 'area', 'overall_equal_weight']
ALPHAS = [1., 10., 100., 1000., 10000., 100000.]


def soft_rank(x, temperature):
    """Ascending ranks; self comparison cancels with .5; ties get mean ranks."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return .5 + ((x[:, None] - x[None]) / temperature).sigmoid().sum(1)


def average_rank(y):
    d = y[:, None] - y[None]
    return .5 + ((d > 0).to(y.dtype) + .5 * (d == 0).to(y.dtype)).sum(1)


def soft_spearman_loss(pred, truth, temperature):
    ranks = soft_rank(pred, temperature) / len(pred)
    target = average_rank(truth.detach()) / len(truth)
    a, b = ranks - ranks.mean(0), target - target.mean(0)
    bv = b.square().sum(0)
    valid = bv > 1e-8
    if not valid.any():
        return pred.sum() * 0
    denom = (a.square().sum(0) * bv).clamp_min(1e-12).sqrt()
    corr = (a * b).sum(0) / denom
    return (1 - corr[valid]).mean()


def pairwise_loss(pred, truth, temperature):
    direction = (truth[:, None] - truth[None]).sign()
    difference = pred[:, None] - pred[None]
    valid = direction != 0
    if not valid.any():
        return pred.sum() * 0
    return F.softplus(-direction[valid] * difference[valid] / temperature).mean()


def score_loss(pred, target, config):
    regression = F.smooth_l1_loss(pred, target, beta=.2)
    if config['method'] == 'regression':
        return regression
    ranking = pairwise_loss(pred, target, config['temperature']) if config['method'] == 'pairwise' else soft_spearman_loss(pred, target, config['temperature'])
    return regression + config['weight'] * ranking


class PairedScoreRegressor(nn.Module):
    def __init__(self, clinical_dim):
        super().__init__()
        self.appearance = nn.Sequential(nn.Linear(768, 64), nn.LayerNorm(64), nn.GELU())
        self.clinical = nn.Sequential(nn.Linear(clinical_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(224, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 2))

    def forward(self, data):
        before, after = self.appearance(data['before']), self.appearance(data['after'])
        return self.head(torch.cat([before, after, after-before, self.clinical(data['clinical'])], 1))


def partition(features, clinical, train, test):
    ca, cb, state = prepare(features, clinical, train, test)
    scale = scaler_state(np.concatenate([features[p][0][train] for p in ['before', 'after']]))
    a, b = {'clinical': ca['clinical']}, {'clinical': cb['clinical']}
    for phase in ['before', 'after']:
        x = features[phase][0]
        a[phase] = ((x[train]-scale['mean'])/scale['std']).astype(np.float32)
        b[phase] = ((x[test]-scale['mean'])/scale['std']).astype(np.float32)
    state['shared_appearance_scaler'] = scale
    return a, b, state


def transform(features, clinical, state):
    """Apply saved training preprocessing; no fitting on inference records."""
    encoded=[]
    for column,categories in zip(state['clinical_columns'],state['clinical_categories']):
        values=clinical[column].to_numpy()
        encoded.extend((values==category).astype(np.float32) for category in categories)
    age_columns=[c for c in clinical if c not in state['clinical_columns']]
    assert len(age_columns)==1
    age=pd.to_numeric(clinical[age_columns[0]],errors='coerce').to_numpy(np.float32)
    matrix=np.stack(encoded+[np.nan_to_num(age,nan=state['age_median'])],1)
    c=state['scalers']['clinical'];s=state['shared_appearance_scaler']
    result={'clinical':((matrix-c['mean'])/c['std']).astype(np.float32)}
    for phase in ['before','after']:
        result[phase]=((features[phase][0]-s['mean'])/s['std']).astype(np.float32)
    return result


def fit(a, y, b, config, seed, epochs, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = PairedScoreRegressor(a['clinical'].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.001)
    data = {k: torch.from_numpy(v).to(device) for k,v in a.items()}
    target = torch.from_numpy((y.astype(np.float32)-5)/5).to(device)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train(); running = 0.
        for ix in torch.randperm(len(y), generator=generator).split(64):
            ix = ix.to(device); optimizer.zero_grad()
            loss = score_loss(model({k:v[ix] for k,v in data.items()}), target[ix], config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite score/rank loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            running += float(loss.detach()) * len(ix)
        if epoch in [0, epochs-1]:
            history.append(dict(epoch=epoch+1, loss=running/len(y)))
    model.eval()
    with torch.inference_mode():
        pred = model({k:torch.from_numpy(v).to(device) for k,v in b.items()}).cpu().numpy()*5+5
    assert np.isfinite(pred).all()
    return pred, {k:v.cpu() for k,v in model.state_dict().items()}, history


def add_composite(x):
    return np.c_[x, x.mean(1)]


def regression_metrics(y, pred):
    rho = float(spearmanr(y,pred).statistic) if np.ptp(pred)>0 and np.ptp(y)>0 else 0.
    return dict(spearman=rho, mae=float(np.abs(y-pred).mean()), rmse=float(np.sqrt(np.mean((y-pred)**2))), within1=float(np.mean(np.abs(y-pred)<=1)), prediction_std=float(pred.std()), constant_prediction=bool(np.ptp(pred)==0))


def joint(y, pred):
    pred = pred.clip(0,10)
    return dict(spearman=float(np.mean([regression_metrics(y[:,j],pred[:,j])['spearman'] for j in range(2)])), mae=float(np.abs(y-pred).mean()))


def configurations(method):
    weights = [0.] if method == 'regression' else [.05,.2,.5]
    temperatures = [.2] if method != 'soft_spearman' else [.1,.3]
    return [dict(method=method, lr=lr, weight=w, temperature=t) for lr in [1e-4,3e-4] for w in weights for t in temperatures]


def choose(y, predictions, configs, maximum_mae):
    candidates = [dict(config=c, metrics=joint(y,p)) for c,p in zip(configs,predictions)]
    allowed = [i for i,c in enumerate(candidates) if c['metrics']['mae'] <= maximum_mae + 1e-12]
    winner = max(allowed,key=lambda i:(candidates[i]['metrics']['spearman'],-candidates[i]['metrics']['mae'])) if allowed else min(range(len(candidates)),key=lambda i:candidates[i]['metrics']['mae'])
    return winner, candidates, bool(allowed)


def select_inner(a, y, b, truth, seed, epochs, device, progress=None):
    selected, candidates, probes = {}, [], {}
    maximum_mae = None
    for method in METHODS:
        configs = configurations(method); preds = []
        for config in configs:
            p, _, _ = fit(a,y,b,config,seed,epochs,device); preds.append(p)
        if method == 'regression':
            maximum_mae = 1.05 * min(joint(truth,p)['mae'] for p in preds)
        ix, rows, feasible = choose(truth,preds,configs,maximum_mae)
        selected[method] = dict(**rows[ix], mae_guard=maximum_mae, guard_feasible=feasible)
        candidates.extend(rows); probes[method] = preds[ix].clip(0,10)
        if progress:
            print(progress, method, 'inner selected', selected[method], flush=True)
    rp = ridge_grid(blocks(a),y,blocks(b),ALPHAS)
    rc = [dict(method='ridge',alpha=x) for x in ALPHAS]
    ix, rows, feasible = choose(truth,rp,rc,maximum_mae)
    selected['ridge'] = dict(**rows[ix],mae_guard=maximum_mae,guard_feasible=feasible)
    candidates.extend(rows); probes['ridge'] = rp[ix]
    return selected,candidates,probes


def blocks(data):
    return [data['before'],data['after'],data['after']-data['before'],data['clinical']]


def fit_ridge(a,y,b,alpha):
    x = np.concatenate(blocks(a),1).astype(np.float64)
    z = np.concatenate(blocks(b),1).astype(np.float64)
    mean,ym = x.mean(0),y.mean(0);x-=mean
    vals,vecs = eigh(mm(x,x.T),check_finite=False)
    dual = vecs@((vecs.T@(y-ym))/(np.maximum(vals,0)[:,None]+alpha))
    coef = mm(x.T,dual)
    state = dict(input_mean=mean,target_mean=ym,coefficient=coef,alpha=alpha)
    return ((z-mean)@coef+ym).clip(0,10),state


def summarize(out):
    frame = pd.read_csv(out/'predictions.csv')
    summary = []
    for keys,r in frame.groupby(['method','family','seed','target']):
        summary.append(dict(zip(['method','family','seed','target'],keys),n=len(r),**regression_metrics(r.truth.to_numpy(),r.prediction.to_numpy())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    cs = []
    for keys,r in pd.read_csv(out/'class_predictions.csv').groupby(['method','family','seed','scheme','target']):
        cs.append(dict(zip(['method','family','seed','scheme','target'],keys),n=len(r),**class_metrics(r.truth.to_numpy(),r.prediction.to_numpy())))
    pd.DataFrame(cs).to_csv(out/'classification_summary.csv',index=False)
    print(pd.DataFrame(summary).query("family != 'neural'").to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default='paired_score_rank_10cv')
    parser.add_argument('--fold-limit',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--inner-epochs',type=int,default=60)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261011,20261012,20261013])
    args=parser.parse_args();torch.set_num_threads(4);threadpool_limits(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    records,features,clinical=load();y=records[['color','area']].to_numpy(np.float32)
    out=BASE/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Use a new run; do not overwrite experiments')
    (out/'protocol.json').write_text(json.dumps(dict(epochs=args.epochs,inner_epochs=args.inner_epochs,seeds=args.seeds,fold_limit=args.fold_limit,methods=METHODS,configurations={m:configurations(m) for m in METHODS},score_scaling='(score-5)/5; beta .2 =1 score point',rank_temperature_units='normalized scores; .1=.5 score points,.3=1.5 points',soft_rank='0.5+sum_j sigmoid((p_i-p_j)/tau), Pearson against exact average target ranks',ties='average true ranks; exclude tied target pairs from pairwise loss',selection='joint true inner Spearman, subject to mean MAE <=1.05*best baseline inner MAE; infeasible fallback min MAE',ridge_alphas=ALPHAS,architecture='shared frozen global CLIP adapter, before/after/difference plus baseline clinical fields; output two continuous scores',upstream='frozen CLIP/LAM, no image encoder fine tuning',classification='fixed3/6 and train-only tertiles, no fitted decision bias or moved class cuts',outer_folds='existing inspected neural_real_phase_10cv/splits.csv; exploratory',device=device),indent=2))
    splits=pd.read_csv(BASE/'outputs/neural_real_phase_10cv/splits.csv')
    rows,cr,foldrows,innerrows,choices=[],[],[],[],[]

    def add(pred,ids,train,fold,method,family,seed):
        pred=add_composite(pred.clip(0,10));truth=add_composite(y)
        for j,target in enumerate(TARGETS):
            foldrows.append(dict(fold=int(fold),method=method,family=family,seed=seed,target=target,**regression_metrics(truth[ids,j],pred[:,j])))
            rows.extend(dict(fold=int(fold),method=method,family=family,seed=seed,target=target,record_key=records.record_key[i],truth=float(t),prediction=float(p)) for i,t,p in zip(ids,truth[ids,j],pred[:,j]))
        for scheme in ['fixed','train_tertiles']:
            cuts=cutoffs(truth,train,scheme);trueclass=labels(truth[ids],cuts);predclass=labels(pred,cuts)
            for j,target in enumerate(TARGETS):
                cr.extend(dict(fold=int(fold),method=method,family=family,seed=seed,scheme=scheme,target=target,record_key=records.record_key[i],truth=int(t),prediction=int(p),cut_low=float(cuts[j,0]),cut_high=float(cuts[j,1])) for i,t,p in zip(ids,trueclass[:,j],predclass[:,j]))

    for fold,frame in list(splits.groupby('fold'))[:args.fold_limit]:
        train=np.flatnonzero(records.record_key.isin(frame.loc[frame.role.eq('train'),'record_key']))
        test=np.flatnonzero(records.record_key.isin(frame.loc[frame.role.eq('test'),'record_key']))
        assert not set(records.split_group.iloc[train])&set(records.split_group.iloc[test])
        itr,iva=next(GroupShuffleSplit(1,test_size=.2,random_state=20261010+int(fold)).split(train,groups=records.split_group.iloc[train]))
        it,iv=train[itr],train[iva]
        assert not set(records.split_group.iloc[it])&set(records.split_group.iloc[iv])
        for role,ids in [('train',it),('validation',iv)]:innerrows.extend(dict(fold=int(fold),role=role,record_key=records.record_key[i]) for i in ids)
        ia,ib,_=partition(features,clinical,it,iv);a,b,state=partition(features,clinical,train,test)
        selected,candidates,ips=select_inner(ia,y[it],ib,y[iv],20261012+int(fold),args.inner_epochs,device,progress=f'fold {fold+1}')
        outer={}
        for method in METHODS:
            allpred=[];c=selected[method]['config']
            for seed in args.seeds:
                raw,weights,history=fit(a,y[train],b,c,seed+100*int(fold),args.epochs,device)
                p=raw.clip(0,10);allpred.append(p)
                probe={k:v[:2] for k,v in b.items()}
                torch.save(dict(method=method,config=c,fold=int(fold),seed=seed,epochs=args.epochs,clinical_dim=a['clinical'].shape[1],model=weights,preprocessing=state,input_probe=probe,raw_score_probe=raw[:2],history=history,clip_fraction=float(np.mean(raw!=p)),train_keys=records.record_key.iloc[train].tolist(),test_keys=records.record_key.iloc[test].tolist()),out/f'fold_{fold:02d}_{method}_seed{seed}.pt')
                add(p,test,train,fold,method,'neural',seed)
            outer[method]=np.mean(allpred,0);add(outer[method],test,train,fold,method,'neural_ensemble',-1)
            print('fold',fold+1,method,'outer complete',flush=True)
        p,linear=fit_ridge(a,y[train],b,selected['ridge']['config']['alpha']);outer['ridge']=p
        with open(out/f'fold_{fold:02d}_ridge.pkl','wb') as f:pickle.dump(dict(linear=linear,preprocessing=state,input_probe={k:v[:2] for k,v in b.items()},score_probe=p[:2],train_keys=records.record_key.iloc[train].tolist(),test_keys=records.record_key.iloc[test].tolist()),f,protocol=5)
        add(p,test,train,fold,'ridge','linear',-1)
        named={};combined=[]
        for j,target in enumerate(TARGETS[:2]):
            feasible=[m for m in outer if regression_metrics(y[iv,j],ips[m][:,j])['mae']<=1.05*regression_metrics(y[iv,j],ips['regression'][:,j])['mae']]
            winner=max(feasible,key=lambda m:(regression_metrics(y[iv,j],ips[m][:,j])['spearman'],-regression_metrics(y[iv,j],ips[m][:,j])['mae']))
            named[target]=winner;combined.append(outer[winner][:,j])
        add(np.stack(combined,1),test,train,fold,'inner_selected','selected_pipeline',-1)
        choices.append(dict(fold=int(fold),selected=selected,candidates=candidates,inner_selected=named,inner_train_keys=records.record_key.iloc[it].tolist(),inner_validation_keys=records.record_key.iloc[iv].tolist()))
        pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False);pd.DataFrame(cr).to_csv(out/'class_predictions.csv',index=False)
        pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(innerrows).to_csv(out/'inner_splits.csv',index=False)
        (out/'choices.json').write_text(json.dumps(choices,indent=2))
        print('fold',fold+1,'complete',flush=True)
    summarize(out)
    print('Score-rank experiments complete',flush=True)


if __name__=='__main__':main()
