"""Patient-group LOOCV followed by source refit and matched external transfer.

Reuses the verified Ridge equations and train-only preprocessing. Each held-out
patient has its own 3-fold grouped inner alpha selection. Supports resumption.
"""
from pathlib import Path
import argparse, hashlib, json, pickle, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fusion_model as aligned
import ridge_multicenter_transfer as core
from local_score_rank import add_composite, regression_metrics
from local_three_class import cutoffs, labels, metrics as class_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='clip_farl_3d_shanghai_group_loocv_external50_10seed_verified_20261013')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--outer-limit', type=int, default=0, help='Positive values run a source-only smoke test; zero means all groups')
    args = p.parse_args()
    aligned.configure()
    threadpool_limits(4)
    records, features, clinical, audit = core.load_multicenter()
    scores = records[['color', 'area']].to_numpy(np.float32)
    truth = add_composite(scores)
    groups = records.split_group.astype(str).to_numpy()
    sh = np.flatnonzero(records.center.eq('Shanghai'))
    patients = np.unique(groups[sh])
    out = core.BASE / 'outputs' / args.run
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(records.to_csv(index=False).encode())
    digest.update(clinical.to_csv(index=False).encode())
    for name in sorted(features):
        digest.update(name.encode())
        parts = features[name] if name in ['before', 'after'] else [features[name]]
        for part in parts:
            digest.update(np.ascontiguousarray(part).tobytes())
    signature = digest.hexdigest()
    protocol = dict(input_sha256=signature, source='Shanghai', source_outer_cv='leave one patient group out; all records from held-out group excluded', source_groups=len(patients), source_records=len(sh), source_selection='3-fold grouped inner CV within each training partition; joint Spearman with 5% MAE guard', source_final='full Shanghai inner-selected refit, never best test fold', external_train_records=50, seeds=core.SEEDS, target_selection='3-fold grouped training-only CV', preprocessing='source-fitted preprocessing frozen in target adaptation', features='whole-image CLIP + clinical + aligned delta-a CLIP and Lab statistics + dense geometry144 + UV240 + FaRL global before512/after512/difference512', classification='score-derived fixed3/6 and source-training tertiles', outer_limit=args.outer_limit, evidence='previously explored cohorts; development-stage validation')
    pp = out / 'protocol.json'
    if pp.exists():
        old = json.loads(pp.read_text())
        if not args.resume:
            raise FileExistsError('Use a fresh run or --resume')
        if old.get('input_sha256') != signature or old.get('outer_limit') != args.outer_limit:
            raise ValueError('Resume input or fold budget mismatch')
    pp.write_text(json.dumps(protocol, indent=2))
    (out / 'input_audit.json').write_text(json.dumps(audit, indent=2))
    rows, cats, splitrows, choices = [], [], [], []
    replay_error = 0.

    def add(pred, ids, source_train, tag, center, seed=-1, fold=-1):
        prediction = add_composite(pred.clip(0, 10))
        for j, target in enumerate(core.TARGETS):
            rows.extend(dict(protocol=tag, center=center, seed=seed, fold=fold, target=target, record_key=records.record_key[i], split_group=groups[i], truth=float(t), prediction=float(v)) for i, t, v in zip(ids, truth[ids, j], prediction[:, j]))
        for scheme in ['fixed', 'train_tertiles']:
            cuts = cutoffs(truth, source_train, scheme)
            tc, pc = labels(truth[ids], cuts), labels(prediction, cuts)
            for j, target in enumerate(core.TARGETS):
                cats.extend(dict(protocol=tag, center=center, seed=seed, fold=fold, scheme=scheme, target=target, record_key=records.record_key[i], truth=int(t), prediction=int(v), cut_low=float(cuts[j, 0]), cut_high=float(cuts[j, 1])) for i, t, v in zip(ids, tc[:, j], pc[:, j]))

    def fitted(path, train, test, prior=None, frozen=None):
        nonlocal replay_error
        if path.exists():
            with path.open('rb') as f:
                state = pickle.load(f)
            assert state['input_sha256'] == signature
            assert state['training_keys'] == records.record_key.iloc[train].tolist()
            assert state['test_keys'] == records.record_key.iloc[test].tolist()
            x = core.matrix(core.transform(features, clinical, state['preprocessing']))
            pred = core.raw_prediction(x[test], state['linear']).clip(0, 10)
            choice = state['selection']
        else:
            if frozen is None:
                alpha, choice = core.source_selection(features, clinical, scores, records, train)
                a, b, prep = core.partition(features, clinical, train, test)
                xtrain, xtest = core.matrix(a), core.matrix(b)
            else:
                x, prep = frozen
                xtrain, xtest = x[train], x[test]
                alpha, choice = core.transfer_selection(xtrain, scores[train], groups[train], prior)
            predictions, models = core.linear_grid(xtrain, scores[train], xtest, [alpha], prior)
            pred = predictions[0]
            state = dict(input_sha256=signature, linear=models[0], preprocessing=prep, training_keys=records.record_key.iloc[train].tolist(), test_keys=records.record_key.iloc[test].tolist(), selection=choice)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as f:
                pickle.dump(state, f, protocol=5)
            temporary.replace(path)
        replay = aligned.predict_saved_model(path, {k: (v[0][test] if k in ['before', 'after'] else v[test]) for k, v in features.items()}, clinical.iloc[test])
        replay_error = max(replay_error, float(np.abs(replay[:, :2] - pred).max(initial=0)))
        assert replay_error < 1e-5
        return pred, state, choice

    def flush():
        frame = pd.DataFrame(rows)
        frame.to_csv(out / 'predictions.csv', index=False)
        pd.DataFrame(cats).to_csv(out / 'class_predictions.csv', index=False)
        (out / 'choices.json').write_text(json.dumps(choices, indent=2))

    for fold, patient in enumerate(patients):
        if args.outer_limit and fold >= args.outer_limit:
            break
        test = sh[groups[sh] == patient]
        train = sh[groups[sh] != patient]
        assert not set(groups[train]) & set(groups[test])
        pred, state, choice = fitted(out / f'shanghai_fold_{fold:03d}.pkl', train, test)
        add(pred, test, train, 'shanghai_group_loocv', 'Shanghai', fold=fold)
        choices.append(dict(stage='shanghai_group_loocv', fold=fold, held_out_group=patient, **choice))
        flush()
        print(f'LOOCV {fold+1}/{len(patients)}: {len(test)} held-out records', flush=True)
    if args.outer_limit:
        (out / 'validation.json').write_text(json.dumps(dict(status='source-only smoke complete; not full LOOCV', replay_max_abs_error=replay_error, folds=min(args.outer_limit, len(patients))), indent=2))
        return
    oof = pd.DataFrame(rows).query("protocol == 'shanghai_group_loocv' and target == 'color'")
    assert len(oof) == len(sh) and oof.record_key.is_unique
    all_ids = np.arange(len(records))
    _, source, choice = fitted(out / 'shanghai_source.pkl', sh, all_ids)
    x = core.matrix(core.transform(features, clinical, source['preprocessing']))
    choices.append(dict(stage='source_full', **choice))
    for center in ['Chongqing', 'Zhejiang']:
        ids = np.flatnonzero(records.center.eq(center))
        add(core.raw_prediction(x[ids], source['linear']), ids, sh, 'direct_full', center)
        for seed in core.SEEDS:
            train, test = core.fifty_record_groups(ids, groups, seed)
            assert len(train) == 50 and not set(groups[train]) & set(groups[test])
            pred, _, choice = fitted(out / f'{center}_seed{seed}_transfer50.pkl', train, test, source['linear'], (x, source['preprocessing']))
            add(core.raw_prediction(x[test], source['linear']), test, sh, 'direct_matched', center, seed=seed)
            add(pred, test, sh, 'transfer50', center, seed=seed)
            choices.append(dict(stage='transfer50', center=center, seed=seed, **choice))
            for role, indices in [('train', train), ('test', test)]:
                splitrows.extend(dict(center=center, seed=seed, role=role, record_key=records.record_key[i], split_group=groups[i]) for i in indices)
            flush()
            print(center, seed, 'complete', flush=True)
    pd.DataFrame(splitrows).to_csv(out / 'external_splits.csv', index=False)
    frame = pd.DataFrame(rows)
    summaries = []
    for keys, q in frame.groupby(['protocol', 'center', 'seed', 'target']):
        summaries.append(dict(zip(['protocol', 'center', 'seed', 'target'], keys), n_test=len(q), **regression_metrics(q.truth.to_numpy(), q.prediction.to_numpy())))
    pd.DataFrame(summaries).to_csv(out / 'split_metrics.csv', index=False)
    classification = []
    for keys, q in pd.DataFrame(cats).groupby(['protocol', 'center', 'seed', 'scheme', 'target']):
        classification.append(dict(zip(['protocol', 'center', 'seed', 'scheme', 'target'], keys), n=len(q), **class_metrics(q.truth.to_numpy(), q.prediction.to_numpy())))
    pd.DataFrame(classification).to_csv(out / 'classification_metrics.csv', index=False)
    (out / 'validation.json').write_text(json.dumps(dict(status='complete', source_oof_records=len(sh), source_groups=len(patients), replay_max_abs_error=replay_error, external_seeds_per_center=len(core.SEEDS)), indent=2))
    print('Full patient-group LOOCV and external transfer complete', flush=True)

if __name__ == '__main__':
    main()
