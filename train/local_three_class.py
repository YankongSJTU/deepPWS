"""Paired classification, ordinal and ranking exploration with grouped selection."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.linalg import eigh
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from threadpoolctl import threadpool_limits
from local_neural_phase_cv import BASE, load, scaler_state
from local_nested_stability import prepare
from local_pointnet_explore import load_points, PointEncoder
from local_spatial_ridge import patches, mm

TARGETS = ['color', 'area', 'overall_equal_weight']
VARIANTS = ['global_ce', 'global_balanced_ce', 'global_ordinal', 'global_rank', 'local_rank', 'points_rank', 'full_rank']
SCHEMES = ['fixed', 'train_tertiles']
ALPHAS = [1., 10., 100., 1000., 10000.]

def cutoffs(y, train, scheme):
    if scheme == 'fixed':
        return np.tile([3., 6.], (y.shape[1], 1))
    cuts = np.quantile(y[train], [1 / 3, 2 / 3], axis=0).T
    assert (cuts[:, 0] < cuts[:, 1]).all()
    return cuts

def labels(y, cuts):
    return ((y > cuts[:, 0]).astype(np.int64) + (y > cuts[:, 1]).astype(np.int64))

def metrics(y, pred):
    precision, recall, f1, n = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return dict(accuracy=float(np.mean(y == pred)), balanced_accuracy=float(recall.mean()), macro_f1=float(f1.mean()), recall80_classes=int((recall >= .8).sum()), recall85_classes=int((recall >= .85).sum()), precision80_classes=int((precision >= .8).sum()), precision85_classes=int((precision >= .85).sum()), **{f'{name}_{j}': float(v[j]) for name, v in [('precision', precision), ('recall', recall), ('f1', f1), ('support', n)] for j in range(3)})

def bias_search(probs, truth, grid=(-1., -.5, 0., .5, 1.), objective='macro_f1'):
    # Minimum bias magnitude breaks macro-F1 ties; zero is available explicitly.
    candidates = []
    for low in grid:
        for high in grid:
            bias = np.array([low, 0., high])
            pred = (np.log(np.clip(probs, 1e-8, 1)) + bias).argmax(1)
            m = metrics(truth, pred)
            candidates.append((m['macro_f1'], -abs(low) - abs(high), low, high, m))
    if objective == 'two_recall':
        best = max(candidates, key=lambda x: (sorted([x[4][f'recall_{j}'] for j in range(3)])[1], x[0], x[1]))
    else:
        best = max(candidates, key=lambda x: x[:2])
    return np.array([best[2], 0., best[3]]), best[4]

def decisions(probs, biases):
    return (np.log(np.clip(probs, 1e-8, 1)) + biases[None]).argmax(-1)

def dataset(features, local, points, clinical, train, test):
    ca, cb, state = prepare(features, clinical, train, test)
    a, b = {'clinical': ca['clinical']}, {'clinical': cb['clinical']}
    states = {}
    for name, data in [('appearance', {p: features[p][0] for p in ['before', 'after']}), ('local', local), ('uv', {p: features[p][2] for p in ['before', 'after']})]:
        scale = scaler_state(np.concatenate([data[p][train] for p in ['before', 'after']]))
        states[name] = scale
        for p in ['before', 'after']:
            a[p + '_' + name] = ((data[p][train] - scale['mean']) / scale['std']).astype(np.float32)
            b[p + '_' + name] = ((data[p][test] - scale['mean']) / scale['std']).astype(np.float32)
    for p in ['before', 'after']:
        a[p + '_points'], b[p + '_points'] = points[p][train], points[p][test]
    state['shared_phase_scalers'] = states
    return a, b, state

class PairedClassifier(nn.Module):
    def __init__(self, clinical_dim, variant):
        super().__init__()
        self.variant = variant
        self.ordinal = variant not in ['global_ce', 'global_balanced_ce']
        self.encoders = nn.ModuleDict({'appearance': nn.Sequential(nn.Linear(768, 64), nn.LayerNorm(64), nn.GELU())})
        self.gates = nn.ParameterDict()
        width = 64
        if variant in ['local_rank', 'points_rank', 'full_rank']:
            self.encoders['local'] = nn.Sequential(nn.Linear(1024, 32), nn.LayerNorm(32), nn.GELU())
            self.gates['local'] = nn.Parameter(torch.tensor(-1.)); width += 32
        if variant in ['points_rank', 'full_rank']:
            self.encoders['points'] = PointEncoder()
            self.gates['points'] = nn.Parameter(torch.tensor(-1.)); width += 32
        if variant == 'full_rank':
            self.encoders['uv'] = nn.Sequential(nn.Linear(768, 32), nn.LayerNorm(32), nn.GELU())
            self.gates['uv'] = nn.Parameter(torch.tensor(-1.)); width += 32
        self.clinical = nn.Sequential(nn.Linear(clinical_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(width * 3 + 32, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 3 if self.ordinal else 9))
        if self.ordinal:
            # A gap below 2*log(2) forbids the middle class from being the mode.
            # Start with a usable middle region; retain ordered learned cuts.
            self.threshold_base = nn.Parameter(torch.full((3,), -1.5))
            self.threshold_gap = nn.Parameter(torch.full((3,), float(np.log(np.expm1(3.)))))

    def forward(self, data):
        parts = []
        for name, encoder in self.encoders.items():
            encoded = [encoder(data[p + '_' + name]) for p in ['before', 'after']]
            if name in self.gates:
                gate = self.gates[name].sigmoid()
                drop = torch.ones((len(encoded[0]), 1), device=encoded[0].device)
                if self.training:
                    drop = (torch.rand_like(drop) >= .15).float() / .85
                encoded = [x * gate * drop for x in encoded]
            parts.extend(encoded + [encoded[1] - encoded[0]])
        out = self.head(torch.cat(parts + [self.clinical(data['clinical'])], dim=1))
        if self.ordinal:
            thresholds = torch.stack([self.threshold_base, self.threshold_base + F.softplus(self.threshold_gap)], dim=1)
            return out[..., None] - thresholds[None]
        return out.reshape(-1, 3, 3)

def probabilities(logits, ordinal):
    if not ordinal:
        return logits.softmax(-1)
    q = logits.sigmoid()
    return torch.stack([1 - q[..., 0], q[..., 0] - q[..., 1], q[..., 1]], dim=-1)

def loss_fn(logits, y, scores, variant, weights):
    ordinal = variant not in ['global_ce', 'global_balanced_ce']
    if ordinal:
        cumulative = (y[..., None] > torch.arange(2, device=y.device)).float()
        loss = F.binary_cross_entropy_with_logits(logits, cumulative)
    else:
        loss = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), reduction='none').reshape_as(y)
        if variant == 'global_balanced_ce':
            loss = loss * torch.stack([weights[j, y[:, j]] for j in range(3)], dim=1)
        loss = loss.mean()
    if variant.endswith('rank'):
        expected = (probabilities(logits, ordinal) * torch.arange(3, device=y.device)).sum(-1)
        direction = (scores[:, None] - scores[None]).sign()
        differences = expected[:, None] - expected[None]
        mask = direction != 0
        if mask.any():
            loss = loss + .2 * F.softplus(-direction[mask] * differences[mask]).mean()
    return loss

def fit(train, y, scores, test, variant, lr, seed, epochs, device):
    torch.manual_seed(seed)
    model = PairedClassifier(train['clinical'].shape[1], variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    data = {k: torch.from_numpy(v).to(device) for k, v in train.items()}
    target = torch.from_numpy(y).to(device)
    continuous = torch.from_numpy(scores.astype(np.float32)).to(device)
    counts = np.stack([np.bincount(y[:, j], minlength=3) for j in range(3)])
    assert (counts > 0).all()
    weights = torch.from_numpy(np.clip(len(y) / (3 * counts), .5, 2).astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(y), generator=generator).split(64):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model({k: v[batch] for k, v in data.items()}), target[batch], continuous[batch], variant, weights)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
    model.eval()
    with torch.inference_mode():
        pred = probabilities(model({k: torch.from_numpy(v).to(device) for k, v in test.items()}), model.ordinal).cpu().numpy()
    return pred, {k: v.cpu() for k, v in model.state_dict().items()}

def ridge_probabilities(train, y, test, alphas):
    def arrays(data):
        b, a = data['before_appearance'], data['after_appearance']
        return np.concatenate([b, a, a - b, data['clinical']], axis=1).astype(np.float64)
    x, z = arrays(train), arrays(test)
    xm = x.mean(0); x -= xm; z -= xm
    target = np.eye(3)[y].reshape(len(y), -1)
    ym = target.mean(0)
    values, vectors = eigh(mm(x, x.T), check_finite=False)
    values = np.maximum(values, 0)
    yp = vectors.T @ (target - ym)
    zp = mm(z, x.T) @ vectors
    ps, states = [], []
    for alpha in alphas:
        dual = vectors @ (yp / (values[:, None] + alpha))
        coef = mm(x.T, dual)
        logits = (zp @ (yp / (values[:, None] + alpha)) + ym).reshape(-1, 3, 3)
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits); p /= p.sum(-1, keepdims=True)
        ps.append(p)
        states.append(dict(input_mean=xm, target_mean=ym, coefficient=coef, alpha=alpha))
    return ps, states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='paired_three_class_corrected_10cv')
    parser.add_argument('--reuse-from', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--inner-epochs', type=int, default=30)
    parser.add_argument('--fold-limit', type=int, default=10)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261011, 20261012, 20261013])
    args = parser.parse_args()
    torch.set_num_threads(4); threadpool_limits(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    records, features, clinical = load()
    local = {p: x.mean(1) for p, x in patches(records).items()}
    points = load_points(records)
    scores = records[['color', 'area']].to_numpy(np.float32)
    scores = np.c_[scores, scores.mean(1)]
    groups = records.split_group.astype(str).to_numpy()
    out = BASE / 'outputs' / args.run; out.mkdir(exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Choose a fresh run')
    (out / 'protocol.json').write_text(json.dumps(dict(epochs=args.epochs, inner_epochs=args.inner_epochs, fold_limit=args.fold_limit, seeds=args.seeds, variants=VARIANTS, schemes=SCHEMES, targets=TARGETS, lr=[.0001, .0003], ridge_alphas=ALPHAS, ranking_weight=.2, selection='joint inner macro-F1 for LR; target-wise inner macro-F1 for pipeline', calibration='inner-only biases low/high [-1,-.5,0,.5,1], then macro-F1; linear biases [-.2,-.1,0,.1,.2]', label_rules='fixed 3/6 or train-only tertiles, composite=(color+area)/2', data='paired before+after only, frozen CLIP/LAM', folds='reused exploratory patient-group outer10', protocol_document='THREE_CLASS_PROTOCOL.md'), indent=2))
    metadata = json.loads((out / 'protocol.json').read_text())
    metadata.update(ordinal_initial_gap=3., ordinal_initial_base=-1.5, reuse_unchanged_ce_from=args.reuse_from, decoders=['raw','calibrated','recall_priority'])
    (out / 'protocol.json').write_text(json.dumps(metadata, indent=2))
    outer = pd.read_csv(BASE / 'outputs/neural_real_phase_10cv/splits.csv')
    preds, fold_scores, choices, inner_rows = [], [], [], []

    def add(prob, ids, fold, scheme, variant, family, seed, cuts, biases, recall_biases=None):
        truth = labels(scores[ids], cuts)
        decoding_rules = [('raw', np.zeros((3, 3))), ('calibrated', biases)]
        if recall_biases is not None:
            decoding_rules.append(('recall_priority', recall_biases))
        for decoding, bias in decoding_rules:
            cls = decisions(prob, bias)
            for j, target in enumerate(TARGETS):
                fold_scores.append(dict(fold=fold, scheme=scheme, variant=variant, family=family, seed=seed, decoding=decoding, target=target, **metrics(truth[:, j], cls[:, j])))
                preds.extend(dict(fold=fold, scheme=scheme, variant=variant, family=family, seed=seed, decoding=decoding, target=target, record_key=records.record_key[i], truth=int(t), prediction=int(p), score=float(scores[i, j]), cut_low=float(cuts[j, 0]), cut_high=float(cuts[j, 1]), prob0=float(q[0]), prob1=float(q[1]), prob2=float(q[2])) for i, t, p, q in zip(ids, truth[:, j], cls[:, j], prob[:, j]))

    for fold, rows in outer.groupby('fold'):
        if fold >= args.fold_limit:
            break
        train = np.flatnonzero(records.record_key.isin(rows.loc[rows.role.eq('train'), 'record_key']))
        test = np.flatnonzero(records.record_key.isin(rows.loc[rows.role.eq('test'), 'record_key']))
        assert not set(groups[train]) & set(groups[test])
        ii, jj = next(GroupShuffleSplit(1, test_size=.2, random_state=20261010 + int(fold)).split(train, groups=groups[train]))
        it, iv = train[ii], train[jj]
        assert not set(groups[it]) & set(groups[iv])
        inner_rows.extend(dict(fold=fold, role=role, record_key=records.record_key[i]) for role, ids in [('train', it), ('validation', iv)] for i in ids)
        a, b, state = dataset(features, local, points, clinical, train, test)
        ia, ib, _ = dataset(features, local, points, clinical, it, iv)
        for scheme in SCHEMES:
            cuts, icuts = cutoffs(scores, train, scheme), cutoffs(scores, it, scheme)
            y, iy, ivy = labels(scores[train], cuts), labels(scores[it], icuts), labels(scores[iv], icuts)
            outer_candidates = {}
            linear, _ = ridge_probabilities(ia, iy, ib, ALPHAS)
            linear_options = []
            for alpha, prob in zip(ALPHAS, linear):
                results = [bias_search(prob[:, j], ivy[:, j], (-.2, -.1, 0., .1, .2)) for j in range(3)]
                linear_options.append((np.mean([m['macro_f1'] for _, m in results]), alpha, np.stack([bias for bias, _ in results]), [m for _, m in results], prob))
            selected = max(linear_options, key=lambda x: x[0])
            _, alpha, biases, inner_metrics, ip = selected
            goal_results = [bias_search(ip[:, j], ivy[:, j], (-.2, -.1, 0., .1, .2), objective='two_recall') for j in range(3)]
            goal_biases = np.stack([bias for bias, _ in goal_results]); goal_metrics = [m for _, m in goal_results]
            ps, weights = ridge_probabilities(a, y, b, [alpha]); prob = ps[0]
            torch.save(dict(fold=int(fold), scheme=scheme, variant='ridge_classifier', preprocessing=state, cutoffs=cuts, biases=biases, recall_biases=goal_biases, linear=weights[0], inner_cutoffs=icuts), out / f'fold_{fold:02d}_{scheme}_ridge_classifier.pt')
            add(prob, test, fold, scheme, 'ridge_classifier', 'linear', -1, cuts, biases, goal_biases)
            outer_candidates['ridge_classifier'] = (prob, biases, inner_metrics, goal_biases, goal_metrics)
            choices.append(dict(fold=int(fold), scheme=scheme, variant='ridge_classifier', alpha=alpha, biases=biases.tolist(), recall_biases=goal_biases.tolist(), inner_macro_f1=[m['macro_f1'] for m in inner_metrics], cutoffs=cuts.tolist(), inner_cutoffs=icuts.tolist()))
            for variant in VARIANTS:
                options = []
                for lr in [.0001, .0003]:
                    ip, _ = fit(ia, iy, scores[it], ib, variant, lr, 20261012 + int(fold), args.inner_epochs, device)
                    results = [bias_search(ip[:, j], ivy[:, j]) for j in range(3)]
                    options.append((np.mean([m['macro_f1'] for _, m in results]), lr, np.stack([bias for bias, _ in results]), [m for _, m in results], ip))
                selected = max(options, key=lambda x: x[0])
                _, lr, biases, inner_metrics, ip = selected
                goal_results = [bias_search(ip[:, j], ivy[:, j], objective='two_recall') for j in range(3)]
                goal_biases = np.stack([bias for bias, _ in goal_results]); goal_metrics = [m for _, m in goal_results]
                choices.append(dict(fold=int(fold), scheme=scheme, variant=variant, lr=lr, validation_joint_macro_f1=[o[0] for o in options], inner_macro_f1=[m['macro_f1'] for m in inner_metrics], biases=biases.tolist(), recall_biases=goal_biases.tolist(), cutoffs=cuts.tolist(), inner_cutoffs=icuts.tolist()))
                probabilities_by_seed = []
                for seed in args.seeds:
                    if args.reuse_from and variant in ['global_ce', 'global_balanced_ce']:
                        source = BASE / 'outputs' / args.reuse_from / f'fold_{fold:02d}_{scheme}_{variant}_seed{seed}.pt'
                        saved = torch.load(source, map_location='cpu', weights_only=False)
                        assert saved['lr'] == lr and saved['epochs'] == args.epochs
                        assert np.allclose(saved['cutoffs'], cuts) and np.allclose(saved['biases'], biases) and np.allclose(saved['recall_biases'], goal_biases)
                        model = PairedClassifier(a['clinical'].shape[1], variant).to(device)
                        model.load_state_dict(saved['model']); model.eval()
                        with torch.inference_mode():
                            prob = probabilities(model({k: torch.from_numpy(v).to(device) for k,v in b.items()}), model.ordinal).cpu().numpy()
                        weights = saved['model']
                    else:
                        prob, weights = fit(a, y, scores[train], b, variant, lr, seed + int(fold) * 100, args.epochs, device)
                    torch.save(dict(fold=int(fold), scheme=scheme, variant=variant, seed=seed, lr=lr, epochs=args.epochs, model=weights, clinical_dim=a['clinical'].shape[1], preprocessing=state, cutoffs=cuts, biases=biases, recall_biases=goal_biases, inner_cutoffs=icuts), out / f'fold_{fold:02d}_{scheme}_{variant}_seed{seed}.pt')
                    add(prob, test, fold, scheme, variant, 'neural', seed, cuts, biases, goal_biases)
                    probabilities_by_seed.append(prob)
                ensemble = np.mean(probabilities_by_seed, axis=0)
                add(ensemble, test, fold, scheme, variant, 'neural_ensemble', -1, cuts, biases, goal_biases)
                outer_candidates[variant] = (ensemble, biases, inner_metrics, goal_biases, goal_metrics)
                print('fold', fold + 1, scheme, variant, 'complete', flush=True)
                pd.DataFrame(preds).to_csv(out / 'predictions.csv', index=False)
                pd.DataFrame(fold_scores).to_csv(out / 'fold_metrics.csv', index=False)
                pd.DataFrame(inner_rows).to_csv(out / 'inner_splits.csv', index=False)
                (out / 'choices.json').write_text(json.dumps(choices, indent=2))
            selected_prob, selected_bias, selected_names, selected_goal_bias = [], [], [], []
            for j, target in enumerate(TARGETS):
                winner = max(outer_candidates, key=lambda v: (outer_candidates[v][2][j]['macro_f1'], sorted([outer_candidates[v][2][j][f'recall_{k}'] for k in range(3)])[1]))
                prob, bias, _, goal_bias, _ = outer_candidates[winner]
                selected_prob.append(prob[:, j]); selected_bias.append(bias[j]); selected_names.append(winner)
                selected_goal_bias.append(goal_bias[j])
            add(np.stack(selected_prob, axis=1), test, fold, scheme, 'inner_selected', 'selected_pipeline', -1, cuts, np.stack(selected_bias), np.stack(selected_goal_bias))
            choices.append(dict(fold=int(fold), scheme=scheme, variant='inner_selected', selected_by_target=dict(zip(TARGETS, selected_names))))
            gp, gb, gn = [], [], []
            for j, target in enumerate(TARGETS):
                winner = max(outer_candidates, key=lambda v: (sorted([outer_candidates[v][4][j][f'recall_{k}'] for k in range(3)])[1], outer_candidates[v][4][j]['macro_f1']))
                prob, _, _, bias, _ = outer_candidates[winner]
                gp.append(prob[:, j]); gb.append(bias[j]); gn.append(winner)
            add(np.stack(gp, axis=1), test, fold, scheme, 'inner_selected_recall', 'selected_pipeline', -1, cuts, np.stack(gb), np.stack(gb))
            choices.append(dict(fold=int(fold), scheme=scheme, variant='inner_selected_recall', selected_by_target=dict(zip(TARGETS, gn))))
            pd.DataFrame(preds).to_csv(out / 'predictions.csv', index=False)
            pd.DataFrame(fold_scores).to_csv(out / 'fold_metrics.csv', index=False)
            (out / 'choices.json').write_text(json.dumps(choices, indent=2))
    frame = pd.DataFrame(preds)
    summary = []
    for key, rows in frame.groupby(['scheme', 'variant', 'family', 'seed', 'decoding', 'target']):
        summary.append(dict(zip(['scheme', 'variant', 'family', 'seed', 'decoding', 'target'], key), n=len(rows), **metrics(rows.truth.to_numpy(), rows.prediction.to_numpy())))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    print('complete', flush=True)

if __name__ == '__main__':
    main()
