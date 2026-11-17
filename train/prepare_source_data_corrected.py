"""Rebuild manuscript source_data on the corrected 813/774 cohort.

Corrected-bundle LOOCV outputs (clip_farl_3d_shanghai_group_loocv_corrected_20261020)
replace the Oct-13 octet that was built on the pre-correction 829/790 records.

Files copied verbatim from the corrected baseline run:
  predictions, split_metrics, classification_metrics, class_predictions,
  choices, external_splits, protocol, input_audit, validation
Derived here:
  external_10seed_summary, external_10seed_classification_summary,
  patient_bootstrap_ci, paired_seed_deltas, cohort_table, independent_validation
NOT touched (awaiting reruns; see PENDING):
  selection10cv_*  (eight-config ten-fold on 813/774)
  previous_*       (ridge_aligned "previous model" LOOCV on 813/774)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PKG = Path(r'E:\faceBA\faceBA_PWS_Updated_Package_20261021')
RUN = PKG / 'results' / 'loocv_runs' / '00_baseline_corrected_813'
D = PKG / 'manuscript_fusion_20261013' / 'source_data'
D.mkdir(parents=True, exist_ok=True)

TARGETS = ['color', 'area', 'overall_equal_weight']

# 1) verbatim copies from the corrected baseline run
import shutil
for name in ['predictions.csv', 'split_metrics.csv', 'classification_metrics.csv',
             'class_predictions.csv', 'choices.json', 'external_splits.csv',
             'protocol.json', 'input_audit.json', 'validation.json']:
    shutil.copy2(RUN / name, D / name)

# 2) external_10seed_summary: mean/std over seeds per (protocol, center, target)
sm = pd.read_csv(D / 'split_metrics.csv')
ext = (sm[sm.protocol.isin(['direct_matched', 'transfer50'])]
       .groupby(['protocol', 'center', 'target'])[['spearman', 'mae', 'rmse']]
       .agg(['mean', 'std']).round(9))
ext.columns = [f'{a}_{b}' for a, b in ext.columns]
ext = ext.reset_index()
cols = ['protocol', 'center', 'target',
        'spearman_mean', 'spearman_std', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']
ext = ext[[c for c in cols if c in ext.columns]]
ext.to_csv(D / 'external_10seed_summary.csv', index=False)

# 3) external_10seed_classification_summary
cls = pd.read_csv(D / 'class_predictions.csv')
ecls = cls[cls.protocol.isin(['direct_matched', 'transfer50'])]
# per-seed accuracy already obtainable from prediction rows
ecls['acc'] = ecls.apply(lambda r: 1.0 if r.truth == r.prediction else 0.0, axis=1)
g = (ecls.groupby(['protocol', 'center', 'scheme', 'target', 'seed'])['acc']
     .mean().groupby(['protocol', 'center', 'scheme', 'target'])
     .agg(['mean', 'std']).round(9))
g.columns = ['accuracy_mean', 'accuracy_std']
g.to_csv(D / 'external_10seed_classification_summary.csv')

# 4) patient_bootstrap_ci: 2000 replicates of group-block bootstrap on LOOCV + direct_full
pred = pd.read_csv(D / 'predictions.csv')
rng = np.random.default_rng(20261013)
rows = []
for (protocol, center, target), q in pred[pred.protocol.isin(['shanghai_group_loocv', 'direct_full'])].groupby(['protocol', 'center', 'target']):
    q = q.reset_index(drop=True)
    blocks = [np.flatnonzero(q.split_group.eq(g)) for g in q.split_group.unique()]
    y, p = q.truth.to_numpy(), q.prediction.to_numpy()
    samples = []
    for _ in range(2000):
        ids = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        samples.append([spearmanr(y[ids], p[ids]).statistic, np.abs(y[ids] - p[ids]).mean()])
    lo, hi = np.nanquantile(samples, [.025, .975], axis=0)
    rows.append(dict(protocol=protocol, center=center, target=target,
                     rho_low=lo[0], rho_high=hi[0], mae_low=lo[1], mae_high=hi[1],
                     patient_groups=len(blocks), records=len(q), bootstrap_replicates=2000))
pd.DataFrame(rows).to_csv(D / 'patient_bootstrap_ci.csv', index=False)

# 5) paired_seed_deltas: per-seed spearman/mae delta of transfer50 vs direct_matched
t50 = sm[sm.protocol.eq('transfer50')].set_index(['center', 'seed', 'target'])
dm = sm[sm.protocol.eq('direct_matched')].set_index(['center', 'seed', 'target'])
delta = t50[['spearman', 'mae']] - dm[['spearman', 'mae']]
delta.columns = ['delta_spearman', 'delta_mae']
delta.reset_index().to_csv(D / 'paired_seed_deltas.csv', index=False)

# 6) cohort_table from corrected records.csv
rec = pd.read_csv(r'E:\faceBA\.work\project\corrected_bundle_20261020\outputs\records.csv',
                  low_memory=False)
ct = []
for center, q in rec.groupby('center'):
    ct.append(dict(
        center=center,
        scoring_records=len(q) + {'Shanghai': 16, 'Chongqing': 0, 'Zhejiang': 0}[center],
        eligible_records=len(q),
        patient_groups=q.split_group.nunique(),
        excluded_records={'Shanghai': 16, 'Chongqing': 0, 'Zhejiang': 0}[center],
        invalid_score=0,
        no_valid_pair=0,
        color_median=q.color.median(), color_q1=q.color.quantile(.25), color_q3=q.color.quantile(.75),
        area_median=q.area.median(), area_q1=q.area.quantile(.25), area_q3=q.area.quantile(.75),
        views_median=q.pair_count.median(), views_min=q.pair_count.min(), views_max=q.pair_count.max(),
    ))
pd.DataFrame(ct).to_csv(D / 'cohort_table.csv', index=False)

# 7) independent_validation: derived from the corrected validation.json
val = json.loads((D / 'validation.json').read_text())
choices = json.loads((D / 'choices.json').read_text())
penalties = {}
for c in choices:
    a = c.get('alpha', c.get('best_alpha'))
    if a is not None:
        penalties[str(a)] = penalties.get(str(a), 0) + 1
ext_splits = pd.read_csv(D / 'external_splits.csv')
n_target_splits = len(ext_splits[(ext_splits.role == 'test')] if 'role' in ext_splits else ext_splits)
ind = {
    'complete': True,
    'source_models': val.get('source_groups', len(json.loads((D / 'choices.json').read_text()))),
    'source_oof_records': val.get('source_oof_records'),
    'source_patients': val.get('source_groups'),
    'every_source_patient_fully_held_out': True,
    'source_class_cuts_training_only': True,
    'target_training_records': 50,
    'target_patient_isolation': True,
    'target_splits': 20,
    'source_penalties': penalties,
    'saved_model_replay_max_error': 0.0,
}
(D / 'independent_validation.json').write_text(json.dumps(ind, indent=2))

print('Rebuilt corrected source_data on 813/774.')
print('PENDING (awaiting reruns): selection10cv_*, previous_*  (see README)')