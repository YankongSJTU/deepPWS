"""Export the fitted source model without training identifiers or pickle loading at serving time."""
from pathlib import Path
import json, pickle, hashlib, shutil, argparse
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent

def prepare():
    source=PROJECT/'results/loocv_runs/00_baseline_corrected_813/shanghai_source.pkl'
    state=pickle.loads(source.read_bytes())
    prep=state['preprocessing']; arrays={
        'coefficient':state['linear']['coefficient'],'intercept':state['linear']['intercept'],
        'appearance_mean':prep['shared_appearance_scaler']['mean'],
        'appearance_std':prep['shared_appearance_scaler']['std'],
        'clinical_mean':prep['scalers']['clinical']['mean'],
        'clinical_std':prep['scalers']['clinical']['std']}
    for i,scaler in enumerate(prep['fusion_extra_scalers']):
        for k in ['median','mean','std']:arrays[f'extra{i}_{k}']=scaler[k]
    modeldir=ROOT/'models'; modeldir.mkdir(exist_ok=True)
    np.savez_compressed(modeldir/'fusion_source.npz',**arrays)
    fields=['skin_type','lesion_site','nerve_zone','phenotype']
    meta={'id':'faceba-fusion-corrected-813','name':'CLIP · FaRL · 3D Fusion',
          'version':'2026.10.20','source_records':813,'source_groups':774,
          'model_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'parameter_sha256':hashlib.sha256((modeldir/'fusion_source.npz').read_bytes()).hexdigest(),
          'age_median':float(prep['age_median']),
          'clinical_fields':[{ 'key':k,'training_column':c,'categories':v}
                            for k,c,v in zip(fields,prep['clinical_columns'],prep['clinical_categories'])],
          'input_features':int(arrays['coefficient'].shape[0]),
          'fixed_cuts':[3,6],'scale':[0,10],
          'source_metrics':{'colour_spearman':.744866,'area_spearman':.739905,'overall_spearman':.757265},
          'report_only_fields':['sex','treatment_count','complications','treatment_method','notes','before_date','after_date','case_code']}
    (modeldir/'model.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print('Exported fitted coefficients and frozen preprocessing:',meta['input_features'],'inputs')

if __name__=='__main__':prepare()
