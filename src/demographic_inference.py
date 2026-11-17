"""Numerical age and recorded III/IV candidates; never infer sex or follow-up."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import expit
class DemographicModel:
    def __init__(self,root):
        root=Path(root);self.meta=json.loads((root/'model.json').read_text(encoding='utf8'));p=root/'candidates.npz'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=self.meta['sha256']:raise ValueError('Candidate checksum mismatch')
        with np.load(p,allow_pickle=False) as z:self.w={k:z[k] for k in z.files}
    def predict(self,features):
        x=np.r_[np.asarray(features['before']).reshape(768),np.asarray(features['aligned_lab']).reshape(2709)[1173:1685]].astype(np.float32)
        x=(x-self.w['mean'])/self.w['scale'];raw=float(x@self.w['age_weight']+self.w['age_bias'])
        age=round(float(np.clip(np.expm1(np.clip(raw,-10,10)),*self.meta['age_range'])),1)
        iv=float(expit(x@self.w['skin_weight']+self.w['skin_bias']))
        return {'version':self.meta['version'],'age':age,'skin_type':'IV' if iv>=.5 else 'III','skin_scores':{'III':1-iv,'IV':iv},'sex_predicted':False,
                'requires_review':True,'skin_label_scope':['III','IV'],'notes':['年龄为照片估计，不是实足年龄。','Fitzpatrick 为既有病历 III/IV 标签的估计，不能替代晒伤/晒黑反应评估。']}
