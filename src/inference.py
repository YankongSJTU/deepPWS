"""Exact NumPy serving path: immutable training preprocessing and fitted Ridge parameters."""
from pathlib import Path
import json, hashlib
import numpy as np

class FusionModel:
    def __init__(self, root):
        self.root=Path(root)
        self.meta=json.loads((self.root/'model.json').read_text(encoding='utf8'))
        path=self.root/'fusion_source.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.meta['parameter_sha256']:
            raise ValueError('Model parameter checksum mismatch')
        with np.load(path,allow_pickle=False) as z:self.w={k:z[k] for k in z.files}

    def predict(self,features,clinical):
        w=self.w
        before=np.asarray(features['before'],np.float32).reshape(1,768)
        after=np.asarray(features['after'],np.float32).reshape(1,768)
        before=((before-w['appearance_mean'])/w['appearance_std']).astype(np.float32)
        after=((after-w['appearance_mean'])/w['appearance_std']).astype(np.float32)
        raw=[];unknown=[]
        for field in self.meta['clinical_fields']:
            value=str(clinical.get(field['key']) or 'missing')
            raw.extend(float(value==c) for c in field['categories'])
            if value not in field['categories']:unknown.append(field['key'])
        age=clinical.get('age');age=self.meta['age_median'] if age is None else float(age)
        raw.append(age)
        c=((np.array(raw,np.float32)[None]-w['clinical_mean'])/w['clinical_std']).astype(np.float32)
        extra=np.asarray(features['aligned_lab'],dtype=np.float64).reshape(1,2709)
        parts=[]
        for i,x in enumerate([extra[:,:1173],extra[:,1173:].astype(np.float32)]):
            x=np.where(np.isfinite(x),x,w[f'extra{i}_median'])
            parts.append((x-w[f'extra{i}_mean'])/w[f'extra{i}_std'])
        matrix=np.concatenate([before,after,after-before,c]+parts,axis=1).astype(np.float64)
        if matrix.shape!=(1,self.meta['input_features']) or not np.isfinite(matrix).all():
            raise ValueError('Invalid inference feature matrix')
        values=(matrix@w['coefficient']+w['intercept']).clip(0,10)[0]
        values=np.r_[values,values.mean()]
        return {'scores':dict(zip(['colour','area','overall'],map(float,values))),
                'categories':dict(zip(['colour','area','overall'],map(int,np.searchsorted([3,6],values,side='left')))),
                'unseen_clinical_fields':unknown,'age_imputed':clinical.get('age') is None,
                'model_id':self.meta['id'],'model_version':self.meta['version'],
                'model_sha256':self.meta['model_sha256'],'parameter_sha256':self.meta['parameter_sha256']}
