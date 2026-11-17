"""Selected CLIP/FaRL global/dense3D fusion; fold-local preprocessing."""
import numpy as np
import ridge_aligned_transfer as aligned
from regional_residual_ridge_smoke import extras,extra_partition
from farl_ridge import load_farl
from project_paths import PROJECT_ROOT

def load_multicenter():
    records,features,clinical,audit=aligned.original_load()
    face=load_farl(records,PROJECT_ROOT/'outputs/farl_features_ep16')
    before,after=face['before_global'],face['after_global']
    features['aligned_lab']=np.c_[extras(records,'aligned_dense_geometry_uv'),before,after,after-before]
    assert features['aligned_lab'].shape==(len(records),2709)
    audit.append(dict(model='clip_farl_global_3d',additional_dimensions=2709,aligned_lab=789,dense_geometry=144,uv=240,farl_global=1536,local_farl=False))
    return records,features,clinical,audit

def configure():
    aligned.partition=partition
    aligned.transform=transform
    aligned.configure()
    aligned.transfer.load_multicenter=load_multicenter

predict_saved_model=aligned.predict_saved_model

def partition(features,clinical,train,test):
    a,b,state=aligned.original_partition(features,clinical,train,test)
    raw=features['aligned_lab']
    aa,bb,sa=extra_partition(raw[:,:1173],train,test)
    fa,fb,sf=extra_partition(raw[:,1173:].astype(np.float32),train,test)
    a['aligned_lab']=np.c_[aa,fa];b['aligned_lab']=np.c_[bb,fb]
    state['fusion_extra_scalers']=[sa,sf]
    return a,b,state

def transform(features,clinical,state):
    data=aligned.original_transform(features,clinical,state)
    raw=features['aligned_lab'];blocks=[]
    for block,s in zip([raw[:,:1173],raw[:,1173:].astype(np.float32)],state['fusion_extra_scalers']):
        block=np.where(np.isfinite(block),block,s['median'])
        blocks.append((block-s['mean'])/s['std'])
    data['aligned_lab']=np.c_[tuple(blocks)]
    return data
