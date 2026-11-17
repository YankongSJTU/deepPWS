"""Portable frozen clinical heads: numeric NPZ parameters, no pickle or downloads."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import erf,expit

def linear(x,w,prefix):return (x@w[prefix+'.weight'].T+w[prefix+'.bias']).astype(np.float32)
def norm(x,w,prefix):
    mean=x.mean(-1,keepdims=True);var=((x-mean)**2).mean(-1,keepdims=True)
    return ((x-mean)/np.sqrt(var+np.float32(1e-5))*w[prefix+'.weight']+w[prefix+'.bias']).astype(np.float32)
def gelu(x):return (np.float32(.5)*x*(np.float32(1)+erf(x/np.float32(np.sqrt(2))))).astype(np.float32)
def branch(x,w,prefix):return linear(gelu(norm(linear(x,w,prefix+'.0'),w,prefix+'.1')),w,prefix+'.4')
def softmax(x):
    y=np.exp(x-x.max(-1,keepdims=True));return y/y.sum(-1,keepdims=True)

class ClinicalModel:
    def __init__(self,root):
        self.root=Path(root);self.meta=json.loads((self.root/'model.json').read_text(encoding='utf8'))
        self.weights=[]
        for item in self.meta['members']:
            path=self.root/item['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Clinical parameter checksum mismatch')
            with np.load(path,allow_pickle=False) as z:self.weights.append({k:z[k] for k in z.files})

    def probabilities(self,before_global,before_local,profile='source'):
        if profile not in self.meta['profiles']:raise ValueError('Unknown clinical profile')
        b=np.asarray(before_global,np.float32).reshape(-1,1280);local=np.asarray(before_local,np.float32).reshape(len(b),12288)
        if not np.isfinite(b).all() or not np.isfinite(local).all():raise ValueError('Nonfinite clinical features')
        values=[]
        for w in self.weights:
            gb=(b-w['global_mean'])/w['global_std']
            pc=local@w['pca_components'].T;pc-=w['pca_mean'][None]@w['pca_components'].T
            x=np.c_[gb,(pc-w['pc_mean'])/w['pc_std']].astype(np.float32)
            shared=gelu(norm(linear(x,w,'shared.0'),w,'shared.1'))
            colour='cq_colour' if profile=='chongqing' else 'colour'
            nerve='zj_nerve' if profile=='zhejiang' else 'nerve'
            values.append({'colour':softmax(branch(x,w,colour)),
                           'side':softmax(linear(shared,w,'side')),
                           'nerve':expit(linear(shared,w,nerve)),
                           'regions':expit(branch(x,w,'regions'))})
        return {k:np.mean([v[k] for v in values],axis=0) for k in values[0]}

    def predict(self,features,profile='source'):
        if profile=='disabled':return {'status':'disabled','version':self.meta['version']}
        if 'clinical_before_local' not in features:
            return {'status':'unavailable','version':self.meta['version'],'message':'此记录由旧版生成，重新分析照片后可获得临床候选。'}
        before=np.r_[np.asarray(features['before']).reshape(768),np.asarray(features['aligned_lab']).reshape(2709)[1173:1685]].astype(np.float32)
        p={k:v[0] for k,v in self.probabilities(before,features['clinical_before_local'],profile).items()}
        result={'status':'ready','version':self.meta['version'],'profile_id':profile,'profile_name':self.meta['profiles'][profile]['name'],
                'profile_description':self.meta['profiles'][profile]['description'],'requires_confirmation':True,
                'probabilities_calibrated':False,'inputs':'treatment-before images only','confirmed_fields':[],
                'notes':['候选结果需核对后使用；模型分数不代表临床准确率。','区域为粗分区，不自动改写病灶部位。','年龄与病历 III/IV 标签由独立估计模型处理；性别只接受医生手填。']}
        for k in ['colour','side','nerve','regions']:
            names=self.meta['labels'][k];scores=[{'label':name,'score':float(v)} for name,v in zip(names,p[k])]
            selected=[names[int(np.argmax(p[k]))]] if k in ['colour','side'] else [name for name,v in zip(names,p[k]) if v>=.5]
            result[k]={'selected':selected,'scores':scores}
        result['suggested_fields']={'phenotype':result['colour']['selected'][0]}
        if result['nerve']['selected']:result['suggested_fields']['nerve_zone']='+'.join(result['nerve']['selected'])
        return result
