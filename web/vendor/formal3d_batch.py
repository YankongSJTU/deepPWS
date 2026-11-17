"""Full-cohort LAM canonical Gaussian reconstruction (one image pair per record)."""
import os,json,time,csv
from pathlib import Path
import torch,numpy as np,pandas as pd
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file
from lam.models.modeling_lam import ModelLAM

for _name,_typ in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if not hasattr(np,_name): setattr(np,_name,_typ)
BASE=Path(__import__('os').environ['FACEBA_RUNTIME_ROOT'])
def load_model():
    cfg=OmegaConf.load(BASE/'configs/inference/lam-20k-8gpu.yaml'); cfg.model.human_model_path=str(BASE/'model_zoo/human_parametric_models')
    m=ModelLAM(**cfg.model).cuda().eval(); ck=load_file(str(BASE/'model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors'),device='cpu'); sd=m.state_dict()
    for k,v in ck.items():
        if k in sd and sd[k].shape==v.shape: sd[k].copy_(v)
    return m
def one(m,path):
    im=np.asarray(Image.open(path).convert('RGB').resize((518,518)),np.float32)/255.; image=torch.from_numpy(im).permute(2,0,1).unsqueeze(0).cuda(); flame={'betas':torch.zeros((1,300),device='cuda')}
    with torch.inference_mode():
        feats=m.forward_encode_image(image); qp,_=m.renderer.get_query_points(flame,device=torch.device('cuda')); latent=m.forward_transformer(feats,camera_embeddings=None,query_points=qp,query_feats=None); gs,qp,_,_=m.renderer.forward_gs(latent,qp,flame,additional_features={}); g=gs[0]; v=g.xyz.detach().cpu().numpy(); c=g.shs.detach().cpu().numpy(); c=c[:,0,:] if c.ndim==3 else c[:,:3]; c=1/(1+np.exp(-c))
    return v,c
def main():
    root=Path(os.environ.get('DATA_ROOT','/export/home/kongyan/project/faceBA/data')); out=BASE/'outputs/formal3d_full'; out.mkdir(parents=True,exist_ok=True); rec=pd.read_csv(BASE/'outputs/records.csv'); rec=rec[rec.status=='eligible'].copy(); lim=int(os.environ.get('FORMAL_MAX_RECORDS','0') or 0); rec=rec.head(lim) if lim else rec
    m=load_model(); rows=[]; done=set()
    for j,r in enumerate(rec.itertuples(),1):
        pairs=json.loads(r.pairs); pair=pairs[0]; key=str(r.record_key).replace(':','_');
        for phase in ('before','after'):
            rel=pair[phase]; dest=out/f'{key}_{phase}.npz';
            if dest.exists(): continue
            try:
                v,c=one(m,root/rel); np.savez_compressed(dest,vertices=v,colors=c); rows.append({'record_key':r.record_key,'center':r.center,'phase':phase,'path':str(dest),'n_points':len(v),'status':'ok'})
            except Exception as e: rows.append({'record_key':r.record_key,'center':r.center,'phase':phase,'path':str(dest),'n_points':0,'status':f'error:{type(e).__name__}'})
        if j%10==0: pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False); print(f'processed {j}/{len(rec)}',flush=True)
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False); print(f'done {len(rows)} images')
if __name__=='__main__': main()
