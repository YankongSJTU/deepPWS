import json, os
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image
BASE=Path(__import__('os').environ['FACEBA_RUNTIME_ROOT'])
def bake(zpath,outpath,size=256):
 z=np.load(zpath); xyz=z['vertices'].astype(np.float32); rgb=np.clip(z['colors'].astype(np.float32),0,1); r=np.linalg.norm(xyz,axis=1)+1e-8; u=np.arctan2(xyz[:,1],xyz[:,0])/(2*np.pi)+.5; v=.5-np.arcsin(np.clip(xyz[:,2]/r,-1,1))/np.pi; px=np.clip((u*(size-1)).round().astype(int),0,size-1); py=np.clip((v*(size-1)).round().astype(int),0,size-1); a=np.zeros((size,size,3),np.float32); w=np.zeros((size,size),np.float32)
 for x,y,c in zip(px,py,rgb): a[y,x]+=c; w[y,x]+=1
 m=w>0; a[m]/=w[m,None]
 if m.any(): a=np.where(m[...,None],a,cv2.inpaint((a*255).astype(np.uint8),255-m.astype(np.uint8)*255,3,cv2.INPAINT_TELEA).astype(np.float32)/255.)
 Image.fromarray(np.clip(a*255,0,255).astype(np.uint8)).save(outpath); return float(m.mean())
def main():
 out=BASE/'outputs/formal3d_full'; uv=out/'uv'; uv.mkdir(exist_ok=True); man=pd.read_csv(out/'manifest_complete.csv' if (out/'manifest_complete.csv').exists() else out/'manifest.csv'); rows=[]
 for i,r in enumerate(man.itertuples(),1):
  src=Path(r.path); dest=uv/(src.stem+'.png')
  try:
   if not src.exists(): raise FileNotFoundError(src)
   cov=bake(src,dest); rows.append({'record_key':r.record_key,'center':r.center,'phase':r.phase,'uv_path':str(dest),'coverage':cov,'completion_fraction':1-cov,'status':'ok'})
  except Exception as e: rows.append({'record_key':r.record_key,'center':r.center,'phase':r.phase,'uv_path':str(dest),'coverage':0,'completion_fraction':1,'status':type(e).__name__})
  if i%100==0: pd.DataFrame(rows).to_csv(out/'uv_manifest.csv',index=False); print(f'uv {i}/{len(man)}',flush=True)
 pd.DataFrame(rows).to_csv(out/'uv_manifest.csv',index=False); print('done',len(rows))
if __name__=='__main__': main()
