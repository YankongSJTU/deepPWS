"""Aligned, topology-based regional LAM geometry and separate populated/inpaint UV.

Model-derived shape proxies, not metric skin shape or physiological measurements.
"""
from pathlib import Path
import json,pickle,argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image
from inspect_flame_template import template

BASE=Path(__import__('os').environ['FACEBA_RUNTIME_ROOT']);OUT=Path(__import__('os').environ['FACEBA_JOB_ROOT'])

REGIONS=['face','forehead','left_cheek_proxy','right_cheek_proxy','nose','perioral_proxy']
def topology():
    v,f=template();edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    assert len(v)+len(edges)==20018
    points=np.vstack([v,v[edges].mean(1)])
    edge_index={tuple(e):i+len(v) for i,e in enumerate(edges)}
    ab=np.array([edge_index[tuple(sorted(p))] for p in f[:,[0,1]]]);bc=np.array([edge_index[tuple(sorted(p))] for p in f[:,[1,2]]]);ac=np.array([edge_index[tuple(sorted(p))] for p in f[:,[0,2]]])
    ff=np.vstack([np.c_[f[:,0],ab,ac],np.c_[f[:,1],bc,ab],np.c_[f[:,2],ac,bc],np.c_[ab,bc,ac]])
    ee=np.unique(np.sort(np.vstack([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1),axis=0)
    with (BASE/'model_zoo/human_parametric_models/flame_vhap/FLAME_masks.pkl').open('rb') as file:m=pickle.load(file,encoding='latin1')
    masks={}
    for name in ['face','forehead','nose','lips']:
        a=np.zeros(len(v),bool);a[np.asarray(m[name],int)]=True;masks[name]=np.r_[a,a[edges].all(1)]
    x,y,z=points.T;face=masks['face']
    masks['left_cheek_proxy']=face&(x<-.022)&(y>-.07)&(y<.018)&~masks['nose']&~masks['lips']
    masks['right_cheek_proxy']=face&(x>.022)&(y>-.07)&(y<.018)&~masks['nose']&~masks['lips']
    masks['perioral_proxy']=face&(abs(x)<.040)&(y>-.072)&(y<-.023)
    # Repeated endpoints yield a fixed topological neighbourhood, not kNN each image.
    src=np.r_[ee[:,0],ee[:,1]];dst=np.r_[ee[:,1],ee[:,0]];degree=np.bincount(src,minlength=len(points))
    return points,ff,ee,masks,src,dst,degree

T,F,E,M,SRC,DST,DEG=topology()
def shape_fields(v):
    cross=np.cross(v[F[:,1]]-v[F[:,0]],v[F[:,2]]-v[F[:,0]])
    normal=np.zeros_like(v)
    for j in range(3):
        for k in range(3):normal[:,k]+=np.bincount(F[:,j],weights=cross[:,k],minlength=len(v))
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    centroid=np.column_stack([np.bincount(SRC,weights=v[DST,k],minlength=len(v)) for k in range(3)])/np.maximum(DEG[:,None],1)
    laplacian=v-centroid
    curvature=np.einsum('ij,ij->i',laplacian,normal)
    return normal,curvature

def align(before,after):
    ids=M['face'];b=before[ids];a=after[ids];bm=b.mean(0);am=a.mean(0)
    u,_,vt=np.linalg.svd((a-am).T@(b-bm));r=u@vt
    if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
    # Rigid alignment preserves scale changes; no similarity scale fitting.
    return (after-am)@r+bm

def uv_pixel(v,size=256):
    radius=np.linalg.norm(v,axis=1)+1e-8;u=np.arctan2(v[:,1],v[:,0])/(2*np.pi)+.5;w=.5-np.arcsin(np.clip(v[:,2]/radius,-1,1))/np.pi
    return np.clip(np.round(u*(size-1)).astype(int),0,size-1),np.clip(np.round(w*(size-1)).astype(int),0,size-1)

def populated(v,size=256):
    x,y=uv_pixel(v,size);mask=np.zeros((size,size),bool);mask[y,x]=True;return mask

def region_uv_maps(size=256):
    x,y=uv_pixel(T,size);count=np.zeros((size,size),int);count[y,x]=1
    # Fixed label-free template coordinate atlas: nearest template UV point per texel.
    xx,yy=np.meshgrid(np.arange(size),np.arange(size));lookup=cKDTree(np.c_[x,y]).query(np.c_[xx.ravel(),yy.ravel()])[1].reshape(size,size)
    return {r:M[r][lookup] for r in REGIONS}
UVM=region_uv_maps()
lm=np.load(BASE/'model_zoo/human_parametric_models/flame_vhap/landmark_embedding_with_eyes.npy',allow_pickle=True).item()
_,original_faces=template();LMF=original_faces[np.asarray(lm['full_lmk_faces_idx']).ravel()];LMB=np.asarray(lm['full_lmk_bary_coords']).reshape(-1,3)
LMT=(T[LMF]*LMB[:,:,None]).sum(1);LMREG={r:M[r][cKDTree(T).query(LMT)[1]] for r in REGIONS}

def moment(out,key,values):
    out[key+'_mean']=float(np.mean(values)) if len(values) else np.nan
    out[key+'_var']=float(np.var(values)) if len(values) else np.nan

def extract(record_key):
    key=record_key.replace(':','_');folder=BASE/'outputs/formal3d_full'
    z=[]
    for phase in ['before','after']:
        with np.load(folder/f'{key}_{phase}.npz') as file:v=file['vertices'].astype(float);assert v.shape==T.shape and np.isfinite(v).all();z.append(v)
    b,raw=z;a=align(b,raw);nb,cb=shape_fields(b);na,ca=shape_fields(a);d=a-b;length=np.linalg.norm(d,axis=1)
    angle=np.arccos(np.clip(np.einsum('ij,ij->i',nb,na),-1,1));strain=np.log(np.maximum(np.linalg.norm(a[E[:,1]]-a[E[:,0]],axis=1),1e-9)/np.maximum(np.linalg.norm(b[E[:,1]]-b[E[:,0]],axis=1),1e-9))
    ld=np.linalg.norm((a[LMF]*LMB[:,:,None]).sum(1)-(b[LMF]*LMB[:,:,None]).sum(1),axis=1)
    g={};uv={};masks=[populated(v) for v in z];textures=[np.asarray(Image.open(folder/'uv'/f'{key}_{p}.png').convert('RGB'),dtype=float)/255 for p in ['before','after']]
    assert all(t.shape==(256,256,3) for t in textures)
    for region in REGIONS:
        mask=M[region];edge=mask[E].all(1)
        for name,values in [('displacement',length),('normal_angle_rad',angle),('curvature_before',cb),('curvature_after',ca),('curvature_change',ca-cb),('curvature_change_abs',np.abs(ca-cb)),('normal_displacement',np.einsum('ij,ij->i',d,nb))]:moment(g,region+'_'+name,values[mask])
        for j,axis in enumerate(['x','y','z']):moment(g,region+'_displacement_'+axis,d[mask,j])
        moment(g,region+'_edge_log_strain',strain[edge]);moment(g,region+'_landmark_displacement',ld[LMREG[region]])
        regionmask=UVM[region]
        for pi,phase in enumerate(['before','after']):
            uv[region+'_'+phase+'_populated_fraction']=float(masks[pi][regionmask].mean()) if regionmask.any() else np.nan
            for kind,subset in [('populated',regionmask&masks[pi]),('inpainted',regionmask&~masks[pi])]:
                for c,channel in enumerate('rgb'):moment(uv,f'{region}_{phase}_{kind}_{channel}',textures[pi][:,:,c][subset])
        for kind,subset in [('both_populated',regionmask&masks[0]&masks[1]),('both_inpainted',regionmask&~masks[0]&~masks[1])]:
            uv[f'{region}_change_{kind}_fraction']=float(subset.sum()/max(regionmask.sum(),1))
            for c,channel in enumerate('rgb'):moment(uv,f'{region}_change_{kind}_{channel}',(textures[1]-textures[0])[:,:,c][subset])
    return g,uv

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    records=pd.read_csv(BASE/'outputs/records.csv');records=records[records.status.eq('eligible')]
    if args.limit:records=records.head(args.limit)
    rows=[]
    for i,r in enumerate(records.itertuples(),1):
        dest=OUT/(r.record_key.replace(':','_')+'.npz')
        if not dest.exists():
            g,u=extract(r.record_key);np.savez_compressed(dest,geometry=np.array(list(g.values())),uv=np.array(list(u.values())))
            (OUT/'feature_names.json').write_text(json.dumps(dict(geometry=list(g),uv=list(u)),indent=2))
        with np.load(dest) as z:rows.append(dict(record_key=r.record_key,geometry_columns=z['geometry'].size,uv_columns=z['uv'].size,geometry_nonfinite=int((~np.isfinite(z['geometry'])).sum()),uv_nonfinite=int((~np.isfinite(z['uv'])).sum())))
        if i%50==0:print('Dense regional records',i,'/',len(records),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'audit.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps(dict(topology='original5023 FLAME vertices plus14995 lexicographic edge midpoints; 20018 LAM indices',regions={r:int(M[r].sum()) for r in REGIONS},cheeks_and_perioral='fixed anatomical coordinate proxies; forehead/nose from FLAME masks',alignment='rigid proper rotation plus translation; no scale fit; align after to before using all face vertices',curvature='signed uniform topological Laplacian projected onto area-weighted normal; proxy, not differential metric curvature',strain='log edge length ratio after/before',landmarks='70 barycentric model-derived FLAME anchors; no observed photograph landmark claim',uv='reconstructed spherical atlas; Gaussian-populated pixels versus Telea-inpainted pixels; no camera visibility masks',missing='empty regional anchor/UV subsets are NaN and must be imputed only inside training partitions',inputs='one reconstructed pair per record, matching formal3d_batch.py first valid pair',limitations='LAM geometry/SH-derived texture proxies; not physiological or metric 3D measurements'),indent=2))
    print('Dense features complete',len(rows),flush=True)
if __name__=='__main__':main()
