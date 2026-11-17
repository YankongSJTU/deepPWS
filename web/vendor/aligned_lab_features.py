"""After-to-before 68-landmark affine alignment, signed Lab residuals and frozen CLIP.

Only the first valid reconstructed pair per record, as in the dense-3D branch.
No outcome labels are used. a* is a redness proxy, not hemoglobin concentration.
"""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image,ImageOps
import torch
BASE=Path(__import__('os').environ['FACEBA_RUNTIME_ROOT']);OUT=Path(__import__('os').environ['FACEBA_JOB_ROOT'])

def load_landmark_models():
    sys.path.insert(0,str(BASE/'3DDFA_V2'))
    if not hasattr(np,'long'):np.long=np.int64
    from FaceBoxes import FaceBoxes
    from TDDFA import TDDFA
    boxes=FaceBoxes();model=TDDFA(arch='mobilenet',widen_factor=1.,checkpoint_fp=str(BASE/'3DDFA_V2/weights/mb1_120x120.pth'),bfm_fp=str(BASE/'3DDFA_V2/configs/bfm_noneck_v3.pkl'),size=120,num_params=62,gpu_mode=False)
    torch.set_grad_enabled(True)
    return boxes,model
def read(path,size=512):
    with Image.open(path) as im:im=ImageOps.exif_transpose(im).convert('RGB');w,h=im.size;im=im.resize((int(w*size/max(w,h)),int(h*size/max(w,h))),Image.Resampling.BICUBIC);return np.asarray(im)
def landmarks(rgb,boxes,model):
    bgr=np.ascontiguousarray(rgb[:,:,::-1])
    with torch.inference_mode():faces=boxes(bgr)
    if len(faces)==0:raise ValueError('no_face')
    faces=np.asarray(faces);score=(faces[:,2]-faces[:,0])*(faces[:,3]-faces[:,1])*faces[:,4];face=faces[int(score.argmax())]
    with torch.inference_mode():param,roi=model(bgr,[face]);xyz=model.recon_vers(param,roi,dense_flag=False)[0].T
    assert xyz.shape==(68,3) and np.isfinite(xyz).all()
    return xyz[:,:2],float(face[4])
def moments(values):
    return np.r_[values.mean(0),values.std(0),np.quantile(values,[.1,.5,.9],axis=0).ravel()]
def source(pair,phase):
    relative=pair[phase];expected=pair[phase+'_hash']
    for path in [BASE.parent/relative,BASE.parent/'data'/relative]:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return path
    raise FileNotFoundError('no SHA-matched source for '+relative)
def one(pair,boxes,model):
    before=read(source(pair,'before'));after=read(source(pair,'after'));b,bc=landmarks(before,boxes,model);a,ac=landmarks(after,boxes,model)
    # Use eyes/nose/brows/mouth, excluding the more pose-sensitive jaw contour.
    selected=np.arange(17,68);matrix,inliers=cv2.estimateAffinePartial2D(a[selected],b[selected],method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=2000,confidence=.99,refineIters=10)
    if matrix is None:raise ValueError('alignment_failed')
    h,w=before.shape[:2];warped=cv2.warpAffine(after,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=cv2.warpAffine(np.ones(after.shape[:2],np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)>0
    face=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(face,cv2.convexHull(b.round().astype(np.int32)),1);mask=face.astype(bool)&valid
    if mask.sum()<300:raise ValueError('insufficient_overlap')
    bl=cv2.cvtColor(before.astype(np.float32)/255,cv2.COLOR_RGB2LAB);al=cv2.cvtColor(warped.astype(np.float32)/255,cv2.COLOR_RGB2LAB);delta=al-bl;de=np.linalg.norm(delta,axis=2)
    data=np.c_[delta[mask],de[mask]];s=moments(data)
    # Fixed +/-40 a* units; neutral grey outside overlap. Positive means redder after.
    da=np.where(mask,delta[:,:,1],0);g=np.clip(da/40,-1,1)
    pseudorgb=np.stack([.5+.5*g,.5-.25*np.abs(g),.5-.5*g],axis=2)
    image=Image.fromarray(np.clip(pseudorgb*255,0,255).astype(np.uint8)).resize((224,224),Image.Resampling.BICUBIC)
    mapped=cv2.transform(a[selected][None].astype(np.float32),matrix)[0];residual=float(np.sqrt(np.mean(np.sum((mapped-b[selected])**2,axis=1)))/max(np.linalg.norm(b[36:42].mean(0)-b[42:48].mean(0)),1e-8))
    meta=dict(before_face_confidence=bc,after_face_confidence=ac,affine_inlier_fraction=float(inliers.mean()),landmark_residual_interocular_units=residual,overlap_fraction=float(mask.mean()),mean_delta_L=float(data[:,0].mean()),mean_delta_a=float(data[:,1].mean()),mean_delta_b=float(data[:,2].mean()),mean_delta_E76=float(data[:,3].mean()))
    return image,s,matrix,meta
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--centers',nargs='+',default=['Shanghai']);args=p.parse_args()
    torch.set_num_threads(3);records=pd.read_csv(BASE/'outputs/records.csv');records=records[records.status.eq('eligible')&records.center.isin(args.centers)]
    if args.limit:records=records.head(args.limit)
    boxes,landmark_model=load_landmark_models()
    from local_real_features import ClipVision
    clip=ClipVision('cuda' if torch.cuda.is_available() else 'cpu');audits=[]
    for i,r in enumerate(records.itertuples(),1):
        key=r.record_key.replace(':','_');npz=OUT/(key+'.npz');meta=OUT/(key+'.json');png=OUT/(key+'_delta_a.png');pair=json.loads(r.pairs)[0]
        if not npz.exists():
            try:
                im,stats,matrix,info=one(pair,boxes,landmark_model);im.save(png)
                with torch.inference_mode():embedding,_=clip([png],spatial=True)
                np.savez_compressed(npz,embedding=embedding[0],lab_stats=stats,affine=matrix)
                info.update(record_key=r.record_key,status='ok',before_hash=pair['before_hash'],after_hash=pair['after_hash']);meta.write_text(json.dumps(info,indent=2))
            except Exception as e:meta.write_text(json.dumps(dict(record_key=r.record_key,status='failed',reason=str(e)),indent=2))
        audits.append(json.loads(meta.read_text()))
        if i%25==0:print('Aligned Lab records',i,'/',len(records),'successful',sum(x['status']=='ok' for x in audits),flush=True)
    all_audits=[json.loads(path.read_text()) for center in ['Shanghai','Chongqing','Zhejiang'] for path in OUT.glob(center+'_*.json')]
    pd.DataFrame(all_audits).to_csv(OUT/'audit.csv',index=False)
    fields=['delta_L','delta_a','delta_b','delta_E76'];names=[kind+'_'+field for kind in ['mean','std','q10','q50','q90'] for field in fields]
    (OUT/'protocol.json').write_text(json.dumps(dict(alignment='3DDFA_V2 68 projected BFM landmarks; proper similarity affine after-to-before RANSAC using landmark indices17..67; no outcome-driven fitting',coordinates='EXIF-aware RGB, preserve aspect ratio at512 maximum side; warp after image into before image grid',mask='before-landmark convex hull intersect warped valid pixels; not lesion mask and may omit forehead beyond eyebrows',color='OpenCV float sRGB to CIELAB, uncalibrated photographs; delta=after_warped-before; DeltaE76 Euclidean Lab',redness='a* red-green appearance proxy; no quantitative hemoglobin claim',residual_encoding='signed delta a fixed40 units; red positive, blue negative, neutral-grey invalid; resize224; frozen unnormalised CLIP projection',numeric_features=names,failures='must retain all records through train-only imputation plus availability indicator; no outcome-based exclusions',pair_policy='first valid record pair matching LAM reconstruction',physiology='lighting and acquisition confounding unresolved'),indent=2))
    print('Aligned Lab complete',len(audits),'successful',sum(x['status']=='ok' for x in audits),flush=True)
if __name__=='__main__':main()
