"""Display-only landmark registration; never changes model input images or scores."""
from pathlib import Path
import json,math
import numpy as np
import cv2
from PIL import Image,ImageOps
VERSION='2026.10.26-align1'
def write_meta(folder,data):
    dest=Path(folder)/'comparison';dest.mkdir(exist_ok=True)
    tmp=dest/'metadata.tmp';tmp.write_text(json.dumps({'version':VERSION,**data},ensure_ascii=False),encoding='utf8');tmp.replace(dest/'metadata.json')
def similarity(a,b):
    ae=np.array([a[36:42].mean(0),a[42:48].mean(0)]);be=np.array([b[36:42].mean(0),b[42:48].mean(0)])
    av=ae[1]-ae[0];bv=be[1]-be[0];den=float(av@av)
    if den<16 or np.linalg.norm(bv)<4:raise ValueError('eyes_too_small')
    real=float(av@bv)/den;imag=float(av[0]*bv[1]-av[1]*bv[0])/den
    linear=np.array([[real,-imag],[imag,real]],np.float64);translation=be[0]-linear@ae[0]
    return np.c_[linear,translation]
def generate(folder,lab,boxes,model):
    folder=Path(folder);req=json.loads((folder/'request.json').read_text(encoding='utf8'));rgb={};landmarks={};confidence={}
    for phase in ['before','after']:
        path=folder/req['images'][phase][0];small=lab.read(path);points,conf=lab.landmarks(small,boxes,model)
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1536,1536),Image.Resampling.LANCZOS);rgb[phase]=np.asarray(im)
        h,w=rgb[phase].shape[:2];landmarks[phase]=points*np.array([w/small.shape[1],h/small.shape[0]])
        confidence[phase]=conf
    a,b=landmarks['after'],landmarks['before'];matrix=similarity(a,b);mapped=cv2.transform(a[None].astype(np.float64),matrix)[0]
    ocular=float(np.linalg.norm(b[36:42].mean(0)-b[42:48].mean(0)))
    nose=float(np.sqrt(np.mean(np.sum((mapped[27:36]-b[27:36])**2,axis=1))))
    stable=np.r_[27:48];rms=float(np.sqrt(np.mean(np.sum((mapped[stable]-b[stable])**2,axis=1))))
    scale=float(np.linalg.norm(matrix[:,0]));rotation=float(math.degrees(math.atan2(matrix[1,0],matrix[0,0])))
    h,w=rgb['before'].shape[:2];valid=cv2.warpAffine(np.ones(rgb['after'].shape[:2],np.uint8),matrix,(w,h),flags=cv2.INTER_NEAREST)
    face=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(face,cv2.convexHull(b.round().astype(np.int32)),1)
    coverage=float((valid*face).sum()/max(face.sum(),1))
    metrics={'eye_center_error_px':float(np.max(np.linalg.norm(np.array([mapped[36:42].mean(0),mapped[42:48].mean(0)])-np.array([b[36:42].mean(0),b[42:48].mean(0)]),axis=1))),
             'stable_landmark_rms_px':rms,'stable_landmark_rms_interocular':rms/ocular,'nose_rms_px':nose,'nose_rms_interocular':nose/ocular,
             'scale_after_to_before':scale,'rotation_degrees':rotation,'face_coverage':coverage,'face_confidence':confidence}
    accepted=rms/ocular<=.08 and nose/ocular<=.08 and abs(rotation)<=45 and .1<=scale<=10 and coverage>=.90 and min(confidence.values())>=.6
    meta={'status':'ready' if accepted else 'unreliable','metrics':metrics,'canvas':[w,h],'method':'eye-centre similarity; uniform scale, rotation, translation; no nonrigid warping',
          'message':'双眼中心已对齐，鼻部与稳定关键点误差检查通过。' if accepted else '拍摄角度、遮挡或关键点差异较大，未通过配准检查；请使用同视角照片或原图对比。'}
    if accepted:
        dest=folder/'comparison';dest.mkdir(exist_ok=True)
        warped=cv2.warpAffine(rgb['after'],matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(237,240,231))
        Image.fromarray(rgb['before']).save(dest/'before.png');Image.fromarray(warped).save(dest/'after.png')
        for phase,photo,points,color in [('before',rgb['before'],b,(35,149,111)),('after',warped,mapped,(221,112,67))]:
            overlay=photo.copy()
            for i,point in enumerate(points):cv2.circle(overlay,tuple(point.round().astype(int)),3,color,-1,cv2.LINE_AA)
            Image.fromarray(overlay).save(dest/(phase+'_landmarks.png'))
        meta['landmarks']={'before':b.tolist(),'after_aligned':mapped.tolist()};meta['matrix']=matrix.tolist()
    write_meta(folder,meta);return meta
