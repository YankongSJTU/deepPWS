"""One isolated photo-to-feature job. Uses the original trained extraction routines.

All derived files stay in the current job directory. No training cache is modified.
"""
from pathlib import Path
import os,sys,json,importlib.util,time
import numpy as np

APP=Path(os.environ.get('FACEBA_APP_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(APP))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def main():
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf8',errors='replace')
    job=Path(sys.argv[1]).resolve(); info=json.loads((job/'request.json').read_text(encoding='utf8'))
    runtime=Path(os.environ.get('FACEBA_RUNTIME_ROOT',str(APP/'runtime'))).resolve()
    os.environ['FACEBA_RUNTIME_ROOT']=str(runtime)
    os.environ['FACEBA_JOB_ROOT']=str(job)
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    standalone=(APP/'standalone_models/lam_encoder.pt').is_file()
    if os.environ.get('FACEBA_OFFLINE')=='1':
        sys.path.insert(0,str(APP))
        from backend.offline_guard import enable
        enable()
    # An offline DINOv2 checkpoint is included in the full runtime bundle.
    if (runtime/'torch_cache').exists():os.environ['TORCH_HOME']=str(runtime/'torch_cache')
    sys.path[:0]=[str(APP/'vendor'),str(runtime),str(runtime/'vendor')]
    def progress(stage,value):
        p=job/'progress.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps({'stage':stage,'progress':value}),encoding='utf8');tmp.replace(p)
    import torch
    torch.set_num_threads(3)
    requested=os.environ.get('FACEBA_DEVICE','auto')
    device='cuda' if requested!='cpu' and torch.cuda.is_available() else 'cpu'
    if not standalone and device=='cpu':raise RuntimeError('完整照片分析需要可用的 NVIDIA GPU；请在 GPU 环境启动服务。')
    progress('正在加载图像编码器',12)
    from local_real_features import ClipVision
    from farl_encoder import FaRLVision
    clip=ClipVision(device)
    farl=FaRLVision(runtime/'weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth',device)
    paths={p:[job/n for n in info['images'][p]] for p in ['before','after']}
    progress('正在提取治疗前后影像特征',25)
    before=np.mean(clip(paths['before'],spatial=True)[0],axis=0).astype(np.float32)
    after=np.mean(clip(paths['after'],spatial=True)[0],axis=0).astype(np.float32)
    from PIL import Image,ImageOps
    face=[]
    for phase in ['before','after']:
        values=[]
        for path in paths[phase]:
            with Image.open(path) as im:values.append(farl([im])[0][0])
        face.append(np.asarray(values,np.float32).mean(0))
    # Clinical spatial head was trained on an EXIF-corrected, square512 first BEFORE view.
    # Extract it independently of after images and alignment, preserving all score features.
    progress('正在提取治疗前临床候选特征',34)
    with Image.open(paths['before'][0]) as im:
        first_before=ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
    clinical_local=farl([first_before])[1][0].astype(np.float32)
    assert clinical_local.shape==(16,768) and np.isfinite(clinical_local).all()
    progress('正在进行面部配准与色彩分析',40)
    import aligned_lab_features as lab
    boxes,landmark_model=lab.load_landmark_models()
    # Verify a face in both first-pair images before any reconstructed result is used.
    for p in ['before','after']:lab.landmarks(lab.read(paths[p][0]),boxes,landmark_model)
    lab.source=lambda pair,phase:paths[phase][0]
    alignment_ok=True; warnings=[]
    try:
        im,stats,affine,qc=lab.one({},boxes,landmark_model)
        residual_path=job/'delta_a.png';im.save(residual_path)
        embed=clip([residual_path],spatial=True)[0][0]
        aligned=np.r_[embed,stats,1.]
    except (ValueError,cv_error()) as e:
        alignment_ok=False;qc={'alignment_available':False}
        aligned=np.r_[np.full(788,np.nan),0.]
        warnings.append('面部配准未成功，色彩残差分支按训练时规则使用缺失值填补。')
    # Independent display transform; original scoring images and features are unchanged.
    from backend.visual_comparison import generate,write_meta
    try:generate(job,lab,boxes,landmark_model)
    except Exception:write_meta(job,{'status':'unreliable','message':'未能生成可靠的关键点配准对比。'})
    progress('正在重建三维面部与纹理',55)
    from formal3d_uv_batch import bake
    if standalone:
        # Sequential stage loading keeps the original float32 weights and releases GPU RAM.
        del clip,farl
        import gc
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        sys.path.insert(0,str(APP))
        from backend.standalone_lam import reconstruct
        reconstructed=reconstruct([paths[p][0] for p in ['before','after']],APP/'standalone_models',device,progress)
    else:
        from formal3d_batch import load_model,one
        model=load_model()
    out=job/'outputs/formal3d_full';(out/'uv').mkdir(parents=True,exist_ok=True)
    coverage={}
    for i,phase in enumerate(['before','after']):
        vertices,colours=reconstructed[i] if standalone else one(model,paths[phase][0])
        if vertices.shape!=(20018,3) or not np.isfinite(vertices).all():raise ValueError('无效的三维重建输出')
        dest=out/f'case_{phase}.npz';np.savez_compressed(dest,vertices=vertices,colors=colours)
        coverage[phase]=bake(dest,out/'uv'/f'case_{phase}.png')
        progress('正在生成区域几何与纹理描述',68+i*10)
    import dense_regional_features as dense
    dense.BASE=job
    geometry,uv=dense.extract('case')
    extras=np.r_[aligned,np.array(list(geometry.values())),np.array(list(uv.values())),face[0],face[1],face[1]-face[0]]
    assert extras.shape==(2709,)
    np.savez_compressed(job/'features.npz',before=before,after=after,aligned_lab=extras,clinical_before_local=clinical_local)
    (job/'quality.json').write_text(json.dumps({'alignment_available':alignment_ok,'alignment':qc,
        'uv_coverage':coverage,'warnings':warnings,'views_before':len(paths['before']),'views_after':len(paths['after']),
        'device':device,'standalone_models':standalone,'external_network_disabled':os.environ.get('FACEBA_OFFLINE')=='1',
        'feature_pipeline_version':'2026.10.25','clinical_features_before_only':True},ensure_ascii=False),encoding='utf8')
    progress('正在计算融合评分',92)

def cv_error():
    import cv2
    return cv2.error

if __name__=='__main__':main()
