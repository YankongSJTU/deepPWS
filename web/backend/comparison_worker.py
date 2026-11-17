from pathlib import Path
import os,sys,json
APP=Path(os.environ.get('FACEBA_APP_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(APP))
def main():
    folder=Path(sys.argv[1]).resolve();runtime=Path(os.environ['FACEBA_RUNTIME_ROOT'])
    os.environ['FACEBA_JOB_ROOT']=str(folder)
    if os.environ.get('FACEBA_OFFLINE')=='1':
        from backend.offline_guard import enable
        enable()
    sys.path[:0]=[str(APP/'vendor'),str(runtime),str(runtime/'vendor')]
    import torch
    torch.set_num_threads(3)
    import aligned_lab_features as lab
    from backend.visual_comparison import generate,write_meta
    try:
        boxes,model=lab.load_landmark_models();generate(folder,lab,boxes,model)
    except Exception:
        write_meta(folder,{'status':'unreliable','message':'未能建立可靠的面部关键点配准，请检查照片是否清晰、同视角且无遮挡。'})
if __name__=='__main__':main()
