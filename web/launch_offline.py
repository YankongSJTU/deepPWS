"""Entry point for the relocatable, self-contained Windows offline distribution."""
from pathlib import Path
import os,sys,json,argparse

ROOT=Path(__file__).resolve().parent
for stream in [sys.stdout,sys.stderr]:
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf8',errors='replace')
sys.path.insert(0,str(ROOT))
os.environ['FACEBA_APP_ROOT']=str(ROOT)
os.environ['FACEBA_RUNTIME_ROOT']=str(ROOT/'runtime')
os.environ['FACEBA_DATA_ROOT']=str(ROOT/'data')
os.environ['FACEBA_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['MPLBACKEND']='Agg'
os.environ.pop('FACEBA_WORKER_LAUNCHER',None)
from backend.offline_guard import enable
enable()

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        sys.argv=[str(ROOT/'backend/feature_worker.py')]+sys.argv[2:]
        from backend.feature_worker import main as worker
        worker();return
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8875)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--open',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    os.environ['FACEBA_DEVICE']=a.device
    from backend.app import readiness,APP_VERSION
    ready=readiness()
    if a.check:
        print(json.dumps({'python':sys.executable,'root':str(ROOT),'runtime':ready,'external_network_disabled':True},ensure_ascii=False,indent=2))
        if not ready['ready']:raise SystemExit(1)
        return
    import urllib.request,webbrowser
    for candidate_port in range(a.port,a.port+10):
        url=f'http://127.0.0.1:{candidate_port}'
        try:
            existing=json.load(urllib.request.urlopen(url+'/api/health',timeout=2))
            if existing.get('offline') and existing.get('app_version')==APP_VERSION:
                if a.open:webbrowser.open(url)
                print('Existing offline app:',url);return
            print('Earlier app occupies port',candidate_port,'; checking the next local port.')
        except (OSError,ValueError):
            import socket
            with socket.socket() as probe:
                try:probe.bind(('127.0.0.1',candidate_port))
                except OSError:continue
            a.port=candidate_port;break
    else:
        print('No free local port. Close an older app and retry.');raise SystemExit(1)
    if not ready['ready']:
        print('Local runtime check failed:',json.dumps(ready,ensure_ascii=False))
        raise SystemExit(1)
    print('DeepPWS Offline: '+url,flush=True)
    print('Device: '+ready['device']+' | All photos and inference stay on this computer.',flush=True)
    if a.open:
        import threading
        threading.Timer(2,lambda:webbrowser.open(url)).start()
    import uvicorn
    uvicorn.run('backend.app:app',host='127.0.0.1',port=a.port,workers=1,log_level='info')

if __name__=='__main__':main()
