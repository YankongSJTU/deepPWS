from pathlib import Path
import argparse,os,sys

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='faceBA Clinical Studio')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--open',action='store_true');a=p.parse_args()
    if a.host not in ['127.0.0.1','localhost','::1'] and not os.environ.get('FACEBA_API_TOKEN'):
        p.error('For shared access set FACEBA_API_TOKEN before binding a network address.')
    import uvicorn
    print(f'faceBA Clinical Studio: http://{a.host}:{a.port}',flush=True)
    if a.open:
        import threading,webbrowser
        threading.Timer(2,lambda:webbrowser.open(f'http://127.0.0.1:{a.port}')).start()
    uvicorn.run('backend.app:app',host=a.host,port=a.port,workers=1,log_level='info')

if __name__=='__main__':main()
