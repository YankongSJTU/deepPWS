from pathlib import Path
from contextlib import asynccontextmanager,contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from html import escape
import io,json,os,secrets,sqlite3,subprocess,sys,threading,uuid,shutil,logging,importlib.util
import numpy as np
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,HTTPException,Request,UploadFile,File,Form,Depends
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from .inference import FusionModel
from .clinical_inference import ClinicalModel
from .demographic_inference import DemographicModel
from .image_upload import prepare_image,MAX_BYTES
from .schemas import ClinicalInfo,CandidateConfirmation,ExcelExportRequest
from .excel_export import workbook
from .visual_comparison import VERSION as COMPARISON_VERSION,write_meta as comparison_meta

ROOT=Path(os.environ.get('FACEBA_APP_ROOT',Path(__file__).resolve().parents[1])).resolve()
DATA=Path(os.environ.get('FACEBA_DATA_ROOT',ROOT/'data')).resolve()
DATA.mkdir(parents=True,exist_ok=True)
JOBS=DATA/'cases';JOBS.mkdir(exist_ok=True)
TOKEN=os.environ.get('FACEBA_API_TOKEN','')
log=logging.getLogger('faceba')
model=FusionModel(ROOT/'models')
clinical_model=ClinicalModel(ROOT/'clinical_models') if (ROOT/'clinical_models/model.json').exists() else None
demographic_model=DemographicModel(ROOT/'demographic_models') if (ROOT/'demographic_models/model.json').exists() else None
APP_VERSION='2026.10.26-workbench1'
pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='faceba-inference')
comparison_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='deeppws-comparison')
comparison_lock=threading.Lock()
db_lock=threading.Lock()

@contextmanager
def db():
    c=sqlite3.connect(DATA/'cases.sqlite3',timeout=30)
    c.row_factory=sqlite3.Row
    try:
        with c:yield c
    finally:c.close()

def init_db():
    with db() as c:
        c.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY,created TEXT NOT NULL,clinical TEXT NOT NULL,status TEXT NOT NULL,stage TEXT NOT NULL,progress INTEGER NOT NULL DEFAULT 0,result TEXT,error TEXT)')
        c.execute("UPDATE cases SET status='failed',stage='服务重启，请重新提交分析',error='分析被服务重启中断' WHERE status IN ('queued','running')")

def update(case_id,**values):
    with db() as c:
        c.execute('UPDATE cases SET '+','.join(f'{k}=?' for k in values)+' WHERE id=?',[*values.values(),case_id])

def get_case(case_id):
    try:uuid.UUID(case_id)
    except ValueError:raise HTTPException(404,'未找到此分析记录')
    with db() as c:r=c.execute('SELECT * FROM cases WHERE id=?',(case_id,)).fetchone()
    if not r:raise HTTPException(404,'未找到此分析记录')
    row=dict(r);row['clinical']=json.loads(row['clinical'])
    row['result']=json.loads(row['result']) if row['result'] else None
    folder=JOBS/case_id
    if row['status']=='running' and (folder/'progress.json').exists():
        try:row.update(json.loads((folder/'progress.json').read_text(encoding='utf8')))
        except (ValueError,OSError):pass
    req=folder/'request.json'
    row['images']=json.loads(req.read_text(encoding='utf8'))['images'] if req.exists() else {}
    row['image_preprocessing']=json.loads(req.read_text(encoding='utf8')).get('image_preprocessing',[]) if req.exists() else []
    row['comparison']=comparison_status(folder)
    return row

def comparison_status(folder):
    path=folder/'comparison/metadata.json'
    if path.exists():
        try:
            meta=json.loads(path.read_text(encoding='utf8'))
            if meta.get('version')==COMPARISON_VERSION:
                if meta.get('status')=='queued' and meta.get('owner_pid')!=os.getpid():return {'status':'missing','version':COMPARISON_VERSION}
                return meta
        except (OSError,ValueError):pass
    return {'status':'missing','version':COMPARISON_VERSION}

def comparison_job(folder):
    env=os.environ.copy();env['FACEBA_RUNTIME_ROOT']=str(runtime_root())
    command=[sys.executable,str(ROOT/'backend/comparison_worker.py'),str(folder)]
    runner=os.environ.get('FACEBA_WORKER_LAUNCHER')
    if runner:command=[runner,str(ROOT/'backend/comparison_worker.py'),str(folder)]
    try:
        with (folder/'comparison/worker.log').open('w',encoding='utf8') as f:
            job=subprocess.run(command,cwd=runtime_root(),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if job.returncode or comparison_status(folder)['status']=='queued':comparison_meta(folder,{'status':'unreliable','message':'配准未能完成，请检查同视角面部照片。'})
    except Exception:comparison_meta(folder,{'status':'unreliable','message':'配准未能完成，可稍后重试。'})

def runtime_root():
    configured=os.environ.get('FACEBA_RUNTIME_ROOT')
    return Path(configured).resolve() if configured else ROOT/'runtime'

def readiness():
    runtime=runtime_root()
    standalone=(ROOT/'standalone_models/lam_encoder.pt').is_file()
    expected=[runtime/'weights/clip/model.safetensors',runtime/'weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth',
              runtime/'3DDFA_V2/weights/mb1_120x120.pth',runtime/'3DDFA_V2/FaceBoxes/weights/FaceBoxesProd.pth']
    if standalone:expected.extend([ROOT/'standalone_models/lam_decoder.pt'])
    else:expected.append(runtime/'model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors')
    missing=[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in expected if not p.is_file()]
    try:
        import torch
        gpu=torch.cuda.is_available()
        use_gpu=gpu and os.environ.get('FACEBA_DEVICE','auto')!='cpu'
        device=torch.cuda.get_device_name(0) if use_gpu else 'CPU'
    except ImportError:gpu=False;device='未安装 PyTorch'
    required=['torch','torchvision','pandas','cv2','safetensors','scipy']
    if not standalone:required+=['omegaconf','einops','pytorch3d','trimesh']
    missing_dependencies=[m for m in required if importlib.util.find_spec(m) is None]
    if clinical_model is None:missing.append('clinical_models/model.json')
    if demographic_model is None:missing.append('demographic_models/model.json')
    ready=not missing and (gpu or standalone) and not missing_dependencies
    return {'ready':ready,'weights_ready':not missing,'gpu_available':gpu,'device':device,
            'missing_assets':missing,'missing_dependencies':missing_dependencies,
            'standalone_models':standalone,'offline':os.environ.get('FACEBA_OFFLINE')=='1','clinical_model_ready':clinical_model is not None,
            'message':'完整分析环境就绪' if ready else '请先配置完整模型运行环境'}

def cached_features_ready(folder):
    if not (folder/'features.npz').is_file() or not (folder/'quality.json').is_file():return False
    try:
        with np.load(folder/'features.npz',allow_pickle=False) as z:
            return {'before','after','aligned_lab','clinical_before_local'}<=set(z.files)
    except (OSError,ValueError):return False

def score_analysis(folder,clinical):
    with np.load(folder/'features.npz',allow_pickle=False) as z:features={k:z[k] for k in z.files}
    estimates=demographic_model.predict(features) if demographic_model else None
    effective=dict(clinical);sources={k:('manual' if clinical.get(k) not in [None,'','unspecified'] else 'missing') for k in ['age','sex','skin_type','lesion_site','nerve_zone','phenotype']}
    if estimates:
        for key in ['age','skin_type']:
            if sources[key]=='missing':effective[key]=estimates[key];sources[key]='model_estimate'
    result=model.predict(features,effective)
    result['clinical_estimates']=estimates
    result['effective_clinical']=effective
    result['clinical_value_sources']=sources
    result['clinical_suggestions']=clinical_model.predict(features,clinical.get('clinical_profile','source')) if clinical_model else {'status':'unavailable'}
    result['quality']=json.loads((folder/'quality.json').read_text(encoding='utf8'))
    metadata=folder/'upload_metadata.json'
    if metadata.exists():
        result['quality']['image_preprocessing']=json.loads(metadata.read_text(encoding='utf8'))
        count=sum(item['resized'] for item in result['quality']['image_preprocessing'])
        if count:result['quality'].setdefault('warnings',[]).append(f'{count} 张大图已按比例缩小后分析，原始照片保存在本机 originals 目录。')
    result['completed_at']=datetime.now(timezone.utc).isoformat()
    result['clinical_inputs_used']=['age','skin_type','lesion_site','nerve_zone','phenotype']
    result['report_only_fields']=[*model.meta['report_only_fields'],'clinical_profile','photo_interval_days']
    result['photo_interval_days']=clinical.get('photo_interval_days')
    if result['photo_interval_days'] is None and clinical.get('before_date') and clinical.get('after_date'):
        from datetime import date
        result['photo_interval_days']=(date.fromisoformat(clinical['after_date'])-date.fromisoformat(clinical['before_date'])).days
    result['photo_interval_source']='manual_days' if clinical.get('photo_interval_days') is not None else 'manual_dates' if result['photo_interval_days'] is not None else 'missing'
    result['app_version']=APP_VERSION
    return result

def process(case_id):
    folder=JOBS/case_id
    try:
        update(case_id,status='running',stage='准备影像分析',progress=5,error=None)
        env=os.environ.copy();env['FACEBA_RUNTIME_ROOT']=str(runtime_root())
        command=[sys.executable,str(ROOT/'backend/feature_worker.py'),str(folder)]
        # Launching the child under the same Python works for both native and GPU6 glibc environments.
        runner=os.environ.get('FACEBA_WORKER_LAUNCHER')
        if runner:command=[runner,str(ROOT/'backend/feature_worker.py'),str(folder)]
        cached=cached_features_ready(folder)
        if cached:
            update(case_id,stage='复用本机影像特征，更新评分与临床候选',progress=92)
            returncode=0
        else:
            with (folder/'worker.log').open('w',encoding='utf8') as logfile:
                completed=subprocess.run(command,cwd=runtime_root(),env=env,stdout=logfile,stderr=subprocess.STDOUT,timeout=7200 if (ROOT/'standalone_models/lam_encoder.pt').exists() else 1800,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            returncode=completed.returncode
        if returncode:
            tail=(folder/'worker.log').read_text(encoding='utf8',errors='replace').splitlines()[-1:]
            log.error('Feature extraction failed for %s: %s',case_id,tail)
            detail=tail[0] if tail else ''
            if 'no_face' in detail:message='未能在照片中检测到清晰面部，请上传同角度、无遮挡的面部照片。'
            elif 'CUDA out of memory' in detail:message='GPU 显存不足，请稍后重试。'
            else:message='影像分析未完成，请检查模型环境或联系管理员查看任务日志。'
            update(case_id,status='failed',stage='分析未完成',error=message)
            return
        case=get_case(case_id)
        result=score_analysis(folder,case['clinical']);result['reused_image_features']=cached
        update(case_id,status='completed',stage='分析完成',progress=100,result=json.dumps(result,ensure_ascii=False))
    except subprocess.TimeoutExpired:
        update(case_id,status='failed',stage='分析超时',error='分析超时，请检查设备资源后重试。')
    except Exception:
        log.exception('Analysis error %s',case_id)
        update(case_id,status='failed',stage='分析未完成',error='分析未完成，已保留病例，可重新提交。')

@asynccontextmanager
async def lifespan(app):
    init_db();yield
    pool.shutdown(wait=False,cancel_futures=True)
    comparison_pool.shutdown(wait=False,cancel_futures=True)

app=FastAPI(title='DeepPWS Clinical Studio',version=APP_VERSION,lifespan=lifespan)

@app.middleware('http')
async def headers(request,call_next):
    if request.headers.get('content-length','0').isdigit() and int(request.headers.get('content-length','0'))>MAX_BYTES*11:
        return JSONResponse({'detail':'上传内容过大'},status_code=413)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['X-Frame-Options']='DENY'
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store'
    return response

def authorize(request:Request):
    if TOKEN and not secrets.compare_digest(request.headers.get('authorization',''),'Bearer '+TOKEN):
        raise HTTPException(401,'请输入工作台访问密钥')
    origin=request.headers.get('origin')
    if request.method not in ['GET','HEAD','OPTIONS'] and origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
        raise HTTPException(403,'不允许跨站提交')

@app.get('/api/health')
def health():return {'status':'ok','authentication_required':bool(TOKEN),'model_loaded':True,'offline':os.environ.get('FACEBA_OFFLINE')=='1','app_version':APP_VERSION,'clinical_model_version':clinical_model.meta['version'] if clinical_model else None}

@app.get('/api/model',dependencies=[Depends(authorize)])
def model_info():return {'model':model.meta,'runtime':readiness(),'app_version':APP_VERSION,'clinical_model':clinical_model.meta if clinical_model else None,'demographic_model':demographic_model.meta if demographic_model else None}

@app.get('/api/cases',dependencies=[Depends(authorize)])
def cases():
    with db() as c:ids=[r['id'] for r in c.execute('SELECT id FROM cases ORDER BY created DESC')]
    return {'cases':[get_case(i) for i in ids]}

@app.post('/api/exports/excel',dependencies=[Depends(authorize)])
def export_excel(selection:ExcelExportRequest):
    if selection.scope=='all':
        with db() as c:ids=[r['id'] for r in c.execute("SELECT id FROM cases WHERE status='completed' AND result IS NOT NULL ORDER BY created DESC")]
    else:ids=list(dict.fromkeys(selection.case_ids))
    records=[get_case(i) for i in ids]
    if not records:raise HTTPException(404,'没有可导出的已完成记录')
    if any(c['status']!='completed' or not c['result'] for c in records):raise HTTPException(409,'只能导出已完成分析的记录，请刷新后重新选择')
    content=workbook(records,selection.scope,APP_VERSION,model.meta['age_median'])
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    return Response(content,media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',headers={'Content-Disposition':f'attachment; filename="DeepPWS_{selection.scope}_{stamp}.xlsx"'})

@app.post('/api/cases/{case_id}/comparison',dependencies=[Depends(authorize)])
def prepare_comparison(case_id:str):
    with comparison_lock:
        case=get_case(case_id)
        if case['status']!='completed':raise HTTPException(409,'分析完成后可生成配准对比')
        current=case['comparison']
        if current['status'] not in ['missing']:return current
        folder=JOBS/case_id;comparison_meta(folder,{'status':'queued','owner_pid':os.getpid(),'message':'正在检测关键点并统一前后照片尺度…'})
        comparison_pool.submit(comparison_job,folder)
        return comparison_status(folder)

@app.get('/api/cases/{case_id}/comparison',dependencies=[Depends(authorize)])
def get_comparison(case_id:str):return get_case(case_id)['comparison']

@app.get('/api/cases/{case_id}/comparison/images/{name}',dependencies=[Depends(authorize)])
def comparison_image(case_id:str,name:str):
    case=get_case(case_id)
    if name not in ['before.png','after.png','before_landmarks.png','after_landmarks.png']:raise HTTPException(404,'未找到对比图')
    if case['comparison']['status']!='ready':raise HTTPException(409,'尚无可靠的配准图')
    path=JOBS/case_id/'comparison'/name
    if not path.exists():raise HTTPException(404,'对比图不存在，请重新分析')
    return FileResponse(path)

async def store_image(upload:UploadFile,folder:Path,name:str):
    size=0;chunks=[]
    while chunk:=await upload.read(1024*1024):
        size+=len(chunk)
        if size>MAX_BYTES:raise HTTPException(413,'每张照片最大 100 MB')
        chunks.append(chunk)
    content=b''.join(chunks)
    filename,metadata=await run_in_threadpool(prepare_image,content,folder,name)
    path=folder/'upload_metadata.json'
    entries=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
    entries.append(metadata);path.write_text(json.dumps(entries,ensure_ascii=False),encoding='utf8')
    return filename

@app.post('/api/cases',status_code=201,dependencies=[Depends(authorize)])
async def create_case(clinical:str=Form(...),before:list[UploadFile]=File(default=[]),after:list[UploadFile]=File(default=[])):
    try:info=ClinicalInfo.model_validate_json(clinical)
    except (ValidationError,ValueError):raise HTTPException(422,'请检查病例信息、日期和年龄等字段')
    if len(before)>5 or len(after)>5:
        raise HTTPException(422,'每个阶段最多保存 5 张照片')
    case_id=str(uuid.uuid4());folder=JOBS/case_id;folder.mkdir()
    try:
        images={}
        for phase,files in [('before',before),('after',after)]:
            images[phase]=[await store_image(f,folder,f'{phase}_{i}') for i,f in enumerate(files)]
        record={'clinical':info.model_dump(mode='json'),'images':images,'image_preprocessing':json.loads((folder/'upload_metadata.json').read_text(encoding='utf8')) if (folder/'upload_metadata.json').exists() else []}
        (folder/'request.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf8')
        with db() as c:
            c.execute('INSERT INTO cases(id,created,clinical,status,stage,progress) VALUES(?,?,?,?,?,?)',
                      (case_id,datetime.now(timezone.utc).isoformat(),info.model_dump_json(),'draft','已保存，等待分析',0))
        return get_case(case_id)
    except Exception:
        shutil.rmtree(folder);raise

@app.post('/api/cases/{case_id}/analyze',dependencies=[Depends(authorize)])
def analyze(case_id:str):
    case=get_case(case_id)
    if case['status'] in ['queued','running']:return case
    if not (1<=len(case['images'].get('before',[]))<=5 and len(case['images'].get('before',[]))==len(case['images'].get('after',[]))):
        raise HTTPException(422,'开始分析前，请上传 1–5 组数量相同的治疗前后照片')
    if not readiness()['ready']:raise HTTPException(503,'完整模型运行环境未就绪，病例已保存。请配置 GPU 和模型文件后继续分析。')
    with db() as c:
        changed=c.execute("UPDATE cases SET status='queued',stage='已加入分析队列',progress=0,error=NULL WHERE id=? AND status NOT IN ('queued','running')",(case_id,)).rowcount
    if changed:
        (JOBS/case_id/'progress.json').unlink(missing_ok=True)
        pool.submit(process,case_id)
    return get_case(case_id)

@app.get('/api/cases/{case_id}',dependencies=[Depends(authorize)])
def detail(case_id:str):return get_case(case_id)

@app.patch('/api/cases/{case_id}/clinical',dependencies=[Depends(authorize)])
def edit_clinical(case_id:str,info:ClinicalInfo):
    case=get_case(case_id)
    if case['status'] in ['queued','running']:raise HTTPException(409,'分析过程中暂时不能修改病例')
    used=['age','skin_type','lesion_site','nerve_zone','phenotype','clinical_profile']
    changed=any(info.model_dump(mode='json')[k]!=case['clinical'].get(k,'source' if k=='clinical_profile' else None) for k in used)
    update(case_id,clinical=info.model_dump_json(),**({'status':'draft','result':None,'stage':'模型输入已更改，请重新分析','progress':0} if changed else {}))
    if not changed and case['result']:
        result=case['result'];result['effective_clinical']={**result.get('effective_clinical',case['clinical']),**{k:v for k,v in info.model_dump(mode='json').items() if k not in used}}
        result['photo_interval_days']=info.photo_interval_days
        if result['photo_interval_days'] is None and info.before_date and info.after_date:result['photo_interval_days']=(info.after_date-info.before_date).days
        result['photo_interval_source']='manual_days' if info.photo_interval_days is not None else 'manual_dates' if result['photo_interval_days'] is not None else 'missing'
        result.setdefault('clinical_value_sources',{})['sex']='manual' if info.sex!='unspecified' else 'missing'
        update(case_id,result=json.dumps(result,ensure_ascii=False))
    req=JOBS/case_id/'request.json';record=json.loads(req.read_text(encoding='utf8'))
    record['clinical']=info.model_dump(mode='json');req.write_text(json.dumps(record,ensure_ascii=False),encoding='utf8')
    return get_case(case_id)

@app.post('/api/cases/{case_id}/confirm-suggestions',dependencies=[Depends(authorize)])
def confirm_suggestions(case_id:str,confirmation:CandidateConfirmation):
    with db_lock:
        case=get_case(case_id)
        if case['status']!='completed':raise HTTPException(409,'请等待当前分析完成')
        suggestions=case['result'].get('clinical_suggestions',{})
        if suggestions.get('status')!='ready' or suggestions.get('profile_id')!=confirmation.profile_id or suggestions.get('version')!=clinical_model.meta['version']:
            raise HTTPException(409,'候选版本已变化，请刷新报告后确认')
        if not cached_features_ready(JOBS/case_id):raise HTTPException(409,'请先重新分析照片以生成新版特征')
        values={}
        for field in set(confirmation.fields):
            value=suggestions.get('suggested_fields',{}).get(field)
            if not value:raise HTTPException(422,'所选字段当前没有可填入的候选值')
            categories=next(f['categories'] for f in model.meta['clinical_fields'] if f['key']==field)
            if field=='phenotype' and value=='粉红' and value not in categories and '粉红色' in categories:value='粉红色'
            if value not in categories:raise HTTPException(422,'候选值不在评分模型类别中，请在病例资料里手动确认')
            values[field]=value
        info=ClinicalInfo.model_validate({**case['clinical'],**values})
        result=score_analysis(JOBS/case_id,info.model_dump(mode='json'));result['reused_image_features']=True
        result['clinical_suggestions']['confirmed_fields']=sorted(set(suggestions.get('confirmed_fields',[]))|set(values))
        update(case_id,clinical=info.model_dump_json(),result=json.dumps(result,ensure_ascii=False))
        req=JOBS/case_id/'request.json';record=json.loads(req.read_text(encoding='utf8'));record['clinical']=info.model_dump(mode='json');req.write_text(json.dumps(record,ensure_ascii=False),encoding='utf8')
        return get_case(case_id)

@app.delete('/api/cases/{case_id}',dependencies=[Depends(authorize)])
def delete(case_id:str):
    case=get_case(case_id)
    if case['status'] in ['queued','running']:raise HTTPException(409,'请等待分析结束再删除')
    if case['comparison']['status']=='queued':raise HTTPException(409,'配准正在进行，请完成后再删除')
    with db() as c:c.execute('DELETE FROM cases WHERE id=?',(case_id,))
    shutil.rmtree(JOBS/case_id)
    return {'deleted':True}

@app.get('/api/cases/{case_id}/images/{name}',dependencies=[Depends(authorize)])
def image(case_id:str,name:str):
    case=get_case(case_id)
    names=sum(case['images'].values(),[])
    if name not in names:raise HTTPException(404,'未找到照片')
    return FileResponse(JOBS/case_id/name)

@app.get('/api/cases/{case_id}/report.json',dependencies=[Depends(authorize)])
def json_report(case_id:str):
    case=get_case(case_id)
    if case['status']!='completed':raise HTTPException(409,'分析完成后可下载报告')
    payload={'report_version':'2.0','case_id':case_id,'created':case['created'],'clinical':case['clinical'],'analysis':case['result'],'visual_comparison':case['comparison']}
    return JSONResponse(payload,headers={'Content-Disposition':f'attachment; filename="DeepPWS-{case_id[:8]}.json"'})

@app.get('/api/cases/{case_id}/report.html',response_class=HTMLResponse,dependencies=[Depends(authorize)])
def printable_report(case_id:str):
    case=get_case(case_id)
    if case['status']!='completed':raise HTTPException(409,'分析完成后可下载报告')
    r=case['result'];c=r.get('effective_clinical',case['clinical']);score=r['scores']
    labels={'case_code':'病例编号','sex':'性别','age':'年龄（岁）','lesion_site':'病灶部位','skin_type':'Fitzpatrick 分型','nerve_zone':'三叉神经分区',
            'phenotype':'颜色分型','clinical_profile':'临床候选模型','treatment_count':'治疗次数','treatment_method':'治疗方式','complications':'并发症','complication_notes':'并发症说明','before_date':'治疗前照片日期','after_date':'治疗后照片日期','photo_interval_days':'手填拍照间隔（天）','notes':'备注'}
    sex={'female':'女','male':'男','unspecified':'未记录'}
    def value(k,v):
        if r.get('clinical_value_sources',{}).get(k)=='model_estimate':return escape(str(v))+'（照片估计，待医生核对）'
        if k=='sex':v=sex.get(v,v)
        if k=='clinical_profile':v=clinical_model.meta['profiles'].get(v,{}).get('name','不生成候选') if clinical_model else v
        if isinstance(v,list):v='、'.join(v)
        return escape(str(v)) if v is not None and v!='' else '未记录'
    rows=''.join('<tr><th>'+labels.get(k,k)+'</th><td>'+value(k,v)+'</td></tr>' for k,v in c.items())
    if r.get('clinical_estimates'):rows+='<tr><th>自动估计说明</th><td>年龄为照片估计；Fitzpatrick 仅估计病历 III/IV 标签，需结合晒伤/晒黑反应核对。性别、并发症与拍照间隔不预测。</td></tr>'
    suggestions=r.get('clinical_suggestions',{});candidate_html=''
    if suggestions.get('status')=='ready':
        candidate_html='<h2>治疗前临床候选（需核对）</h2><p>模型：'+escape(suggestions['profile_name'])+'；版本：'+escape(suggestions['version'])+'。模型分数未校准，候选未确认前不替代病例字段。</p><table>'
        for key,label in [('colour','颜色'),('side','左右'),('nerve','神经分区'),('regions','区域范围')]:
            candidate_html+='<tr><th>'+label+'</th><td>'+escape('、'.join(suggestions[key]['selected']) or '未给出阳性候选')+'</td></tr>'
        candidate_html+='</table><p>已确认填入：'+escape('、'.join({'phenotype':'颜色','nerve_zone':'神经分区'}[k] for k in suggestions.get('confirmed_fields',[])) or '无')+'。区域为粗分区，不自动改写病灶部位。</p>'
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DeepPWS 分析报告</title><style>
body{font:15px/1.7 Arial,"Microsoft YaHei",sans-serif;max-width:850px;margin:48px auto;color:#213a36;padding:24px}header{border-bottom:2px solid #456e59;padding-bottom:20px}small{color:#73817b}.scores{display:flex;gap:24px;margin:32px 0}.score{background:#f0f4ef;padding:20px;flex:1}.score b{display:block;font-size:32px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #e8ece6;text-align:left}th{width:180px}footer{font-size:12px;margin-top:30px}@media print{button{display:none}body{margin:0}}</style>
<header><small>DeepPWS / CLINICAL STUDIO</small><h1>治疗反应分析报告</h1><p>病例：'''+escape(c['case_code'])+'''</p></header><div class="scores">'''+''.join(f'<div class="score">{label}<b>{score[k]:.2f}<small> / 10</small></b></div>' for k,label in [('colour','颜色评分'),('area','面积评分'),('overall','综合评分')])+'''</div><h2>病例与临床信息</h2><table>'''+rows+'''</table>'''+candidate_html+'''<footer>模型版本：'''+escape(r['model_version'])+'''。综合评分为颜色与面积评分的等权平均。年龄、肤型、部位、分区及颜色分型参与模型；性别、治疗次数、并发症等仅用于报告记录。本结果用于研究辅助评估。<br>生成时间：'''+escape(r['completed_at'])+'''</footer><button onclick="window.print()">打印 / 另存为 PDF</button></html>'''

app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
