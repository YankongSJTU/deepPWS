const paths={
sparkles:'m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6L12 3ZM20 2v4m-2-2h4',
layers:'m12 3 10 6-10 6L2 9l10-6ZM2 13l10 6 10-6M2 17l10 6 10-6',
box:'m12 3 9 5v9l-9 5-9-5V8l9-5ZM3 8l9 5 9-5M12 13v9M7.5 5.5l9 5',
activity:'M3 12h4l3-8 4 16 3-8h4',settings:'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1 1-3Z',
chevrons:'m8 9 4-4 4 4m-8 6 4 4 4-4',menu:'M4 6h16M4 12h16M4 18h16',history:'M3 12a9 9 0 1 0 2-5M3 3v5h5M12 7v5l3 2',
'arrow-up-right':'M7 17 17 7M7 7h10v10','arrow-right':'M4 12h16m-6-6 6 6-6 6','arrow-left':'M20 12H4m6-6-6 6 6 6',
'plus-circle':'M12 8v8M8 12h8M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z',plus:'M12 5v14M5 12h14',
'chevron-down':'m6 9 6 6 6-6',info:'M12 11v6m0-10h.01M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z',
'image-plus':'M21 13V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h8M3 16l5-5 8 8m-4-4 3-3 3 3M8 7h.01M19 16v6m-3-3h6',
'scan-face':'M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3M8 9h.01M16 9h.01M12 10v3m-4 3c2 2 6 2 8 0',
'shield-check':'m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6',save:'M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2ZM7 3v6h10V3M7 21v-8h10v8',
palette:'M12 3a9 9 0 1 0 0 18h1a2 2 0 0 0 1-3l-1-1a1 1 0 0 1 1-2h3a4 4 0 0 0 4-4c0-5-4-8-9-8ZM7 10h.01M10 7h.01M15 7h.01M18 10h.01',
scan:'M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3M7 12h10',
circle:'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z','check-circle':'m8 12 3 3 5-6M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z',
leaf:'M20 3C10 3 3 7 3 14a7 7 0 0 0 7 7c7 0 10-8 10-18ZM4 20 15 9',search:'M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14Zm5 12 6 6',
folder:'M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z',
download:'M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5','file-text':'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 12h8M8 16h8',
edit:'m16 3 5 5-12 12-6 1 1-6L16 3Zm-3 3 5 5',trash:'M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7'
};
function icon(name){return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name]||paths.circle}"/></svg>`}
function hydrate(root=document){root.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon))}
const $=id=>document.getElementById(id);
function defaultCaseCode(){const d=new Date(),pad=(n,w=2)=>String(n).padStart(w,'0');return `PWS-${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}-${pad(d.getMilliseconds(),3)}`}
function ensureCaseCode(){if(!$('case-code').value.trim())$('case-code').value=defaultCaseCode()}
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={files:{before:[],after:[]},urls:{before:[],after:[]},cases:[],model:null,clinicalModel:null,runtime:null,active:null,poll:null,busy:false,editing:null,savedSignature:null,savedId:null};
let token=sessionStorage.getItem('DeepPWS-token')||'';
let toastTimeout;
function toast(message,error=false){$('toast').textContent=message;$('toast').className='visible'+(error?' error':'');clearTimeout(toastTimeout);toastTimeout=setTimeout(()=>$('toast').className='',4500)}
async function api(path,options={}){
  const headers=new Headers(options.headers||{});if(token)headers.set('Authorization','Bearer '+token);
  const response=await fetch('/api'+path,{...options,headers});
  if(response.status===401){if(!$('auth-dialog').open)$('auth-dialog').showModal();throw Error('请输入工作台访问密钥')}
  if(!response.ok){let detail;try{detail=(await response.json()).detail}catch{}throw Error(typeof detail==='string'?detail:`请求未完成（${response.status}）`)}
  if(options.blob)return response.blob();
  return response.json();
}
function showView(name){
  document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!=='view-'+name);
  document.querySelectorAll('.nav-item').forEach(el=>el.classList.toggle('active',el.dataset.view===name));
  $('breadcrumb-current').textContent=({new:'新建评估',history:'分析记录',model:'模型信息',result:'分析报告'})[name];
  document.querySelector('.sidebar').classList.remove('open');
  clearTimeout(state.poll);state.poll=null;
  if(name==='history')refreshHistory().catch(e=>toast(e.message,true));
  if(name==='model')renderModel();
  window.scrollTo({top:0,behavior:'smooth'});
}
function setBusy(busy){state.busy=busy;$('analyze-button').disabled=busy;$('save-draft').disabled=busy;$('analyze-button').innerHTML=busy?icon('activity')+'正在准备分析…':icon('sparkles')+'一键分析'+icon('arrow-right')}
function clinical(){
  const form=new FormData($('assessment-form'));let o=Object.fromEntries(form.entries());
  ['age','treatment_count','photo_interval_days'].forEach(k=>o[k]=o[k]===''?null:Number(o[k]));
  ['before_date','after_date'].forEach(k=>o[k]=o[k]||null);
  o.complications=[...$('complication-options').querySelectorAll('input:checked')].map(el=>el.value);
  return o;
}
function checkForm(requirePhotos){
  ensureCaseCode();
  if(!$('assessment-form').reportValidity())return false;
  const c=clinical();
  if(c.before_date&&c.after_date&&c.before_date>c.after_date){toast('治疗后照片日期不能早于治疗前照片日期',true);return false}
  if(requirePhotos&&(!state.files.before.length||state.files.before.length!==state.files.after.length)){toast('请上传数量相同的治疗前后照片，支持 1–5 组',true);$('drop-before').scrollIntoView({behavior:'smooth',block:'center'});return false}
  return true;
}
function updateReadiness(){
  const checks=[$('case-code').value.trim().length>0,state.files.before.length>0,state.files.after.length>0&&state.files.before.length===state.files.after.length];
  ['profile','before','after'].forEach((k,i)=>{const el=$('check-'+k);el.classList.toggle('complete',checks[i]);el.querySelector('[data-icon]').innerHTML=icon(checks[i]?'check-circle':'circle')});
  const percent=Math.round(checks.filter(Boolean).length/3*100);$('readiness-percent').textContent=percent+'%';$('readiness-fill').style.width=percent+'%';
  $('input-summary').textContent=percent===100?'资料已准备好，可以开始分析。':checks[0]?`已添加 ${state.files.before.length} 张治疗前 / ${state.files.after.length} 张治疗后照片。`:'从填写病例编号开始。';
  $('step-1').classList.toggle('done',checks[0]);$('step-2').classList.toggle('current',checks[0]);$('step-2').classList.toggle('done',checks[1]&&checks[2]);
  $('step-3').classList.toggle('current',percent===100);
}
function photosChanged(){if(state.editing){state.editing=null;state.savedId=null;toast('更换影像后将保存为一份新评估，原记录保留。')}updateReadiness()}
async function photoPreview(file){
  const bitmap=await createImageBitmap(file,{resizeWidth:512,resizeQuality:'high'});
  try{const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;canvas.getContext('2d').drawImage(bitmap,0,0);return await new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('preview')),'image/jpeg',.88))}
  finally{bitmap.close()}
}
function renderPhotos(phase){
  state.urls[phase].forEach(URL.revokeObjectURL);state.urls[phase]=[];
  const zone=$('drop-'+phase),grid=$(phase+'-previews');zone.classList.toggle('has-files',state.files[phase].length>0);
  $(phase+'-count').textContent=state.files[phase].length+' / 5';grid.innerHTML='';
  state.files[phase].forEach((file,index)=>{
    const tile=document.createElement('div');tile.className='photo-tile';
    const img=document.createElement('img');img.alt=`${phase==='before'?'治疗前':'治疗后'}照片 ${index+1}`;tile.append(img);
    photoPreview(file).then(blob=>{if(!tile.isConnected||!state.files[phase].includes(file))return;const url=URL.createObjectURL(blob);state.urls[phase].push(url);img.src=url}).catch(()=>{img.alt='无法预览，仍可上传由后台处理'});
    const span=document.createElement('span');span.textContent=`${index+1}. ${file.name}`;tile.append(span);
    const button=document.createElement('button');button.type='button';button.textContent='×';button.setAttribute('aria-label','移除此照片');button.onclick=e=>{e.stopPropagation();state.files[phase].splice(index,1);renderPhotos(phase);photosChanged()};tile.append(button);grid.append(tile);
  });
  if(state.files[phase].length<5){const add=document.createElement('button');add.type='button';add.className='add-photo';add.innerHTML=icon('plus')+'添加同阶段照片';add.onclick=e=>{e.stopPropagation();$(phase+'-files').click()};grid.append(add)}
}
async function addFiles(phase,files){
  const incoming=[...files];if(state.files[phase].length+incoming.length>5){toast('每个阶段最多上传 5 张照片',true);return}
  for(const file of incoming){if(!['image/jpeg','image/png','image/webp'].includes(file.type)){toast('请上传 JPG、PNG 或 WebP 图片',true);return}if(file.size>100*1024*1024){toast('每张照片不能超过 100 MB',true);return}}
  state.files[phase].push(...incoming);renderPhotos(phase);photosChanged();$(phase+'-files').value='';
}
async function saveCase(){
  const c=clinical();
  if(state.editing){const saved=await api('/cases/'+state.editing+'/clinical',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(c)});return saved}
  const signature=JSON.stringify([c,...['before','after'].map(p=>state.files[p].map(f=>[f.name,f.size,f.lastModified]))]);
  if(signature===state.savedSignature&&state.savedId)return api('/cases/'+state.savedId);
  const form=new FormData();form.append('clinical',JSON.stringify(c));
  ['before','after'].forEach(p=>state.files[p].forEach(f=>form.append(p,f)));
  const saved=await api('/cases',{method:'POST',body:form});state.savedId=saved.id;state.savedSignature=signature;return saved;
}
async function submitAssessment(e){
  e.preventDefault();if(state.busy||!checkForm(true))return;setBusy(true);
  let saved;
  try{saved=await saveCase();await api('/cases/'+saved.id+'/analyze',{method:'POST'});await openCase(saved.id);refreshHistory(false).catch(()=>{})}
  catch(e){toast(e.message,true);if(saved){await openCase(saved.id)}}
  finally{setBusy(false)}
}
function formatDate(s){return new Date(s).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})}
const statusNames={draft:'待分析',queued:'排队中',running:'分析中',completed:'已完成',failed:'未完成'};
function statusTag(s){return `<span class="status ${s}">${escapeHtml(statusNames[s]||s)}</span>`}
const sexNames={female:'女',male:'男',unspecified:'未记录'};
async function refreshHistory(render=true){state.cases=(await api('/cases')).cases;$('record-count').textContent=state.cases.length;if(render)renderHistory()}
function renderHistory(){
  const counts=[state.cases.length,state.cases.filter(c=>c.status==='completed').length,state.cases.filter(c=>['draft','failed'].includes(c.status)).length];
  $('history-stats').innerHTML=['全部病例','已完成评估','待继续评估'].map((n,i)=>`<div class="stat-card"><span class="stat-icon">${icon(['folder','check-circle','activity'][i])}</span><div><small>${n}</small><b>${counts[i]}</b></div></div>`).join('');
  const term=$('history-search').value.toLowerCase();const cases=state.cases.filter(c=>(c.clinical.case_code+' '+c.clinical.lesion_site).toLowerCase().includes(term));
  $('history-empty').hidden=state.cases.length>0;
  $('history-body').innerHTML=cases.map(c=>`<tr><td><input class="export-checkbox" type="checkbox" data-export-case="${c.id}" aria-label="选择导出 ${escapeHtml(c.clinical.case_code)}" ${c.status==='completed'?'':'disabled'}></td><td><b>${escapeHtml(c.clinical.case_code)}</b><small>${formatDate(c.created)}</small></td><td>${escapeHtml(c.clinical.lesion_site)||'未记录'}</td><td>${c.clinical.age??'—'} / ${sexNames[c.clinical.sex]}</td><td>${c.clinical.treatment_count===null?'—':c.clinical.treatment_count+' 次'}</td><td>${c.result?`<b>${c.result.scores.overall.toFixed(2)}</b><small>满分 10</small>`:'—'}</td><td>${statusTag(c.status)}</td><td><button class="text-button" data-open="${c.id}">${c.status==='completed'?'查看报告':'继续评估'}</button>${!['running','queued'].includes(c.status)?`<button class="text-button danger" aria-label="删除 ${escapeHtml(c.clinical.case_code)}" data-delete="${c.id}">删除</button>`:''}</td></tr>`).join('')||(!state.cases.length?'':'<tr><td colspan="8">没有找到匹配的记录。</td></tr>');
  $('history-body').querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>openCase(b.dataset.open).catch(e=>toast(e.message,true)));
  $('history-body').querySelectorAll('[data-delete]').forEach(b=>b.onclick=async()=>{if(!confirm('删除这份病例及其照片、报告？此操作不能撤销。'))return;try{await api('/cases/'+b.dataset.delete,{method:'DELETE'});await refreshHistory();toast('病例已删除')}catch(e){toast(e.message,true)}});
  bindHistoryExport(cases);
}
function runtimeIndicator(){const r=state.runtime,el=$('runtime-indicator');el.classList.toggle('ready',Boolean(r?.ready));el.querySelector('span').textContent=r?.ready?'模型已就绪':'模型环境待配置';el.title=r?.device||''}
function renderModel(){
  if(!state.model)return;
  const m=state.model,r=state.runtime;
  $('model-content').innerHTML=`<section class="model-hero"><div class="eyebrow">PRETRAINED / FROZEN PARAMETERS</div><h2>${escapeHtml(m.name)}</h2><p>从成对照片中提取全局影像、对齐色彩变化和区域三维特征，结合训练时确定的临床字段，输出颜色与面积评分。综合评分为两者的等权平均。</p><div class="model-fields" style="margin-top:20px"><span>CLIP</span><span>FaRL global</span><span>Aligned Lab</span><span>3D / UV</span><span>Clinical + Ridge</span></div></section><div class="model-grid"><section class="panel"><h3>参与模型的临床信息</h3><div class="model-fields"><span>年龄</span><span>Fitzpatrick 肤型</span><span>病灶部位</span><span>三叉神经分区</span><span>颜色分型</span></div><p class="comparison-note" style="margin-top:17px">年龄与 Fitzpatrick 留空时使用独立照片估计，清楚标记来源并供医生修正。未见过的手填类别按原编码规则处理。</p><h3 style="margin-top:27px">用于病例与报告的补充信息</h3><div class="model-fields report-only"><span>性别</span><span>治疗次数</span><span>治疗方式</span><span>并发症</span><span>随访日期</span><span>临床备注</span></div><p class="comparison-note" style="margin-top:17px">这些信息会随病例保存，不作为当前模型的新输入。</p></section><section class="panel"><h3>模型与运行状态</h3><table class="model-spec"><tr><td>模型版本</td><td>${escapeHtml(m.version)}</td></tr><tr><td>上海校正源队列</td><td>${m.source_records} 条 / ${m.source_groups} 组</td></tr><tr><td>拟合输入维度</td><td>${m.input_features}</td></tr><tr><td>源 LOOCV 总体 Spearman</td><td>${m.source_metrics.overall_spearman.toFixed(4)}</td></tr><tr><td>当前设备</td><td>${escapeHtml(r.device)}</td></tr><tr><td>运行状态</td><td>${r.ready?'完整分析环境就绪':'需要配置运行环境'}</td></tr></table>${r.ready?'':`<div class="env-note">病例保存和资料管理可用。开始真实照片分析前，需要完整编码器、GPU 和运行依赖。<div class="missing-list">${[...r.missing_assets,...(r.missing_dependencies||[]).map(m=>'Python: '+m)].map(escapeHtml).join('<br>')}</div></div>`}<div class="comparison-note" style="margin-top:15px;word-break:break-all">参数指纹：${escapeHtml(m.parameter_sha256)}</div></section></div>`;
  if(state.clinicalModel){const cm=state.clinicalModel;$('model-content').insertAdjacentHTML('beforeend',`<section class="panel clinical-model-info"><div class="result-panel-title"><span>已内置临床候选识别</span><span class="tag tag-green">${escapeHtml(cm.version)}</span></div><p>仅使用治疗前影像生成颜色、左右、神经分区和粗区域候选。三个源模型共同推理，所有参数保存在本机。</p><div class="clinical-profile-list">${Object.values(cm.profiles).map(p=>`<div><b>${escapeHtml(p.name)}</b><p>${escapeHtml(p.description)}</p></div>`).join('')}</div><p class="comparison-note">重庆与浙江适配是探索性研究候选，预测分数未校准。年龄与 Fitzpatrick 标签可手填，留空使用照片估计并需核对；性别只手填。主疗效评分使用完整 CLIP/FaRL/3D/UV 融合模型。</p></section>`)}
}
async function openCase(id){
  const c=await api('/cases/'+id);state.active=c;showView('result');renderResult(c);
  if(['queued','running'].includes(c.status))pollCase(id);
}
function pollCase(id){clearTimeout(state.poll);state.poll=setTimeout(async()=>{try{const c=await api('/cases/'+id);if(state.active?.id!==id)return;state.active=c;renderResult(c);if(['queued','running'].includes(c.status))pollCase(id);else refreshHistory(false).catch(()=>{})}catch(e){toast(e.message,true)}},3000)}
const resultBlobUrls=[];
async function caseImage(id,name){const blob=await api('/cases/'+id+'/images/'+encodeURIComponent(name),{blob:true});const url=URL.createObjectURL(blob);resultBlobUrls.push(url);return url}
function renderResult(c){
  resultBlobUrls.splice(0).forEach(URL.revokeObjectURL);
  $('result-title').textContent=c.clinical.case_code;$('result-subtitle').textContent=`${formatDate(c.created)} 创建 · ${c.clinical.lesion_site||'未记录部位'} · ${statusNames[c.status]}`;
  $('download-json').hidden=$('download-report').hidden=c.status!=='completed';$('edit-case').hidden=['running','queued'].includes(c.status);
  if(['running','queued'].includes(c.status)){$('result-content').innerHTML=`<section class="panel progress-panel"><div class="progress-orb">${icon('scan-face')}</div><h2>${escapeHtml(c.stage)}</h2><p>正在使用内置模型分析治疗前后影像</p><div class="progress-track"><span style="width:${c.progress}%"></span></div><div class="progress-percent">${c.progress}%</div><p class="progress-note">首次加载编码器可能需要一些时间。您可以离开此页，稍后在分析记录中查看结果。</p></section>`;return}
  if(c.status!=='completed'){$('result-content').innerHTML=`<section class="panel error-panel"><h3>${c.status==='failed'?'这次分析未能完成':'病例已保存，准备好时继续。'}</h3><p>${escapeHtml(c.error||'补充前后配对照片后，即可开始完整模型分析。')}</p><button class="button button-primary" id="resume-analysis">${icon('sparkles')}开始 / 重试分析</button><button class="button button-outline" id="edit-draft">${icon('edit')}完善病例资料</button></section>`;$('resume-analysis').onclick=async()=>{try{await api('/cases/'+c.id+'/analyze',{method:'POST'});await openCase(c.id)}catch(e){toast(e.message,true)}};$('edit-draft').onclick=()=>editCase(c).catch(e=>toast(e.message,true));return}
  const r=c.result,info=r.effective_clinical||c.clinical;
  const fields=[['病例编号',info.case_code],['年龄 / 性别',`${info.age??'未记录'}${info.age===null?'':' 岁'} / ${sexNames[info.sex]}`],['病灶部位',info.lesion_site],['治疗次数',info.treatment_count===null?'未记录':info.treatment_count+' 次'],['Fitzpatrick 肤型',info.skin_type],['三叉神经分区',info.nerve_zone],['颜色分型',info.phenotype],['治疗方式',info.treatment_method],['并发症',info.complications.join('、')],['治疗前 / 后日期',`${info.before_date||'未记录'} / ${info.after_date||'未记录'}`],['并发症说明',info.complication_notes],['临床备注',info.notes]];
  const unknownLabels={skin_type:'肤型',lesion_site:'部位',nerve_zone:'分区',phenotype:'颜色分型'};
  const warnings=[...(r.quality?.warnings||[])];if(r.age_imputed)warnings.push('年龄未填写，使用训练时保存的中位数。');if(r.unseen_clinical_fields.length)warnings.push('以下字段未提供或不在训练类别中，已按原编码规则处理：'+r.unseen_clinical_fields.map(k=>unknownLabels[k]).join('、')+'。');
  $('result-content').innerHTML=`<div class="score-grid">${[['colour','颜色评分','palette'],['area','面积评分','scan'],['overall','综合评分','activity']].map(([k,n,i])=>`<article class="score-card ${k==='overall'?'featured':''}"><div class="score-top"><span>${n}</span>${icon(i)}</div><div class="score-value">${r.scores[k].toFixed(2)}<small>/ 10</small></div><div class="score-meter"><span style="width:${r.scores[k]*10}%"></span></div><div class="score-foot"><span>${['低分组','中分组','高分组'][r.categories[k]]}</span><span>${k==='overall'?'颜色与面积的等权平均':'固定模型输出'}</span></div></article>`).join('')}</div><div class="result-grid"><section class="panel"><div class="result-panel-title"><span>治疗前后影像对比</span><span class="tag tag-green" id="comparison-status">准备配准</span></div><div class="image-comparison"><img id="compare-after" alt="治疗后照片"><div class="comparison-before" id="comparison-before"><img id="compare-before" alt="治疗前照片"></div><div class="comparison-divider" id="comparison-divider"></div><div class="comparison-labels"><span>BEFORE</span><span>AFTER</span></div></div><input class="comparison-range" id="comparison-range" type="range" min="0" max="100" value="50" aria-label="调整治疗前后照片对比比例"><p class="comparison-note" id="comparison-note">正在准备关键点配准对比…</p><div class="result-quality"><span>${r.quality.alignment_available?'面部配准可用':'配准分支已填补'}</span><span>前 ${r.quality.views_before} / 后 ${r.quality.views_after} 张</span><span>三维与纹理特征已提取</span></div></section><section class="panel"><div class="result-panel-title"><span>本次病例资料</span><span class="tag tag-neutral">CASE PROFILE</span></div><dl class="clinical-details">${fields.map(([n,v])=>`<div class="${['临床备注','并发症说明'].includes(n)?'full':''}"><dt>${n}</dt><dd>${escapeHtml(v)||'未记录'}</dd></div>`).join('')}</dl></section></div><div class="result-footnote">${warnings.map(escapeHtml).join('<br>')}${warnings.length?'<br>':''}性别、治疗次数、并发症等为报告记录字段；评分使用已训练模型及原有五项临床输入。分组由评分阈值 3 和 6 派生。本结果用于研究辅助评估。</div>`;
  const candidates=document.createElement('div');candidates.innerHTML=clinicalSuggestionsHtml(c);$('result-content').insertBefore(candidates,$('result-content').querySelector('.result-grid'));bindClinicalSuggestions(c);renderClinicalReview(c);
  renderComparison(c);
}
function clinicalSuggestionsHtml(c){
  const s=c.result?.clinical_suggestions;
  if(s?.status==='disabled')return '<section class="panel clinical-suggestions"><h3>本次未启用颜色与分区候选</h3><p>可在编辑资料中切换候选模型。</p></section>';
  if(!s||s.status!=='ready'||(state.clinicalModel&&s.version!==state.clinicalModel.version))return `<section class="panel clinical-suggestions"><h3>临床候选识别已更新</h3><p>此记录尚未生成当前版本的临床候选，重新分析后可查看。</p><button class="button button-outline" id="refresh-clinical">重新分析照片</button></section>`;
  const names={colour:'病灶颜色',side:'左右范围',nerve:'三叉神经分区',regions:'区域范围'};
  const confirmed=s.confirmed_fields||[];
  const choices=[['phenotype','颜色'],['nerve_zone','神经分区']].filter(([k])=>s.suggested_fields?.[k]);
  return `<section class="panel clinical-suggestions"><div class="result-panel-title"><div><span>治疗前临床候选</span><p class="candidate-subtitle">${escapeHtml(s.profile_name)} · ${escapeHtml(s.version)}</p></div><span class="tag tag-neutral">需核对</span></div><div class="candidate-grid">${Object.entries(names).map(([k,n])=>`<article class="candidate-card"><h3>${n}</h3><div class="candidate-selection">${escapeHtml(s[k].selected.join('、')||'未给出阳性候选')}</div><div class="candidate-scores">${[...s[k].scores].sort((a,b)=>b.score-a.score).slice(0,k==='regions'?5:3).map(v=>`<div><span>${escapeHtml(v.label)}</span><div class="candidate-bar"><i style="width:${Math.max(0,Math.min(100,v.score*100))}%"></i></div><small>${(v.score*100).toFixed(1)}%</small></div>`).join('')}</div></article>`).join('')}</div><p class="comparison-note">以上为模型分数，不代表诊断准确率。区域是粗分区，不自动改写病灶部位；年龄与 III/IV 肤型标签估计见下方核对区；性别保留手填。</p><div class="candidate-confirmation"><div class="candidate-checkboxes">${choices.map(([k,n])=>`<label><input type="checkbox" name="confirm-candidate" value="${k}" ${!c.clinical[k]&&!confirmed.includes(k)?'checked':''}><span>填入${n}候选：<b>${escapeHtml(s.suggested_fields[k])}</b>${confirmed.includes(k)?' · 已确认':''}</span></label>`).join('')}</div><button type="button" class="button button-primary" id="confirm-candidates">确认所选候选并更新评分</button></div><small class="candidate-help">仅填入您勾选的字段；已填写的病例信息默认不覆盖。确认后复用本机影像特征，无需重新做三维分析。</small></section>`;
}
function bindClinicalSuggestions(c){
  if($('refresh-clinical'))$('refresh-clinical').onclick=async()=>{try{await api('/cases/'+c.id+'/analyze',{method:'POST'});await openCase(c.id)}catch(e){toast(e.message,true)}};
  const b=$('confirm-candidates');if(!b)return;
  const inputs=[...document.querySelectorAll('[name="confirm-candidate"]')];
  const changed=()=>b.disabled=!inputs.some(i=>i.checked);inputs.forEach(i=>i.onchange=changed);changed();
  b.onclick=async()=>{b.disabled=true;try{const fields=inputs.filter(i=>i.checked).map(i=>i.value);const updated=await api('/cases/'+c.id+'/confirm-suggestions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({fields,profile_id:c.result.clinical_suggestions.profile_id})});state.active=updated;renderResult(updated);await refreshHistory(false);toast('已确认所选候选，并使用本机影像特征更新评分。')}catch(e){toast(e.message,true);changed()}};
}
async function editCase(c){
  state.editing=null;state.savedId=null;state.savedSignature=null;$('assessment-form').reset();
  const form=$('assessment-form');
  for(const[k,v]of Object.entries(c.clinical)){const el=form.elements.namedItem(k);if(el&&k!=='complications')el.value=v??''}
  $('complication-options').querySelectorAll('input').forEach(el=>el.checked=c.clinical.complications.includes(el.value));updateComplicationDetail();
  for(const phase of ['before','after']){state.files[phase]=[];for(const name of c.images[phase]||[]){const blob=await api('/cases/'+c.id+'/images/'+encodeURIComponent(name),{blob:true});state.files[phase].push(new File([blob],name,{type:blob.type}))}renderPhotos(phase)}
  state.editing=c.id;updateReadiness();showView('new');toast('正在编辑 '+c.clinical.case_code+'；更改模型输入后需重新分析。');
}
function resetForm(){state.editing=null;state.savedId=null;state.savedSignature=null;$('assessment-form').reset();ensureCaseCode();['before','after'].forEach(p=>{state.files[p]=[];renderPhotos(p)});updateComplicationDetail();updateReadiness();showView('new')}
function updateComplicationDetail(){const selected=$('complication-options').querySelectorAll('input:checked');$('complication-detail').hidden=![...selected].some(el=>el.value!=='无')}
async function download(path,filename){try{const blob=await api(path,{blob:true}),url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000)}catch(e){toast(e.message,true)}}
async function boot(){
  try{const health=await api('/health');if(health.authentication_required&&!token){$('auth-dialog').showModal();return}
    const data=await api('/model');state.model=data.model;state.clinicalModel=data.clinical_model;state.runtime=data.runtime;runtimeIndicator();
    const ids={skin_type:'skin-type',lesion_site:'site-options',nerve_zone:'nerve-zone',phenotype:'phenotype'};
    for(const field of data.model.clinical_fields){const el=$(ids[field.key]);el.innerHTML=field.key==='lesion_site'?'':`<option value="">${field.key==='skin_type'?'留空时估计 III/IV 标签':'未记录'}</option>`;const categories=field.key==='skin_type'?['I','II','III','IV','V','VI']:field.categories.filter(c=>c!=='missing');categories.forEach(c=>{const option=document.createElement('option');option.value=c;option.textContent=c;el.append(option)})}
    await refreshHistory(false);
  }catch(e){toast(e.message,true);$('runtime-indicator').querySelector('span').textContent='服务未连接'}
}
hydrate();
ensureCaseCode();
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{if(b.dataset.view==='new')resetForm();else showView(b.dataset.view)});
$('mobile-menu').onclick=()=>document.querySelector('.sidebar').classList.toggle('open');
$('open-history').onclick=$('back-to-history').onclick=()=>showView('history');
$('new-from-history').onclick=$('empty-new').onclick=resetForm;
$('assessment-form').addEventListener('submit',submitAssessment);
$('assessment-form').addEventListener('input',updateReadiness);
$('assessment-form').addEventListener('keydown',e=>{if(e.key==='Enter'&&e.target.tagName==='INPUT')e.preventDefault()});
$('save-draft').onclick=async()=>{if(state.busy||!checkForm(false))return;setBusy(true);try{const c=await saveCase();await refreshHistory(false);toast('病例已保存，可在分析记录中继续。');await openCase(c.id)}catch(e){toast(e.message,true)}finally{setBusy(false)}};
$('complication-options').querySelectorAll('input').forEach(input=>input.onchange=()=>{if(input.checked){$('complication-options').querySelectorAll('input').forEach(other=>{if(other!==input&&(input.value==='无'||other.value==='无'))other.checked=false})}updateComplicationDetail()});
for(const phase of ['before','after']){const zone=$('drop-'+phase),input=$(phase+'-files');zone.onclick=e=>{if(!e.target.closest('button'))input.click()};zone.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();input.click()}};input.onchange=()=>addFiles(phase,input.files);zone.ondragover=e=>{e.preventDefault();zone.classList.add('drag-over')};zone.ondragleave=()=>zone.classList.remove('drag-over');zone.ondrop=e=>{e.preventDefault();zone.classList.remove('drag-over');addFiles(phase,e.dataTransfer.files)}}
$('history-search').oninput=renderHistory;
$('edit-case').onclick=()=>editCase(state.active).catch(e=>toast(e.message,true));
$('download-json').onclick=()=>download('/cases/'+state.active.id+'/report.json','DeepPWS-'+state.active.id.slice(0,8)+'.json');
$('download-report').onclick=async()=>{const win=window.open('about:blank','_blank');if(win)win.document.body.textContent='正在生成报告…';try{const blob=await api('/cases/'+state.active.id+'/report.html',{blob:true}),url=URL.createObjectURL(blob);if(win){win.opener=null;win.location=url}else{const a=document.createElement('a');a.href=url;a.download='DeepPWS-report.html';a.click()}setTimeout(()=>URL.revokeObjectURL(url),120000)}catch(e){if(win)win.close();toast(e.message,true)}};
$('auth-form').onsubmit=async e=>{e.preventDefault();token=$('access-token').value.trim();try{await api('/cases');sessionStorage.setItem('DeepPWS-token',token);$('auth-dialog').close();$('auth-error').textContent='';await boot()}catch(e){$('auth-error').textContent='密钥不正确，请重试。';token='';sessionStorage.removeItem('DeepPWS-token')}};
$('auth-dialog').addEventListener('cancel',e=>e.preventDefault());
updateReadiness();boot();
