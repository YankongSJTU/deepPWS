const exportSelection=new Set();
function bindHistoryExport(cases){
  const completed=new Set(state.cases.filter(c=>c.status==='completed').map(c=>c.id));
  [...exportSelection].forEach(id=>{if(!completed.has(id))exportSelection.delete(id)});
  const head=$('export-select-visible');const visible=cases.filter(c=>c.status==='completed');
  const update=()=>{
    $('export-selected').disabled=exportSelection.size===0;$('export-selected').textContent=`导出选中（${exportSelection.size}）`;
    $('export-all').disabled=completed.size===0;
    head.checked=visible.length>0&&visible.every(c=>exportSelection.has(c.id));head.indeterminate=visible.some(c=>exportSelection.has(c.id))&&!head.checked;
  };
  document.querySelectorAll('[data-export-case]').forEach(el=>{el.checked=exportSelection.has(el.dataset.exportCase);el.onchange=()=>{el.checked?exportSelection.add(el.dataset.exportCase):exportSelection.delete(el.dataset.exportCase);update()}});
  head.onchange=()=>{visible.forEach(c=>head.checked?exportSelection.add(c.id):exportSelection.delete(c.id));renderHistory()};
  $('export-all').onclick=()=>exportExcel('all');$('export-selected').onclick=()=>exportExcel('selected');update();
}
async function exportExcel(scope){
  const button=scope==='all'?$('export-all'):$('export-selected');button.disabled=true;
  try{
    const blob=await api('/exports/excel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scope,case_ids:scope==='selected'?[...exportSelection]:[]}),blob:true});
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`DeepPWS_${scope}_${new Date().toISOString().slice(0,10)}.xlsx`;link.click();setTimeout(()=>URL.revokeObjectURL(url),60000);toast('Excel 文件已导出。');
  }catch(e){toast(e.message,true)}finally{bindHistoryExport(state.cases.filter(c=>(c.clinical.case_code+' '+c.clinical.lesion_site).toLowerCase().includes($('history-search').value.toLowerCase())))}
}
let comparisonSerial=0;
async function renderComparison(c){
  const serial=++comparisonSerial,stage=document.querySelector('.image-comparison');if(!stage)return;
  stage.innerHTML='<div class="comparison-loading">正在准备关键点配准…</div>';
  const controls=document.createElement('div');controls.className='comparison-controls';
  controls.innerHTML='<label>对比方式<select id="comparison-mode"><option value="aligned">关键点配准</option><option value="points">配准＋关键点</option><option value="original">原始视角（未配准）</option></select></label><label>同步放大<input id="comparison-zoom" type="range" min="1" max="3" step="0.1" value="1"></label>';
  stage.parentNode.insertBefore(controls,stage);const mode=$('comparison-mode'),zoom=$('comparison-zoom'),note=$('comparison-note'),tag=$('comparison-status');
  mode.disabled=true;
  const fresh=()=>state.active?.id===c.id&&comparisonSerial===serial&&stage.isConnected;
  let meta=c.comparison||{status:'missing'};const urls={};let displaySerial=0;
  async function load(path){if(!urls[path]){const blob=await api(path,{blob:true});if(!fresh())return null;urls[path]=URL.createObjectURL(blob);resultBlobUrls.push(urls[path])}return urls[path]}
  async function display(){
    const rev=++displaySerial;const original=mode.value==='original';
    const paths=['before','after'].map(phase=>original?'/cases/'+c.id+'/images/'+encodeURIComponent(c.images[phase][0]):'/cases/'+c.id+'/comparison/images/'+phase+(mode.value==='points'?'_landmarks':'')+'.png');
    const [before,after]=await Promise.all(paths.map(load));if(!fresh()||rev!==displaySerial||!before||!after)return;
    stage.innerHTML=`<img id="compare-after" alt="治疗后${original?'原始视角':'配准图'}"><div class="comparison-before" id="comparison-before"><img id="compare-before" alt="治疗前${original?'原始视角':'参考图'}"></div><div class="comparison-divider" id="comparison-divider"></div><div class="comparison-labels"><span>BEFORE</span><span>AFTER</span></div>`;
    $('compare-before').src=before;$('compare-after').src=after;
    const update=()=>{$('comparison-before').style.clipPath=`inset(0 ${100-$('comparison-range').value}% 0 0)`;$('comparison-divider').style.left=$('comparison-range').value+'%';stage.querySelectorAll('img').forEach(im=>im.style.transform=`scale(${zoom.value})`)};
    $('comparison-range').oninput=update;zoom.oninput=update;update();
    tag.textContent=original?'未配准原始视角':'关键点配准已通过';
    note.textContent=original?(meta.status==='unreliable'?meta.message+' 当前为未配准原始视角。':'当前为未配准原始视角；切换关键点配准可统一前后缩放和位置。'):`以治疗前照片为参考，双眼中心对齐；治疗后缩放 ${meta.metrics.scale_after_to_before.toFixed(2)}×，旋转 ${meta.metrics.rotation_degrees.toFixed(1)}°。鼻部误差 ${(meta.metrics.nose_rms_interocular*100).toFixed(1)}% 眼距。仅平移、旋转、等比例缩放，不改变评分输入。`;
  }
  mode.onchange=()=>display().catch(e=>toast(e.message,true));
  try{
    if(meta.status==='missing')meta=await api('/cases/'+c.id+'/comparison',{method:'POST'});
    for(let i=0;meta.status==='queued'&&i<100;i++){
      if(!fresh())return;tag.textContent='正在配准';note.textContent='正在检测面部关键点并统一尺度；既有评分保持不变。';
      await new Promise(resolve=>setTimeout(resolve,1500));if(!fresh())return;meta=await api('/cases/'+c.id+'/comparison');
    }
    if(!fresh())return;
    mode.disabled=false;
    if(meta.status!=='ready'){
      mode.value='original';mode.querySelectorAll('option').forEach(o=>{if(o.value!=='original')o.disabled=true});
      if(meta.status!=='unreliable')meta={status:'unreliable',message:'配准尚未完成，可稍后重新打开记录查看。'};
    }
    await display();
  }catch(e){if(fresh()){tag.textContent='配准未完成';note.textContent=e.message;stage.innerHTML='<div class="comparison-loading">配准图暂不可用，请稍后重新打开记录。</div>'}}
}
