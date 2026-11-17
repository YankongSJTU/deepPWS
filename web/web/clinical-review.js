/* Doctor edits are explicit; original manual inputs and model provenance stay separate. */
function renderClinicalReview(c){
  const r=c.result,values=r.effective_clinical||c.clinical,sources=r.clinical_value_sources||{},est=r.clinical_estimates;
  const panel=document.createElement('section');panel.className='panel clinical-suggestions clinical-review';
  const note=k=>{const source=sources[k]||([null,undefined,'','unspecified'].includes(c.clinical[k])?'missing':'manual');return source==='model_estimate'?'模型估计 · 待核对':source==='manual'?'医生填写':'未填写'};
  const input=(key,title,type='text',extra='')=>`<label class="field"><span>${title}</span><input name="${key}" type="${type}" value="${escapeHtml(values[key]??'')}" ${extra}></label>`;
  const option=(v,label,selected)=>`<option value="${escapeHtml(v)}" ${v===selected?'selected':''}>${escapeHtml(label)}</option>`;
  const complications=[...new Set([...$('complication-options').querySelectorAll('input')].map(el=>el.value).concat(values.complications||[]))];
  panel.innerHTML=`<div class="result-panel-title"><span>临床信息核对与修正</span><span class="tag tag-neutral">医生可修改</span></div>
    <div class="estimate-summary"><div><small>本次使用年龄 · ${note('age')}</small><b>${values.age===null||values.age===undefined?'未记录':escapeHtml(values.age)+' 岁'}</b></div><div><small>本次使用 Fitzpatrick · ${note('skin_type')}</small><b>${escapeHtml(values.skin_type)||'未记录'}</b></div><div><small>性别 · 仅手填</small><b>${escapeHtml(sexNames[values.sex]||'未记录')}</b></div></div>
    <p class="comparison-note">医生手填值优先。留空时，年龄与 Fitzpatrick III/IV 标签使用照片估计；年龄不是实足年龄，III/IV 候选不能替代晒伤/晒黑反应评估。${est?'估计值已参与本次评分，核对后可直接修正。':'这份旧版报告尚未使用新增估计；保存核对资料或重新分析后更新评分。'}性别、并发症与拍照间隔不从照片预测。</p>
    ${!est?'<p class="comparison-note">这是旧版报告，重新分析后才能生成年龄与 III/IV 标签估计。</p>':''}
    <form id="clinical-review-form"><div class="form-grid three-col">
      ${input('age','年龄（岁）','number','min="0" max="120" step="0.1"')}
      <label class="field"><span>性别（可留空）</span><select name="sex">${[['unspecified','未记录'],['female','女'],['male','男']].map(([v,n])=>option(v,n,values.sex)).join('')}</select></label>
      <label class="field"><span>Fitzpatrick（医生核对）</span><select name="skin_type">${['','I','II','III','IV','V','VI'].map(v=>option(v,v||'留空，使用 III/IV 标签估计',values.skin_type)).join('')}</select></label>
      ${input('lesion_site','病灶部位','text','maxlength="200"')}
      <label class="field"><span>三叉神经分区</span><select name="nerve_zone">${[...new Set(['',...state.model.clinical_fields.find(f=>f.key==='nerve_zone').categories.filter(v=>v!=='missing'),values.nerve_zone||''])].map(v=>option(v,v||'未记录',values.nerve_zone)).join('')}</select></label>
      <label class="field"><span>病灶颜色分型</span><select name="phenotype">${[...new Set(['',...state.model.clinical_fields.find(f=>f.key==='phenotype').categories.filter(v=>v!=='missing'),values.phenotype||''])].map(v=>option(v,v||'未记录',values.phenotype)).join('')}</select></label>
      ${input('photo_interval_days','手填拍照间隔（天，可留空）','number','min="0" max="36500" step="1"')}
      ${input('before_date','手填治疗前照片日期','date')}${input('after_date','手填治疗后照片日期','date')}
    </div><p class="comparison-note">拍照间隔：${r.photo_interval_days===null||r.photo_interval_days===undefined?'未填写':escapeHtml(r.photo_interval_days)+' 天'}${r.photo_interval_source==='manual_dates'?'（由医生填写的日期计算）':''}；仅记录，不影响评分。</p>
    <div class="field-label">并发症（医生手填，可不选）</div><div class="chip-group">${complications.map(v=>`<label class="check-chip"><input type="checkbox" name="review_complications" value="${escapeHtml(v)}" ${(values.complications||[]).includes(v)?'checked':''}><span>${escapeHtml(v)}</span></label>`).join('')}</div>
    <label class="field notes-field"><span>并发症说明（选填）</span><textarea name="complication_notes" rows="2" maxlength="1500">${escapeHtml(values.complication_notes||'')}</textarea></label>
    <div class="candidate-confirmation"><small class="candidate-help">点击确认后，上方保留的估计值视为医生已核对；留空字段仍使用估计。评分输入变更时复用本机特征；新版缓存可用时无需重新进行三维分析。</small><button type="submit" class="button button-primary">确认／修正资料并更新评分</button></div></form>`;
  $('result-content').insertBefore(panel,$('result-content').querySelector('.result-grid'));
  const form=panel.querySelector('form');
  form.onsubmit=async e=>{
    e.preventDefault();if(!form.reportValidity())return;
    const data=new FormData(form),edited={...c.clinical};
    ['age','photo_interval_days'].forEach(k=>edited[k]=data.get(k)===''?null:Number(data.get(k)));
    ['sex','skin_type','lesion_site','nerve_zone','phenotype','complication_notes'].forEach(k=>edited[k]=data.get(k));
    ['before_date','after_date'].forEach(k=>edited[k]=data.get(k)||null);
    edited.complications=data.getAll('review_complications');
    if(edited.complications.includes('无')&&edited.complications.length>1){toast('不能同时选择“无”和其他并发症',true);return}
    const button=form.querySelector('button[type="submit"]');button.disabled=true;
    try{
      const updated=await api('/cases/'+c.id+'/clinical',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(edited)});
      if(updated.status!=='completed')await api('/cases/'+c.id+'/analyze',{method:'POST'});
      await openCase(c.id);await refreshHistory(false);toast('已保存医生核对资料，评分将使用本机缓存更新。');
    }catch(err){toast(err.message,true);button.disabled=false}
  };
}
