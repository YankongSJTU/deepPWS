"""Offline XLSX export using standard-library OOXML; text never becomes formulas."""
import io,re,zipfile,math
from xml.sax.saxutils import escape
from datetime import datetime,timezone

FIELDS=[('病例编号','case_code'),('性别（手填）','sex'),('年龄原始输入','age'),('Fitzpatrick原始输入','skin_type'),('病灶部位','lesion_site'),('三叉神经分区','nerve_zone'),('颜色分型','phenotype'),('治疗次数','treatment_count'),('治疗方式','treatment_method'),('并发症（手填）','complications'),('并发症说明','complication_notes'),('治疗前照片日期','before_date'),('治疗后照片日期','after_date'),('拍照间隔原始输入（天）','photo_interval_days'),('临床备注','notes')]
SOURCES={'manual':'医生填写','model_estimate':'模型估计，待核对','missing':'未填写'}
def rows(cases,age_median):
    headers=['记录ID','创建时间','分析完成时间']+[n for n,k in FIELDS]+['本次使用年龄','年龄来源','本次使用Fitzpatrick','Fitzpatrick来源','照片年龄估计','III/IV标签估计','颜色评分','面积评分','综合评分','临床候选配置','颜色候选','神经分区候选','区域候选','已确认候选字段','拍照间隔（天）','间隔来源','评分模型版本','应用版本','分析提示']
    data=[headers]
    for case in cases:
        c=case['clinical'];r=case['result'];effective=r.get('effective_clinical',c);sources=r.get('clinical_value_sources',{});est=r.get('clinical_estimates') or {};s=r.get('clinical_suggestions',{})
        vals=[]
        for label,k in FIELDS:
            value=c.get(k)
            if k=='sex':value={'female':'女','male':'男','unspecified':'未记录'}.get(value,'未记录')
            if isinstance(value,list):value='、'.join(value)
            vals.append(value)
        def source(k):
            if k=='age' and effective.get('age') is None and r.get('age_imputed'):return '训练集中位数填补，非照片预测'
            return SOURCES.get(sources.get(k), '医生填写' if c.get(k) is not None and c.get(k)!='' else '未填写')
        def selected(k):return '、'.join(s.get(k,{}).get('selected',[]))
        used_age=age_median if effective.get('age') is None and r.get('age_imputed') else effective.get('age')
        data.append([case['id'],case['created'],r.get('completed_at'),*vals,used_age,source('age'),effective.get('skin_type'),source('skin_type'),est.get('age'),est.get('skin_type'),r['scores']['colour'],r['scores']['area'],r['scores']['overall'],s.get('profile_name',c.get('clinical_profile','source')),selected('colour'),selected('nerve'),selected('regions'),'、'.join(s.get('confirmed_fields',[])),r.get('photo_interval_days'),{'manual_days':'医生手填','manual_dates':'由手填日期计算','missing':'未填写'}.get(r.get('photo_interval_source'),'未填写'),r.get('model_version'),r.get('app_version'),'；'.join(r.get('quality',{}).get('warnings',[]))])
        comp=case.get('comparison',{});q=r.get('quality',{})
        data[-1].extend([selected('side'),json_scores(s),comp.get('status','missing'),comp.get('metrics',{}).get('stable_landmark_rms_interocular'),len(case.get('images',{}).get('before',[])),len(case.get('images',{}).get('after',[]))])
    data[0].extend(['左右范围候选','临床候选模型分数（JSON）','显示配准状态','关键点误差（眼距比例）','治疗前照片数量','治疗后照片数量'])
    assert all(len(row)==len(data[0]) for row in data)
    return data

def json_scores(s):
    import json
    return json.dumps({key:s[key]['scores'] for key in ['colour','side','nerve','regions'] if key in s},ensure_ascii=False)

def column(n):
    out=''
    while n:n,k=divmod(n-1,26);out=chr(65+k)+out
    return out
def text(v):return escape(re.sub('[\x00-\x08\x0b\x0c\x0e-\x1f]','',str(v)))
def sheet(table):
    body=[]
    for i,row in enumerate(table,1):
        cells=[]
        for j,value in enumerate(row,1):
            if value is None:continue
            ref=f'{column(j)}{i}'
            if i>1 and isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):cells.append(f'<c r="{ref}" s="2"><v>{value}</v></c>')
            else:cells.append(f'<c r="{ref}" t="inlineStr" s="{1 if i==1 else 0}"><is><t xml:space="preserve">{text(value)}</t></is></c>')
        body.append(f'<row r="{i}">'+''.join(cells)+'</row>')
    end=f'{column(len(table[0]))}{len(table)}'
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:'+end+'"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="'+str(len(table[0]))+'" width="23" customWidth="1"/></cols><sheetData>'+''.join(body)+'</sheetData><autoFilter ref="A1:'+end+'"/></worksheet>'
def workbook(cases,scope,version,age_median):
    about=[['项目','说明'],['产品','DeepPWS'],['导出范围','全部已完成记录' if scope=='all' else '用户勾选的已完成记录'],['记录数量',len(cases)],['导出时间UTC',datetime.now(timezone.utc).isoformat()],['应用版本',version],['预测值来源','原始医生输入与本次使用值分列；模型估计并非已确认的病历事实。'],['Fitzpatrick','照片仅估计既有 III/IV 标签，需结合晒伤/晒黑反应核对。'],['手动字段','性别、并发症和拍照间隔不由照片预测。'],['评分','保留原始浮点数；Excel默认显示两位小数。'],['影像','不导出照片或影像缓存。']]
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="分析记录" sheetId="1" r:id="rId1"/><sheet name="导出说明" sheetId="2" r:id="rId2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml','<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF254C40"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/><xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        z.writestr('xl/worksheets/sheet1.xml',sheet(rows(cases,age_median)));z.writestr('xl/worksheets/sheet2.xml',sheet(about))
    return b.getvalue()
