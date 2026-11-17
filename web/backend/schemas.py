from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from datetime import date,datetime
from typing import Literal

class ClinicalInfo(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    case_code:str=Field(default_factory=lambda:'PWS-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'),min_length=1,max_length=80)
    sex:Literal['female','male','unspecified']='unspecified'
    age:float|None=Field(default=None,ge=0,le=120)
    clinical_profile:Literal['source','chongqing','zhejiang','disabled']='source'
    lesion_site:str=Field(default='',max_length=200)
    skin_type:str=Field(default='',max_length=20)
    nerve_zone:str=Field(default='',max_length=50)
    phenotype:str=Field(default='',max_length=50)
    treatment_count:int|None=Field(default=None,ge=0,le=200)
    treatment_method:str=Field(default='',max_length=150)
    complications:list[str]=Field(default_factory=list,max_length=15)
    complication_notes:str=Field(default='',max_length=1500)
    before_date:date|None=None
    after_date:date|None=None
    photo_interval_days:int|None=Field(default=None,ge=0,le=36500)
    notes:str=Field(default='',max_length=3000)

    @field_validator('case_code',mode='before')
    @classmethod
    def default_case_code(cls,value):
        return value if value and str(value).strip() else 'PWS-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    @field_validator('complications')
    @classmethod
    def validate_complications(cls,values):
        if any(len(x)>100 for x in values):raise ValueError('Complication name too long')
        if '无' in values and len(values)>1:raise ValueError('不能同时选择“无”和其他并发症')
        return list(dict.fromkeys(values))

    @model_validator(mode='after')
    def dates_valid(self):
        if self.before_date and self.after_date and self.after_date<self.before_date:
            raise ValueError('治疗后日期不能早于治疗前日期')
        return self

class CandidateConfirmation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    fields:list[Literal['phenotype','nerve_zone']]=Field(min_length=1,max_length=2)
    profile_id:Literal['source','chongqing','zhejiang']

class ExcelExportRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scope:Literal['all','selected']='selected'
    case_ids:list[str]=Field(default_factory=list,max_length=10000)
    @model_validator(mode='after')
    def selection_valid(self):
        if self.scope=='selected' and not self.case_ids:raise ValueError('请勾选已完成的记录')
        if self.scope=='all' and self.case_ids:raise ValueError('全部导出不应指定记录')
        return self
