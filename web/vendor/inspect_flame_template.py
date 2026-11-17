"""Read only numeric topology from the trusted local FLAME asset.

Unused chumpy expression graphs are inert placeholders, not evaluated.
"""
import pickle
from pathlib import Path
import numpy as np
BASE=Path(__import__('os').environ['FACEBA_RUNTIME_ROOT'])
class Inert:
    def __new__(cls,*args,**kwargs):return object.__new__(cls)
    def __setstate__(self,state):self.state=state
class NumericAssetUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('chumpy'):return Inert
        return super().find_class(module,name)
def template():
    with (BASE/'model_zoo/human_parametric_models/flame_vhap/flame2023.pkl').open('rb') as f:z=NumericAssetUnpickler(f,encoding='latin1').load()
    v=z['v_template'];faces=z['f']
    assert isinstance(v,np.ndarray) and isinstance(faces,np.ndarray)
    assert v.ndim==2 and v.shape[1]==3 and faces.ndim==2 and faces.shape[1]==3
    return v.astype(float),faces.astype(int)
if __name__=='__main__':
    v,f=template();e=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    print('template',v.shape,'faces',f.shape,'edges',e.shape,'subdivision points',len(v)+len(e));print('bounds',v.min(0),v.max(0))
