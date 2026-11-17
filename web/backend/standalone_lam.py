"""Torch-only canonical reconstruction with all computation graphs stored locally."""
from pathlib import Path
import gc,json,hashlib
import numpy as np
from PIL import Image
import torch

def load_graph(path,device):
    path=Path(path)
    # Torch's Windows C++ filename API does not consistently support Unicode.
    # Python file handles preserve arbitrary user directory names.
    if not str(path).isascii():
        with path.open('rb') as stream:return torch.jit.load(stream,map_location=device).eval()
    return torch.jit.load(str(path),map_location=device).eval()

def reconstruct(paths,modelroot,device,progress=None):
    modelroot=Path(modelroot)
    encoder=load_graph(modelroot/'lam_encoder.pt',device)
    features=[]
    with torch.inference_mode():
        for index,path in enumerate(paths):
            with Image.open(path) as im:
                rgb=np.asarray(im.convert('RGB').resize((518,518)),np.float32)/255.
            image=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)
            features.append(encoder(image).cpu())
            if progress:progress('正在提取三维影像表征',55+index*5)
        del encoder,image;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        decoder=load_graph(modelroot/'lam_decoder.pt',device)
        output=[]
        for index,feature in enumerate(features):
            vertices,colors=decoder(feature.to(device))
            output.append((vertices.cpu().numpy(),colors.cpu().numpy()))
            if progress:progress('正在生成三维几何与纹理',67+index*8)
        del decoder;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    return output
