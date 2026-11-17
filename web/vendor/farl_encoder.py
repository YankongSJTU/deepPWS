"""Offline frozen FaRL ViT-B/16 image encoder, without downloading base CLIP.

Implements the public OpenAI-compatible visual state dictionary. Real weights
are required; incompatible/missing weights never fall back to random features.
"""
from pathlib import Path
import hashlib,math
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F

def checkpoint_state(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    state=value.get('state_dict',value)
    clean={}
    for key,value in state.items():
        key=key.removeprefix('module.')
        if key.startswith('visual.') and torch.is_tensor(value):clean[key[7:]]=value
    if not clean:raise ValueError('No OpenAI-compatible visual state in the FaRL checkpoint')
    return clean

class FaRLVision:
    def __init__(self,path,device='cpu'):
        self.path=Path(path)
        if not self.path.is_file():raise FileNotFoundError('FaRL weights missing: '+str(path))
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._setup(checkpoint_state(self.path),device,True)

    def _setup(self,state,device,official_architecture):
        self.device=device
        self.w={key:value.detach().float().to(device) for key,value in state.items()}
        self.width,self.patch=self.w['conv1.weight'].shape[0],self.w['conv1.weight'].shape[-1]
        self.grid=math.isqrt(self.w['positional_embedding'].shape[0]-1)
        assert self.grid**2+1==len(self.w['positional_embedding'])
        layers={int(key.split('.')[2]) for key in self.w if key.startswith('transformer.resblocks.')}
        self.layers=len(layers)
        assert layers==set(range(self.layers))
        self.heads=self.width//64
        self.output_dim=self.w['proj'].shape[1]
        if official_architecture and (self.width,self.patch,self.grid,self.layers,self.output_dim)!=(768,16,14,12,512):
            raise ValueError('Expected the official FaRL ViT-B/16 visual architecture')
        required=['class_embedding','conv1.weight','positional_embedding','ln_pre.weight','ln_pre.bias','ln_post.weight','ln_post.bias','proj']
        for i in range(self.layers):
            prefix=f'transformer.resblocks.{i}.'
            required += [prefix+key for key in ['ln_1.weight','ln_1.bias','ln_2.weight','ln_2.bias','attn.in_proj_weight','attn.in_proj_bias','attn.out_proj.weight','attn.out_proj.bias','mlp.c_fc.weight','mlp.c_fc.bias','mlp.c_proj.weight','mlp.c_proj.bias']]
        missing=set(required)-set(self.w)
        if missing:raise ValueError('Missing visual weights: '+','.join(sorted(missing)))

    def layernorm(self,x,prefix):
        return F.layer_norm(x,(self.width,),self.w[prefix+'.weight'],self.w[prefix+'.bias'],eps=1e-5)

    def preprocess(self,image):
        image=ImageOps.exif_transpose(image).convert('RGB')
        size=self.patch*self.grid
        scale=size/min(image.size)
        image=image.resize((int(image.width*scale),int(image.height*scale)),Image.Resampling.BICUBIC)
        x,y=(image.width-size)//2,(image.height-size)//2
        array=np.asarray(image.crop((x,y,x+size,y+size)),dtype=np.float32)/255
        array=(array-np.array([.48145466,.4578275,.40821073],np.float32))/np.array([.26862954,.26130258,.27577711],np.float32)
        return torch.from_numpy(array.transpose(2,0,1).copy())

    @torch.inference_mode()
    def encode_tensor(self,x):
        x=x.to(self.device)
        x=F.conv2d(x,self.w['conv1.weight'],stride=self.patch).flatten(2).transpose(1,2)
        token=self.w['class_embedding'].expand(len(x),1,-1)
        x=torch.cat([token,x],1)+self.w['positional_embedding']
        x=self.layernorm(x,'ln_pre')
        for i in range(self.layers):
            prefix=f'transformer.resblocks.{i}.'
            h=self.layernorm(x,prefix+'ln_1')
            qkv=F.linear(h,self.w[prefix+'attn.in_proj_weight'],self.w[prefix+'attn.in_proj_bias'])
            q,k,v=[part.reshape(len(x),len(x[0]),self.heads,-1).transpose(1,2) for part in qkv.chunk(3,-1)]
            h=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)
            x=x+F.linear(h,self.w[prefix+'attn.out_proj.weight'],self.w[prefix+'attn.out_proj.bias'])
            h=F.linear(self.layernorm(x,prefix+'ln_2'),self.w[prefix+'mlp.c_fc.weight'],self.w[prefix+'mlp.c_fc.bias'])
            h=h*torch.sigmoid(1.702*h)
            x=x+F.linear(h,self.w[prefix+'mlp.c_proj.weight'],self.w[prefix+'mlp.c_proj.bias'])
        global_feature=self.layernorm(x[:,0],'ln_post')@self.w['proj']
        tokens=x[:,1:].transpose(1,2).reshape(len(x),self.width,self.grid,self.grid)
        local=F.adaptive_avg_pool2d(tokens,(4,4)).flatten(2).transpose(1,2)
        return global_feature.cpu().numpy(),local.cpu().numpy()

    def __call__(self,images):
        return self.encode_tensor(torch.stack([self.preprocess(image) for image in images]))
