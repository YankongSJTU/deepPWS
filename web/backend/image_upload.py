"""Keep original large photos and prepare bounded, correctly oriented analysis copies."""
from pathlib import Path
import io
from PIL import Image,ImageOps,UnidentifiedImageError
from fastapi import HTTPException
MAX_BYTES=100*1024*1024
MAX_UPLOAD_PIXELS=300_000_000
RESIZE_ABOVE_PIXELS=30_000_000
ANALYSIS_LONG_EDGE=4096
Image.MAX_IMAGE_PIXELS=MAX_UPLOAD_PIXELS

def prepare_image(content:bytes,folder:Path,name:str):
    try:
        with Image.open(io.BytesIO(content)) as im:
            if im.format not in ['JPEG','PNG','WEBP']:raise HTTPException(422,'仅支持 JPG、PNG、WebP 照片')
            original_size=im.size;fmt=im.format
            if min(original_size)<64:raise HTTPException(422,'照片宽高须至少 64 像素')
            if im.width*im.height>MAX_UPLOAD_PIXELS:raise HTTPException(422,'照片超过 3 亿像素，请先缩小后上传')
            im.verify()
        resized=original_size[0]*original_size[1]>RESIZE_ABOVE_PIXELS
        extension={'JPEG':'jpg','PNG':'png','WEBP':'webp'}[fmt];filename=name+'.'+extension
        with Image.open(io.BytesIO(content)) as im:
            if resized:
                # JPEG decoder subsampling avoids allocating the entire high-resolution raster.
                if fmt=='JPEG':im.draft('RGB',(ANALYSIS_LONG_EDGE,ANALYSIS_LONG_EDGE))
                image=ImageOps.exif_transpose(im)
                image.thumbnail((ANALYSIS_LONG_EDGE,ANALYSIS_LONG_EDGE),Image.Resampling.LANCZOS)
                if fmt=='JPEG':
                    image=image.convert('RGB');options={'quality':95,'subsampling':0}
                elif fmt=='PNG':options={'compress_level':3}
                else:options={'quality':95}
                if im.info.get('icc_profile'):options['icc_profile']=im.info['icc_profile']
                image.save(folder/filename,format=fmt,**options);stored_size=image.size
                originals=folder/'originals';originals.mkdir(exist_ok=True)
                (originals/filename).write_bytes(content)
            else:
                im.load();(folder/filename).write_bytes(content);stored_size=original_size
    except HTTPException:raise
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
        raise HTTPException(422,'照片文件无法读取，请重新导出后上传')
    return filename,{'image':filename,'original_size':list(original_size),'analysis_size':list(stored_size),'resized':resized,'original_retained':resized}
