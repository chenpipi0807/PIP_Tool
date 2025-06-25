import os
import requests
import numpy as np
import hashlib
import torch
from PIL import Image
from io import BytesIO

class PIP_LoadImageURL:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
        self.cache_dir = os.path.join(self.output_dir, "url_images_cache")
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
            },
            "optional": {
                "refresh": ("BUTTON", {"label": "刷新图片"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "load_image"
    CATEGORY = "PIP工具/图像处理"
    
    def load_image(self, url, refresh=False):
        try:
            # 为URL创建一个唯一的文件名
            url_hash = hashlib.md5(url.encode()).hexdigest()
            file_extension = self._get_extension_from_url(url)
            cache_filename = os.path.join(self.cache_dir, f"{url_hash}{file_extension}")
            
            # 如果缓存不存在或者要求刷新，则下载图片
            if not os.path.exists(cache_filename) or refresh:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # 如果请求不成功则抛出异常
                
                # 保存到缓存
                with open(cache_filename, 'wb') as f:
                    f.write(response.content)
                
                # 从响应内容直接加载图片
                img = Image.open(BytesIO(response.content))
            else:
                # 从缓存加载图片
                img = Image.open(cache_filename)
            
            # 转换为ComfyUI使用的格式 (RGBA -> RGB)
            if img.mode == 'RGBA':
                img = self._alpha_to_white(img)
            
            # 确保是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为ComfyUI期望的格式 (PyTorch tensor)
            img_np = np.array(img).astype(np.float32) / 255.0
            # 转换为 BCHW 格式并转为 PyTorch tensor
            tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return (tensor,)
            
        except Exception as e:
            print(f"加载URL图像时出错: {e}")
            # 返回一个1x1的红色错误图像
            error_img = np.ones((1, 1, 1, 3), dtype=np.float32)
            error_img[0, 0, 0] = [1.0, 0.0, 0.0]  # 红色
            # 转换为PyTorch tensor
            error_tensor = torch.from_numpy(error_img)
            return (error_tensor,)
    
    def _get_extension_from_url(self, url):
        """从URL获取文件扩展名"""
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']
        url_lower = url.lower()
        
        for ext in extensions:
            if url_lower.endswith(ext):
                return ext
        
        # 如果没有匹配的扩展名，默认使用.jpg
        return '.jpg'
    
    def _alpha_to_white(self, image):
        """将透明背景转换为白色背景"""
        if image.mode != 'RGBA':
            return image
            
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 是alpha通道
        return background
