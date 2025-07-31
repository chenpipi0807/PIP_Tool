import os
import glob
from PIL import Image
import numpy as np
import torch


class PIP_LoadLocalImage:
    """
    PIP 本地图像加载节点
    
    功能:
    - 扫描指定文件夹中的图像文件
    - 按文件名A-Z排序
    - 使用seed值选择图像，支持ComfyUI内置的seed控制
    - 输出当前计数、图片名称和图片内容
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "C:/path/to/images"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999
                })
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"}
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("current_count", "image_name", "image")
    FUNCTION = "load_local_image"
    CATEGORY = "PIP_Tool"
    
    def load_local_image(self, folder_path, seed, prompt=None, extra_pnginfo=None, my_unique_id=None):
        """
        从本地路径加载图像，使用seed值选择图像
        """
        try:
            # 获取所有支持的图像文件
            image_files = self._get_image_files(folder_path)
            
            if not image_files:
                # 如果没有找到图像文件，返回默认值
                print(f"[PIP_LoadLocalImage] 警告: 在路径 '{folder_path}' 中未找到图像文件")
                return ("0/0", "无图像", self._create_default_image())
            
            # 按文件名A-Z排序
            image_files.sort()
            total_count = len(image_files)
            
            # 直接使用seed值作为索引，超过范围时取模
            current_index = seed % total_count
            print(f"[PIP_LoadLocalImage] 使用seed: {seed}, 图片总数: {total_count}, 选中索引: {current_index}")
            current_file = image_files[current_index]
            
            # 获取文件名（不含路径）
            image_name = os.path.basename(current_file)
            
            # 生成计数字符串
            current_count = f"{current_index + 1}/{total_count}"
            
            # 加载图像
            image = self._load_image_file(current_file)
            
            print(f"[PIP_LoadLocalImage] 文件夹路径: {folder_path}")
            print(f"[PIP_LoadLocalImage] 总图像数量: {total_count}")
            print(f"[PIP_LoadLocalImage] 当前计数: {current_count}")
            print(f"[PIP_LoadLocalImage] 图像名称: {image_name}")
            print(f"[PIP_LoadLocalImage] 图像尺寸: {image.shape}")
            
            return (current_count, image_name, image)
            
        except Exception as e:
            print(f"[PIP_LoadLocalImage] 错误: {str(e)}")
            return ("错误", "加载失败", self._create_default_image())
    

    def _get_image_files(self, folder_path):
        """
        获取文件夹中所有支持的图像文件
        """
        if not os.path.exists(folder_path):
            print(f"[PIP_LoadLocalImage] 错误: 路径不存在 '{folder_path}'")
            return []
        
        if not os.path.isdir(folder_path):
            print(f"[PIP_LoadLocalImage] 错误: 不是有效的文件夹路径 '{folder_path}'")
            return []
        
        # 支持的图像格式
        supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
        image_files = []
        
        # 搜索所有支持的格式（不区分大小写）
        for pattern in supported_formats:
            # 小写
            files = glob.glob(os.path.join(folder_path, pattern))
            image_files.extend(files)
            # 大写
            files = glob.glob(os.path.join(folder_path, pattern.upper()))
            image_files.extend(files)
        
        # 去重（因为可能有重复）
        image_files = list(set(image_files))
        
        print(f"[PIP_LoadLocalImage] 找到 {len(image_files)} 个图像文件")
        return image_files
    
    def _load_image_file(self, file_path):
        """
        加载图像文件并转换为ComfyUI格式
        """
        try:
            # 使用PIL加载图像
            pil_image = Image.open(file_path)
            
            # 转换为RGB模式（确保3通道）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            
            # 转换为torch tensor并添加batch维度
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            print(f"[PIP_LoadLocalImage] 加载图像失败 '{file_path}': {str(e)}")
            return self._create_default_image()
    
    def _create_default_image(self):
        """
        创建默认的错误图像（纯黑色512x512）
        """
        # 创建512x512的黑色图像
        default_image = np.zeros((512, 512, 3), dtype=np.float32)
        image_tensor = torch.from_numpy(default_image).unsqueeze(0)
        return image_tensor
