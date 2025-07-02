import torch
import numpy as np

class PIP_RGBA_to_RGB:
    """
    RGBA转RGB节点
    将4通道的RGBA图像转换为3通道的RGB图像
    支持多种背景混合模式
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background_mode": (["white", "black", "transparent_as_white", "custom"], {
                    "default": "white"
                }),
            },
            "optional": {
                "custom_color_r": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "custom_color_g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "custom_color_b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)
    FUNCTION = "convert_rgba_to_rgb"
    CATEGORY = "图像处理"
    
    def convert_rgba_to_rgb(self, image, background_mode="white", 
                           custom_color_r=1.0, custom_color_g=1.0, custom_color_b=1.0):
        """
        将RGBA图像转换为RGB图像
        
        Args:
            image: 输入图像tensor (batch, height, width, channels)
            background_mode: 背景处理模式
            custom_color_r/g/b: 自定义背景颜色的RGB值
        
        Returns:
            转换后的RGB图像tensor
        """
        # 确保输入是4维tensor
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        batch_size, height, width, channels = image.shape
        
        # 如果已经是RGB图像（3通道），直接返回
        if channels == 3:
            print("输入图像已经是RGB格式，无需转换")
            return (image,)
        
        # 如果不是RGBA图像（4通道），报错
        if channels != 4:
            raise ValueError(f"不支持的图像格式，通道数: {channels}。仅支持RGB(3通道)和RGBA(4通道)")
        
        print(f"检测到RGBA图像，正在转换为RGB...")
        
        # 分离RGBA通道
        rgb_channels = image[:, :, :, :3]  # RGB通道
        alpha_channel = image[:, :, :, 3:4]  # Alpha通道
        
        # 根据背景模式处理
        if background_mode == "white":
            background_color = torch.ones_like(rgb_channels)
        elif background_mode == "black":
            background_color = torch.zeros_like(rgb_channels)
        elif background_mode == "transparent_as_white":
            # 透明区域作为白色，不做Alpha混合
            result = rgb_channels.clone()
            # 将完全透明的像素设为白色
            transparent_mask = alpha_channel < 0.01
            result[transparent_mask.expand(-1, -1, -1, 3)] = 1.0
            print(f"转换完成: RGBA -> RGB (透明区域作为白色)")
            return (result,)
        elif background_mode == "custom":
            background_color = torch.zeros_like(rgb_channels)
            background_color[:, :, :, 0] = custom_color_r  # R
            background_color[:, :, :, 1] = custom_color_g  # G
            background_color[:, :, :, 2] = custom_color_b  # B
        else:
            raise ValueError(f"不支持的背景模式: {background_mode}")
        
        # Alpha混合: result = foreground * alpha + background * (1 - alpha)
        alpha_expanded = alpha_channel.expand(-1, -1, -1, 3)
        result = rgb_channels * alpha_expanded + background_color * (1 - alpha_expanded)
        
        # 确保值在[0,1]范围内
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"转换完成: RGBA -> RGB (背景模式: {background_mode})")
        return (result,)


# 为了与PIP_Tool的命名规范保持一致
class PIP_RGBAtoRGB:
    """PIP_RGBA_to_RGB的别名，保持命名一致性"""
    
    def __init__(self):
        self.converter = PIP_RGBA_to_RGB()
    
    @classmethod
    def INPUT_TYPES(cls):
        return PIP_RGBA_to_RGB.INPUT_TYPES()
    
    RETURN_TYPES = PIP_RGBA_to_RGB.RETURN_TYPES
    RETURN_NAMES = PIP_RGBA_to_RGB.RETURN_NAMES
    FUNCTION = "convert_rgba_to_rgb"
    CATEGORY = PIP_RGBA_to_RGB.CATEGORY
    
    def convert_rgba_to_rgb(self, *args, **kwargs):
        return self.converter.convert_rgba_to_rgb(*args, **kwargs)
