import torch
import numpy as np
from PIL import Image

class PIP_EdgeExpand:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["上", "下", "左", "右", "上下", "左右"],),
                "expand_pixels": ("INT", {
                    "default": 400,
                    "min": 1,
                    "max": 2000,
                    "step": 1
                }),
                "expand_color": (["黑", "白", "灰"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "expand_image"
    CATEGORY = "PIP Tool"
    
    def expand_image(self, image, direction, expand_pixels, expand_color):
        # 确保expand_pixels是整数类型
        expand_pixels = int(expand_pixels)
        
        # 将tensor转换为numpy数组
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = image
        
        # 获取图像尺寸
        height, width = image_np.shape[:2]
        
        # 确定颜色值
        if expand_color == "黑":
            fill_color = 0.0
        elif expand_color == "白":
            fill_color = 1.0
        else:  # 灰
            fill_color = 0.5
        
        # 根据方向计算扩展参数
        if direction == "上":
            new_height = height + expand_pixels
            new_width = width
            # 创建新图像
            if len(image_np.shape) == 3:
                expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_color, dtype=image_np.dtype)
                expanded_image[expand_pixels:, :, :] = image_np
            else:
                expanded_image = np.full((new_height, new_width), fill_color, dtype=image_np.dtype)
                expanded_image[expand_pixels:, :] = image_np
                
        elif direction == "下":
            new_height = height + expand_pixels
            new_width = width
            if len(image_np.shape) == 3:
                expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_color, dtype=image_np.dtype)
                expanded_image[:height, :, :] = image_np
            else:
                expanded_image = np.full((new_height, new_width), fill_color, dtype=image_np.dtype)
                expanded_image[:height, :] = image_np
                
        elif direction == "左":
            new_height = height
            new_width = width + expand_pixels
            if len(image_np.shape) == 3:
                expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_color, dtype=image_np.dtype)
                expanded_image[:, expand_pixels:, :] = image_np
            else:
                expanded_image = np.full((new_height, new_width), fill_color, dtype=image_np.dtype)
                expanded_image[:, expand_pixels:] = image_np
                
        elif direction == "右":
            new_height = height
            new_width = width + expand_pixels
            if len(image_np.shape) == 3:
                expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_color, dtype=image_np.dtype)
                expanded_image[:, :width, :] = image_np
            else:
                expanded_image = np.full((new_height, new_width), fill_color, dtype=image_np.dtype)
                expanded_image[:, :width] = image_np
                
        elif direction == "上下":
            # 上下各扩一半
            expand_half = expand_pixels // 2
            new_height = height + expand_pixels
            new_width = width
            if len(image_np.shape) == 3:
                expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_color, dtype=image_np.dtype)
                expanded_image[expand_half:expand_half + height, :, :] = image_np
            else:
                expanded_image = np.full((new_height, new_width), fill_color, dtype=image_np.dtype)
                expanded_image[expand_half:expand_half + height, :] = image_np
                
        elif direction == "左右":
            # 左右各扩一半
            expand_half = expand_pixels // 2
            new_height = height
            new_width = width + expand_pixels
            if len(image_np.shape) == 3:
                expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_color, dtype=image_np.dtype)
                expanded_image[:, expand_half:expand_half + width, :] = image_np
            else:
                expanded_image = np.full((new_height, new_width), fill_color, dtype=image_np.dtype)
                expanded_image[:, expand_half:expand_half + width] = image_np
        
        # 确保输出是3D数组 (batch, height, width, channels)
        if len(expanded_image.shape) == 2:
            # 如果是灰度图，添加通道维度
            expanded_image = np.expand_dims(expanded_image, axis=-1)
            expanded_image = np.repeat(expanded_image, 3, axis=-1)
        
        # 添加batch维度
        if len(expanded_image.shape) == 3:
            expanded_image = np.expand_dims(expanded_image, axis=0)
        
        # 转换为tensor
        result_tensor = torch.from_numpy(expanded_image.astype(np.float32))
        
        print(f"PIP边缘扩图: 方向={direction}, 像素={expand_pixels}, 颜色={expand_color}")
        print(f"原始尺寸: {width}x{height}, 扩展后尺寸: {result_tensor.shape[2]}x{result_tensor.shape[1]}")
        
        return (result_tensor,)

# 导出节点类
NODE_CLASS_MAPPINGS = {
    "PIP_EdgeExpand": PIP_EdgeExpand
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_EdgeExpand": "PIP 边缘扩图"
}
