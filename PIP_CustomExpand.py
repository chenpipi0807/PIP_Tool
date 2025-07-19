import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

class PIP_CustomExpand:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_top": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "expand_bottom": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "expand_left": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "expand_right": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "fill_color": (["黑", "白", "灰"],),
                "mask_invert": ("BOOLEAN", {
                    "default": False
                }),
                "mask_expand": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "mask_feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("expanded_image", "expansion_mask")
    FUNCTION = "custom_expand"
    CATEGORY = "PIP Tool"
    
    def custom_expand(self, image, expand_top, expand_bottom, expand_left, expand_right, fill_color, mask_invert, mask_expand, mask_feather):
        # 确保所有参数都是整数类型
        expand_top = int(expand_top)
        expand_bottom = int(expand_bottom)
        expand_left = int(expand_left)
        expand_right = int(expand_right)
        
        # 将tensor转换为numpy数组
        if isinstance(image, torch.Tensor):
            # 处理批次维度
            if len(image.shape) == 4:
                # 取第一张图片
                image_np = image[0].cpu().numpy()
            else:
                image_np = image.squeeze().cpu().numpy()
        else:
            image_np = image
        
        # 获取原始图像尺寸
        original_height, original_width = image_np.shape[:2]
        
        # 计算新的图像尺寸
        new_height = original_height + expand_top + expand_bottom
        new_width = original_width + expand_left + expand_right
        
        # 确定填充颜色值
        if fill_color == "黑":
            fill_value = 0.0
        elif fill_color == "白":
            fill_value = 1.0
        else:  # 灰
            fill_value = 0.5
        
        # 创建扩展后的图像
        if len(image_np.shape) == 3:
            # 彩色图像
            expanded_image = np.full((new_height, new_width, image_np.shape[2]), fill_value, dtype=image_np.dtype)
            # 将原图放置在正确位置
            expanded_image[expand_top:expand_top + original_height, 
                         expand_left:expand_left + original_width, :] = image_np
        else:
            # 灰度图像
            expanded_image = np.full((new_height, new_width), fill_value, dtype=image_np.dtype)
            expanded_image[expand_top:expand_top + original_height, 
                         expand_left:expand_left + original_width] = image_np
        
        # 创建基础mask图像 (白色表示原图区域，黑色表示扩展区域)
        mask = np.zeros((new_height, new_width), dtype=np.float32)
        mask[expand_top:expand_top + original_height, 
             expand_left:expand_left + original_width] = 1.0
        
        # 处理mask扩展/收缩
        if mask_expand != 0:
            if mask_expand > 0:
                # 扩展mask（膨胀）
                structure = np.ones((3, 3))
                for _ in range(abs(mask_expand)):
                    mask = binary_dilation(mask, structure=structure).astype(np.float32)
            else:
                # 收缩mask（腐蚀）
                structure = np.ones((3, 3))
                for _ in range(abs(mask_expand)):
                    mask = binary_erosion(mask, structure=structure).astype(np.float32)
        
        # 处理mask羽化
        if mask_feather > 0:
            mask = gaussian_filter(mask, sigma=mask_feather)
            # 确保值在0-1范围内
            mask = np.clip(mask, 0.0, 1.0)
        
        # 处理mask反转
        if mask_invert:
            mask = 1.0 - mask
        
        # 确保扩展图像是3D数组 (height, width, channels)
        if len(expanded_image.shape) == 2:
            # 如果是灰度图，添加通道维度并转换为RGB
            expanded_image = np.expand_dims(expanded_image, axis=-1)
            expanded_image = np.repeat(expanded_image, 3, axis=-1)
        
        # 添加batch维度
        if len(expanded_image.shape) == 3:
            expanded_image = np.expand_dims(expanded_image, axis=0)
        
        # 为mask添加batch维度
        mask = np.expand_dims(mask, axis=0)
        
        # 转换为tensor
        result_image = torch.from_numpy(expanded_image.astype(np.float32))
        result_mask = torch.from_numpy(mask.astype(np.float32))
        
        print(f"PIP自定义扩图: 上={expand_top}, 下={expand_bottom}, 左={expand_left}, 右={expand_right}")
        print(f"填充颜色: {fill_color}")
        print(f"Mask反转: {mask_invert}, Mask扩展: {mask_expand}, Mask羽化: {mask_feather}")
        print(f"原始尺寸: {original_width}x{original_height}")
        print(f"扩展后尺寸: {new_width}x{new_height}")
        print(f"扩展图像shape: {result_image.shape}")
        print(f"Mask shape: {result_mask.shape}")
        
        return (result_image, result_mask)

# 导出节点类
NODE_CLASS_MAPPINGS = {
    "PIP_CustomExpand": PIP_CustomExpand
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_CustomExpand": "PIP 自定义扩图（mask）"
}
