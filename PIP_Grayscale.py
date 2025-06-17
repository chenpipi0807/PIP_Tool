import torch
from PIL import Image
import numpy as np

class PIP_Grayscale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert_to_grayscale"
    CATEGORY = "图像处理"

    def convert_to_grayscale(self, image):
        # 确保图像是正确的维度 (batch_size, height, width, channels)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 获取批次大小
        batch_size = image.shape[0]
        
        # 创建一个空张量来存储结果
        result = []
        
        # 处理每个批次
        for i in range(batch_size):
            # 获取当前图像
            img = image[i]
            
            # 转换为PIL图像
            pil_img = self._tensor_to_pil(img)
            
            # 转换为灰度图像
            # 使用 L 模式直接转换为灰度图，然后转回 RGB 以保持格式一致
            gray_img = pil_img.convert('L').convert('RGB')
            
            # 转回张量
            gray_tensor = self._pil_to_tensor(gray_img)
            
            # 添加到结果列表
            result.append(gray_tensor)
        
        # 将结果堆叠为批次
        if len(result) > 0:
            result_tensor = torch.stack(result)
        else:
            # 如果没有图像处理，返回空张量
            result_tensor = torch.zeros((0, 0, 0, 3), dtype=torch.float32)
            
        return (result_tensor,)
    
    def _tensor_to_pil(self, tensor):
        # 确保tensor在0-1范围内
        tensor = tensor.clamp(0, 1)
        # 转换为numpy数组并调整为0-255
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        # 创建PIL图像
        return Image.fromarray(img_np, mode='RGB')
    
    def _pil_to_tensor(self, pil_img):
        # 确保图像是RGB模式
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # 转为numpy数组
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        # 转为PyTorch张量
        return torch.from_numpy(img_np)
