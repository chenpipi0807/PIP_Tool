import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np

class PIP_SeamlessConcatenation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_percentage": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "slider"
                }),
                "blur_radius": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "max_dimension": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width_int", "height_int")
    FUNCTION = "seamless_concatenate"
    CATEGORY = "图像处理"

    def seamless_concatenate(self, image1, image2, blend_percentage, blur_radius, max_dimension):
        # 确保图像是正确的维度
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)

        # 只取每个批次的第一张图像
        img1 = image1[0]  # (H, W, C)
        img2 = image2[0]  # (H, W, C)
        
        # 转换为PIL进行处理（保持原有的稳定逻辑）
        img1_pil = self._tensor_to_pil(img1)
        img2_pil = self._tensor_to_pil(img2)
        
        # 调整图像尺寸
        img1_pil, img2_pil = self._resize_for_concat(img1_pil, img2_pil, max_dimension)
        
        # 执行无缝拼接
        result_img = self._simple_seamless_concat(img1_pil, img2_pil, blend_percentage, blur_radius)
        
        # 转回tensor
        result_tensor = self._pil_to_tensor(result_img).unsqueeze(0)
        
        # 获取最终尺寸
        width, height = result_img.size
        
        return (result_tensor, width, height)
    
    def _tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        tensor = tensor.clamp(0, 1)
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='RGB')
    
    def _pil_to_tensor(self, pil_img):
        """将PIL图像转换为tensor"""
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)
    
    def _resize_for_concat(self, img1, img2, max_dimension):
        """调整两张图像的尺寸"""
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # 统一高度
        target_height = min(h1, h2)
        new_w1 = int(w1 * target_height / h1)
        new_w2 = int(w2 * target_height / h2)
        
        # 检查总宽度
        total_width = new_w1 + new_w2
        if total_width > max_dimension:
            scale_factor = max_dimension / total_width
            target_height = int(target_height * scale_factor)
            new_w1 = int(new_w1 * scale_factor)
            new_w2 = int(new_w2 * scale_factor)
        
        img1_resized = img1.resize((new_w1, target_height), Image.LANCZOS)
        img2_resized = img2.resize((new_w2, target_height), Image.LANCZOS)
        
        return img1_resized, img2_resized
    
    def _simple_seamless_concat(self, img1, img2, blend_percentage, blur_radius):
        """使用alpha混合的无缝拼接"""
        w1, h = img1.size
        w2, h = img2.size
        
        print(f"[DEBUG] 图像尺寸: img1={w1}x{h}, img2={w2}x{h}")
        
        # 计算重叠区域宽度
        overlap_width = int(min(w1, w2) * blend_percentage / 100.0)
        overlap_width = min(overlap_width, w1 // 2, w2 // 2)  # 限制最大重叠
        
        print(f"[DEBUG] 重叠宽度: {overlap_width} (blend_percentage={blend_percentage}%)")
        
        if overlap_width <= 0:
            # 如果没有重叠，直接拼接
            total_width = w1 + w2
            result = Image.new('RGB', (total_width, h))
            result.paste(img1, (0, 0))
            result.paste(img2, (w1, 0))
            print(f"[DEBUG] 无重叠直接拼接，总宽度: {total_width}")
            return result
        
        # 计算最终尺寸（有重叠）
        final_width = w1 + w2 - overlap_width
        print(f"[DEBUG] 最终尺寸: {final_width}x{h} (重叠{overlap_width}px)")
        
        # 创建最终图像
        result = Image.new('RGB', (final_width, h))
        
        # 贴上左图（完整）
        result.paste(img1, (0, 0))
        print(f"[DEBUG] 左图贴在 (0, 0)")
        
        # 准备右图的重叠部分和非重叠部分
        right_overlap_start = w1 - overlap_width
        
        # 提取重叠区域
        left_overlap = img1.crop((w1 - overlap_width, 0, w1, h))
        right_overlap = img2.crop((0, 0, overlap_width, h))
        
        print(f"[DEBUG] 重叠区域: 左图({w1 - overlap_width}, 0, {w1}, {h}), 右图(0, 0, {overlap_width}, {h})")
        
        # 创建渐变蒙版（从左到右：黑到白）
        mask = self._create_gradient_mask(overlap_width, h, blur_radius)
        
        # 使用蒙版融合重叠区域
        blended_overlap = Image.composite(right_overlap, left_overlap, mask)
        
        # 贴上融合后的重叠区域
        result.paste(blended_overlap, (right_overlap_start, 0))
        print(f"[DEBUG] 融合区域贴在 ({right_overlap_start}, 0)")
        
        # 贴上右图的剩余部分
        if overlap_width < w2:
            right_remaining = img2.crop((overlap_width, 0, w2, h))
            result.paste(right_remaining, (w1, 0))
            print(f"[DEBUG] 右图剩余部分贴在 ({w1}, 0)")
        
        return result
    
    def _create_gradient_mask(self, width, height, blur_radius):
        """创建渐变蒙版"""
        print(f"[DEBUG] 创建渐变蒙版: {width}x{height}, 模糊半径={blur_radius}")
        
        # 创建从左到右的线性渐变
        mask = Image.new('L', (width, height))
        mask_data = []
        
        for y in range(height):
            for x in range(width):
                # 线性渐变：左边=0(黑)，右边=255(白)
                value = int(255 * x / (width - 1)) if width > 1 else 127
                mask_data.append(value)
        
        mask.putdata(mask_data)
        
        # 对蒙版应用模糊，产生柔和的过渡
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            print(f"[DEBUG] 蒙版已模糊，半径={blur_radius}")
        
        return mask
