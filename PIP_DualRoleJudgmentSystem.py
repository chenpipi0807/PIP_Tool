import torch
import json
import os
import re
from difflib import get_close_matches
from PIL import Image, ImageFilter
import numpy as np

class PIP_DualRoleJudgmentSystem:
    def __init__(self):
        self.role_names = self._load_role_names()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SYS": ("IMAGE",),
                "USR": ("IMAGE",),
                "SPT-M-15": ("IMAGE",),
                "SPT-M-25": ("IMAGE",),
                "SPT-M-35": ("IMAGE",),
                "SPT-M-60": ("IMAGE",),
                "SPT-F-15": ("IMAGE",),
                "SPT-F-25": ("IMAGE",),
                "SPT-F-35": ("IMAGE",),
                "SPT-F-60": ("IMAGE",),
                "condition": ("STRING", {"default": "SPT-F-25|SPT-M-25", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width_int", "height_int")
    FUNCTION = "process_roles"
    CATEGORY = "角色处理"

    def _load_role_names(self):
        """从 RoleName.json 加载角色名称"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "RoleName.json")
            
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return [role["name"] for role in data["roles"]]
        except Exception as e:
            print(f"加载角色名称时出错: {e}")
            # 如果无法加载，使用默认角色列表
            return ["SYS", "USR", "SPT-M-15", "SPT-M-25", "SPT-M-35", "SPT-M-60", 
                    "SPT-F-15", "SPT-F-25", "SPT-F-35", "SPT-F-60"]

    def _match_role(self, role_str):
        """尝试匹配角色名称，支持不区分大小写和近似匹配"""
        # 直接匹配
        if role_str in self.role_names:
            return role_str
        
        # 不区分大小写匹配
        lowercase_roles = {r.lower(): r for r in self.role_names}
        if role_str.lower() in lowercase_roles:
            return lowercase_roles[role_str.lower()]
        
        # 处理常见错误模式（如空格、连字符、大小写等）
        normalized_str = re.sub(r'[\s\-_]', '', role_str).lower()
        normalized_roles = {re.sub(r'[\s\-_]', '', r).lower(): r for r in self.role_names}
        
        if normalized_str in normalized_roles:
            return normalized_roles[normalized_str]
        
        # 近似匹配
        closest_matches = get_close_matches(role_str.lower(), [r.lower() for r in self.role_names], n=1, cutoff=0.6)
        if closest_matches:
            for original in self.role_names:
                if original.lower() == closest_matches[0]:
                    return original
        
        # 如果仍未找到匹配，尝试查找子串匹配
        for name in self.role_names:
            if role_str.lower() in name.lower() or name.lower() in role_str.lower():
                return name
                
        # 未找到匹配时返回 None
        return None

    def process_roles(self, condition, **kwargs):
        """处理条件并返回对应的图像"""
        # 解析条件字符串
        roles = condition.split("|")
        selected_roles = []
        
        # 最多处理前两个角色条件
        for role_str in roles[:2]:
            role_str = role_str.strip()
            matched_role = self._match_role(role_str)
            if matched_role:
                selected_roles.append(matched_role)
        
        # 获取对应的图像
        images = []
        for role_name in selected_roles[:2]:  # 只取前两个
            if role_name in kwargs:
                images.append(kwargs[role_name])
            else:
                # 如果找不到角色对应的图像，使用第一个可用图像
                for key, value in kwargs.items():
                    if key in self.role_names:
                        images.append(value)
                        break
        
        # 如果没有匹配到任何角色
        if len(images) == 0:
            # 使用第一个可用的角色图像作为默认
            for key, value in kwargs.items():
                if key in self.role_names:
                    images.append(value)
                    break
            
            # 如果仍然没有图像，返回空图像
            if len(images) == 0:
                empty_tensor = torch.zeros((512, 512, 3))
                return (empty_tensor.unsqueeze(0), 512, 512)
        
        # 只有一个角色的情况：直接返回该图像
        if len(images) == 1:
            img_tensor = images[0]
            if img_tensor.dim() == 3:  # 确保有批次维度
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.dim() == 4 and img_tensor.size(0) > 1:
                img_tensor = img_tensor[0].unsqueeze(0)  # 只取第一张
            
            # 从tensor获取尺寸
            height, width = img_tensor.shape[1:3]
            return (img_tensor, width, height)
        
        # 两个角色的情况：进行无缝拼接
        if len(images) >= 2:
            img1 = images[0]
            img2 = images[1]
            
            # 确保图像是正确的维度
            if img1.dim() == 4:
                img1 = img1[0]  # 移除批次维度
            if img2.dim() == 4:
                img2 = img2[0]  # 移除批次维度
            
            # 转换为PIL进行处理
            img1_pil = self._tensor_to_pil(img1)
            img2_pil = self._tensor_to_pil(img2)
            
            # 默认参数
            blend_percentage = 30.0  # 重叠百分比
            blur_radius = 10.0      # 模糊半径
            max_dimension = 1024     # 最大尺寸
            
            # 调整图像尺寸
            img1_pil, img2_pil = self._resize_for_concat(img1_pil, img2_pil, max_dimension)
            
            # 执行无缝拼接
            result_img = self._simple_seamless_concat(img1_pil, img2_pil, blend_percentage, blur_radius)
            
            # 转回tensor
            result_tensor = self._pil_to_tensor(result_img).unsqueeze(0)
            
            # 获取最终尺寸
            width, height = result_img.size
            
            return (result_tensor, width, height)
        
        # 如果执行到这里，说明有异常情况，返回空图像
        empty_tensor = torch.zeros((512, 512, 3))
        return (empty_tensor.unsqueeze(0), 512, 512)
        
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
        
        # 计算重叠区域宽度
        overlap_width = int(min(w1, w2) * blend_percentage / 100.0)
        overlap_width = min(overlap_width, w1 // 2, w2 // 2)  # 限制最大重叠
        
        if overlap_width <= 0:
            # 如果没有重叠，直接拼接
            total_width = w1 + w2
            result = Image.new('RGB', (total_width, h))
            result.paste(img1, (0, 0))
            result.paste(img2, (w1, 0))
            return result
        
        # 计算最终尺寸（有重叠）
        final_width = w1 + w2 - overlap_width
        
        # 创建最终图像
        result = Image.new('RGB', (final_width, h))
        
        # 贴上左图（完整）
        result.paste(img1, (0, 0))
        
        # 准备右图的重叠部分和非重叠部分
        right_overlap_start = w1 - overlap_width
        
        # 提取重叠区域
        left_overlap = img1.crop((w1 - overlap_width, 0, w1, h))
        right_overlap = img2.crop((0, 0, overlap_width, h))
        
        # 创建渐变蒙版（从左到右：黑到白）
        mask = self._create_gradient_mask(overlap_width, h, blur_radius)
        
        # 使用蒙版融合重叠区域
        blended_overlap = Image.composite(right_overlap, left_overlap, mask)
        
        # 贴上融合后的重叠区域
        result.paste(blended_overlap, (right_overlap_start, 0))
        
        # 贴上右图的剩余部分
        if overlap_width < w2:
            right_remaining = img2.crop((overlap_width, 0, w2, h))
            result.paste(right_remaining, (w1, 0))
        
        return result
    
    def _create_gradient_mask(self, width, height, blur_radius):
        """创建渐变蒙版"""
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
        
        return mask
