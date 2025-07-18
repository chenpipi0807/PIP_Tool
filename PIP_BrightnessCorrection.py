import torch
import numpy as np
from PIL import Image
import cv2

class PIP_BrightnessCorrection:
    """
    PIP 亮度补偿节点
    自动或手动调整图像亮度，让过暗或过亮的图像恢复正常
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "brightness_score": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "brightness_value": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "correction_mode": (["稳健", "激进"], {
                    "default": "稳健"
                }),
                "strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("corrected_image", "correction_info", "brightness_change")
    FUNCTION = "correct_brightness"
    CATEGORY = "PIP_Tool"
    
    def correct_brightness(self, image, brightness_score=0, brightness_value=0.5, correction_mode="稳健", strength=0.7):
        try:
            # 处理批次图像
            corrected_images = []
            correction_infos = []
            brightness_changes = []
            
            for i in range(image.shape[0]):
                img_tensor = image[i]
                
                # 转换为numpy数组
                img_np = img_tensor.cpu().numpy()
                
                # 确保值范围在0-1之间
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                
                # 分析当前图像亮度
                current_brightness = self._analyze_brightness(img_np)
                
                # 始终使用guided模式：基于检测节点结果进行智能调整
                brightness_adj, contrast_adj, gamma_adj = self._calculate_guided_correction(
                    brightness_score, brightness_value, current_brightness, correction_mode
                )
                
                # 应用强度调整
                brightness_adj *= strength
                contrast_adj *= strength
                gamma_adj = (gamma_adj - 1.0) * strength + 1.0
                
                # 应用亮度调整
                corrected_img = self._apply_corrections(
                    img_np, brightness_adj, contrast_adj, gamma_adj
                )
                
                # 计算实际亮度变化
                new_brightness = self._analyze_brightness(corrected_img)
                brightness_change = new_brightness - current_brightness
                
                # 转换回tensor格式
                corrected_tensor = torch.from_numpy(corrected_img).float()
                corrected_images.append(corrected_tensor)
                
                # 生成调整信息
                info = self._generate_correction_info(
                    current_brightness, new_brightness, brightness_adj, 
                    contrast_adj, gamma_adj, f"guided-{correction_mode}"
                )
                correction_infos.append(info)
                brightness_changes.append(brightness_change)
                
                print(f"[PIP_BrightnessCorrection] 图像 {i+1}: 亮度 {current_brightness:.3f} -> {new_brightness:.3f}")
            
            # 合并结果
            result_image = torch.stack(corrected_images, dim=0)
            combined_info = "\n".join(correction_infos)
            avg_brightness_change = sum(brightness_changes) / len(brightness_changes)
            
            return (result_image, combined_info, float(avg_brightness_change))
            
        except Exception as e:
            print(f"[PIP_BrightnessCorrection] 错误: {str(e)}")
            return (image, f"调整失败: {str(e)}", 0.0)
    
    def _analyze_brightness(self, img_np):
        """分析图像亮度"""
        if len(img_np.shape) == 3:
            # 使用感知亮度公式
            r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            perceived = 0.299 * r + 0.587 * g + 0.114 * b
            return np.mean(perceived)
        else:
            return np.mean(img_np)
    
    def _calculate_guided_correction(self, brightness_score, brightness_value, current_brightness, correction_mode):
        """基于检测节点的结果计算调整参数（支持稳健/激进模式）"""
        # 优先使用检测节点的结果，如果为默认值则使用当前检测
        if brightness_score == 0 and brightness_value == 0.5:
            # 没有输入检测结果，使用当前检测
            target_brightness = 0.5
            brightness_diff = target_brightness - current_brightness
        else:
            # 使用检测节点的结果
            target_brightness = 0.5  # 目标亮度
            brightness_diff = target_brightness - brightness_value
        
        # 根据评分确定调整强度
        if abs(brightness_score) <= 10:
            # 亮度正常，不需要调整
            return 0.0, 0.0, 1.0
        
        # 根据模式设置调整参数
        if correction_mode == "激进":
            # 激进模式：更强的调整效果
            max_brightness_adjustment = 0.35  # 翔了2.3倍
            base_strength = 0.8  # 增加0.3
            max_contrast_adjustment = 0.15  # 翔了近二倍
            contrast_multiplier = 1.8
            gamma_range = (0.7, 1.3)  # 更大的伽马范围
            gamma_strength = 1.5
        else:
            # 稳健模式：保守的调整（原来的设置）
            max_brightness_adjustment = 0.15
            base_strength = 0.5
            max_contrast_adjustment = 0.08
            contrast_multiplier = 1.0
            gamma_range = (0.85, 1.15)
            gamma_strength = 1.0
        
        # 计算亮度调整
        score_factor = min(abs(brightness_score) / 100.0, 1.0)
        brightness_adjustment = np.clip(
            brightness_diff * base_strength * score_factor, 
            -max_brightness_adjustment, 
            max_brightness_adjustment
        )
        
        # 计算对比度调整
        if brightness_score < -30:
            # 图像过暗，适当增加对比度
            contrast_base = 0.03 if correction_mode == "稳健" else 0.08
            contrast_adjustment = contrast_base * contrast_multiplier * (abs(brightness_score) - 30) / 70
        elif brightness_score > 30:
            # 图像过亮，适当降低对比度
            contrast_base = 0.03 if correction_mode == "稳健" else 0.08
            contrast_adjustment = -contrast_base * contrast_multiplier * (brightness_score - 30) / 70
        else:
            contrast_adjustment = 0.0
        
        contrast_adjustment = np.clip(contrast_adjustment, -max_contrast_adjustment, max_contrast_adjustment)
        
        # 计算伽马调整
        if brightness_score < -40:
            # 过暗图像，降低伽马值提亮暗部
            gamma_base = 0.92 if correction_mode == "稳健" else 0.8
            gamma_reduction = 0.08 if correction_mode == "稳健" else 0.15
            gamma_adjustment = gamma_base - gamma_reduction * gamma_strength * (abs(brightness_score) - 40) / 60
        elif brightness_score > 40:
            # 过亮图像，提高伽马值压暗亮部
            gamma_base = 1.08 if correction_mode == "稳健" else 1.2
            gamma_increase = 0.08 if correction_mode == "稳健" else 0.15
            gamma_adjustment = gamma_base + gamma_increase * gamma_strength * (brightness_score - 40) / 60
        else:
            gamma_adjustment = 1.0
        
        gamma_adjustment = np.clip(gamma_adjustment, gamma_range[0], gamma_range[1])
        
        return brightness_adjustment, contrast_adjustment, gamma_adjustment
    

    
    def _apply_corrections(self, img_np, brightness_adj, contrast_adj, gamma_adj):
        """应用亮度、对比度和伽马调整（优化版）"""
        corrected = img_np.copy()
        
        # 先保存原始亮度信息用于最后的软混合
        if len(img_np.shape) == 3:
            original_lum = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        else:
            original_lum = img_np
        
        # 1. 应用亮度调整（更加平滑）
        if brightness_adj != 0:
            # 使用指数调整而非线性调整，更加自然
            if brightness_adj > 0:
                corrected = corrected + brightness_adj * (1 - corrected)  # 保护高光
            else:
                corrected = corrected + brightness_adj * corrected  # 保护暗部
        
        # 2. 应用对比度调整（渐进式）
        if contrast_adj != 0:
            # 渐进式对比度调整，在极端区域减弱效果
            contrast_mask = np.ones_like(corrected)
            if len(corrected.shape) == 3:
                current_lum = 0.299 * corrected[:,:,0] + 0.587 * corrected[:,:,1] + 0.114 * corrected[:,:,2]
                # 在极端亮度区域减弱对比度调整
                extreme_mask = np.logical_or(current_lum < 0.1, current_lum > 0.9)
                contrast_mask = np.where(extreme_mask, 0.3, 1.0)
                contrast_mask = np.stack([contrast_mask] * 3, axis=-1)
            
            adjusted_contrast = contrast_adj * contrast_mask
            corrected = (corrected - 0.5) * (1 + adjusted_contrast) + 0.5
        
        # 3. 应用伽马调整（保护极端值）
        if gamma_adj != 1.0:
            # 保护极端亮度值，避免数值不稳定
            corrected_safe = np.clip(corrected, 0.01, 0.99)
            gamma_corrected = np.power(corrected_safe, 1.0 / gamma_adj)
            # 在极端区域使用原始值
            corrected = np.where((corrected <= 0.01) | (corrected >= 0.99), corrected, gamma_corrected)
        
        # 4. 渐进式音调保护（固定开启）
        corrected = self._preserve_tones_advanced(img_np, corrected, True, True)
        
        # 5. 最后的软混合，避免过度调整
        blend_strength = 0.9  # 稍微保留原始信息
        corrected = img_np * (1 - blend_strength) + corrected * blend_strength
        
        # 确保值在有效范围内
        corrected = np.clip(corrected, 0, 1)
        
        return corrected
    
    def _preserve_tones_advanced(self, original, corrected, preserve_highlights, preserve_shadows):
        """优化的音调保护函数"""
        result = corrected.copy()
        
        if len(original.shape) == 3:
            # 计算亮度掉罩
            original_lum = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
            corrected_lum = 0.299 * corrected[:,:,0] + 0.587 * corrected[:,:,1] + 0.114 * corrected[:,:,2]
        else:
            original_lum = original
            corrected_lum = corrected
        
        if preserve_highlights:
            # 渐进式高光保护 (亮度 > 0.7)
            highlight_start = 0.7
            highlight_end = 0.95
            
            # 创建渐进高光摸罩
            highlight_mask = np.zeros_like(original_lum)
            highlight_region = (original_lum >= highlight_start) & (original_lum <= highlight_end)
            highlight_mask[highlight_region] = (original_lum[highlight_region] - highlight_start) / (highlight_end - highlight_start)
            highlight_mask[original_lum > highlight_end] = 1.0
            
            if len(original.shape) == 3:
                highlight_mask = np.stack([highlight_mask] * 3, axis=-1)
            
            # 渐进混合，在高光区域更多保留原始信息
            blend_factor = 0.7 * highlight_mask  # 在高光区域保留70%的原始信息
            result = original * blend_factor + result * (1 - blend_factor)
        
        if preserve_shadows:
            # 渐进式阴影保护 (亮度 < 0.3)
            shadow_start = 0.3
            shadow_end = 0.05
            
            # 创建渐进阴影摸罩
            shadow_mask = np.zeros_like(original_lum)
            shadow_region = (original_lum <= shadow_start) & (original_lum >= shadow_end)
            shadow_mask[shadow_region] = (shadow_start - original_lum[shadow_region]) / (shadow_start - shadow_end)
            shadow_mask[original_lum < shadow_end] = 1.0
            
            if len(original.shape) == 3:
                shadow_mask = np.stack([shadow_mask] * 3, axis=-1)
            
            # 渐进混合，在阴影区域更多保留原始信息
            blend_factor = 0.7 * shadow_mask  # 在阴影区域保留70%的原始信息
            result = original * blend_factor + result * (1 - blend_factor)
        
        return result
    
    def _generate_correction_info(self, old_brightness, new_brightness, 
                                 brightness_adj, contrast_adj, gamma_adj, mode):
        """生成调整信息"""
        info_lines = [
            f"调整模式: {mode}",
            f"原始亮度: {old_brightness:.3f}",
            f"调整后亮度: {new_brightness:.3f}",
            f"亮度变化: {new_brightness - old_brightness:+.3f}",
            f"应用参数:",
            f"  - 亮度调整: {brightness_adj:+.3f}",
            f"  - 对比度调整: {contrast_adj:+.3f}",
            f"  - 伽马调整: {gamma_adj:.3f}"
        ]
        
        # 添加效果评估
        brightness_change = new_brightness - old_brightness
        if abs(brightness_change) < 0.05:
            info_lines.append("效果: 微调")
        elif abs(brightness_change) < 0.15:
            info_lines.append("效果: 适度调整")
        else:
            info_lines.append("效果: 显著改善")
        
        return "\n".join(info_lines)
