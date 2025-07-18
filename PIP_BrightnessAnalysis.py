import torch
import numpy as np
from PIL import Image
import cv2

class PIP_BrightnessAnalysis:
    """
    PIP 亮度检测节点
    分析图像亮度并返回-100到+100的评分
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "method": (["luminance", "average", "perceived", "histogram"], {
                    "default": "perceived"
                }),
                "region": (["full", "center", "face_priority"], {
                    "default": "full"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("brightness_score", "brightness_value", "analysis_text")
    FUNCTION = "analyze_brightness"
    CATEGORY = "PIP_Tool"
    
    def analyze_brightness(self, image, method="perceived", region="full"):
        try:
            # 转换图像格式
            if len(image.shape) == 4:
                # 处理批次图像，取第一张
                img_tensor = image[0]
            else:
                img_tensor = image
            
            # 转换为numpy数组 (H, W, C)
            img_np = img_tensor.cpu().numpy()
            
            # 确保值范围在0-1之间
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # 获取分析区域
            if region == "center":
                h, w = img_np.shape[:2]
                center_h, center_w = h // 4, w // 4
                img_np = img_np[center_h:h-center_h, center_w:w-center_w]
            elif region == "face_priority":
                # 尝试检测人脸区域
                img_np = self._get_face_region(img_np)
            
            # 计算亮度值
            brightness_value = self._calculate_brightness(img_np, method)
            
            # 转换为-100到+100评分
            brightness_score = self._convert_to_score(brightness_value, method)
            
            # 生成分析文本
            analysis_text = self._generate_analysis_text(brightness_score, brightness_value, method)
            
            print(f"[PIP_BrightnessAnalysis] 亮度评分: {brightness_score}, 原始值: {brightness_value:.3f}")
            print(f"[PIP_BrightnessAnalysis] 分析方法: {method}, 区域: {region}")
            
            return (brightness_score, float(brightness_value), analysis_text)
            
        except Exception as e:
            print(f"[PIP_BrightnessAnalysis] 错误: {str(e)}")
            return (0, 0.5, f"分析失败: {str(e)}")
    
    def _get_face_region(self, img_np):
        """尝试检测人脸区域，如果没有则使用中心区域"""
        try:
            import cv2
            
            # 转换为灰度图进行人脸检测
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 使用Haar级联检测器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # 使用第一个检测到的人脸
                x, y, w, h = faces[0]
                # 扩大人脸区域
                padding = min(w, h) // 4
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_np.shape[1] - x, w + 2 * padding)
                h = min(img_np.shape[0] - y, h + 2 * padding)
                return img_np[y:y+h, x:x+w]
            else:
                # 没有检测到人脸，使用中心区域
                h, w = img_np.shape[:2]
                center_h, center_w = h // 4, w // 4
                return img_np[center_h:h-center_h, center_w:w-center_w]
                
        except Exception:
            # 如果人脸检测失败，使用中心区域
            h, w = img_np.shape[:2]
            center_h, center_w = h // 4, w // 4
            return img_np[center_h:h-center_h, center_w:w-center_w]
    
    def _calculate_brightness(self, img_np, method):
        """计算图像亮度"""
        if method == "average":
            # 简单平均值
            return np.mean(img_np) / 255.0
            
        elif method == "luminance":
            # 使用标准亮度公式 (ITU-R BT.709)
            if len(img_np.shape) == 3:
                r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                return np.mean(luminance) / 255.0
            else:
                return np.mean(img_np) / 255.0
                
        elif method == "perceived":
            # 感知亮度 (更符合人眼感知)
            if len(img_np.shape) == 3:
                r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
                # 使用感知亮度公式
                perceived = 0.299 * r + 0.587 * g + 0.114 * b
                return np.mean(perceived) / 255.0
            else:
                return np.mean(img_np) / 255.0
                
        elif method == "histogram":
            # 基于直方图的亮度分析
            if len(img_np.shape) == 3:
                # 转换为灰度
                gray = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
            else:
                gray = img_np
            
            # 计算直方图
            hist, bins = np.histogram(gray, bins=256, range=(0, 256))
            
            # 计算加权平均亮度
            weighted_sum = np.sum(hist * bins[:-1])
            total_pixels = np.sum(hist)
            
            return (weighted_sum / total_pixels) / 255.0 if total_pixels > 0 else 0.5
    
    def _convert_to_score(self, brightness_value, method):
        """将亮度值转换为-100到+100的评分"""
        # 理想亮度范围 (根据不同方法调整)
        if method == "histogram":
            ideal_range = (0.45, 0.65)  # 直方图方法的理想范围
        else:
            ideal_range = (0.4, 0.6)    # 其他方法的理想范围
        
        ideal_center = (ideal_range[0] + ideal_range[1]) / 2
        ideal_tolerance = (ideal_range[1] - ideal_range[0]) / 2
        
        # 计算偏离程度
        deviation = brightness_value - ideal_center
        
        if abs(deviation) <= ideal_tolerance:
            # 在理想范围内，评分接近0
            score = int((deviation / ideal_tolerance) * 20)  # -20到+20范围
        else:
            # 超出理想范围
            if deviation > 0:
                # 过亮
                excess = deviation - ideal_tolerance
                max_excess = 1.0 - ideal_center - ideal_tolerance
                score = int(20 + (excess / max_excess) * 80)  # 20到100
                score = min(100, score)
            else:
                # 过暗
                deficit = abs(deviation) - ideal_tolerance
                max_deficit = ideal_center - ideal_tolerance
                score = int(-20 - (deficit / max_deficit) * 80)  # -100到-20
                score = max(-100, score)
        
        return score
    
    def _generate_analysis_text(self, score, brightness_value, method):
        """生成分析文本"""
        if score >= 60:
            level = "严重过亮"
        elif score >= 30:
            level = "过亮"
        elif score >= 10:
            level = "稍亮"
        elif score >= -10:
            level = "正常"
        elif score >= -30:
            level = "稍暗"
        elif score >= -60:
            level = "过暗"
        else:
            level = "严重过暗"
        
        recommendation = ""
        if score > 20:
            recommendation = "建议降低亮度"
        elif score < -20:
            recommendation = "建议提高亮度"
        else:
            recommendation = "亮度适中"
        
        return f"亮度等级: {level} (评分: {score})\n原始亮度: {brightness_value:.3f}\n检测方法: {method}\n建议: {recommendation}"
