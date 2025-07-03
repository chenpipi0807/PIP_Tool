import torch
import numpy as np
from PIL import Image

class PIP_Pixelate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "method": (["average", "center", "dominant"], {
                    "default": "average"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelate_image"
    CATEGORY = "PIP_Tool"

    def pixelate_image(self, image, pixel_size, intensity, method):
        """
        对图像进行像素化处理
        
        Args:
            image: 输入图像tensor
            pixel_size: 像素块大小
            intensity: 像素化强度 (0-1)
            method: 颜色计算方法
        """
        # 转换为numpy数组
        img_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        batch_results = []
        
        for i in range(image.shape[0]):
            if image.shape[0] > 1:
                img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            height, width, channels = img_np.shape
            
            # 创建像素化后的图像
            pixelated = img_np.copy()
            
            # 按网格处理
            for y in range(0, height, pixel_size):
                for x in range(0, width, pixel_size):
                    # 确定当前块的边界
                    y_end = min(y + pixel_size, height)
                    x_end = min(x + pixel_size, width)
                    
                    # 提取当前块
                    block = img_np[y:y_end, x:x_end]
                    
                    # 根据方法计算颜色
                    if method == "average":
                        # 计算平均颜色
                        avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                    elif method == "center":
                        # 使用中心点颜色
                        center_y = (y_end - y) // 2
                        center_x = (x_end - x) // 2
                        avg_color = block[center_y, center_x]
                    elif method == "dominant":
                        # 寻找主导色（最常见的颜色）
                        avg_color = self._get_dominant_color(block)
                    
                    # 填充整个块
                    pixelated[y:y_end, x:x_end] = avg_color
            
            # 根据强度混合原图和像素化图像
            if intensity < 1.0:
                pixelated = (intensity * pixelated + (1 - intensity) * img_np).astype(np.uint8)
            
            # 转换回tensor
            result_tensor = torch.from_numpy(pixelated.astype(np.float32) / 255.0).unsqueeze(0)
            batch_results.append(result_tensor)
        
        # 合并批次结果
        final_result = torch.cat(batch_results, dim=0)
        
        return (final_result,)
    
    def _get_dominant_color(self, block):
        """获取块中的主导色"""
        # 将块重塑为像素列表
        pixels = block.reshape(-1, block.shape[-1])
        
        # 简化颜色空间以减少计算量
        # 将每个通道量化为16个级别
        simplified = (pixels // 16) * 16
        
        # 找到最常见的颜色
        unique_colors, counts = np.unique(simplified, axis=0, return_counts=True)
        dominant_idx = np.argmax(counts)
        
        return unique_colors[dominant_idx]


class PIP_PixelateAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "method": (["average", "center", "dominant", "median"], {
                    "default": "average"
                }),
                "preserve_edges": ("BOOLEAN", {
                    "default": False
                }),
                "color_reduction": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 256,
                    "step": 8,
                    "display": "number"
                }),
                "shape": (["square", "diamond", "circle"], {
                    "default": "square"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pixelate_advanced"
    CATEGORY = "PIP_Tool"

    def pixelate_advanced(self, image, pixel_size, intensity, method, preserve_edges, color_reduction, shape):
        """
        高级像素化处理，支持更多选项
        """
        img_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        batch_results = []
        
        for i in range(image.shape[0]):
            if image.shape[0] > 1:
                img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            height, width, channels = img_np.shape
            
            # 边缘检测（如果启用）
            edges = None
            if preserve_edges:
                edges = self._detect_edges(img_np)
            
            # 颜色减少
            if color_reduction < 256:
                img_np = self._reduce_colors(img_np, color_reduction)
            
            pixelated = img_np.copy()
            
            # 按网格处理
            for y in range(0, height, pixel_size):
                for x in range(0, width, pixel_size):
                    y_end = min(y + pixel_size, height)
                    x_end = min(x + pixel_size, width)
                    
                    # 检查是否在边缘附近
                    if preserve_edges and edges is not None:
                        edge_block = edges[y:y_end, x:x_end]
                        if np.any(edge_block > 128):  # 如果检测到边缘，使用更小的像素块
                            continue
                    
                    block = img_np[y:y_end, x:x_end]
                    
                    # 计算颜色
                    if method == "average":
                        color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                    elif method == "center":
                        center_y = (y_end - y) // 2
                        center_x = (x_end - x) // 2
                        color = block[center_y, center_x]
                    elif method == "dominant":
                        color = self._get_dominant_color(block)
                    elif method == "median":
                        color = np.median(block.reshape(-1, channels), axis=0).astype(np.uint8)
                    
                    # 根据形状填充
                    if shape == "square":
                        pixelated[y:y_end, x:x_end] = color
                    elif shape == "diamond":
                        self._fill_diamond(pixelated, y, x, y_end, x_end, color)
                    elif shape == "circle":
                        self._fill_circle(pixelated, y, x, y_end, x_end, color)
            
            # 混合
            if intensity < 1.0:
                original = (image[i].cpu().numpy() * 255).astype(np.uint8) if image.shape[0] > 1 else (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pixelated = (intensity * pixelated + (1 - intensity) * original).astype(np.uint8)
            
            result_tensor = torch.from_numpy(pixelated.astype(np.float32) / 255.0).unsqueeze(0)
            batch_results.append(result_tensor)
        
        final_result = torch.cat(batch_results, dim=0)
        return (final_result,)
    
    def _detect_edges(self, img):
        """简单的边缘检测"""
        gray = np.mean(img, axis=2)
        
        # Sobel边缘检测
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # 使用简单卷积
        edges_x = np.abs(self._simple_conv(gray, sobel_x))
        edges_y = np.abs(self._simple_conv(gray, sobel_y))
        
        return np.sqrt(edges_x**2 + edges_y**2)
    
    def _simple_conv(self, img, kernel):
        """简单的2D卷积"""
        result = np.zeros_like(img)
        pad = kernel.shape[0] // 2
        
        for i in range(pad, img.shape[0] - pad):
            for j in range(pad, img.shape[1] - pad):
                result[i, j] = np.sum(img[i-pad:i+pad+1, j-pad:j+pad+1] * kernel)
        
        return result
    
    def _reduce_colors(self, img, levels):
        """减少颜色数量"""
        factor = 256 // levels
        return (img // factor) * factor
    
    def _get_dominant_color(self, block):
        """获取主导色"""
        pixels = block.reshape(-1, block.shape[-1])
        simplified = (pixels // 16) * 16
        unique_colors, counts = np.unique(simplified, axis=0, return_counts=True)
        dominant_idx = np.argmax(counts)
        return unique_colors[dominant_idx]
    
    def _fill_diamond(self, img, y1, x1, y2, x2, color):
        """菱形填充"""
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        radius = min((y2 - y1), (x2 - x1)) // 2
        
        for y in range(y1, y2):
            for x in range(x1, x2):
                if abs(y - center_y) + abs(x - center_x) <= radius:
                    img[y, x] = color
    
    def _fill_circle(self, img, y1, x1, y2, x2, color):
        """圆形填充"""
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        radius = min((y2 - y1), (x2 - x1)) // 2
        
        for y in range(y1, y2):
            for x in range(x1, x2):
                if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
                    img[y, x] = color
