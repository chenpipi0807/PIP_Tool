import torch
from PIL import Image
import numpy as np

class PIP_ImageConcatenation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["左右", "右左", "上下", "下上"],),
                "max_dimension": ("INT", {
                    "default": 996,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "gap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "background_color": (["白色", "黑色", "透明"],),
            },
            "optional": {
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width_int", "height_int")
    FUNCTION = "concatenate_images"
    CATEGORY = "图像处理"

    def concatenate_images(self, image1, image2, direction, max_dimension, gap, background_color, image3=None):
        # 确保图像是正确的维度 (batch_size, height, width, channels)
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)
        if image3 is not None and image3.dim() == 3:
            image3 = image3.unsqueeze(0)

        # 只取每个批次的第一张图像
        img1 = image1[0]
        img2 = image2[0]
        img3 = image3[0] if image3 is not None else None

        # 转换为PIL图像进行处理
        img1_pil = self._tensor_to_pil(img1)
        img2_pil = self._tensor_to_pil(img2)
        img3_pil = self._tensor_to_pil(img3) if img3 is not None else None
        
        # 计算图像尺寸
        images = [img1_pil, img2_pil]
        if img3_pil:
            images.append(img3_pil)
        
        # 确定是水平还是垂直连接
        horizontal = direction in ["左右", "右左"]
        
        # 确定拼接方向的顺序
        if direction == "左右":
            images = [img1_pil, img2_pil] + ([img3_pil] if img3_pil else [])
        elif direction == "右左":
            images = ([img3_pil] if img3_pil else []) + [img2_pil, img1_pil]
        elif direction == "上下":
            images = [img1_pil, img2_pil] + ([img3_pil] if img3_pil else [])
        elif direction == "下上":
            images = ([img3_pil] if img3_pil else []) + [img2_pil, img1_pil]
        
        # 调整图像尺寸，确保拼接边对齐
        resized_images = self._resize_images_for_concatenation(images, horizontal, max_dimension)
        
        # 设置背景颜色
        bg_color_map = {
            "白色": (255, 255, 255),
            "黑色": (0, 0, 0),
            "透明": None
        }
        bg_color = bg_color_map[background_color]
        
        # 拼接图像
        concat_img = self._concatenate_pil_images(resized_images, horizontal, gap, bg_color)
        
        # 获取最终尺寸
        width, height = concat_img.size
        
        # 转回PyTorch张量
        result_tensor = self._pil_to_tensor(concat_img)
        
        # 添加批次维度
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor, width, height)
    
    def _tensor_to_pil(self, tensor):
        if tensor is None:
            return None
        # 确保tensor在0-1范围内
        tensor = tensor.clamp(0, 1)
        # 转换为numpy数组并调整为0-255
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        # 创建PIL图像
        return Image.fromarray(img_np, mode='RGB')
    
    def _pil_to_tensor(self, pil_img):
        if pil_img is None:
            return None
        # 转换为RGBA模式以处理透明度
        if pil_img.mode == 'RGBA':
            # 创建白色背景
            bg = Image.new('RGBA', pil_img.size, (255, 255, 255, 255))
            # 将图像与背景合并
            pil_img = Image.alpha_composite(bg.convert('RGBA'), pil_img.convert('RGBA')).convert('RGB')
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # 转为numpy数组
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        # 转为PyTorch张量
        return torch.from_numpy(img_np)
    
    def _resize_images_for_concatenation(self, images, horizontal, max_dimension):
        if not images:
            return []
            
        resized_images = []
        
        # 计算所有图像的宽高比
        aspect_ratios = [img.width / img.height for img in images]
        
        if horizontal:
            # 找到所有图像中的最小高度作为目标高度
            heights = [img.height for img in images]
            widths = [img.width for img in images]
            
            # 计算按比例缩放后的总宽度
            total_width_ratio = sum([ar for ar in aspect_ratios])
            target_height = min(heights)
            
            # 检查是否需要进一步缩放以符合max_dimension
            if target_height > max_dimension:
                scale_factor = max_dimension / target_height
                target_height = max_dimension
            else:
                scale_factor = 1
                
            # 调整所有图像高度一致
            for img, ar in zip(images, aspect_ratios):
                new_height = target_height
                new_width = int(new_height * ar)
                resized = img.resize((new_width, new_height), Image.LANCZOS)
                resized_images.append(resized)
                
            # 检查总宽度是否超过max_dimension，如果是则进一步缩放
            total_width = sum([img.width for img in resized_images])
            if total_width > max_dimension:
                scale_factor = max_dimension / total_width
                resized_images = []
                for img, ar in zip(images, aspect_ratios):
                    new_height = int(target_height * scale_factor)
                    new_width = int(new_height * ar)
                    resized = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_images.append(resized)
        else:
            # 找到所有图像中的最小宽度作为目标宽度
            heights = [img.height for img in images]
            widths = [img.width for img in images]
            
            # 计算按比例缩放后的总高度
            total_height_ratio = sum([1/ar for ar in aspect_ratios])
            target_width = min(widths)
            
            # 检查是否需要进一步缩放以符合max_dimension
            if target_width > max_dimension:
                scale_factor = max_dimension / target_width
                target_width = max_dimension
            else:
                scale_factor = 1
                
            # 调整所有图像宽度一致
            for img, ar in zip(images, aspect_ratios):
                new_width = target_width
                new_height = int(new_width / ar)
                resized = img.resize((new_width, new_height), Image.LANCZOS)
                resized_images.append(resized)
                
            # 检查总高度是否超过max_dimension，如果是则进一步缩放
            total_height = sum([img.height for img in resized_images])
            if total_height > max_dimension:
                scale_factor = max_dimension / total_height
                resized_images = []
                for img, ar in zip(images, aspect_ratios):
                    new_width = int(target_width * scale_factor)
                    new_height = int(new_width / ar)
                    resized = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_images.append(resized)
                    
        return resized_images
    
    def _concatenate_pil_images(self, images, horizontal, gap, bg_color):
        if not images:
            return None
            
        # 计算拼接后的尺寸
        if horizontal:
            total_width = sum([img.width for img in images]) + gap * (len(images) - 1)
            max_height = max([img.height for img in images])
            
            # 创建新图像
            mode = 'RGBA' if bg_color is None else 'RGB'
            bg_color = bg_color if bg_color is not None else (0, 0, 0, 0)  # 透明背景用RGBA
            
            result = Image.new(mode, (total_width, max_height), bg_color)
            
            # 拼接图像
            x_offset = 0
            for img in images:
                if img.mode != mode:
                    img = img.convert(mode)
                    
                # 垂直居中
                y_offset = (max_height - img.height) // 2
                result.paste(img, (x_offset, y_offset))
                x_offset += img.width + gap
        else:
            max_width = max([img.width for img in images])
            total_height = sum([img.height for img in images]) + gap * (len(images) - 1)
            
            # 创建新图像
            mode = 'RGBA' if bg_color is None else 'RGB'
            bg_color = bg_color if bg_color is not None else (0, 0, 0, 0)  # 透明背景用RGBA
            
            result = Image.new(mode, (max_width, total_height), bg_color)
            
            # 拼接图像
            y_offset = 0
            for img in images:
                if img.mode != mode:
                    img = img.convert(mode)
                    
                # 水平居中
                x_offset = (max_width - img.width) // 2
                result.paste(img, (x_offset, y_offset))
                y_offset += img.height + gap
                
        return result
