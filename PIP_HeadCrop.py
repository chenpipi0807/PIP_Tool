import os
import torch
import numpy as np
from PIL import Image
import cv2
import math
import sys

# 导入YOLO模型
try:
    from ultralytics import YOLO
    has_ultralytics = True
except ImportError:
    has_ultralytics = False
    print("未安装ultralytics库，正在尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    try:
        from ultralytics import YOLO
        has_ultralytics = True
        print("ultralytics安装成功！")
    except ImportError:
        print("ultralytics安装失败，请手动安装")

# 确保opencv-python存在
try:
    import cv2
except ImportError:
    print("未安装opencv-python库，正在尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

class PIP_HeadCrop:
    def __init__(self):
        # 获取当前文件所在的目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.current_dir, "models", "face_detect_v0_n", "model.pt")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            print(f"错误：模型文件不存在于路径：{self.model_path}")
            self.model = None
        else:
            try:
                # 加载YOLO模型
                if has_ultralytics:
                    self.model = YOLO(self.model_path)
                    print("YOLO模型加载成功！")
                else:
                    self.model = None
                    print("未能加载YOLO模型，缺少ultralytics库")
            except Exception as e:
                print(f"加载YOLO模型时出错：{e}")
                self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_factor": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop_head"
    CATEGORY = "图像处理"

    def crop_head(self, image, padding_factor=0.2):
        # 确保模型已加载
        if self.model is None:
            print("YOLO模型未加载，返回原图")
            # 创建一个与原图尺寸相同的全零mask
            if image.dim() == 3:
                image = image.unsqueeze(0)
            batch_size, height, width = image.shape[0], image.shape[1], image.shape[2]
            empty_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
            return (image, empty_mask)
            
        # 确保图像是正确的维度 (batch_size, height, width, channels)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 获取批次大小
        batch_size = image.shape[0]
        
        # 创建一个空列表来存储结果
        result_images = []
        result_masks = []
        
        # 处理每个批次
        for i in range(batch_size):
            # 获取当前图像
            img = image[i]
            
            # 转换为PIL图像
            pil_img = self._tensor_to_pil(img)
            original_width, original_height = pil_img.size
            
            # 创建与原图尺寸相同的mask
            mask = Image.new('L', (original_width, original_height), 0)  # 黑色背景
            
            # 转换为OpenCV格式进行处理
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # 运行YOLO模型进行人脸检测
            detections = self.model(cv_img, verbose=False)
            
            # 提取所有检测到的头部
            heads = []
            for det in detections:
                boxes = det.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 计算边界框面积
                        area = (x2 - x1) * (y2 - y1)
                        heads.append((x1, y1, x2, y2, area, confidence))
            
            if heads:
                # 选择面积最大的头部
                heads.sort(key=lambda x: x[4], reverse=True)
                x1, y1, x2, y2, _, _ = heads[0]
                
                # 计算检测到的头部尺寸
                head_width = x2 - x1
                head_height = y2 - y1
                
                # 应用填充因子来扩展检测框
                padding_x = int(head_width * padding_factor)
                padding_y = int(head_height * padding_factor)
                
                # 计算裁剪坐标（直接使用矩形，不强制正方形）
                crop_x1 = max(0, x1 - padding_x)
                crop_y1 = max(0, y1 - padding_y)
                crop_x2 = min(original_width, x2 + padding_x)
                crop_y2 = min(original_height, y2 + padding_y)
                
                # 在mask上绘制扩展后的裁切区域为白色
                mask_np = np.array(mask)
                mask_np[crop_y1:crop_y2, crop_x1:crop_x2] = 255
                mask = Image.fromarray(mask_np, mode='L')
                
                # 裁剪图像
                cropped_img = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # 转回张量
                result_tensor = self._pil_to_tensor(cropped_img)
                mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)
                
                result_images.append(result_tensor)
                result_masks.append(mask_tensor)
            else:
                # 如果没有检测到头部，返回原图和空mask
                result_images.append(img)
                empty_mask = torch.zeros((original_height, original_width), dtype=torch.float32)
                result_masks.append(empty_mask)
        
        # 将结果堆叠为批次
        if len(result_images) > 0:
            result_tensor = torch.stack(result_images)
            mask_tensor = torch.stack(result_masks)
        else:
            # 如果没有图像处理，返回空张量
            result_tensor = torch.zeros((0, 0, 0, 3), dtype=torch.float32)
            mask_tensor = torch.zeros((0, 0, 0), dtype=torch.float32)
            
        return (result_tensor, mask_tensor)
    
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
