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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fallback_to_cpu = False  # 标记是否需要使用CPU回退
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            print(f"错误：模型文件不存在于路径：{self.model_path}")
            self.model = None
        else:
            try:
                # 加载YOLO模型
                if has_ultralytics:
                    self.model = YOLO(self.model_path)
                    print(f"YOLO模型加载成功！（设备：{self.device}）")
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
            
            # 运行YOLO模型进行人脸检测（智能设备选择）
            detections = self._smart_inference(cv_img)
            
            # 提取所有检测到的头部
            heads = []
            for det in detections:
                # 检查是否是空检测结果
                if hasattr(det, 'boxes') and det.boxes is not None:
                    boxes = det.boxes
                    if len(boxes) > 0:
                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            # 计算边界框面积
                            area = (x2 - x1) * (y2 - y1)
                            heads.append((x1, y1, x2, y2, area, confidence))
                else:
                    # 处理空检测结果的情况
                    print("检测到空结果，跳过该检测")
                    continue
            
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
    
    def _smart_inference(self, cv_img):
        """
        智能推理：优先使用CUDA，遇到NMS兼容性问题时自动回退到CPU
        如果仍然失败，则使用更低的置信度和兼容性设置
        """
        # 如果已经标记需要使用CPU回退，直接使用CPU
        if self.fallback_to_cpu:
            try:
                return self.model(cv_img, verbose=False, device='cpu')
            except Exception as e:
                # 如果CPU也失败，尝试使用更兼容的设置
                print(f"CPU模式也失败，尝试兼容性设置: {e}")
                return self._fallback_inference(cv_img)
        
        # 尝试使用首选设备（通常是CUDA）
        try:
            return self.model(cv_img, verbose=False, device=self.device)
        except Exception as e:
            # 检查是否是NMS兼容性问题
            error_msg = str(e)
            if "torchvision::nms" in error_msg:
                print(f"检测到NMS兼容性问题，自动回退到CPU模式")
                self.fallback_to_cpu = True
                self.device = 'cpu'  # 更新设备状态
                try:
                    return self.model(cv_img, verbose=False, device='cpu')
                except Exception as cpu_error:
                    print(f"CPU回退也失败，使用兼容性推理: {cpu_error}")
                    return self._fallback_inference(cv_img)
            else:
                # 其他类型的错误，直接抛出
                raise e
    
    def _fallback_inference(self, cv_img):
        """
        终极回退方案：使用更低的设置和兼容性模式
        """
        try:
            # 尝试使用更低的置信度和不同的参数设置
            print("尝试使用兼容性推理设置...")
            # 降低置信度，增加兼容性
            return self.model(cv_img, verbose=False, device='cpu', conf=0.1, iou=0.7, half=False)
        except Exception as e1:
            try:
                # 最后的尝试：使用最基本的设置
                print(f"兼容性推理失败: {e1}，尝试最基本设置...")
                return self.model.predict(cv_img, verbose=False, device='cpu', save=False, conf=0.25)
            except Exception as e2:
                # 如果所有方法都失败，返回空结果但不崩溃
                print(f"所有推理方法都失败: {e2}，返回空检测结果")
                # 创建一个空的检测结果对象，防止后续代码崩溃
                class EmptyResult:
                    def __init__(self):
                        self.boxes = None
                        self.xyxy = []
                        
                    def __len__(self):
                        return 0
                        
                    def __getitem__(self, index):
                        return []
                
                return [EmptyResult()]
