import torch
import requests
import base64
import json
import time
import os
from PIL import Image
import numpy as np
from io import BytesIO

class PIP_Kontext:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "ein fantastisches bild"}),
                "input_image": ("IMAGE",),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 9999999999}),
                "guidance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "aspect_ratio": (["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"], {"default": "1:1"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "图像处理/AI生成"

    def generate_image(self, prompt, input_image, seed=42, guidance=3.0, steps=50, aspect_ratio="1:1", output_format="jpeg", prompt_upsampling=False, safety_tolerance=2):
        try:
            # 获取API密钥
            api_key = self._get_api_key()
            if not api_key:
                raise Exception("API密钥未找到，请在kontext.txt文件中设置您的API密钥")

            # 转换输入图像为base64
            base64_image = self._image_to_base64(input_image)

            # 准备API请求
            url = "https://api.bfl.ai/v1/flux-kontext-pro"
            
            payload = {
                "prompt": prompt,
                "input_image": base64_image,
                "seed": seed,
                "guidance": guidance,
                "steps": steps,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "prompt_upsampling": prompt_upsampling,
                "safety_tolerance": safety_tolerance
            }
            
            headers = {
                "x-key": api_key,
                "Content-Type": "application/json"
            }

            print(f"正在向Kontext API发送请求...")
            
            # 发送初始请求
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
            result = response.json()
            task_id = result.get("id")
            polling_url = result.get("polling_url")
            
            if not task_id or not polling_url:
                raise Exception(f"API响应格式错误: {result}")
            
            print(f"任务已提交，任务ID: {task_id}")
            print(f"正在等待图像生成...")
            
            # 轮询任务状态
            generated_image = self._poll_task_status(polling_url, api_key)
            
            # 转换为ComfyUI张量格式
            image_tensor = self._url_to_tensor(generated_image)
            
            print("图像生成完成！")
            return (image_tensor,)
            
        except Exception as e:
            print(f"Kontext API错误: {str(e)}")
            # 返回一个空白图像作为错误处理
            error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (error_image,)

    def _get_api_key(self):
        """从kontext.txt文件中读取API密钥"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            key_file = os.path.join(script_dir, "kontext.txt")
            
            if os.path.exists(key_file):
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                    return api_key if api_key else None
            return None
        except Exception as e:
            print(f"读取API密钥失败: {str(e)}")
            return None

    def _image_to_base64(self, image_tensor):
        """将ComfyUI图像张量转换为base64字符串"""
        # 确保图像是正确的维度
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # 取第一张图片
        
        # 转换为PIL图像
        pil_image = self._tensor_to_pil(image_tensor)
        
        # 转换为base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        base64_str = base64.b64encode(img_data).decode('utf-8')
        
        return f"data:image/png;base64,{base64_str}"

    def _tensor_to_pil(self, tensor):
        """将张量转换为PIL图像"""
        tensor = tensor.clamp(0, 1)
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='RGB')

    def _poll_task_status(self, polling_url, api_key, max_wait_time=300, poll_interval=5):
        """轮询任务状态直到完成"""
        headers = {"x-key": api_key}
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(polling_url, headers=headers)
                
                if response.status_code != 200:
                    print(f"轮询状态失败: {response.status_code}")
                    time.sleep(poll_interval)
                    continue
                
                result = response.json()
                status = result.get("status")
                
                if status == "Ready":
                    # 任务完成
                    if "result" in result and "sample" in result["result"]:
                        return result["result"]["sample"]
                    else:
                        raise Exception("API返回格式错误：缺少结果图像")
                        
                elif status == "Error":
                    error_msg = result.get("result", {}).get("error", "未知错误")
                    raise Exception(f"任务执行失败: {error_msg}")
                    
                elif status in ["Pending", "Request Moderated"]:
                    # 任务仍在进行中
                    print(f"任务状态: {status}，继续等待...")
                    time.sleep(poll_interval)
                    continue
                    
                else:
                    print(f"未知状态: {status}，继续等待...")
                    time.sleep(poll_interval)
                    continue
                    
            except requests.RequestException as e:
                print(f"轮询请求失败: {str(e)}")
                time.sleep(poll_interval)
                continue
        
        raise Exception(f"任务超时（{max_wait_time}秒）")

    def _url_to_tensor(self, image_url):
        """从URL下载图像并转换为张量"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 从响应内容创建PIL图像
            pil_image = Image.open(BytesIO(response.content))
            
            # 确保是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为张量
            img_np = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_np).unsqueeze(0)  # 添加批次维度
            
            return tensor
            
        except Exception as e:
            print(f"下载图像失败: {str(e)}")
            # 返回空白图像
            return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
