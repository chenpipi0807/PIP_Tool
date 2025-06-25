import os
import requests
import json
import hashlib

class PIP_LoadJSONURL:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
        self.cache_dir = os.path.join(self.output_dir, "url_json_cache")
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"multiline": False}),
            },
            "optional": {
                "refresh": ("BUTTON", {"label": "刷新JSON"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("JSON字符串",)
    FUNCTION = "load_json"
    CATEGORY = "PIP工具/网络"
    
    def load_json(self, url, refresh=False):
        try:
            # 为URL创建一个唯一的文件名
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_filename = os.path.join(self.cache_dir, f"{url_hash}.json")
            
            # 如果缓存不存在或者要求刷新，则下载JSON
            if not os.path.exists(cache_filename) or refresh:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # 如果请求不成功则抛出异常
                
                # 验证JSON格式
                json_content = response.text
                # 尝试解析JSON以确保它是有效的
                json.loads(json_content)
                
                # 保存到缓存
                with open(cache_filename, 'w', encoding='utf-8') as f:
                    f.write(json_content)
            else:
                # 从缓存加载JSON
                with open(cache_filename, 'r', encoding='utf-8') as f:
                    json_content = f.read()
            
            return (json_content,)
            
        except json.JSONDecodeError as e:
            print(f"无效的JSON格式: {e}")
            return ("{\"error\": \"无效的JSON格式\"}",)
        except Exception as e:
            print(f"加载URL JSON时出错: {e}")
            return ("{\"error\": \"" + str(e).replace("\"", "'") + "\"}",)
