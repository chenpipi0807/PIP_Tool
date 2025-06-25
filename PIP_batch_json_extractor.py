import json

class PIP_batchJSONExtractor:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True}),
                "batch_id": ("STRING", {"default": "1"}),
                "key1": ("STRING", {"default": ""}),
            },
            "optional": {
                "key2": ("STRING", {"default": ""}),
                "key3": ("STRING", {"default": ""}),
                "key4": ("STRING", {"default": ""}),
                "key5": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("值1", "值2", "值3", "值4", "值5")
    FUNCTION = "extract_from_json"
    CATEGORY = "PIP工具/数据处理"
    
    def extract_from_json(self, json_input, batch_id, key1, key2="", key3="", key4="", key5=""):
        try:
            # 处理JSON输入（可能是文件路径或JSON字符串）
            json_data = None
            
            # 尝试作为JSON字符串解析
            try:
                json_data = json.loads(json_input)
            except json.JSONDecodeError:
                # 如果不是有效的JSON字符串，尝试作为文件路径
                try:
                    with open(json_input, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except (FileNotFoundError, IsADirectoryError, PermissionError):
                    return f"JSON解析错误：无法解析输入", "", "", "", ""
            
            # 确保json_data是列表
            if not isinstance(json_data, list):
                json_data = [json_data]
                
            # 查找匹配批次ID的对象
            target_item = None
            for item in json_data:
                if isinstance(item, dict) and item.get("批次ID") == batch_id:
                    target_item = item
                    break
            
            if not target_item:
                return f"未找到批次ID为{batch_id}的数据", "", "", "", ""
            
            # 提取请求的键值
            results = [""] * 5
            keys = [key1, key2, key3, key4, key5]
            
            for i, key in enumerate(keys):
                if key:  # 如果键名不为空
                    if key in target_item:
                        # 将值转换为字符串
                        value = target_item[key]
                        results[i] = str(value)
                    else:
                        results[i] = f"键'{key}'不存在"
            
            return tuple(results)
            
        except Exception as e:
            print(f"JSON提取错误: {str(e)}")
            return f"处理时出错: {str(e)}", "", "", "", ""
