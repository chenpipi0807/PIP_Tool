import json

class PIP_NovelBatchValidator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "placeholder": "粘贴完整的小说批次JSON数据..."
                }),
                "batch_id": ("STRING", {
                    "default": "690",
                    "placeholder": "输入批次ID"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("batch_json",)
    FUNCTION = "extract_batch_json"
    CATEGORY = "小说处理"
    
    def extract_batch_json(self, json_data, batch_id):
        """根据批次ID提取对应的完整JSON对象"""
        try:
            # 解析JSON数据
            data = json.loads(json_data)
            
            # 确保数据是列表格式
            if not isinstance(data, list):
                return (f"错误：JSON数据必须是数组格式，当前是: {type(data).__name__}",)
            
            # 搜索匹配的批次ID
            for item in data:
                if not isinstance(item, dict):
                    continue
                    
                # 检查不同可能的批次ID字段名
                item_batch_id = None
                for key in ["批次ID", "batch_id", "#"]:
                    if key in item:
                        item_batch_id = str(item[key])
                        break
                
                # 如果找到匹配的批次ID
                if item_batch_id == batch_id:
                    # 返回完整的JSON对象（格式化）
                    return (json.dumps(item, ensure_ascii=False, indent=2),)
            
            # 如果没有找到匹配的批次ID
            available_ids = []
            for item in data:
                if isinstance(item, dict):
                    for key in ["批次ID", "batch_id", "#"]:
                        if key in item:
                            available_ids.append(str(item[key]))
                            break
            
            return (f"错误：未找到批次ID '{batch_id}'。\n可用的批次ID: {', '.join(available_ids[:10])}{'...' if len(available_ids) > 10 else ''}",)
            
        except json.JSONDecodeError as e:
            return (f"JSON解析错误: {str(e)}",)
        except Exception as e:
            return (f"处理错误: {str(e)}",)
