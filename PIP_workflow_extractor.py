import json

class PIP_WorkflowExtractor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "粘贴工作流JSON数据到这里"
                    }
                ),
                "node_id_1": ("STRING", {"default": "none"}),
                "key_1": ("STRING", {"default": "none"}),
                "node_id_2": ("STRING", {"default": "none"}),
                "key_2": ("STRING", {"default": "none"}),
                "node_id_3": ("STRING", {"default": "none"}),
                "key_3": ("STRING", {"default": "none"}),
                "node_id_4": ("STRING", {"default": "none"}),
                "key_4": ("STRING", {"default": "none"}),
                "node_id_5": ("STRING", {"default": "none"}),
                "key_5": ("STRING", {"default": "none"}),
            },
            "optional": {
                "refresh": ("BUTTON", {"label": "刷新JSON数据"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("变量1", "变量2", "变量3", "变量4", "变量5")
    FUNCTION = "extract_values"
    CATEGORY = "PIP工具/JSON处理"

    def extract_values(self, json_data, node_id_1, key_1, node_id_2, key_2, node_id_3, key_3, node_id_4, key_4, node_id_5, key_5, refresh=False):
        # 解析JSON数据
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            return ("无效的JSON数据",) * 5

        # Helper function to extract value
        def extract_value(node_id, key):
            if node_id != "none" and key != "none" and node_id in data:
                node = data[node_id]
                if 'inputs' in node and key in node['inputs']:
                    return node['inputs'][key]
                elif key in node:
                    return node[key]
                elif 'class_type' in node and key == 'class_type':
                    return node['class_type']
                elif key == 'all_inputs':
                    return json.dumps(node['inputs']) if 'inputs' in node else "{}"
            return "未找到键值"

        # Extract values for each node_id and key pair
        results = [
            extract_value(node_id_1, key_1),
            extract_value(node_id_2, key_2),
            extract_value(node_id_3, key_3),
            extract_value(node_id_4, key_4),
            extract_value(node_id_5, key_5)
        ]

        return tuple(results)
