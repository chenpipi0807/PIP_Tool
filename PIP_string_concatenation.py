import torch

class PIP_StringConcatenation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": "", "multiline": True}),
                "string2": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "string3": ("STRING", {"default": "", "multiline": True}),
                "string4": ("STRING", {"default": "", "multiline": True}),
                "string5": ("STRING", {"default": "", "multiline": True}),
                "auto_add_commas": (["开", "关"], {"default": "开"}),
                "remove_newlines": (["开", "关"], {"default": "关"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_string",)
    FUNCTION = "concatenate_strings"
    CATEGORY = "字符串处理"

    def concatenate_strings(self, string1, string2, string3="", string4="", string5="", auto_add_commas="开", remove_newlines="关"):
        # 确保所有输入都是字符串类型
        strings = [str(string1), str(string2), str(string3), str(string4), str(string5)]
        
        # 如果开启了换行符移除功能，移除所有换行符
        if remove_newlines == "开":
            strings = [s.replace('\n', '').replace('\r', '') for s in strings]
        
        # 检查并处理逗号（只检查前三个字符串）
        if auto_add_commas == "开":
            for i in range(min(2, len(strings) - 1)):  # 只处理前两个字符串和下一个的关系
                if strings[i] and strings[i+1]:  # 确保两个字符串都非空
                    if not strings[i].rstrip().endswith(',') and strings[i+1].strip():
                        strings[i] = strings[i].rstrip() + ','
        
        # 拼接非空字符串
        result = ""
        for s in strings:
            if s:  # 只拼接非空字符串
                result += s
        
        return (result,)
